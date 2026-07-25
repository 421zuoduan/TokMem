#!/usr/bin/env python3
"""Generate per-sample compositional predictions from a saved TokMem checkpoint."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
sys.path.insert(0, str(COMPOSITIONAL_DIR))

from eval import calculate_tool_metrics, compare_function_calls_advanced  # noqa: E402
from backbone_registry import resolve_function_calling_model_class  # noqa: E402
from backbone_prompting import format_user_assistant_prompt  # noqa: E402
from checkpoint_io import (  # noqa: E402
    checkpoint_tool_names,
    load_checkpoint_into_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a saved compositional TokMem checkpoint over a JSON test split."
    )
    parser.add_argument("--run-config", required=True, help="Path to run_config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to round checkpoint .pt")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--data-path", default=None, help="Override test JSON path")
    parser.add_argument("--method", default=None, help="Method label stored in each record")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Generation limit override. By default, reuse max_new_tokens from "
            "the training run config (or 256 for older configs)."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def resolve_data_path(run_config, override):
    if override:
        return Path(override)

    args = run_config["args"]
    data_dir = Path(args["data_dir"])
    rounds = run_config.get("rounds") or []
    if rounds:
        tools_range = rounds[-1]["tools"]
    else:
        tools_range = args["training_rounds"].split(",")[-1].split(":", 1)[0]

    per_round_calls = args.get("test_max_function_calls_per_round")
    if per_round_calls:
        test_calls = int(
            [value.strip() for value in per_round_calls.split(",") if value.strip()][-1]
        )
    else:
        test_calls = int(args.get("test_max_function_calls", 4))
    return data_dir / "test" / f"function_calling_test_tools{tools_range}_{test_calls}calls.json"


def resolve_max_new_tokens(run_config, override):
    if override is not None:
        return override
    return int(run_config["args"].get("max_new_tokens", 256))


def torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def build_lora_config(args):
    if not args.get("use_lora", False):
        return None
    config = {
        "r": int(args.get("lora_r", 8)),
        "alpha": int(args.get("lora_alpha", 32)),
        "dropout": float(args.get("lora_dropout", 0.1)),
        "target_modules": [
            item.strip()
            for item in args.get("lora_target_modules", "o_proj").split(",")
        ],
    }
    if args.get("lora_layer_indices") is not None:
        config["layer_indices"] = [
            int(item.strip())
            for item in args["lora_layer_indices"].split(",")
        ]
    return config


def build_model(run_config, checkpoint, tokenizer, device, dtype):
    args = run_config["args"]
    tools = checkpoint_tool_names(
        checkpoint,
        fallback=checkpoint.get("tools"),
    )
    model_class = resolve_function_calling_model_class(args["model_name"])
    model = model_class(
        model_name=args["model_name"],
        num_tools=len(tools),
        tool_names=tools,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        decouple_embeddings=bool(args.get("decouple_embeddings", False)),
        lora_config=build_lora_config(args),
        use_eoc=bool(args.get("use_eoc", False)),
        use_logit_bias=bool(args.get("use_logit_bias", False)),
        use_tool_head_replacement=bool(args.get("use_tool_head_replacement", False)),
        use_memory_bank_constraint=bool(
            args.get("use_memory_bank_constraint", False)
        ),
        memory_bank_probability_threshold=float(
            args.get("memory_bank_probability_threshold", 0.5)
        ),
        logit_bias_network=args.get("logit_bias_network", "linear"),
        logit_bias_scale=float(args.get("logit_bias_scale", 1.0)),
    )
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return model


def generate_one(model, tokenizer, item, device, max_new_tokens):
    user_text = format_user_assistant_prompt(
        tokenizer,
        item["user_input"],
        model=model,
    )
    encoded = tokenizer(user_text, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        return model.generate_with_tool_prediction(
            encoded["input_ids"],
            encoded["attention_mask"],
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=False,
        )[0]


def prediction_record(index, item, result, tokenizer, method, candidate_tools):
    expected_tools = item.get("tools", [item.get("tool_name", "unknown")])
    expected_calls = item.get("function_calls", [item.get("function_call", "{}")])
    predicted_tools = [tool_info["tool_name"] for tool_info in result.get("predicted_tools", [])]
    predicted_calls = result.get("function_calls", [])
    tool_tokens = [
        tokenizer.decode([tool_info["token_id"]])
        for tool_info in result.get("predicted_tools", [])
    ]

    call_eval = compare_function_calls_advanced(
        predicted_calls,
        expected_calls,
        ignore_order=True,
    )
    tool_metrics = calculate_tool_metrics(
        predicted_tools=predicted_tools,
        expected_tools=expected_tools,
        candidate_tools=candidate_tools,
    )

    return {
        "index": index,
        "method": method,
        "user_input": item["user_input"],
        "expected_tools": expected_tools,
        "expected_calls": expected_calls,
        "predicted_tools": predicted_tools,
        "predicted_calls": predicted_calls,
        "tool_tokens": tool_tokens,
        "full_generated_sequence": result.get("full_generated_sequence", ""),
        "tool_sequence_exact": predicted_tools == expected_tools,
        "tool_multiset_exact": bool(tool_metrics["tool_exact_match_acc"] >= 1.0),
        "call_exact": bool(call_eval.exact_match),
        "f1": call_eval.f1_score,
        "tool_f1": tool_metrics["tool_f1_score"],
        "parse_errors": call_eval.details.get("parse_errors", {}),
    }


def main():
    args = parse_args()
    run_config = load_json(args.run_config)
    data_path = resolve_data_path(run_config, args.data_path)
    max_new_tokens = resolve_max_new_tokens(run_config, args.max_new_tokens)
    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    method = args.method or run_config.get("run_name") or Path(args.run_config).parent.name
    dtype = torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        run_config["args"]["model_name"],
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(run_config, checkpoint, tokenizer, args.device, dtype)
    del checkpoint
    gc.collect()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_tools = list(getattr(model, "tool_names", []))

    with open(output_path, "w") as handle:
        for index, item in enumerate(data):
            result = generate_one(
                model,
                tokenizer,
                item,
                args.device,
                max_new_tokens,
            )
            record = prediction_record(index, item, result, tokenizer, method, candidate_tools)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(f"Generated {index + 1}/{len(data)} predictions")

    print(f"Wrote {len(data)} predictions to {output_path}")


if __name__ == "__main__":
    main()
