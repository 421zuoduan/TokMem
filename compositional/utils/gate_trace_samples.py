#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compositional.dataset import NativeFunctionCallingDataset
from compositional.eval import calculate_argument_accuracy, compare_function_calls_advanced
from compositional.utils.js_explore import (
    find_run_config_path,
    load_checkpoint_bundle,
    resolve_test_data_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace per-site gate probabilities for fully-correct and not-fully-correct samples.",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to round checkpoint")
    parser.add_argument("--run_config_path", type=str, default=None, help="Optional explicit run_config.json path")
    parser.add_argument("--data_path", type=str, default=None, help="Optional explicit JSON data path")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path. Defaults next to the checkpoint as gate_trace_samples.json",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Optional model path override")
    parser.add_argument("--device", type=str, default=None, help="Device override, default: cuda if available else cpu")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype override, e.g. bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="balanced",
        choices=["balanced", "full_correct", "not_full_correct", "all"],
        help="Which sample buckets to export",
    )
    parser.add_argument(
        "--max_samples_per_bucket",
        type=int,
        default=5,
        help="Max samples to export per correctness bucket when selection_mode=balanced",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Max samples to export for single-bucket or all-samples mode",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("gate_trace_samples")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def to_abs_repo_path(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_sample_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    tools = sample.get("tools", [])
    function_calls = sample.get("function_calls", [])
    if not isinstance(tools, list):
        tools = [tools]
    if not isinstance(function_calls, list):
        function_calls = [function_calls]
    return {
        **sample,
        "tools": tools,
        "function_calls": function_calls,
    }


def build_dataset_item(
    sample: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    mode: str,
    max_length: int,
) -> Dict[str, Any]:
    dataset = NativeFunctionCallingDataset(
        data_path=None,
        tokenizer=tokenizer,
        max_length=max_length,
        model=model,
        mode=mode,
        use_eoc=bool(getattr(model, "use_eoc", False)),
    )
    dataset.data = [sample]
    return dataset[0]


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def is_tool_token_id(model: Any, token_id: int) -> bool:
    return bool(token_id in getattr(model, "token_id_to_tool_id", {}))


def build_gate_site_records(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    hidden_states: torch.Tensor,
) -> List[Dict[str, Any]]:
    eoc_token_id = getattr(model, "eoc_token_id", None)
    if eoc_token_id is None or getattr(model, "routing_probe", None) is None:
        return []

    probe_from = getattr(model, "probe_from", "eoc")
    shift_labels = labels[1:]
    valid_mask = shift_labels != -100
    records: List[Dict[str, Any]] = []
    eoc_ordinal = 0

    def add_site(site_type: str, boundary_position: int, next_token_id: int, eoc_index: Optional[int]) -> None:
        source_position = boundary_position if probe_from == "eoc" else boundary_position + 1
        if source_position >= hidden_states.size(0):
            return
        routing_hidden = hidden_states[source_position].unsqueeze(0)
        routing_logit = float(model._get_routing_probe_scores(routing_hidden).item())
        routing_prob = 1.0 / (1.0 + math.exp(-routing_logit))
        target_is_tool = is_tool_token_id(model, next_token_id)
        source_token_id = int(input_ids[source_position].item())
        pred_is_tool = routing_prob >= float(getattr(model, "gate_threshold", 0.5))
        record = {
            "site_index": len(records),
            "site_type": site_type,
            "eoc_index": eoc_index,
            "boundary_position": int(boundary_position),
            "source_position": int(source_position),
            "source_token_id": source_token_id,
            "source_token_text": decode_token(tokenizer, source_token_id),
            "next_gold_token_id": int(next_token_id),
            "next_gold_token_text": decode_token(tokenizer, next_token_id),
            "next_gold_is_tool": bool(target_is_tool),
            "next_gold_tool_name": (
                model.get_tool_name_from_token_id(next_token_id) if target_is_tool else None
            ),
            "logit": routing_logit,
            "prob": routing_prob,
            "pred_is_tool": bool(pred_is_tool),
            "is_correct": bool(pred_is_tool == target_is_tool),
        }
        records.append(record)

    valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_positions.numel() > 0:
        initial_pos = int(valid_positions[0].item())
        if labels[initial_pos].item() == -100:
            next_token_id = int(shift_labels[initial_pos].item())
            if next_token_id != -100:
                add_site("bos", initial_pos, next_token_id, None)

    eoc_positions = torch.nonzero(
        (labels[:-1] == eoc_token_id) & valid_mask,
        as_tuple=False,
    ).flatten()
    for position in eoc_positions.tolist():
        next_token_id = int(shift_labels[position].item())
        if next_token_id == -100:
            continue
        add_site("after_eoc", position, next_token_id, eoc_ordinal)
        eoc_ordinal += 1

    return records


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    sample: Dict[str, Any],
    device: str,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> Dict[str, Any]:
    eval_item = build_dataset_item(
        sample=sample,
        tokenizer=tokenizer,
        model=model,
        mode="eval",
        max_length=max_length,
    )
    input_ids = eval_item["input_ids"].unsqueeze(0).to(device)
    attention_mask = eval_item["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        batch_results = model.generate_with_tool_prediction(
            user_tokens=input_ids,
            user_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    result = batch_results[0]
    predicted_tools = [tool_info["tool_name"] for tool_info in result.get("predicted_tools", [])]
    predicted_calls = list(result.get("function_calls", []))
    expected_tools = list(sample["tools"])
    expected_calls = list(sample["function_calls"])

    tool_match = Counter(predicted_tools) == Counter(expected_tools)
    call_eval = compare_function_calls_advanced(predicted_calls, expected_calls, ignore_order=True)
    argument_eval = calculate_argument_accuracy(predicted_calls, expected_calls)

    return {
        "predicted_tools": predicted_tools,
        "predicted_calls": predicted_calls,
        "full_generated_sequence": result.get("full_generated_sequence", ""),
        "tool_match": bool(tool_match),
        "full_correct": bool(call_eval.exact_match),
        "f1_score": float(call_eval.f1_score),
        "precision": float(call_eval.precision),
        "recall": float(call_eval.recall),
        "arguments_accuracy": float(argument_eval["arguments_accuracy"]),
        "matched_arguments": int(argument_eval["matched_arguments"]),
        "total_target_arguments": int(argument_eval["total_target_arguments"]),
        "parse_errors": int(call_eval.details.get("parse_errors", {}).get("outputs", 0)),
    }


def collect_sample_record(
    model: Any,
    tokenizer: Any,
    sample: Dict[str, Any],
    sample_index: int,
    generation: Dict[str, Any],
    device: str,
    max_length: int,
) -> Dict[str, Any]:
    train_item = build_dataset_item(
        sample=sample,
        tokenizer=tokenizer,
        model=model,
        mode="train",
        max_length=max_length,
    )
    input_ids = train_item["input_ids"].to(device)
    attention_mask = train_item["attention_mask"].to(device)
    labels = train_item["labels"].to(device)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states[-1][0]
    gate_sites = build_gate_site_records(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        labels=labels,
        hidden_states=hidden_states,
    )

    return {
        "sample_index": int(sample_index),
        "user_input": sample["user_input"],
        "gold_tools": list(sample["tools"]),
        "gold_function_calls": list(sample["function_calls"]),
        "gold_call_count": int(len(sample["tools"])),
        "correctness_bucket": "full_correct" if generation["full_correct"] else "not_full_correct",
        "generation": generation,
        "gate_site_count": int(len(gate_sites)),
        "gate_sites": gate_sites,
    }


def should_keep_sample(
    selection_mode: str,
    counts: Dict[str, int],
    generation_full_correct: bool,
    max_samples_per_bucket: int,
    max_samples: int,
) -> bool:
    bucket = "full_correct" if generation_full_correct else "not_full_correct"
    if selection_mode == "balanced":
        return counts[bucket] < max_samples_per_bucket
    if selection_mode == "full_correct":
        return generation_full_correct and counts[bucket] < max_samples
    if selection_mode == "not_full_correct":
        return (not generation_full_correct) and counts[bucket] < max_samples
    total = counts["full_correct"] + counts["not_full_correct"]
    return total < max_samples


def is_collection_complete(
    selection_mode: str,
    counts: Dict[str, int],
    max_samples_per_bucket: int,
    max_samples: int,
) -> bool:
    if selection_mode == "balanced":
        return (
            counts["full_correct"] >= max_samples_per_bucket
            and counts["not_full_correct"] >= max_samples_per_bucket
        )
    if selection_mode == "full_correct":
        return counts["full_correct"] >= max_samples
    if selection_mode == "not_full_correct":
        return counts["not_full_correct"] >= max_samples
    return (counts["full_correct"] + counts["not_full_correct"]) >= max_samples


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    checkpoint_path = to_abs_repo_path(args.checkpoint_path)
    if checkpoint_path is None:
        raise ValueError("--checkpoint_path is required")

    run_config_path = find_run_config_path(
        checkpoint_path=checkpoint_path,
        explicit_run_config_path=to_abs_repo_path(args.run_config_path),
    )
    run_args = None
    if run_config_path is not None and run_config_path.exists():
        run_args = json.loads(run_config_path.read_text(encoding="utf-8")).get("args", {})

    data_path = to_abs_repo_path(args.data_path)
    if data_path is None:
        data_path = resolve_test_data_path(checkpoint_path, None, run_args)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = (
        to_abs_repo_path(args.output_path)
        if args.output_path is not None
        else checkpoint_path.parent / "gate_trace_samples.json"
    )

    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Run config: %s", run_config_path)
    logger.info("Data path: %s", data_path)
    logger.info("Output path: %s", output_path)

    model, tokenizer, checkpoint_payload, samples, loaded_run_args = load_checkpoint_bundle(
        checkpoint_path=checkpoint_path,
        run_config_path=run_config_path,
        test_data_path=data_path,
        device=device,
        dtype_name=args.dtype,
        model_name_override=args.model_name,
        logger=logger,
    )
    max_length = int((loaded_run_args or {}).get("max_length", 1024))

    samples = [normalize_sample_fields(sample) for sample in samples]
    sample_indices = list(range(len(samples)))
    rng = random.Random(args.seed)
    rng.shuffle(sample_indices)

    counts = {"full_correct": 0, "not_full_correct": 0}
    selected_records: List[Dict[str, Any]] = []

    for offset, sample_index in enumerate(sample_indices, start=1):
        sample = samples[sample_index]
        generation = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            device=device,
            max_length=max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        if not should_keep_sample(
            selection_mode=args.selection_mode,
            counts=counts,
            generation_full_correct=generation["full_correct"],
            max_samples_per_bucket=args.max_samples_per_bucket,
            max_samples=args.max_samples,
        ):
            continue

        record = collect_sample_record(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            sample_index=sample_index,
            generation=generation,
            device=device,
            max_length=max_length,
        )
        bucket = record["correctness_bucket"]
        counts[bucket] += 1
        selected_records.append(record)
        logger.info(
            "Collected %s sample %d/%d: dataset_index=%d, gate_sites=%d",
            bucket,
            counts[bucket],
            args.max_samples_per_bucket if args.selection_mode == "balanced" else args.max_samples,
            sample_index,
            record["gate_site_count"],
        )
        if is_collection_complete(
            selection_mode=args.selection_mode,
            counts=counts,
            max_samples_per_bucket=args.max_samples_per_bucket,
            max_samples=args.max_samples,
        ):
            break
        if offset % 25 == 0:
            logger.info(
                "Scanned %d/%d samples, collected full_correct=%d, not_full_correct=%d",
                offset,
                len(sample_indices),
                counts["full_correct"],
                counts["not_full_correct"],
            )

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "run_config_path": str(run_config_path) if run_config_path is not None else None,
        "data_path": str(data_path),
        "selection_mode": args.selection_mode,
        "max_samples_per_bucket": int(args.max_samples_per_bucket),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "device": device,
        "max_length": int(max_length),
        "generation_config": {
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "do_sample": bool(args.do_sample),
        },
        "model_config": {
            "use_eoc": bool(getattr(model, "use_eoc", False)),
            "use_gate": bool(getattr(model, "use_gate", False)),
            "probe_from": getattr(model, "probe_from", None),
            "gate_threshold": float(getattr(model, "gate_threshold", 0.5)),
            "gate_network": getattr(model, "gate_network", None),
        },
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "collected_counts": counts,
        "samples": selected_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d sample records to %s",
        len(selected_records),
        output_path,
    )


if __name__ == "__main__":
    main()
