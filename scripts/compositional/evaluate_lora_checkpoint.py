#!/usr/bin/env python3
"""Evaluate a saved compositional PEFT adapter on its pretrained backbone."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
sys.path.insert(0, str(COMPOSITIONAL_DIR))

from checkpoint_io import load_lora_adapter  # noqa: E402
from lora_sequential import (  # noqa: E402
    FunctionCallingDataset,
    collate_fn,
    eval_lora_model,
)
from run_layout import write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a pretrained backbone plus a saved LoRA adapter and run the "
            "maintained compositional evaluation."
        )
    )
    parser.add_argument("--run-config", required=True, help="Training run_config.json")
    parser.add_argument(
        "--adapter-checkpoint",
        required=True,
        help="PEFT adapter directory produced by lora_sequential.py",
    )
    parser.add_argument("--output", required=True, help="Evaluation JSON output path")
    parser.add_argument("--data-path", default=None, help="Optional test split override")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def final_round_spec(run_config):
    run_args = run_config["args"]
    rounds = run_config.get("rounds") or []
    if rounds:
        tools_range = rounds[-1]["tools"]
    else:
        tools_range = (
            run_args["training_rounds"].split(",")[-1].split(":", 1)[0]
        )

    per_round_calls = run_args.get("test_max_function_calls_per_round")
    if per_round_calls:
        test_calls = int(
            [
                value.strip()
                for value in per_round_calls.split(",")
                if value.strip()
            ][-1]
        )
    else:
        test_calls = int(run_args.get("test_max_function_calls", 4))
    return tools_range, test_calls


def resolve_data_path(run_config, override):
    if override:
        return Path(override)
    tools_range, test_calls = final_round_spec(run_config)
    return (
        Path(run_config["args"]["data_dir"])
        / "test"
        / f"function_calling_test_tools{tools_range}_{test_calls}calls.json"
    )


def torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    run_config = load_json(args.run_config)
    run_args = run_config["args"]
    model_name = run_args["model_name"]
    data_path = resolve_data_path(run_config, args.data_path)
    if not data_path.is_file():
        raise SystemExit(f"Test split not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "local_files_only": True,
    }
    if args.device == "cpu":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).to("cpu")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **model_kwargs,
        )
    model = load_lora_adapter(
        base_model,
        args.adapter_checkpoint,
        expected_base_model_name=model_name,
        is_trainable=False,
    )
    model.eval()

    max_length = args.max_length or int(run_args.get("max_length", 512))
    eval_batch_size = args.eval_batch_size or int(
        run_args.get("eval_batch_size", 32)
    )
    dataset = FunctionCallingDataset(
        str(data_path),
        tokenizer,
        max_length,
        "eval",
        model_type=base_model.config.model_type,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    metrics = eval_lora_model(
        model,
        tokenizer,
        dataloader,
        device=args.device,
    )

    output_path = Path(args.output).resolve()
    write_json(
        str(output_path),
        {
            "experiment_type": "lora_checkpoint_evaluation",
            "run_config": str(Path(args.run_config).resolve()),
            "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
            "base_model": model_name,
            "data_path": str(data_path.resolve()),
            "metrics": metrics,
        },
    )
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
