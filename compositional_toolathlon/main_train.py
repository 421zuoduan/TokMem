#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from backbone_registry import resolve_function_calling_model_class  # noqa: E402
from checkpoint_io import build_checkpoint_payload  # noqa: E402

from compositional_toolathlon.audit_dataset import episode_id_hash  # noqa: E402
from compositional_toolathlon.episode_to_steps import (  # noqa: E402
    record_content_hash,
)
from compositional_toolathlon.dataset import (  # noqa: E402
    create_step_dataloader,
    read_jsonl,
)
from compositional_toolathlon.manifest import load_manifest  # noqa: E402
from compositional_toolathlon.target_policy import (  # noqa: E402
    TARGET_POLICY_HASH,
    TARGET_POLICY_NAME,
    TARGET_POLICY_VERSION,
    TARGET_TOOL_NAMES,
    resolve_target_tool_ids,
)
from compositional_toolathlon.training import train_stepwise_model  # noqa: E402


METHOD_FLAGS = {
    "tokmem": {"use_eoc": False, "use_logit_bias": False},
    "eoc_only": {"use_eoc": True, "use_logit_bias": False},
    "tapmem": {"use_eoc": True, "use_logit_bias": True},
}


def validate_data_audit_for_training(
    audit_payload: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if audit_payload.get("passed") is not True:
        raise ValueError("training requires a passing dataset coverage audit")
    if audit_payload.get("schema_version") != 2:
        raise ValueError(
            "training requires dataset coverage audit schema_version=2; "
            "rerun audit_dataset so coverage is checked on train only"
        )
    thresholds = audit_payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("dataset coverage audit requires thresholds")
    if thresholds.get("successful_episode_scope") != "train":
        raise ValueError("dataset coverage audit must count successful episodes on train")
    for field in (
        "min_successful_episodes",
        "min_argument_shapes",
        "min_templates",
    ):
        value = thresholds.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"dataset coverage audit requires positive threshold {field}"
            )
    if thresholds["min_successful_episodes"] < 3:
        raise ValueError(
            "dataset coverage audit must require at least 3 successful train "
            "episodes per target tool"
        )
    if thresholds.get("require_distractor_role") is not True:
        raise ValueError(
            "dataset coverage audit must require every target tool to appear "
            "as a train distractor"
        )
    execution_requirements = audit_payload.get("execution_requirements")
    if (
        not isinstance(execution_requirements, dict)
        or execution_requirements.get("real_execution") is not True
    ):
        raise ValueError(
            "dataset coverage audit must require real execution evidence"
        )
    if audit_payload.get("tool_manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("dataset audit and training manifest hashes differ")

    expected_policy_fields = {
        "target_policy_name": TARGET_POLICY_NAME,
        "target_policy_version": TARGET_POLICY_VERSION,
        "target_policy_hash": TARGET_POLICY_HASH,
        "target_tool_count": len(TARGET_TOOL_NAMES),
    }
    mismatched_policy_fields = sorted(
        field
        for field, expected in expected_policy_fields.items()
        if audit_payload.get(field) != expected
    )
    if mismatched_policy_fields:
        raise ValueError(
            "dataset coverage audit does not use the frozen target policy: "
            + ", ".join(mismatched_policy_fields)
        )

    expected_target_ids = set(resolve_target_tool_ids(manifest))
    per_tool = audit_payload.get("per_tool")
    if not isinstance(per_tool, dict) or set(per_tool) != expected_target_ids:
        raise ValueError(
            "dataset coverage audit per_tool denominator differs from frozen policy"
        )


def validate_step_records_for_split(
    audit_payload: dict[str, Any],
    split: str,
    step_records: list[dict[str, Any]],
) -> None:
    expected_split_hashes = audit_payload.get("split_episode_id_hashes", {})
    expected_step_hashes = audit_payload.get("split_step_content_hashes", {})
    expected_step_counts = audit_payload.get("split_step_counts", {})
    actual_episode_hash = episode_id_hash(
        {record["episode_id"] for record in step_records}
    )
    if expected_split_hashes.get(split) != actual_episode_hash:
        raise ValueError(
            f"{split} steps do not match the episodes approved by data audit"
        )
    if expected_step_counts.get(split) != len(step_records):
        raise ValueError(
            f"{split} step count does not match the approved data audit"
        )
    if expected_step_hashes.get(split) != record_content_hash(step_records):
        raise ValueError(
            f"{split} step content does not match the approved data audit"
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def initialize_memory_rows(model: Any) -> None:
    parameters = (
        [
            model.trainable_tool_input_embeddings,
            model.trainable_tool_output_embeddings,
        ]
        if model.decouple_embeddings
        else [model.trainable_tool_embeddings]
    )
    seen = set()
    with torch.no_grad():
        for parameter in parameters:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            original_dtype = parameter.dtype
            initialized = parameter.float()
            torch.nn.init.orthogonal_(initialized)
            parameter.copy_(initialized.to(original_dtype))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TokMem/EOC-only/TapMem on verified Toolathlon step data"
    )
    parser.add_argument("--method", choices=sorted(METHOD_FLAGS), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-steps", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--decouple-embeddings", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logit-bias-loss-weight", type=float, default=0.1)
    parser.add_argument("--logit-bias-scale", type=float, default=1.0)
    parser.add_argument("--logit-bias-network", choices=("linear", "mlp"), default="linear")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")

    set_seed(args.seed)
    flags = METHOD_FLAGS[args.method]
    manifest = load_manifest(args.manifest)
    audit_path = Path(args.data_audit)
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_data_audit_for_training(audit_payload, manifest)
    train_records = read_jsonl(args.train_steps)
    validate_step_records_for_split(audit_payload, "train", train_records)
    validate_step_records_for_split(
        audit_payload,
        "validation",
        [],
    )
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    tool_names = [record["stable_id"] for record in manifest["tools"]]
    capacity = 247 if flags["use_eoc"] else 248
    if len(tool_names) > capacity:
        raise ValueError(
            f"{args.method} supports at most {capacity} tools, manifest has {len(tool_names)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    model_class = resolve_function_calling_model_class(args.model_name)
    model = model_class(
        model_name=args.model_name,
        num_tools=len(tool_names),
        tool_names=tool_names,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype_from_name(args.dtype),
        decouple_embeddings=args.decouple_embeddings,
        lora_config=None,
        use_eoc=flags["use_eoc"],
        use_logit_bias=flags["use_logit_bias"],
        use_tool_head_replacement=False,
        logit_bias_network=args.logit_bias_network,
        logit_bias_scale=args.logit_bias_scale,
    )
    initialize_memory_rows(model)

    train_loader = create_step_dataloader(
        args.train_steps,
        args.manifest,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_eoc=flags["use_eoc"],
        mode="train",
        shuffle=True,
        balance_by_episode=False,
        sampler_seed=args.seed,
    )
    train_metrics = train_stepwise_model(
        model=model,
        dataloader=train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_logit_bias=flags["use_logit_bias"],
        use_logit_train_add=flags["use_logit_bias"],
        detach=True,
        logit_bias_loss_weight=args.logit_bias_loss_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    episode_acceptance = {}
    evaluator_status = {}
    target_call_status = Counter()
    for record in train_records:
        episode_id = record["episode_id"]
        episode_acceptance[episode_id] = bool(
            record.get("episode_accepted", True)
        )
        evaluator_status[episode_id] = bool(
            record.get("episode_evaluator_passed", True)
        )
        target_call_status[
            "successful" if record.get("target_call_success", True) else "failed"
        ] += 1
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": args.method,
        "model_name": os.path.realpath(args.model_name),
        "manifest": os.path.realpath(args.manifest),
        "manifest_hash": manifest["manifest_hash"],
        "tool_count": len(tool_names),
        "train_steps": os.path.realpath(args.train_steps),
        "train_step_count": len(train_records),
        "train_episode_count": len(
            {record["episode_id"] for record in train_records}
        ),
        "steps_per_epoch": len(train_records),
        "data_audit": os.path.realpath(args.data_audit),
        "data_audit_sha256": audit_sha256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "decouple_embeddings": args.decouple_embeddings,
        "use_eoc": flags["use_eoc"],
        "use_logit_bias": flags["use_logit_bias"],
        "use_logit_train_add": flags["use_logit_bias"],
        "detach": True,
        "logit_bias_network": args.logit_bias_network,
        "logit_bias_scale": args.logit_bias_scale,
        "logit_bias_loss_weight": args.logit_bias_loss_weight,
        "lora_config": None,
        "interface": "no-tool-doc-memory-token",
        "training_unit": "next_tool_call_step",
        "sampling_policy": "shuffle_without_replacement_per_step",
        "validation_policy": "none",
        "checkpoint_selection_policy": "final_epoch",
        "training_episode_policy": audit_payload.get(
            "training_episode_policy",
            {
                "include_rejected": False,
                "include_failed_calls": False,
            },
        ),
        "episode_acceptance_counts": audit_payload.get(
            "episode_acceptance_counts",
            {},
        ),
        "evaluator_status_counts": {
            "passed": sum(evaluator_status.values()),
            "failed": len(evaluator_status) - sum(evaluator_status.values()),
        },
        "target_call_status_counts": {
            "successful": target_call_status["successful"],
            "failed": target_call_status["failed"],
        },
        "train_step_content_hash": record_content_hash(train_records),
        "optimizer": "AdamW",
        "optimizer_weight_decay": 0.0,
        "scheduler": "linear",
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": train_metrics["optimizer_steps"],
        "warmup_steps": train_metrics["optimizer_steps"] // 10,
        "checkpoint_format": "trainable_only",
        "frozen_backbone_saved": False,
        "optimizer_state_saved": False,
        "scheduler_state_saved": False,
    }
    write_json(run_dir / "run_config.json", run_config)
    results = {"training": train_metrics}
    write_json(run_dir / "metrics.json", results)

    checkpoint = build_checkpoint_payload(
        model,
        round_number=int(
            train_metrics["checkpoint_selection"]["best_epoch"] or args.epochs
        ),
        round_tools=tool_names,
        results=results,
        checkpoint_format="trainable_only",
        base_model_name=run_config["model_name"],
    )
    checkpoint["toolathlon"] = {
        "manifest_hash": manifest["manifest_hash"],
        "interface": "no-tool-doc-memory-token",
        "method": args.method,
    }
    torch.save(checkpoint, run_dir / "checkpoint_trainable.pt")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
