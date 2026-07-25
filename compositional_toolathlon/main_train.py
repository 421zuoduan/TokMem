#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
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
from compositional_toolathlon.dataset import (  # noqa: E402
    create_step_dataloader,
    read_jsonl,
)
from compositional_toolathlon.manifest import load_manifest  # noqa: E402
from compositional_toolathlon.training import (  # noqa: E402
    evaluate_stepwise_loss,
    train_stepwise_model,
)


METHOD_FLAGS = {
    "tokmem": {"use_eoc": False, "use_logit_bias": False},
    "eoc_only": {"use_eoc": True, "use_logit_bias": False},
    "tapmem": {"use_eoc": True, "use_logit_bias": True},
}


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
    parser.add_argument("--validation-steps", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
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
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")

    set_seed(args.seed)
    flags = METHOD_FLAGS[args.method]
    manifest = load_manifest(args.manifest)
    audit_path = Path(args.data_audit)
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("passed") is not True:
        raise ValueError("training requires a passing dataset coverage audit")
    if audit_payload.get("schema_version") != 2:
        raise ValueError(
            "training requires dataset coverage audit schema_version=2; "
            "rerun audit_dataset so coverage is checked on train only"
        )
    if audit_payload.get("thresholds", {}).get("successful_episode_scope") != "train":
        raise ValueError("dataset coverage audit must count successful episodes on train")
    if audit_payload.get("tool_manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("dataset audit and training manifest hashes differ")
    expected_split_hashes = audit_payload.get("split_episode_id_hashes", {})
    for split, step_path in (
        ("train", args.train_steps),
        ("validation", args.validation_steps),
    ):
        actual_hash = episode_id_hash(
            {
                record["episode_id"]
                for record in read_jsonl(step_path)
            }
        )
        if expected_split_hashes.get(split) != actual_hash:
            raise ValueError(
                f"{split} steps do not match the episodes approved by data audit"
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
        balance_by_episode=True,
        sampler_seed=args.seed,
    )
    validation_loader = create_step_dataloader(
        args.validation_steps,
        args.manifest,
        tokenizer,
        model,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        use_eoc=flags["use_eoc"],
        mode="train",
        balance_by_episode=False,
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
        validation_dataloader=validation_loader,
    )
    validation_metrics = evaluate_stepwise_loss(
        model=model,
        dataloader=validation_loader,
        device=args.device,
        use_logit_bias=flags["use_logit_bias"],
        use_logit_train_add=flags["use_logit_bias"],
        detach=True,
        logit_bias_loss_weight=args.logit_bias_loss_weight,
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
        "validation_steps": os.path.realpath(args.validation_steps),
        "data_audit": os.path.realpath(args.data_audit),
        "data_audit_sha256": audit_sha256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
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
    }
    write_json(run_dir / "run_config.json", run_config)
    results = {
        "training": train_metrics,
        "validation": validation_metrics,
    }
    write_json(run_dir / "metrics.json", results)

    checkpoint = build_checkpoint_payload(
        model,
        round_number=int(
            train_metrics["checkpoint_selection"]["best_epoch"] or args.epochs
        ),
        round_tools=tool_names,
        results=results,
        checkpoint_format="trainable_only",
        base_model_name=args.model_name,
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
