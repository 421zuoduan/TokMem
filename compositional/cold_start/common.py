"""Shared loading helpers for the cold-start command-line tools."""

import json

import torch
from transformers import AutoTokenizer

from backbone_registry import resolve_function_calling_model_class
from checkpoint_io import checkpoint_tool_names, load_checkpoint_into_model


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def torch_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_lora_config(args):
    if not args.get("use_lora", False):
        return None
    config = {
        "r": int(args.get("lora_r", 8)),
        "alpha": int(args.get("lora_alpha", 32)),
        "dropout": float(args.get("lora_dropout", 0.1)),
        "target_modules": [
            value.strip()
            for value in args.get("lora_target_modules", "o_proj").split(",")
        ],
    }
    if args.get("lora_layer_indices") is not None:
        config["layer_indices"] = [
            int(value.strip())
            for value in args["lora_layer_indices"].split(",")
        ]
    return config


def load_checkpoint_model(run_config_path, checkpoint_path, device, dtype):
    """Strictly load the original checkpoint before doing any expansion."""
    run_config = load_json(run_config_path)
    args = run_config["args"]
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    old_tool_names = checkpoint_tool_names(
        checkpoint,
        fallback=checkpoint.get("tools"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args["model_name"],
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = resolve_function_calling_model_class(args["model_name"])
    model = model_class(
        model_name=args["model_name"],
        num_tools=len(old_tool_names),
        tool_names=old_tool_names,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        decouple_embeddings=bool(args.get("decouple_embeddings", False)),
        lora_config=build_lora_config(args),
        use_eoc=bool(args.get("use_eoc", False)),
        use_logit_bias=bool(args.get("use_logit_bias", False)),
        use_tool_head_replacement=bool(
            args.get("use_tool_head_replacement", False)
        ),
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
    del checkpoint

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return run_config, tokenizer, model


def load_selected_tool_names(manifest_path):
    manifest = load_json(manifest_path)
    return list(manifest["new_tools"])
