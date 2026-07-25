#!/usr/bin/env python3
"""Compare Qwen3.5 fast-kernel logits with the Transformers fallback."""

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"loading={args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    input_ids = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107]],
        device="cuda",
        dtype=torch.long,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        fast_logits = model(
            input_ids=input_ids,
            use_cache=False,
        ).logits[:, -1].float()
    torch.cuda.synchronize()
    fast_seconds = time.perf_counter() - start

    linear_layers = [
        layer
        for layer in model.model.layers
        if layer.layer_type == "linear_attention"
    ]
    assert len(linear_layers) == 24
    for layer in linear_layers:
        attention = layer.linear_attn
        fallback_norm = modeling_qwen3_5.Qwen3_5RMSNormGated(
            attention.head_v_dim,
            eps=attention.layer_norm_epsilon,
        ).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            fallback_norm.weight.copy_(attention.norm.weight)
        attention.norm = fallback_norm
        attention.causal_conv1d_fn = None
        attention.causal_conv1d_update = (
            modeling_qwen3_5.torch_causal_conv1d_update
        )
        attention.chunk_gated_delta_rule = (
            modeling_qwen3_5.torch_chunk_gated_delta_rule
        )
        attention.recurrent_gated_delta_rule = (
            modeling_qwen3_5.torch_recurrent_gated_delta_rule
        )

    start = time.perf_counter()
    with torch.inference_mode():
        fallback_logits = model(
            input_ids=input_ids,
            use_cache=False,
        ).logits[:, -1].float()
    torch.cuda.synchronize()
    fallback_seconds = time.perf_counter() - start

    difference = (fast_logits - fallback_logits).abs()
    cosine = F.cosine_similarity(fast_logits, fallback_logits).item()
    fast_top = int(fast_logits.argmax(dim=-1))
    fallback_top = int(fallback_logits.argmax(dim=-1))
    target = torch.tensor([108], device="cuda")
    fast_ce = F.cross_entropy(fast_logits, target).item()
    fallback_ce = F.cross_entropy(fallback_logits, target).item()
    ce_relative_difference = abs(fast_ce - fallback_ce) / max(
        abs(fallback_ce),
        1e-12,
    )

    metrics = {
        "finite": bool(
            torch.isfinite(fast_logits).all()
            and torch.isfinite(fallback_logits).all()
        ),
        "top1_same": fast_top == fallback_top,
        "fast_top": fast_top,
        "fallback_top": fallback_top,
        "cosine": cosine,
        "mean_abs_diff": difference.mean().item(),
        "max_abs_diff": difference.max().item(),
        "fast_ce": fast_ce,
        "fallback_ce": fallback_ce,
        "ce_relative_diff": ce_relative_difference,
        "fast_seconds": fast_seconds,
        "fallback_seconds": fallback_seconds,
        "peak_gpu_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2,
            1,
        ),
    }
    print(metrics, flush=True)

    assert metrics["finite"]
    assert cosine >= 0.999
    assert metrics["mean_abs_diff"] <= 0.05
    assert ce_relative_difference <= 0.01


if __name__ == "__main__":
    main()
