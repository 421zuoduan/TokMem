#!/usr/bin/env python3
"""Validate Qwen3.5 prefill and cached-decode fast-kernel dispatch."""

import argparse
from collections import Counter

import torch
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-linear-layers", type=int, default=24)
    parser.add_argument("--expected-full-layers", type=int, default=8)
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
    torch.cuda.reset_peak_memory_stats()

    counts = Counter()

    def counted(name, function):
        def wrapper(*wrapper_args, **wrapper_kwargs):
            counts[name] += 1
            return function(*wrapper_args, **wrapper_kwargs)

        return wrapper

    linear_layers = []
    full_layers = []
    hook_handles = []
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            linear_layers.append(layer)
            attention = layer.linear_attn
            attention.causal_conv1d_fn = counted(
                "causal_conv1d_fn",
                attention.causal_conv1d_fn,
            )
            attention.causal_conv1d_update = counted(
                "causal_conv1d_update",
                attention.causal_conv1d_update,
            )
            attention.chunk_gated_delta_rule = counted(
                "chunk_gated_delta_rule",
                attention.chunk_gated_delta_rule,
            )
            attention.recurrent_gated_delta_rule = counted(
                "fused_recurrent_gated_delta_rule",
                attention.recurrent_gated_delta_rule,
            )

            def count_norm(_module, _inputs, _output):
                counts["fused_norm"] += 1

            hook_handles.append(
                attention.norm.register_forward_hook(count_norm)
            )
        else:
            full_layers.append(layer)

    assert len(linear_layers) == args.expected_linear_layers
    assert len(full_layers) == args.expected_full_layers
    assert all(
        layer.linear_attn.norm.__class__.__name__ == "FusedRMSNormGated"
        for layer in linear_layers
    )

    input_ids = torch.arange(
        100,
        164,
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    print(
        {
            "linear_layers": len(linear_layers),
            "full_layers": len(full_layers),
            "norm": linear_layers[0].linear_attn.norm.__class__.__name__,
        },
        flush=True,
    )

    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    prefill_counts = dict(counts)
    assert torch.isfinite(prefill.logits).all()
    assert prefill_counts == {
        "causal_conv1d_fn": args.expected_linear_layers,
        "chunk_gated_delta_rule": args.expected_linear_layers,
        "fused_norm": args.expected_linear_layers,
    }
    print({"prefill": prefill_counts}, flush=True)

    counts.clear()
    next_id = torch.tensor([[164]], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        decode = model(
            input_ids=next_id,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    decode_counts = dict(counts)
    assert torch.isfinite(decode.logits).all()
    assert decode_counts == {
        "causal_conv1d_update": args.expected_linear_layers,
        "fused_recurrent_gated_delta_rule": args.expected_linear_layers,
        "fused_norm": args.expected_linear_layers,
    }
    print({"cached_decode": decode_counts}, flush=True)
    print(
        {
            "peak_gpu_mib": round(
                torch.cuda.max_memory_allocated() / 1024**2,
                1,
            ),
            "prefill_top_id": int(prefill.logits[0, -1].argmax()),
            "decode_top_id": int(decode.logits[0, -1].argmax()),
        },
        flush=True,
    )

    for handle in hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
