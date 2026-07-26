#!/usr/bin/env python3
"""Build old-tool query prototypes from first-tool training examples."""

import argparse
import sys
from pathlib import Path

import torch


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from backbone_prompting import format_user_assistant_prompt  # noqa: E402
from cold_start.common import (  # noqa: E402
    load_checkpoint_model,
    load_json,
    torch_dtype,
)
from cold_start.probes import (  # noqa: E402
    _final_hidden_states_without_logits,
    _right_pad,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Average the final hidden states immediately before old tools' "
            "first calls. This script performs inference only."
        )
    )
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--samples-per-tool", type=int, default=20)
    parser.add_argument(
        "--existing",
        default=None,
        help=(
            "Optional smaller prototype artifact whose per-tool query "
            "states are reused before encoding additional samples."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--progress",
        "--progress-every",
        dest="progress_every",
        type=int,
        default=100,
        help="Print progress after approximately this many encoded queries.",
    )
    return parser.parse_args()


def select_first_tool_indices(data, tool_names, samples_per_tool):
    indices_by_tool = {tool_name: [] for tool_name in tool_names}
    wanted = set(tool_names)
    for data_index, item in enumerate(data):
        first_tool = item["tools"][0]
        if (
            first_tool in wanted
            and len(indices_by_tool[first_tool]) < samples_per_tool
        ):
            indices_by_tool[first_tool].append(data_index)
        if all(
            len(indices) == samples_per_tool
            for indices in indices_by_tool.values()
        ):
            break

    missing = {
        tool_name: samples_per_tool - len(indices)
        for tool_name, indices in indices_by_tool.items()
        if len(indices) < samples_per_tool
    }
    if missing:
        raise ValueError(
            "Not enough first-tool examples for: "
            + ", ".join(
                f"{tool_name} (missing {count})"
                for tool_name, count in missing.items()
            )
        )
    return indices_by_tool


def encode_query_hidden_states(
    model,
    tokenizer,
    data,
    tool_names,
    indices_by_tool,
    batch_size,
    progress_every,
):
    sequences = []
    for tool_name in tool_names:
        for data_index in indices_by_tool[tool_name]:
            prompt = format_user_assistant_prompt(
                tokenizer,
                data[data_index]["user_input"],
                model=model,
            )
            sequences.append(
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )

    vectors = []
    device = model.trainable_tool_input_embeddings.device
    last_report = 0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        input_ids, attention_mask, lengths = _right_pad(
            batch,
            tokenizer.pad_token_id,
            device,
        )
        with torch.inference_mode():
            hidden_states = _final_hidden_states_without_logits(
                model,
                input_ids,
                attention_mask,
            )
        row_indices = torch.arange(len(batch), device=device)
        end_indices = torch.tensor(lengths, device=device) - 1
        vectors.append(
            hidden_states[row_indices, end_indices].float().cpu()
        )

        completed = min(start + len(batch), len(sequences))
        if (
            completed == len(sequences)
            or (
                progress_every > 0
                and completed - last_report >= progress_every
            )
        ):
            print(f"Encoded {completed}/{len(sequences)} query states")
            last_report = completed

    all_vectors = torch.cat(vectors, dim=0)
    samples_per_tool = len(indices_by_tool[tool_names[0]])
    hidden_states_by_tool = {}
    prototypes = []
    for tool_index, tool_name in enumerate(tool_names):
        start = tool_index * samples_per_tool
        tool_vectors = all_vectors[start : start + samples_per_tool]
        hidden_states_by_tool[tool_name] = tool_vectors
        prototypes.append(tool_vectors.mean(dim=0))
    return hidden_states_by_tool, torch.stack(prototypes, dim=0)


def main():
    args = parse_args()
    run_config, tokenizer, model = load_checkpoint_model(
        args.run_config,
        args.checkpoint,
        args.device,
        torch_dtype(args.dtype),
    )
    data = load_json(args.data)
    tool_names = list(model.tool_names)
    indices_by_tool = select_first_tool_indices(
        data,
        tool_names,
        args.samples_per_tool,
    )
    existing = None
    existing_count = 0
    if args.existing is not None:
        existing = torch.load(
            args.existing,
            map_location="cpu",
            weights_only=False,
        )
        existing_count = int(
            existing["metadata"]["samples_per_tool"]
        )
    pending_indices_by_tool = {
        tool_name: indices[existing_count:]
        for tool_name, indices in indices_by_tool.items()
    }
    print(
        f"Selected {args.samples_per_tool} first-tool queries for each of "
        f"{len(tool_names)} old tools; reusing {existing_count}"
    )

    new_hidden_states_by_tool, _ = encode_query_hidden_states(
        model,
        tokenizer,
        data,
        tool_names,
        pending_indices_by_tool,
        args.batch_size,
        args.progress_every,
    )
    hidden_states_by_tool = {}
    prototypes = []
    for tool_name in tool_names:
        pieces = []
        if existing is not None:
            pieces.append(existing["hidden_states_by_tool"][tool_name])
        pieces.append(new_hidden_states_by_tool[tool_name])
        tool_hidden_states = torch.cat(pieces, dim=0)
        hidden_states_by_tool[tool_name] = tool_hidden_states
        prototypes.append(tool_hidden_states.mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    artifact = {
        "format": "tokmem_old_tool_query_prototypes",
        "version": 1,
        "tool_names": tool_names,
        "hidden_states_by_tool": hidden_states_by_tool,
        "prototypes": prototypes,
        "data_indices_by_tool": indices_by_tool,
        "metadata": {
            "model_name": run_config["args"]["model_name"],
            "run_config": str(Path(args.run_config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data": str(Path(args.data).resolve()),
            "num_tools": len(tool_names),
            "samples_per_tool": args.samples_per_tool,
            "reused_samples_per_tool": existing_count,
            "existing_artifact": (
                None
                if args.existing is None
                else str(Path(args.existing).resolve())
            ),
            "hidden_size": int(prototypes.shape[-1]),
            "saved_dtype": "float32",
            "model_dtype": args.dtype,
            "selection": "first N dataset examples whose first target tool matches",
            "hidden_state_position": (
                "last non-padding prompt token immediately before the first "
                "assistant tool token"
            ),
            "prototype": "arithmetic mean of raw per-query hidden states",
            "training": False,
            "uses_new_tool_examples": False,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    print(
        f"Wrote {len(tool_names)} query prototypes with shape "
        f"{tuple(prototypes.shape)} to {output_path}"
    )


if __name__ == "__main__":
    main()
