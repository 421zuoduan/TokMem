#!/usr/bin/env python3
"""Build a small training-free cold-start delta from an existing checkpoint."""

import argparse
import gc
import sys
from pathlib import Path

import torch


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from cold_start.common import (  # noqa: E402
    load_checkpoint_model,
    load_json,
    load_selected_tool_names,
    torch_dtype,
)
from cold_start.probes import extract_document_hidden_states  # noqa: E402
from cold_start.runtime import (  # noqa: E402
    COLD_START_FORMAT,
    COLD_START_VERSION,
    append_cold_start_tools,
    make_orthogonal_new_embeddings,
)
from cold_start.similarity import (  # noqa: E402
    centered_cosine_scores,
    topk_gap_weights,
)


DEFAULT_MANIFEST = Path(__file__).with_name("selected_tools_20.json")
DEFAULT_OLD_DESCRIPTIONS = (
    COMPOSITIONAL_DIR / "data" / "tool_descriptions_tools51-100.json"
)
DEFAULT_NEW_DESCRIPTIONS = (
    COMPOSITIONAL_DIR / "data" / "tool_descriptions_tools1-50.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add orthogonally initialized memory-token rows and, for TapMem, "
            "schema-derived TCRA rows without training."
        )
    )
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--old-descriptions",
        default=str(DEFAULT_OLD_DESCRIPTIONS),
    )
    parser.add_argument(
        "--new-descriptions",
        default=str(DEFAULT_NEW_DESCRIPTIONS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    return parser.parse_args()


def _neighbor_records(result, old_tool_names, new_tool_names):
    records = []
    indices = result["neighbor_indices"]
    scores = result["neighbor_scores"]
    weights = result["weights"]
    for new_index, new_name in enumerate(new_tool_names):
        neighbors = []
        for rank, old_index_tensor in enumerate(indices[new_index]):
            old_index = int(old_index_tensor.item())
            neighbors.append(
                {
                    "tool": old_tool_names[old_index],
                    "score": float(scores[new_index, rank].item()),
                    "weight": float(weights[new_index, old_index].item()),
                }
            )
        records.append({"new_tool": new_name, "neighbors": neighbors})
    return records


def main():
    args = parse_args()
    dtype = torch_dtype(args.dtype)
    run_config, tokenizer, model = load_checkpoint_model(
        args.run_config,
        args.checkpoint,
        args.device,
        dtype,
    )
    old_tool_names = list(model.tool_names)
    new_tool_names = load_selected_tool_names(args.manifest)
    seed = (
        int(run_config["args"].get("seed", 0))
        if args.seed is None
        else int(args.seed)
    )

    new_embeddings = make_orthogonal_new_embeddings(
        model,
        len(new_tool_names),
        seed,
    )

    tcra = None
    if model.logit_bias_head is not None:
        old_descriptions = load_json(args.old_descriptions)
        new_descriptions = load_json(args.new_descriptions)
        schemas_by_name = {**old_descriptions, **new_descriptions}
        ordered_schemas = [
            schemas_by_name[name]
            for name in old_tool_names + new_tool_names
        ]
        print("Encoding old and new tool documentation")
        document_hidden_states = extract_document_hidden_states(
            model,
            tokenizer,
            ordered_schemas,
            batch_size=args.probe_batch_size,
        )
        old_count = len(old_tool_names)
        scores = centered_cosine_scores(
            document_hidden_states[:old_count],
            document_hidden_states[old_count:],
        )
        similarity = topk_gap_weights(scores, top_k=args.top_k)
        weights = similarity["weights"]

        old_weight = model.logit_bias_head.weight.detach().float().cpu()
        old_bias = model.logit_bias_head.bias.detach().float().cpu()
        new_tcra_weight = weights @ old_weight
        new_tcra_bias = weights @ old_bias
        neighbors = _neighbor_records(
            similarity,
            old_tool_names,
            new_tool_names,
        )
        tcra = {
            "new_weight": new_tcra_weight,
            "new_bias": new_tcra_bias,
            "aggregation_weights": weights,
            "cosine_scores": similarity["scores"],
            "background_scores": similarity["background_scores"],
            "neighbors": neighbors,
            "top_k": int(args.top_k),
            "hidden_state_source": "canonical_tool_document",
            "document_template_count": 1,
        }
    else:
        new_tcra_weight = None
        new_tcra_bias = None

    registry = append_cold_start_tools(
        model,
        new_tool_names,
        new_embeddings,
        new_tcra_weight,
        new_tcra_bias,
    )
    delta = {
        "format": COLD_START_FORMAT,
        "version": COLD_START_VERSION,
        "base_model": run_config["args"]["model_name"],
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_tool_names": old_tool_names,
        "new_tool_names": new_tool_names,
        "new_embeddings": new_embeddings.detach().cpu(),
        "embedding_initialization": {
            "method": "compositional_orthogonal_init_all_tools",
            "seed": seed,
            "joint_tool_count": len(old_tool_names) + len(new_tool_names),
            "old_rows_restored_from_checkpoint": True,
        },
        "tcra": tcra,
        "registry": registry,
        "training_samples_used": 0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, output_path)
    print(f"Saved cold-start delta to {output_path}")
    print(
        f"Expanded {len(old_tool_names)} old tools with "
        f"{len(new_tool_names)} new tools; EOC token id: {model.eoc_token_id}"
    )
    if tcra is not None:
        for record in tcra["neighbors"]:
            compact = ", ".join(
                f"{item['tool']}={item['weight']:.3f}"
                for item in record["neighbors"]
            )
            print(f"{record['new_tool']}: {compact}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
