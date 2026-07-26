#!/usr/bin/env python3
"""Apply an old-replay-selected strength to a document-convex delta."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from cold_start.common import (  # noqa: E402
    load_checkpoint_model,
    torch_dtype,
)
from cold_start.runtime import (  # noqa: E402
    coherence_partial_renorm_new_embeddings,
    make_convex_new_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-delta", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, model = load_checkpoint_model(
        args.run_config,
        args.checkpoint,
        args.device,
        torch_dtype(args.dtype),
    )
    source_delta = torch.load(
        args.source_delta,
        map_location="cpu",
        weights_only=False,
    )
    with open(args.selection, "r", encoding="utf-8") as handle:
        selection = json.load(handle)

    document_similarity = source_delta["document_similarity"]
    convex_weights = document_similarity.get(
        "convex_anchor_weights",
        document_similarity["aggregation_weights"],
    )
    convex_embeddings = make_convex_new_embeddings(
        model,
        convex_weights,
    )
    strength = float(selection["selected_strength"])
    (
        selected_embeddings,
        donor_target_norms,
        donor_coherence,
    ) = coherence_partial_renorm_new_embeddings(
        model,
        convex_embeddings,
        convex_weights,
        strength,
    )

    delta = copy.deepcopy(source_delta)
    delta["new_embeddings"] = selected_embeddings.detach().cpu()
    delta["embedding_initialization"] = {
        **delta["embedding_initialization"],
        "method": "document-convex-donor-coherence",
        "renorm_rule": "donor-coherence",
        "renorm_strength": strength,
        "donor_target_norms": donor_target_norms,
        "donor_coherence": donor_coherence,
    }
    delta["routing_calibration"] = {
        "method": "old_replay_largest_safe_prefix",
        "selection_file": str(Path(args.selection).resolve()),
        "new_tcra_gain": float(selection.get("tcra_gain", 1.0)),
        "new_logit_penalty": float(
            selection.get("new_logit_penalty", 0.0)
        ),
        "constraints": selection["constraints"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, output_path)
    print(
        f"Saved strength {strength:.4f} cold-start delta to {output_path}"
    )


if __name__ == "__main__":
    main()
