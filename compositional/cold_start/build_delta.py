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
    coherence_partial_renorm_new_embeddings,
    make_convex_new_embeddings,
    make_orthogonal_new_embeddings,
    partial_renorm_new_embeddings,
)
from cold_start.similarity import (  # noqa: E402
    centered_cosine_geometry,
    local_affine_weights,
    loo_residual_krr_weights,
    query_prototype_bridge_weights,
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
            "Add training-free memory-token rows using orthogonal or "
            "document-convex initialization and, for TapMem, schema-derived "
            "TCRA rows."
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
    parser.add_argument(
        "--document-view",
        default="purpose",
        choices=["purpose", "full"],
        help=(
            "Use the tool name and purpose for routing similarity, or include "
            "the full parameter schema."
        ),
    )
    parser.add_argument(
        "--renorm-strength",
        type=float,
        default=1.0,
        help=(
            "Interpolation strength toward the old-tool mean norm when using "
            "document-convex-renorm; 0 keeps the convex norm and 1 fully "
            "matches the old mean."
        ),
    )
    parser.add_argument(
        "--renorm-rule",
        default="old-mean",
        choices=[
            "old-mean",
            "donor-coherence",
        ],
        help=(
            "Restore toward the global old-tool mean norm, or preserve each "
            "convex row's uncertainty using weighted donor coherence."
        ),
    )
    parser.add_argument(
        "--embedding-initialization",
        default="orthogonal",
        choices=[
            "orthogonal",
            "document-convex",
            "document-convex-renorm",
            "document-local-affine",
            "document-loo-residual-krr",
            "document-query-prototype-bridge",
        ],
        help=(
            "Use the original joint orthogonal initialization, or aggregate "
            "old learned embeddings with the same document weights as TCRA. "
            "Local affine fusion allows controlled extrapolation."
        ),
    )
    parser.add_argument(
        "--affine-negative-mass",
        type=float,
        default=0.1,
        help=(
            "Maximum total negative coefficient for document-local-affine. "
            "The same coefficients are used for embedding and TCRA rows."
        ),
    )
    parser.add_argument(
        "--affine-ridge-relative",
        type=float,
        default=0.1,
        help="Ridge weight relative to local document covariance.",
    )
    parser.add_argument(
        "--residual-negative-mass",
        type=float,
        default=0.1,
        help=(
            "Maximum total negative coefficient after leave-one-out residual "
            "transfer. This changes fusion coefficients, not embedding norms."
        ),
    )
    parser.add_argument(
        "--residual-ridge-relative",
        type=float,
        default=0.1,
        help="Ridge weight relative to the residual-transfer kernel diagonal.",
    )
    parser.add_argument(
        "--query-prototypes",
        default=None,
        help=(
            "Old-tool query-prototype artifact required by "
            "document-query-prototype-bridge."
        ),
    )
    parser.add_argument(
        "--bridge-negative-mass",
        type=float,
        default=0.6,
        help="Maximum negative coefficient mass for query-prototype fusion.",
    )
    parser.add_argument(
        "--document-ridge-relative",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--query-ridge-relative",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--bridge-prototype-samples-per-tool",
        type=int,
        default=None,
        help="Use only the first N saved old queries per bridge prototype.",
    )
    parser.add_argument(
        "--bridge-extrapolate-to-cap",
        action="store_true",
    )
    parser.add_argument(
        "--identity-calibration-quantile",
        type=float,
        default=None,
        help=(
            "Enable gate-preserving new-tool identity calibration using this "
            "old-query background quantile."
        ),
    )
    parser.add_argument(
        "--identity-calibration-start-index",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--identity-calibration-end-index",
        type=int,
        default=5,
    )
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


def _identity_calibration_offsets(
    model,
    query_prototype_artifact,
    old_tool_count,
    start_index,
    end_index,
    quantile,
):
    hidden_states = torch.cat(
        [
            query_prototype_artifact["hidden_states_by_tool"][tool_name][
                start_index:end_index
            ]
            for tool_name in query_prototype_artifact["tool_names"]
        ],
        dim=0,
    )
    device = model.trainable_tool_input_embeddings.device
    model_dtype = model.trainable_tool_input_embeddings.dtype
    new_token_ids = torch.tensor(
        model.tool_reserved_token_ids[old_tool_count:],
        dtype=torch.long,
        device=device,
    )
    margin_parts = []
    for start in range(0, hidden_states.shape[0], 32):
        hidden = hidden_states[start : start + 32].to(
            device=device,
            dtype=model_dtype,
        )
        with torch.inference_mode():
            logits = model._get_lm_head_module()(hidden)
            routed_logits = model._apply_logit_bias_to_logits(
                logits,
                hidden,
                torch.ones(
                    hidden.shape[0],
                    dtype=torch.bool,
                    device=device,
                ),
            )
            new_logits = routed_logits[:, new_token_ids].float()
            other_logits = routed_logits.clone()
            other_logits[:, new_token_ids] = torch.finfo(
                other_logits.dtype
            ).min
            other_values = other_logits.max(dim=-1).values.float()
            margin_parts.append(
                (new_logits - other_values.unsqueeze(1)).cpu()
            )
    margins = torch.cat(margin_parts, dim=0)
    return {
        "offsets": torch.quantile(
            margins,
            float(quantile),
            dim=0,
        ),
        "margins": margins,
        "samples": int(margins.shape[0]),
    }


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

    similarity = None
    neighbors = None
    needs_document_similarity = (
        model.logit_bias_head is not None
        or args.embedding_initialization.startswith("document-")
    )
    if needs_document_similarity:
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
            document_view=args.document_view,
        )
        old_count = len(old_tool_names)
        geometry = centered_cosine_geometry(
            document_hidden_states[:old_count],
            document_hidden_states[old_count:],
        )
        scores = geometry["scores"]
        similarity = topk_gap_weights(scores, top_k=args.top_k)
        weights = similarity["weights"]
        neighbors = _neighbor_records(
            similarity,
            old_tool_names,
            new_tool_names,
        )

    parameter_weights = weights if needs_document_similarity else None
    affine_result = None
    residual_result = None
    bridge_result = None
    query_prototype_artifact = None
    affine_negative_mass = None
    convex_weights = weights if needs_document_similarity else None
    if args.embedding_initialization == "document-local-affine":
        affine_result = local_affine_weights(
            scores,
            geometry["old_gram"],
            similarity,
            ridge_relative=args.affine_ridge_relative,
            negative_mass_cap=args.affine_negative_mass,
        )
        parameter_weights = affine_result["weights"]
        affine_negative_mass = affine_result["negative_mass"]
        neighbors = _neighbor_records(
            {
                **similarity,
                "weights": parameter_weights,
            },
            old_tool_names,
            new_tool_names,
        )
    elif args.embedding_initialization == "document-loo-residual-krr":
        residual_result = loo_residual_krr_weights(
            scores,
            geometry["old_gram"],
            similarity,
            top_k=args.top_k,
            ridge_relative=args.residual_ridge_relative,
            negative_mass_cap=args.residual_negative_mass,
        )
        parameter_weights = residual_result["weights"]
        neighbors = _neighbor_records(
            {
                **similarity,
                "weights": parameter_weights,
            },
            old_tool_names,
            new_tool_names,
        )
    elif (
        args.embedding_initialization
        == "document-query-prototype-bridge"
    ):
        if args.query_prototypes is None:
            raise ValueError(
                "--query-prototypes is required for "
                "document-query-prototype-bridge"
            )
        query_prototype_artifact = torch.load(
            args.query_prototypes,
            map_location="cpu",
            weights_only=False,
        )
        if args.bridge_prototype_samples_per_tool is None:
            old_query_prototypes = query_prototype_artifact["prototypes"]
        else:
            old_query_prototypes = torch.stack(
                [
                    query_prototype_artifact[
                        "hidden_states_by_tool"
                    ][tool_name][
                        : args.bridge_prototype_samples_per_tool
                    ].float().mean(dim=0)
                    for tool_name in query_prototype_artifact["tool_names"]
                ],
                dim=0,
            )
        bridge_result = query_prototype_bridge_weights(
            scores,
            geometry["old_gram"],
            old_query_prototypes,
            similarity,
            document_ridge_relative=args.document_ridge_relative,
            query_ridge_relative=args.query_ridge_relative,
            negative_mass_cap=args.bridge_negative_mass,
            extrapolate_to_cap=args.bridge_extrapolate_to_cap,
        )
        parameter_weights = bridge_result["weights"]
        neighbors = _neighbor_records(
            {
                **similarity,
                "weights": parameter_weights,
            },
            old_tool_names,
            new_tool_names,
        )

    if args.embedding_initialization in {
        "document-local-affine",
        "document-loo-residual-krr",
        "document-query-prototype-bridge",
    }:
        new_embeddings = make_convex_new_embeddings(
            model,
            parameter_weights,
        )
        if affine_result is not None:
            embedding_initialization = {
                "method": args.embedding_initialization,
                "top_k": int(args.top_k),
                "weight_source": "canonical_tool_document_hidden_state",
                "aggregation_weights": parameter_weights,
                "convex_anchor_weights": convex_weights,
                "negative_mass_cap": float(args.affine_negative_mass),
                "negative_mass": affine_negative_mass,
                "ridge_relative": float(args.affine_ridge_relative),
                "ridge_values": affine_result["ridge_values"],
                "document_reconstruction_errors": (
                    affine_result["reconstruction_errors"]
                ),
            }
        elif residual_result is not None:
            embedding_initialization = {
                "method": args.embedding_initialization,
                "top_k": int(args.top_k),
                "weight_source": "canonical_tool_document_hidden_state",
                "aggregation_weights": parameter_weights,
                "convex_anchor_weights": convex_weights,
                "negative_mass_cap": float(args.residual_negative_mass),
                "negative_mass": residual_result["negative_mass"],
                "raw_negative_mass": residual_result[
                    "raw_negative_mass"
                ],
                "ridge_relative": float(args.residual_ridge_relative),
                "ridge_value": residual_result["ridge_value"],
                "kernel_bandwidth": residual_result[
                    "kernel_bandwidth"
                ],
                "kernel_confidence": residual_result[
                    "kernel_confidence"
                ],
            }
        else:
            embedding_initialization = {
                "method": args.embedding_initialization,
                "top_k": int(args.top_k),
                "weight_source": (
                    "tool_document_to_old_query_prototype_bridge"
                ),
                "aggregation_weights": parameter_weights,
                "convex_anchor_weights": convex_weights,
                "negative_mass_cap": float(args.bridge_negative_mass),
                "negative_mass": bridge_result["negative_mass"],
                "raw_negative_mass": bridge_result[
                    "raw_negative_mass"
                ],
                "document_ridge_relative": float(
                    args.document_ridge_relative
                ),
                "query_ridge_relative": float(
                    args.query_ridge_relative
                ),
                "document_kernel_bandwidth": bridge_result[
                    "document_kernel_bandwidth"
                ],
                "query_kernel_bandwidth": bridge_result[
                    "query_kernel_bandwidth"
                ],
                "ray_scales": bridge_result["ray_scales"],
                "extrapolate_to_cap": bool(
                    args.bridge_extrapolate_to_cap
                ),
                "query_prototype_artifact": str(
                    Path(args.query_prototypes).resolve()
                ),
                "prototype_samples_per_tool": (
                    args.bridge_prototype_samples_per_tool
                ),
            }
    elif args.embedding_initialization.startswith("document-convex"):
        new_embeddings = make_convex_new_embeddings(model, weights)
        target_norm = None
        donor_target_norms = None
        donor_coherence = None
        if args.embedding_initialization == "document-convex-renorm":
            if args.renorm_rule == "donor-coherence":
                (
                    new_embeddings,
                    donor_target_norms,
                    donor_coherence,
                ) = coherence_partial_renorm_new_embeddings(
                    model,
                    new_embeddings,
                    weights,
                    strength=args.renorm_strength,
                )
            else:
                new_embeddings, target_norm = partial_renorm_new_embeddings(
                    model,
                    new_embeddings,
                    strength=args.renorm_strength,
                )
        embedding_initialization = {
            "method": args.embedding_initialization,
            "top_k": int(args.top_k),
            "weight_source": "canonical_tool_document_hidden_state",
            "document_view": args.document_view,
            "aggregation_weights": weights,
            "renorm_target": (
                None
                if (
                    target_norm is None
                    and donor_target_norms is None
                )
                else (
                    "weighted_donor_embedding_norm"
                    if donor_target_norms is not None
                    else "mean_old_tool_embedding_norm"
                )
            ),
            "target_norm": (
                None if target_norm is None else float(target_norm.item())
            ),
            "renorm_strength": (
                None
                if (
                    target_norm is None
                    and donor_target_norms is None
                )
                else float(args.renorm_strength)
            ),
            "renorm_rule": (
                None
                if (
                    target_norm is None
                    and donor_target_norms is None
                )
                else args.renorm_rule
            ),
            "donor_target_norms": donor_target_norms,
            "donor_coherence": donor_coherence,
        }
    else:
        new_embeddings = make_orthogonal_new_embeddings(
            model,
            len(new_tool_names),
            seed,
        )
        embedding_initialization = {
            "method": "compositional_orthogonal_init_all_tools",
            "seed": seed,
            "joint_tool_count": len(old_tool_names) + len(new_tool_names),
            "old_rows_restored_from_checkpoint": True,
        }

    tcra = None
    if model.logit_bias_head is not None:
        old_weight = model.logit_bias_head.weight.detach().float().cpu()
        old_bias = model.logit_bias_head.bias.detach().float().cpu()
        new_tcra_weight = parameter_weights @ old_weight
        new_tcra_bias = parameter_weights @ old_bias
        tcra = {
            "new_weight": new_tcra_weight,
            "new_bias": new_tcra_bias,
            "aggregation_weights": parameter_weights,
            "convex_anchor_weights": convex_weights,
            "cosine_scores": similarity["scores"],
            "background_scores": similarity["background_scores"],
            "neighbors": neighbors,
            "top_k": int(args.top_k),
            "hidden_state_source": "canonical_tool_document",
            "document_view": args.document_view,
            "document_template_count": 1,
            "negative_mass_cap": (
                float(args.affine_negative_mass)
                if affine_result is not None
                else (
                    float(args.residual_negative_mass)
                    if residual_result is not None
                    else (
                        float(args.bridge_negative_mass)
                        if bridge_result is not None
                        else None
                    )
                )
            ),
            "negative_mass": (
                affine_negative_mass
                if affine_result is not None
                else (
                    residual_result["negative_mass"]
                    if residual_result is not None
                    else (
                        bridge_result["negative_mass"]
                        if bridge_result is not None
                        else None
                    )
                )
            ),
            "ridge_relative": (
                float(args.affine_ridge_relative)
                if affine_result is not None
                else (
                    float(args.residual_ridge_relative)
                    if residual_result is not None
                    else (
                        float(args.query_ridge_relative)
                        if bridge_result is not None
                        else None
                    )
                )
            ),
        }
    else:
        new_tcra_weight = None
        new_tcra_bias = None

    document_similarity = None
    if similarity is not None:
        document_similarity = {
            "aggregation_weights": parameter_weights,
            "convex_anchor_weights": convex_weights,
            "cosine_scores": similarity["scores"],
            "old_cosine_gram": geometry["old_gram"],
            "background_scores": similarity["background_scores"],
            "neighbors": neighbors,
            "top_k": int(args.top_k),
            "hidden_state_source": "canonical_tool_document",
            "document_view": args.document_view,
            "affine_raw_weights": (
                None
                if affine_result is None
                else affine_result["raw_weights"]
            ),
            "negative_mass": affine_negative_mass,
            "affine_ridge_relative": (
                None
                if affine_result is None
                else float(args.affine_ridge_relative)
            ),
            "residual_raw_weights": (
                None
                if residual_result is None
                else residual_result["raw_weights"]
            ),
            "residual_loo_anchor_weights": (
                None
                if residual_result is None
                else residual_result["loo_anchor_weights"]
            ),
            "residual_operator": (
                None
                if residual_result is None
                else residual_result["residual_operator"]
            ),
            "residual_raw_negative_mass": (
                None
                if residual_result is None
                else residual_result["raw_negative_mass"]
            ),
            "residual_kernel_bandwidth": (
                None
                if residual_result is None
                else residual_result["kernel_bandwidth"]
            ),
            "residual_kernel_confidence": (
                None
                if residual_result is None
                else residual_result["kernel_confidence"]
            ),
            "residual_ridge_relative": (
                None
                if residual_result is None
                else float(args.residual_ridge_relative)
            ),
            "query_prototype_bridge": (
                None
                if bridge_result is None
                else {
                    "raw_weights": bridge_result["raw_weights"],
                    "document_bridge_weights": bridge_result[
                        "document_bridge_weights"
                    ],
                    "predicted_query_prototypes": bridge_result[
                        "predicted_query_prototypes"
                    ],
                    "raw_negative_mass": bridge_result[
                        "raw_negative_mass"
                    ],
                    "negative_mass": bridge_result["negative_mass"],
                    "ray_scales": bridge_result["ray_scales"],
                    "extrapolate_to_cap": bool(
                        args.bridge_extrapolate_to_cap
                    ),
                    "document_kernel_bandwidth": bridge_result[
                        "document_kernel_bandwidth"
                    ],
                    "query_kernel_bandwidth": bridge_result[
                        "query_kernel_bandwidth"
                    ],
                    "document_ridge_relative": float(
                        args.document_ridge_relative
                    ),
                    "query_ridge_relative": float(
                        args.query_ridge_relative
                    ),
                    "prototype_artifact": str(
                        Path(args.query_prototypes).resolve()
                    ),
                    "prototype_samples_per_tool": (
                        args.bridge_prototype_samples_per_tool
                    ),
                }
            ),
        }

    registry = append_cold_start_tools(
        model,
        new_tool_names,
        new_embeddings,
        new_tcra_weight,
        new_tcra_bias,
    )
    routing_calibration = None
    identity_calibration_result = None
    if args.identity_calibration_quantile is not None:
        identity_calibration_result = _identity_calibration_offsets(
            model,
            query_prototype_artifact,
            len(old_tool_names),
            args.identity_calibration_start_index,
            args.identity_calibration_end_index,
            args.identity_calibration_quantile,
        )
        routing_calibration = {
            "method": "old_query_gate_preserving_identity_tail",
            "new_tool_identity_offsets": (
                identity_calibration_result["offsets"]
            ),
            "quantile": float(args.identity_calibration_quantile),
            "start_index": args.identity_calibration_start_index,
            "end_index": args.identity_calibration_end_index,
            "samples": identity_calibration_result["samples"],
            "preserves_original_new_tool_maximum": True,
            "uses_new_tool_examples": False,
            "query_prototype_artifact": str(
                Path(args.query_prototypes).resolve()
            ),
        }
    prototype_samples_per_tool = (
        0
        if query_prototype_artifact is None
        else (
            int(
                query_prototype_artifact["metadata"][
                    "samples_per_tool"
                ]
            )
            if args.bridge_prototype_samples_per_tool is None
            else int(args.bridge_prototype_samples_per_tool)
        )
    )
    calibration_samples_per_tool = (
        0
        if identity_calibration_result is None
        else (
            args.identity_calibration_end_index
            - args.identity_calibration_start_index
        )
    )
    delta = {
        "format": COLD_START_FORMAT,
        "version": COLD_START_VERSION,
        "base_model": run_config["args"]["model_name"],
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_tool_names": old_tool_names,
        "new_tool_names": new_tool_names,
        "new_embeddings": new_embeddings.detach().cpu(),
        "embedding_initialization": embedding_initialization,
        "document_similarity": document_similarity,
        "tcra": tcra,
        "routing_calibration": routing_calibration,
        "registry": registry,
        "training_samples_used": 0,
        "new_tool_training_samples_used": 0,
        "base_query_prototype_samples_used": (
            len(old_tool_names) * prototype_samples_per_tool
        ),
        "base_query_calibration_samples_used": (
            len(old_tool_names) * calibration_samples_per_tool
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, output_path)
    print(f"Saved cold-start delta to {output_path}")
    print(
        f"Expanded {len(old_tool_names)} old tools with "
        f"{len(new_tool_names)} new tools; EOC token id: {model.eoc_token_id}"
    )
    if neighbors is not None:
        for record in neighbors:
            compact = ", ".join(
                f"{item['tool']}={item['weight']:.3f}"
                for item in record["neighbors"]
            )
            print(f"{record['new_tool']}: {compact}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
