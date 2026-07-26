#!/usr/bin/env python3
"""Sweep cold-start parameter fusion with one backbone forward per query batch."""

import argparse
import json
import math
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
from cold_start.calibration import (  # noqa: E402
    equalized_old_query_offsets,
    preserve_gate_with_identity_offsets,
    select_per_tool_safe_strengths,
    split_old_query_hidden_states,
)
from cold_start.probes import (  # noqa: E402
    _final_hidden_states_without_logits,
    _right_pad,
)
from cold_start.runtime import (  # noqa: E402
    apply_cold_start_delta,
    coherence_partial_renorm_new_embeddings,
    make_convex_new_embeddings,
    partial_renorm_new_embeddings,
)
from cold_start.similarity import (  # noqa: E402
    local_affine_weights,
    loo_residual_krr_weights,
    query_prototype_bridge_weights,
    topk_gap_weights,
)


DEFAULT_DATA = (
    COMPOSITIONAL_DIR
    / "data"
    / "test"
    / "function_calling_test_tools51-100_plus_cold20_4calls.json"
)

QUERY_PROTOTYPE_BRIDGE_RULES = {
    "query-prototype-bridge",
    "query-prototype-bridge-decoupled",
}
AFFINE_FUSION_RULES = {
    "local-affine",
    "loo-residual-krr",
    "loo-residual-krr-decoupled",
    *QUERY_PROTOTYPE_BRIDGE_RULES,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate many cold-start parameter-fusion candidates while "
            "reusing the same query hidden states and full-vocabulary logits."
        )
    )
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--strength-start", type=float, default=0.0)
    parser.add_argument("--strength-end", type=float, default=1.0)
    parser.add_argument("--strength-step", type=float, default=0.05)
    parser.add_argument("--penalty-start", type=float, default=0.0)
    parser.add_argument("--penalty-end", type=float, default=0.0)
    parser.add_argument("--penalty-step", type=float, default=0.1)
    parser.add_argument("--tcra-gain-start", type=float, default=1.0)
    parser.add_argument("--tcra-gain-end", type=float, default=1.0)
    parser.add_argument("--tcra-gain-step", type=float, default=0.25)
    parser.add_argument(
        "--fusion-rule",
        "--renorm-rule",
        dest="fusion_rule",
        default="local-affine",
        choices=[
            "local-affine",
            "loo-residual-krr",
            "loo-residual-krr-decoupled",
            "query-prototype-bridge",
            "query-prototype-bridge-decoupled",
            "old-mean",
            "donor-coherence",
        ],
        help=(
            "For affine/residual fusion, renorm is disabled and "
            "--strength-* scans the allowed negative coefficient mass eta."
        ),
    )
    parser.add_argument("--affine-top-k", type=int, default=4)
    parser.add_argument(
        "--affine-ridge-relative",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--embedding-negative-mass-cap",
        type=float,
        default=0.0,
        help=(
            "Fixed embedding cap for a decoupled affine fusion rule; the "
            "strength grid controls the TCRA cap."
        ),
    )
    parser.add_argument(
        "--query-prototypes",
        default=None,
        help=(
            "Old-tool query-prototype artifact. Required by the two "
            "query-prototype-bridge fusion rules."
        ),
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
        help=(
            "Use only the first N saved old queries per tool to form bridge "
            "prototypes. The remaining saved states can calibrate routing."
        ),
    )
    parser.add_argument(
        "--bridge-extrapolate-to-cap",
        action="store_true",
        help=(
            "Continue from the convex anchor through the prototype-KRR "
            "solution until the requested negative-mass cap is reached."
        ),
    )
    parser.add_argument(
        "--old-query-calibration",
        default="none",
        choices=[
            "none",
            "tail-equalized",
            "identity-tail",
            "safe-ray",
        ],
        help=(
            "Optionally shift each new-tool logit using only saved old-tool "
            "training-query hidden states. This changes no embedding norm."
        ),
    )
    parser.add_argument("--calibration-start-index", type=int, default=3)
    parser.add_argument("--calibration-end-index", type=int, default=5)
    parser.add_argument(
        "--calibration-shape-samples-per-tool",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--calibration-tool-tail-quantile",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--calibration-target-old-fpr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--calibration-no-boost",
        action="store_true",
        help="Only suppress overactive new tools; never give a negative offset.",
    )
    parser.add_argument(
        "--safe-ray-target-fpr",
        type=float,
        default=0.01,
        help=(
            "Maximum old-query activation rate for each new tool when "
            "--old-query-calibration=safe-ray."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-records", action="store_true")
    parser.add_argument(
        "--routing-state-cache",
        default=None,
        help=(
            "Optional torch cache for query routing states. If the file "
            "exists, reuse it without running tokenization, the backbone, "
            "or the LM head."
        ),
    )
    parser.add_argument(
        "--source-routing-state-cache",
        default=None,
        help=(
            "Optional source cache whose last hidden states are reused when "
            "--routing-state-cache does not exist. The current checkpoint "
            "and delta still recompute the LM-head and TCRA routing state "
            "before writing the target cache."
        ),
    )
    return parser.parse_args()


def _strength_grid(start, end, step):
    count = int(round((end - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def _encode_prompts(model, tokenizer, items):
    sequences = []
    for item in items:
        prompt = format_user_assistant_prompt(
            tokenizer,
            item["user_input"],
            model=model,
        )
        sequences.append(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
    return sequences


def _partition(records, expected_is_new):
    selected = [
        record
        for record in records
        if record["expected_is_new"] is expected_is_new
    ]
    correct = sum(record["correct"] for record in selected)
    return {
        "examples": len(selected),
        "correct": correct,
        "accuracy": correct / max(1, len(selected)),
    }


def _summarize(strength, new_tcra_gain, new_logit_penalty, records):
    correct = sum(record["correct"] for record in records)
    predicted_new = sum(record["predicted_is_new"] for record in records)
    tool_token_predictions = sum(
        record["predicted_tool_token"] for record in records
    )
    old_to_new = sum(
        (not record["expected_is_new"]) and record["predicted_is_new"]
        for record in records
    )
    new_to_new = sum(
        record["expected_is_new"] and record["predicted_is_new"]
        for record in records
    )
    return {
        "strength": strength,
        "new_tcra_gain": new_tcra_gain,
        "new_logit_penalty": new_logit_penalty,
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / max(1, len(records)),
        "new_tools": _partition(records, True),
        "old_tools": _partition(records, False),
        "predicted_new_tools": predicted_new,
        "predicted_new_tool_rate": predicted_new / max(1, len(records)),
        "old_to_new": old_to_new,
        "old_to_new_rate": old_to_new / max(
            1,
            sum(not record["expected_is_new"] for record in records),
        ),
        "new_to_new": new_to_new,
        "new_to_new_rate": new_to_new / max(
            1,
            sum(record["expected_is_new"] for record in records),
        ),
        "tool_token_predictions": tool_token_predictions,
        "tool_token_prediction_rate": (
            tool_token_predictions / max(1, len(records))
        ),
    }


def _query_prototype_bridge_metadata(result):
    return {
        "document_bridge_weights": (
            result["document_bridge_weights"].tolist()
        ),
        "predicted_query_prototypes": (
            result["predicted_query_prototypes"].tolist()
        ),
        "old_query_gram": result["old_query_gram"].tolist(),
        "predicted_query_scores": (
            result["predicted_query_scores"].tolist()
        ),
        "raw_parameter_weights": result["raw_weights"].tolist(),
        "raw_negative_mass": result["raw_negative_mass"].tolist(),
        "document_kernel_bandwidth": float(
            result["document_kernel_bandwidth"].item()
        ),
        "query_kernel_bandwidth": float(
            result["query_kernel_bandwidth"].item()
        ),
        "document_ridge_value": float(
            result["document_ridge_value"].item()
        ),
        "query_ridge_value": float(
            result["query_ridge_value"].item()
        ),
    }


def _old_only_routing_state(model, hidden_states, old_tool_count):
    """Compute the old-reference and best non-new logit for saved states."""
    device = model.trainable_tool_input_embeddings.device
    old_token_ids = torch.tensor(
        model.tool_reserved_token_ids[:old_tool_count],
        dtype=torch.long,
        device=device,
    )
    new_token_ids = torch.tensor(
        model.tool_reserved_token_ids[old_tool_count:],
        dtype=torch.long,
        device=device,
    )
    other_value_parts = []
    old_reference_parts = []
    for start in range(0, hidden_states.shape[0], 32):
        hidden = hidden_states[start : start + 32].to(
            device=device,
            dtype=model.trainable_tool_input_embeddings.dtype,
        )
        with torch.inference_mode():
            logits = model._get_lm_head_module()(hidden)
            if model.logit_bias_head is not None:
                scores = model._get_logit_bias_scores(hidden).float()
                old_scores = scores[:, :old_tool_count]
                old_log_probabilities = torch.log_softmax(
                    old_scores,
                    dim=-1,
                )
                old_bias = (
                    old_log_probabilities + math.log(old_tool_count)
                )
                old_bias = (
                    old_bias * model.logit_bias_scale
                ).to(dtype=logits.dtype)
                logits[:, old_token_ids] += old_bias
                old_reference = (
                    torch.logsumexp(
                        old_scores,
                        dim=-1,
                        keepdim=True,
                    )
                    - math.log(old_tool_count)
                )
            else:
                old_reference = None
            logits[:, new_token_ids] = torch.finfo(logits.dtype).min
            other_value_parts.append(logits.max(dim=-1).values.float().cpu())
            if old_reference is not None:
                old_reference_parts.append(old_reference.float().cpu())
    return {
        "hidden_states": hidden_states.float(),
        "other_values": torch.cat(other_value_parts, dim=0),
        "old_reference": (
            None
            if not old_reference_parts
            else torch.cat(old_reference_parts, dim=0)
        ),
    }


def _new_tool_logits(
    routing_state,
    new_embeddings,
    candidate_tcra,
    model,
):
    routing_dtype = model.trainable_tool_input_embeddings.dtype
    hidden_states = (
        routing_state["hidden_states"].to(dtype=routing_dtype).float()
    )
    quantized_embeddings = new_embeddings.to(dtype=routing_dtype).float()
    logits = hidden_states @ quantized_embeddings.transpose(0, 1)
    if candidate_tcra is not None:
        candidate_weight = candidate_tcra["weight"].to(
            dtype=routing_dtype
        ).float()
        candidate_bias = candidate_tcra["bias"].to(
            dtype=routing_dtype
        ).float()
        scores = (
            hidden_states
            @ candidate_weight.transpose(0, 1)
            + candidate_bias
        )
        logits += (
            scores - routing_state["old_reference"]
        ) * model.logit_bias_scale
    return logits


def main():
    args = parse_args()
    _, tokenizer, model = load_checkpoint_model(
        args.run_config,
        args.checkpoint,
        args.device,
        torch_dtype(args.dtype),
    )
    delta = torch.load(
        args.delta,
        map_location="cpu",
        weights_only=False,
    )
    old_tool_count = len(delta["base_tool_names"])
    document_similarity = delta.get("document_similarity")
    if document_similarity is None:
        raise ValueError("Renorm sweep requires document similarity weights")
    aggregation_weights = document_similarity.get(
        "convex_anchor_weights",
        document_similarity["aggregation_weights"],
    )
    query_prototype_artifact = None
    old_query_prototypes = None
    prototype_bridge_metadata = None
    if args.fusion_rule in QUERY_PROTOTYPE_BRIDGE_RULES:
        if args.query_prototypes is None:
            raise ValueError(
                "--query-prototypes is required for "
                f"{args.fusion_rule}"
            )
        query_prototype_artifact = torch.load(
            args.query_prototypes,
            map_location="cpu",
            weights_only=False,
        )
        if (
            query_prototype_artifact["tool_names"]
            != delta["base_tool_names"]
        ):
            raise ValueError(
                "Query-prototype tools do not match the delta base tools"
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
    plain_new_embeddings = None
    if args.fusion_rule not in AFFINE_FUSION_RULES:
        plain_new_embeddings = make_convex_new_embeddings(
            model,
            aggregation_weights,
        )
    strengths = _strength_grid(
        args.strength_start,
        args.strength_end,
        args.strength_step,
    )
    penalties = _strength_grid(
        args.penalty_start,
        args.penalty_end,
        args.penalty_step,
    )
    tcra_gains = _strength_grid(
        args.tcra_gain_start,
        args.tcra_gain_end,
        args.tcra_gain_step,
    )
    if model.logit_bias_head is None:
        tcra_gains = [1.0]
    embeddings_by_strength = {}
    target_norm = None
    donor_target_norms = None
    donor_coherence = None
    affine_metadata = {}
    tcra_by_strength = {}
    old_tcra_weight = None
    old_tcra_bias = None
    if model.logit_bias_head is not None:
        old_tcra_weight = model.logit_bias_head.weight.detach().float().cpu()
        old_tcra_bias = model.logit_bias_head.bias.detach().float().cpu()
    convex_result = None
    if args.fusion_rule in AFFINE_FUSION_RULES:
        cosine_scores = document_similarity["cosine_scores"]
        old_cosine_gram = document_similarity["old_cosine_gram"]
        convex_result = topk_gap_weights(
            cosine_scores,
            top_k=args.affine_top_k,
        )
    fixed_embedding_bridge_result = None
    if args.fusion_rule == "query-prototype-bridge-decoupled":
        fixed_embedding_bridge_result = query_prototype_bridge_weights(
            cosine_scores,
            old_cosine_gram,
            old_query_prototypes,
            convex_result,
            document_ridge_relative=args.document_ridge_relative,
            query_ridge_relative=args.query_ridge_relative,
            negative_mass_cap=args.embedding_negative_mass_cap,
            extrapolate_to_cap=args.bridge_extrapolate_to_cap,
        )
    for strength in strengths:
        if args.fusion_rule == "local-affine":
            affine_result = local_affine_weights(
                cosine_scores,
                old_cosine_gram,
                convex_result,
                ridge_relative=args.affine_ridge_relative,
                negative_mass_cap=strength,
            )
            parameter_weights = affine_result["weights"]
            embeddings = make_convex_new_embeddings(
                model,
                parameter_weights,
            )
            affine_metadata[str(strength)] = {
                "negative_mass_cap": strength,
                "negative_mass": (
                    affine_result["negative_mass"].tolist()
                ),
                "ridge_values": (
                    affine_result["ridge_values"].tolist()
                ),
                "document_reconstruction_errors": (
                    affine_result["reconstruction_errors"].tolist()
                ),
            }
            if old_tcra_weight is not None:
                tcra_by_strength[strength] = {
                    "weight": parameter_weights @ old_tcra_weight,
                    "bias": parameter_weights @ old_tcra_bias,
                }
        elif args.fusion_rule == "loo-residual-krr":
            residual_result = loo_residual_krr_weights(
                cosine_scores,
                old_cosine_gram,
                convex_result,
                top_k=args.affine_top_k,
                ridge_relative=args.affine_ridge_relative,
                negative_mass_cap=strength,
            )
            parameter_weights = residual_result["weights"]
            embeddings = make_convex_new_embeddings(
                model,
                parameter_weights,
            )
            affine_metadata[str(strength)] = {
                "negative_mass_cap": strength,
                "negative_mass": (
                    residual_result["negative_mass"].tolist()
                ),
                "raw_negative_mass": (
                    residual_result["raw_negative_mass"].tolist()
                ),
                "kernel_bandwidth": float(
                    residual_result["kernel_bandwidth"].item()
                ),
                "kernel_confidence": (
                    residual_result["kernel_confidence"].tolist()
                ),
                "ridge_value": float(
                    residual_result["ridge_value"].item()
                ),
            }
            if old_tcra_weight is not None:
                tcra_by_strength[strength] = {
                    "weight": parameter_weights @ old_tcra_weight,
                    "bias": parameter_weights @ old_tcra_bias,
                }
        elif args.fusion_rule == "loo-residual-krr-decoupled":
            embedding_result = loo_residual_krr_weights(
                cosine_scores,
                old_cosine_gram,
                convex_result,
                top_k=args.affine_top_k,
                ridge_relative=args.affine_ridge_relative,
                negative_mass_cap=args.embedding_negative_mass_cap,
            )
            tcra_result = loo_residual_krr_weights(
                cosine_scores,
                old_cosine_gram,
                convex_result,
                top_k=args.affine_top_k,
                ridge_relative=args.affine_ridge_relative,
                negative_mass_cap=strength,
            )
            embedding_weights = embedding_result["weights"]
            tcra_weights = tcra_result["weights"]
            embeddings = make_convex_new_embeddings(
                model,
                embedding_weights,
            )
            affine_metadata[str(strength)] = {
                "embedding_negative_mass_cap": (
                    args.embedding_negative_mass_cap
                ),
                "embedding_negative_mass": (
                    embedding_result["negative_mass"].tolist()
                ),
                "tcra_negative_mass_cap": strength,
                "tcra_negative_mass": (
                    tcra_result["negative_mass"].tolist()
                ),
                "raw_negative_mass": (
                    tcra_result["raw_negative_mass"].tolist()
                ),
                "kernel_bandwidth": float(
                    tcra_result["kernel_bandwidth"].item()
                ),
                "kernel_confidence": (
                    tcra_result["kernel_confidence"].tolist()
                ),
                "ridge_value": float(
                    tcra_result["ridge_value"].item()
                ),
            }
            if old_tcra_weight is not None:
                tcra_by_strength[strength] = {
                    "weight": tcra_weights @ old_tcra_weight,
                    "bias": tcra_weights @ old_tcra_bias,
                }
        elif args.fusion_rule == "query-prototype-bridge":
            bridge_result = query_prototype_bridge_weights(
                cosine_scores,
                old_cosine_gram,
                old_query_prototypes,
                convex_result,
                document_ridge_relative=args.document_ridge_relative,
                query_ridge_relative=args.query_ridge_relative,
                negative_mass_cap=strength,
                extrapolate_to_cap=args.bridge_extrapolate_to_cap,
            )
            parameter_weights = bridge_result["weights"]
            embeddings = make_convex_new_embeddings(
                model,
                parameter_weights,
            )
            affine_metadata[str(strength)] = {
                "negative_mass_cap": strength,
                "negative_mass": (
                    bridge_result["negative_mass"].tolist()
                ),
                "raw_negative_mass": (
                    bridge_result["raw_negative_mass"].tolist()
                ),
                "parameter_weights": parameter_weights.tolist(),
                "ray_scales": bridge_result["ray_scales"].tolist(),
            }
            if prototype_bridge_metadata is None:
                prototype_bridge_metadata = (
                    _query_prototype_bridge_metadata(bridge_result)
                )
            if old_tcra_weight is not None:
                tcra_by_strength[strength] = {
                    "weight": parameter_weights @ old_tcra_weight,
                    "bias": parameter_weights @ old_tcra_bias,
                }
        elif (
            args.fusion_rule
            == "query-prototype-bridge-decoupled"
        ):
            tcra_bridge_result = query_prototype_bridge_weights(
                cosine_scores,
                old_cosine_gram,
                old_query_prototypes,
                convex_result,
                document_ridge_relative=args.document_ridge_relative,
                query_ridge_relative=args.query_ridge_relative,
                negative_mass_cap=strength,
                extrapolate_to_cap=args.bridge_extrapolate_to_cap,
            )
            embedding_weights = fixed_embedding_bridge_result["weights"]
            tcra_weights = tcra_bridge_result["weights"]
            embeddings = make_convex_new_embeddings(
                model,
                embedding_weights,
            )
            affine_metadata[str(strength)] = {
                "embedding_negative_mass_cap": (
                    args.embedding_negative_mass_cap
                ),
                "embedding_negative_mass": (
                    fixed_embedding_bridge_result[
                        "negative_mass"
                    ].tolist()
                ),
                "tcra_negative_mass_cap": strength,
                "tcra_negative_mass": (
                    tcra_bridge_result["negative_mass"].tolist()
                ),
                "raw_negative_mass": (
                    tcra_bridge_result["raw_negative_mass"].tolist()
                ),
                "embedding_weights": embedding_weights.tolist(),
                "tcra_weights": tcra_weights.tolist(),
                "embedding_ray_scales": (
                    fixed_embedding_bridge_result["ray_scales"].tolist()
                ),
                "tcra_ray_scales": (
                    tcra_bridge_result["ray_scales"].tolist()
                ),
            }
            if prototype_bridge_metadata is None:
                prototype_bridge_metadata = (
                    _query_prototype_bridge_metadata(
                        tcra_bridge_result
                    )
                )
            if old_tcra_weight is not None:
                tcra_by_strength[strength] = {
                    "weight": tcra_weights @ old_tcra_weight,
                    "bias": tcra_weights @ old_tcra_bias,
                }
        elif args.fusion_rule == "donor-coherence":
            (
                embeddings,
                donor_target_norms,
                donor_coherence,
            ) = coherence_partial_renorm_new_embeddings(
                model,
                plain_new_embeddings,
                aggregation_weights,
                strength,
            )
        else:
            embeddings, target_norm = partial_renorm_new_embeddings(
                model,
                plain_new_embeddings,
                strength,
            )
        embeddings_by_strength[strength] = embeddings

    apply_cold_start_delta(model, delta)
    new_tool_set = set(delta["new_tool_names"])
    token_id_to_tool = {
        token_id: model.tool_id_to_name[tool_id]
        for token_id, tool_id in model.token_id_to_tool_id.items()
    }
    new_token_ids = torch.tensor(
        model.tool_reserved_token_ids[old_tool_count:],
        dtype=torch.long,
        device=model.trainable_tool_input_embeddings.device,
    )
    candidate_tool_token_ids = set(model.tool_reserved_token_ids)

    calibration_offsets_by_strength = {}
    calibration_metadata = None
    if args.old_query_calibration in {
        "tail-equalized",
        "identity-tail",
        "safe-ray",
    }:
        shape_hidden_states, threshold_hidden_states = (
            split_old_query_hidden_states(
                query_prototype_artifact,
                args.calibration_start_index,
                args.calibration_end_index,
                args.calibration_shape_samples_per_tool,
            )
        )
    if args.old_query_calibration == "tail-equalized":
        shape_routing_state = _old_only_routing_state(
            model,
            shape_hidden_states,
            old_tool_count,
        )
        threshold_routing_state = _old_only_routing_state(
            model,
            threshold_hidden_states,
            old_tool_count,
        )
        calibration_metadata = {
            "rule": "tail-equalized",
            "start_index": args.calibration_start_index,
            "end_index": args.calibration_end_index,
            "shape_samples_per_tool": (
                args.calibration_shape_samples_per_tool
            ),
            "tool_tail_quantile": (
                args.calibration_tool_tail_quantile
            ),
            "target_old_fpr": args.calibration_target_old_fpr,
            "allow_boost": not args.calibration_no_boost,
            "by_strength": {},
        }
        for strength in strengths:
            candidate_tcra = tcra_by_strength.get(strength)
            shape_logits = _new_tool_logits(
                shape_routing_state,
                embeddings_by_strength[strength],
                candidate_tcra,
                model,
            )
            threshold_logits = _new_tool_logits(
                threshold_routing_state,
                embeddings_by_strength[strength],
                candidate_tcra,
                model,
            )
            calibration_result = equalized_old_query_offsets(
                shape_logits
                - shape_routing_state["other_values"].unsqueeze(1),
                threshold_logits
                - threshold_routing_state["other_values"].unsqueeze(1),
                args.calibration_tool_tail_quantile,
                args.calibration_target_old_fpr,
                allow_boost=not args.calibration_no_boost,
            )
            calibration_offsets_by_strength[strength] = (
                calibration_result["offsets"]
            )
            calibration_metadata["by_strength"][str(strength)] = {
                "offsets": calibration_result["offsets"].tolist(),
                "class_offsets": (
                    calibration_result["class_offsets"].tolist()
                ),
                "global_offset": float(
                    calibration_result["global_offset"].item()
                ),
                "shape_samples": calibration_result["shape_samples"],
                "threshold_samples": (
                    calibration_result["threshold_samples"]
                ),
                "empirical_threshold_any_new_fpr": float(
                    calibration_result[
                        "empirical_threshold_any_new_fpr"
                    ].item()
                ),
            }
    elif args.old_query_calibration == "identity-tail":
        calibration_hidden_states = torch.cat(
            [shape_hidden_states, threshold_hidden_states],
            dim=0,
        )
        calibration_routing_state = _old_only_routing_state(
            model,
            calibration_hidden_states,
            old_tool_count,
        )
        calibration_metadata = {
            "rule": "identity-tail",
            "start_index": args.calibration_start_index,
            "end_index": args.calibration_end_index,
            "samples": int(calibration_hidden_states.shape[0]),
            "tool_tail_quantile": (
                args.calibration_tool_tail_quantile
            ),
            "preserves_original_new_tool_maximum": True,
            "by_strength": {},
        }
        for strength in strengths:
            calibration_logits = _new_tool_logits(
                calibration_routing_state,
                embeddings_by_strength[strength],
                tcra_by_strength.get(strength),
                model,
            )
            calibration_margins = (
                calibration_logits
                - calibration_routing_state["other_values"].unsqueeze(1)
            )
            offsets = torch.quantile(
                calibration_margins,
                float(args.calibration_tool_tail_quantile),
                dim=0,
            )
            calibration_offsets_by_strength[strength] = offsets
            calibration_metadata["by_strength"][str(strength)] = {
                "identity_offsets": offsets.tolist(),
            }
    elif args.old_query_calibration == "safe-ray":
        calibration_hidden_states = torch.cat(
            [shape_hidden_states, threshold_hidden_states],
            dim=0,
        )
        calibration_routing_state = _old_only_routing_state(
            model,
            calibration_hidden_states,
            old_tool_count,
        )
        margins_by_strength = {}
        for strength in strengths:
            calibration_logits = _new_tool_logits(
                calibration_routing_state,
                embeddings_by_strength[strength],
                tcra_by_strength.get(strength),
                model,
            )
            margins_by_strength[strength] = (
                calibration_logits
                - calibration_routing_state["other_values"].unsqueeze(1)
            )
        safe_result = select_per_tool_safe_strengths(
            margins_by_strength,
            args.safe_ray_target_fpr,
        )
        selected_strengths = safe_result["selected_strengths"]
        combined_embeddings = torch.stack(
            [
                embeddings_by_strength[strength][tool_index]
                for tool_index, strength in enumerate(selected_strengths)
            ],
            dim=0,
        )
        combined_tcra = None
        if old_tcra_weight is not None:
            combined_tcra = {
                "weight": torch.stack(
                    [
                        tcra_by_strength[strength]["weight"][tool_index]
                        for tool_index, strength in enumerate(
                            selected_strengths
                        )
                    ],
                    dim=0,
                ),
                "bias": torch.stack(
                    [
                        tcra_by_strength[strength]["bias"][tool_index]
                        for tool_index, strength in enumerate(
                            selected_strengths
                        )
                    ],
                    dim=0,
                ),
            }
        result_key = strengths[-1]
        combined_margins = torch.stack(
            [
                margins_by_strength[strength][:, tool_index]
                for tool_index, strength in enumerate(selected_strengths)
            ],
            dim=1,
        )
        calibration_metadata = {
            "rule": "safe-ray",
            "start_index": args.calibration_start_index,
            "end_index": args.calibration_end_index,
            "samples": int(calibration_hidden_states.shape[0]),
            "target_per_tool_fpr": args.safe_ray_target_fpr,
            "candidate_strengths": strengths,
            "selected_strengths": selected_strengths,
            "selected_fprs": safe_result["selected_fprs"],
            "empirical_any_new_fpr": float(
                (combined_margins.max(dim=1).values > 0)
                .float()
                .mean()
                .item()
            ),
        }
        strengths = [result_key]
        embeddings_by_strength = {result_key: combined_embeddings}
        tcra_by_strength = (
            {}
            if combined_tcra is None
            else {result_key: combined_tcra}
        )

    all_items = load_json(args.data_path)
    end = None if args.limit is None else args.offset + args.limit
    items = all_items[args.offset:end]
    routing_state_cache_path = (
        None
        if args.routing_state_cache is None
        else Path(args.routing_state_cache)
    )
    source_routing_state_cache_path = (
        None
        if args.source_routing_state_cache is None
        else Path(args.source_routing_state_cache)
    )
    if (
        source_routing_state_cache_path is not None
        and routing_state_cache_path is None
    ):
        raise ValueError(
            "--source-routing-state-cache requires "
            "--routing-state-cache"
        )
    cached_state_batches = None
    source_state_batches = None
    routing_state_cache_reused = False
    source_routing_state_cache_reused = False
    if (
        routing_state_cache_path is not None
        and routing_state_cache_path.exists()
    ):
        routing_state_cache = torch.load(
            routing_state_cache_path,
            map_location="cpu",
            weights_only=False,
        )
        cached_state_batches = routing_state_cache["batches"]
        routing_state_cache_reused = True
        print(
            "Loaded "
            f"{len(cached_state_batches)} routing-state batches from "
            f"{routing_state_cache_path}"
        )
    elif source_routing_state_cache_path is not None:
        source_routing_state_cache = torch.load(
            source_routing_state_cache_path,
            map_location="cpu",
            weights_only=False,
        )
        source_state_batches = source_routing_state_cache["batches"]
        source_routing_state_cache_reused = True
        print(
            "Loaded hidden states from "
            f"{len(source_state_batches)} routing-state batches in "
            f"{source_routing_state_cache_path}"
        )
    sequences = (
        None
        if (
            cached_state_batches is not None
            or source_state_batches is not None
        )
        else _encode_prompts(model, tokenizer, items)
    )
    predicted_ids_by_candidate = {
        (strength, gain, penalty): []
        for strength in strengths
        for gain in tcra_gains
        for penalty in penalties
    }

    device = model.trainable_tool_input_embeddings.device
    generated_state_batches = []
    if cached_state_batches is not None:
        state_batch_count = len(cached_state_batches)
    elif source_state_batches is not None:
        state_batch_count = len(source_state_batches)
    else:
        state_batch_count = math.ceil(len(sequences) / args.batch_size)
    completed = 0
    for batch_index in range(state_batch_count):
        with torch.inference_mode():
            if cached_state_batches is None:
                if source_state_batches is None:
                    start = batch_index * args.batch_size
                    batch = sequences[start : start + args.batch_size]
                    input_ids, attention_mask, lengths = _right_pad(
                        batch,
                        tokenizer.pad_token_id,
                        device,
                    )
                    hidden_states = _final_hidden_states_without_logits(
                        model,
                        input_ids,
                        attention_mask,
                    )
                    row_indices = torch.arange(
                        len(batch),
                        device=device,
                    )
                    end_indices = (
                        torch.tensor(lengths, device=device) - 1
                    )
                    last_hidden_states = hidden_states[
                        row_indices,
                        end_indices,
                    ]
                else:
                    last_hidden_states = source_state_batches[
                        batch_index
                    ]["last_hidden_states"].to(device=device)
                base_logits = model._get_lm_head_module()(
                    last_hidden_states
                )
                fixed_logits = base_logits.clone()
                old_reference = None
                fixed_new_tcra_evidence = None
                if model.logit_bias_head is not None:
                    tcra_scores = model._get_logit_bias_scores(
                        last_hidden_states
                    ).float()
                    old_scores = tcra_scores[:, :old_tool_count]
                    old_log_probabilities = torch.log_softmax(
                        old_scores,
                        dim=-1,
                    )
                    old_bias = (
                        old_log_probabilities + math.log(old_tool_count)
                    )
                    old_bias = (
                        old_bias * model.logit_bias_scale
                    ).to(dtype=fixed_logits.dtype)
                    old_token_ids = torch.tensor(
                        model.tool_reserved_token_ids[:old_tool_count],
                        dtype=torch.long,
                        device=device,
                    )
                    fixed_logits[:, old_token_ids] += old_bias
                    old_reference = (
                        torch.logsumexp(
                            old_scores,
                            dim=-1,
                            keepdim=True,
                        )
                        - math.log(old_tool_count)
                    )
                    fixed_new_tcra_evidence = (
                        (
                            tcra_scores[:, old_tool_count:]
                            - old_reference
                        )
                        * model.logit_bias_scale
                    ).to(dtype=fixed_logits.dtype)

                other_logits = fixed_logits.clone()
                other_logits[:, new_token_ids] = torch.finfo(
                    other_logits.dtype
                ).min
                other_values, other_predictions = torch.max(
                    other_logits,
                    dim=-1,
                )
                routing_logit_dtype = fixed_logits.dtype
                if routing_state_cache_path is not None:
                    generated_state_batches.append(
                        {
                            "last_hidden_states": (
                                last_hidden_states.detach().cpu()
                            ),
                            "other_values": other_values.detach().cpu(),
                            "other_predictions": (
                                other_predictions.detach().cpu()
                            ),
                            "old_reference": (
                                None
                                if old_reference is None
                                else old_reference.detach().cpu()
                            ),
                            "fixed_new_tcra_evidence": (
                                None
                                if fixed_new_tcra_evidence is None
                                else fixed_new_tcra_evidence.detach().cpu()
                            ),
                        }
                    )
            else:
                cached_state = cached_state_batches[batch_index]
                last_hidden_states = cached_state[
                    "last_hidden_states"
                ].to(device=device)
                other_values = cached_state["other_values"].to(
                    device=device
                )
                other_predictions = cached_state[
                    "other_predictions"
                ].to(device=device)
                old_reference = cached_state["old_reference"]
                if old_reference is not None:
                    old_reference = old_reference.to(device=device)
                fixed_new_tcra_evidence = cached_state[
                    "fixed_new_tcra_evidence"
                ]
                if fixed_new_tcra_evidence is not None:
                    fixed_new_tcra_evidence = (
                        fixed_new_tcra_evidence.to(device=device)
                    )
                routing_logit_dtype = other_values.dtype

            for strength in strengths:
                new_embeddings = embeddings_by_strength[strength].to(
                    device=device,
                    dtype=last_hidden_states.dtype,
                )
                new_embedding_logits = (
                    last_hidden_states.float()
                    @ new_embeddings.float().transpose(0, 1)
                ).to(dtype=routing_logit_dtype)
                new_tcra_evidence = fixed_new_tcra_evidence
                if strength in tcra_by_strength:
                    candidate_tcra = tcra_by_strength[strength]
                    candidate_weight = candidate_tcra["weight"].to(
                        device=device,
                        dtype=last_hidden_states.dtype,
                    )
                    candidate_bias = candidate_tcra["bias"].to(
                        device=device,
                        dtype=last_hidden_states.dtype,
                    )
                    candidate_scores = (
                        last_hidden_states.float()
                        @ candidate_weight.float().transpose(0, 1)
                        + candidate_bias.float()
                    )
                    new_tcra_evidence = (
                        (candidate_scores - old_reference)
                        * model.logit_bias_scale
                    ).to(dtype=routing_logit_dtype)
                for gain in tcra_gains:
                    gained_new_logits = new_embedding_logits
                    if new_tcra_evidence is not None:
                        gained_new_logits = (
                            gained_new_logits
                            + new_tcra_evidence * gain
                        )
                    if strength in calibration_offsets_by_strength:
                        calibration_offsets = (
                            calibration_offsets_by_strength[strength].to(
                                device=device,
                                dtype=gained_new_logits.dtype,
                            )
                        )
                        if (
                            args.old_query_calibration
                            == "identity-tail"
                        ):
                            gained_new_logits = (
                                preserve_gate_with_identity_offsets(
                                    gained_new_logits,
                                    calibration_offsets,
                                )
                            )
                        else:
                            gained_new_logits = (
                                gained_new_logits - calibration_offsets
                            )
                    new_values, new_local_indices = torch.max(
                        gained_new_logits,
                        dim=-1,
                    )
                    new_predictions = new_token_ids[new_local_indices]
                    for penalty in penalties:
                        predictions = torch.where(
                            new_values - penalty > other_values,
                            new_predictions,
                            other_predictions,
                        )
                        predicted_ids_by_candidate[
                            (strength, gain, penalty)
                        ].extend(predictions.detach().cpu().tolist())
        completed += last_hidden_states.shape[0]
        if args.progress_every > 0 and (
            completed == len(items)
            or completed % args.progress_every == 0
        ):
            print(f"Evaluated {completed}/{len(items)} query states")

    if (
        routing_state_cache_path is not None
        and not routing_state_cache_reused
    ):
        routing_state_cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            {
                "metadata": {
                    "run_config": str(Path(args.run_config).resolve()),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "delta": str(Path(args.delta).resolve()),
                    "data_path": str(Path(args.data_path).resolve()),
                    "offset": args.offset,
                    "limit": len(items),
                    "batch_size": args.batch_size,
                    "old_tool_count": old_tool_count,
                    "has_logit_bias": model.logit_bias_head is not None,
                    "source_routing_state_cache": (
                        None
                        if source_routing_state_cache_path is None
                        else str(
                            source_routing_state_cache_path.resolve()
                        )
                    ),
                },
                "batches": generated_state_batches,
            },
            routing_state_cache_path,
        )
        print(
            "Wrote "
            f"{len(generated_state_batches)} routing-state batches to "
            f"{routing_state_cache_path}"
        )

    summaries = []
    records_by_strength = {}
    for strength in strengths:
        for gain in tcra_gains:
            for penalty in penalties:
                records = []
                candidate = (strength, gain, penalty)
                for local_index, (item, predicted_token_id) in enumerate(
                    zip(items, predicted_ids_by_candidate[candidate])
                ):
                    expected_tool = item["tools"][0]
                    predicted_tool = token_id_to_tool.get(predicted_token_id)
                    records.append(
                        {
                            "index": args.offset + local_index,
                            "expected_tool": expected_tool,
                            "predicted_tool": predicted_tool,
                            "predicted_token_id": predicted_token_id,
                            "predicted_tool_token": (
                                predicted_token_id
                                in candidate_tool_token_ids
                            ),
                            "predicted_is_new": predicted_tool in new_tool_set,
                            "expected_is_new": expected_tool in new_tool_set,
                            "correct": predicted_tool == expected_tool,
                        }
                    )
                summary = _summarize(
                    strength,
                    gain,
                    penalty,
                    records,
                )
                if args.fusion_rule in AFFINE_FUSION_RULES:
                    summary["negative_mass_cap"] = strength
                if args.fusion_rule in {
                    "loo-residual-krr-decoupled",
                    "query-prototype-bridge-decoupled",
                }:
                    summary["embedding_negative_mass_cap"] = (
                        args.embedding_negative_mass_cap
                    )
                    summary["tcra_negative_mass_cap"] = strength
                summaries.append(summary)
                if args.save_records:
                    key = (
                        f"strength={strength},gain={gain},"
                        f"penalty={penalty}"
                    )
                    records_by_strength[key] = records

    output = {
        "metric": "first_full_vocabulary_greedy_tool_routing",
        "run_config": str(Path(args.run_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "delta": str(Path(args.delta).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "routing_state_cache": (
            None
            if routing_state_cache_path is None
            else str(routing_state_cache_path.resolve())
        ),
        "routing_state_cache_reused": routing_state_cache_reused,
        "source_routing_state_cache": (
            None
            if source_routing_state_cache_path is None
            else str(source_routing_state_cache_path.resolve())
        ),
        "source_routing_state_cache_reused": (
            source_routing_state_cache_reused
        ),
        "offset": args.offset,
        "limit": len(items),
        "fusion_rule": args.fusion_rule,
        "renorm_applied": args.fusion_rule in {
            "old-mean",
            "donor-coherence",
        },
        "sweep_parameter": (
            "tcra_negative_mass_cap"
            if args.fusion_rule in {
                "loo-residual-krr-decoupled",
                "query-prototype-bridge-decoupled",
            }
            else (
                "negative_mass_cap"
                if args.fusion_rule in AFFINE_FUSION_RULES
                else "renorm_strength"
            )
        ),
        "affine_top_k": (
            args.affine_top_k
            if args.fusion_rule in AFFINE_FUSION_RULES
            else None
        ),
        "affine_ridge_relative": (
            args.affine_ridge_relative
            if args.fusion_rule in {
                "local-affine",
                "loo-residual-krr",
                "loo-residual-krr-decoupled",
            }
            else None
        ),
        "query_prototypes": (
            str(Path(args.query_prototypes).resolve())
            if args.fusion_rule in QUERY_PROTOTYPE_BRIDGE_RULES
            else None
        ),
        "query_prototype_artifact_format": (
            query_prototype_artifact.get("format")
            if query_prototype_artifact is not None
            else None
        ),
        "query_prototype_artifact_metadata": (
            query_prototype_artifact.get("metadata")
            if query_prototype_artifact is not None
            else None
        ),
        "bridge_prototype_samples_per_tool": (
            args.bridge_prototype_samples_per_tool
            if args.fusion_rule in QUERY_PROTOTYPE_BRIDGE_RULES
            else None
        ),
        "document_ridge_relative": (
            args.document_ridge_relative
            if args.fusion_rule in QUERY_PROTOTYPE_BRIDGE_RULES
            else None
        ),
        "query_ridge_relative": (
            args.query_ridge_relative
            if args.fusion_rule in QUERY_PROTOTYPE_BRIDGE_RULES
            else None
        ),
        "query_prototype_bridge_metadata": prototype_bridge_metadata,
        "bridge_extrapolate_to_cap": (
            args.bridge_extrapolate_to_cap
            if args.fusion_rule in QUERY_PROTOTYPE_BRIDGE_RULES
            else None
        ),
        "old_query_calibration": calibration_metadata,
        "shared_embedding_tcra_weights": (
            args.fusion_rule in {
                "local-affine",
                "loo-residual-krr",
                "query-prototype-bridge",
            }
            and old_tcra_weight is not None
        ),
        "embedding_negative_mass_cap": (
            args.embedding_negative_mass_cap
            if args.fusion_rule in {
                "loo-residual-krr-decoupled",
                "query-prototype-bridge-decoupled",
            }
            else None
        ),
        "target_old_mean_norm": (
            None if target_norm is None else float(target_norm.item())
        ),
        "donor_target_norms": (
            None
            if donor_target_norms is None
            else donor_target_norms.tolist()
        ),
        "donor_coherence": (
            None if donor_coherence is None else donor_coherence.tolist()
        ),
        "affine_metadata": affine_metadata or None,
        "strengths": summaries,
        "new_logit_penalties": penalties,
        "new_tcra_gains": tcra_gains,
    }
    if args.save_records:
        output["records"] = records_by_strength

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    ranked = sorted(
        summaries,
        key=lambda item: (
            item["accuracy"],
            item["new_tools"]["accuracy"],
            item["old_tools"]["accuracy"],
        ),
        reverse=True,
    )
    print(json.dumps(ranked[:10], ensure_ascii=False, indent=2))
    print(f"Wrote parameter-fusion sweep to {output_path}")


if __name__ == "__main__":
    main()
