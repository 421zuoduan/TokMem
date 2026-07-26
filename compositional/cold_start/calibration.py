"""Old-query calibration for cold-start tool logits."""

import math

import torch


def split_old_query_hidden_states(
    artifact,
    start_index,
    end_index,
    shape_samples_per_tool,
):
    """Split each old tool's saved query states into shape/threshold sets."""
    tool_names = artifact["tool_names"]
    shape_parts = []
    threshold_parts = []
    for tool_name in tool_names:
        states = artifact["hidden_states_by_tool"][tool_name][
            start_index:end_index
        ].float()
        shape_parts.append(states[:shape_samples_per_tool])
        threshold_parts.append(states[shape_samples_per_tool:])
    return torch.cat(shape_parts, dim=0), torch.cat(
        threshold_parts,
        dim=0,
    )


def upper_order_statistic(values, quantile):
    """Return a conservative empirical upper quantile."""
    if values.ndim == 0:
        values = values.unsqueeze(0)
    sample_count = values.shape[0]
    rank = math.ceil((sample_count + 1) * float(quantile))
    rank = min(max(rank, 1), sample_count)
    return values.sort(dim=0).values[rank - 1]


def equalized_old_query_offsets(
    shape_margins,
    threshold_margins,
    tool_tail_quantile,
    target_old_fpr,
    allow_boost=True,
):
    """Build per-tool offsets and an any-new conformal threshold.

    Margins are new-tool logits minus the best logit outside the new-tool
    block. The first split equalizes different new tools' upper tails. The
    second split chooses one shared shift that bounds any-new activation on
    old queries.
    """
    class_offsets = upper_order_statistic(
        shape_margins,
        tool_tail_quantile,
    )
    if not allow_boost:
        class_offsets = class_offsets.clamp_min(0.0)

    centered_threshold_margins = (
        threshold_margins - class_offsets.unsqueeze(0)
    )
    maximum_centered_margin = centered_threshold_margins.max(dim=1).values
    global_offset = upper_order_statistic(
        maximum_centered_margin,
        1.0 - float(target_old_fpr),
    )
    offsets = class_offsets + global_offset
    if not allow_boost:
        offsets = offsets.clamp_min(0.0)

    calibrated_threshold_margins = threshold_margins - offsets.unsqueeze(0)
    empirical_any_new_fpr = (
        calibrated_threshold_margins.max(dim=1).values > 0
    ).float().mean()
    return {
        "offsets": offsets,
        "class_offsets": class_offsets,
        "global_offset": global_offset,
        "shape_samples": int(shape_margins.shape[0]),
        "threshold_samples": int(threshold_margins.shape[0]),
        "empirical_threshold_any_new_fpr": empirical_any_new_fpr,
    }


def select_per_tool_safe_strengths(margins_by_strength, target_fpr):
    """Choose the furthest candidate ray point safe for each new tool."""
    strengths = sorted(margins_by_strength)
    tool_count = margins_by_strength[strengths[0]].shape[1]
    selected_strengths = []
    selected_fprs = []
    for tool_index in range(tool_count):
        selected_strength = strengths[0]
        selected_fpr = float(
            (
                margins_by_strength[selected_strength][:, tool_index] > 0
            )
            .float()
            .mean()
            .item()
        )
        for strength in strengths[1:]:
            fpr = float(
                (margins_by_strength[strength][:, tool_index] > 0)
                .float()
                .mean()
                .item()
            )
            if fpr <= target_fpr:
                selected_strength = strength
                selected_fpr = fpr
        selected_strengths.append(selected_strength)
        selected_fprs.append(selected_fpr)
    return {
        "selected_strengths": selected_strengths,
        "selected_fprs": selected_fprs,
    }


def preserve_gate_with_identity_offsets(new_logits, offsets):
    """Re-rank new tools while preserving their original maximum logit."""
    original_maximum = new_logits.max(dim=1, keepdim=True).values
    identity_logits = new_logits - offsets.unsqueeze(0)
    identity_maximum = identity_logits.max(dim=1, keepdim=True).values
    return original_maximum + identity_logits - identity_maximum
