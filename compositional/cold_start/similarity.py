"""Hidden-state similarity and Top-K score-gap aggregation."""

import torch
import torch.nn.functional as functional


def _normalize_rows(values):
    return functional.normalize(values.float(), p=2, dim=-1, eps=1e-12)


def centered_cosine_scores(
    old_hidden_states,
    new_hidden_states,
):
    """Compare tool documents after removing their shared formatting component."""
    return centered_cosine_geometry(
        old_hidden_states,
        new_hidden_states,
    )["scores"]


def centered_cosine_geometry(
    old_hidden_states,
    new_hidden_states,
):
    """Return new-to-old scores and the old-document cosine Gram matrix."""
    old_hidden_states = old_hidden_states.float()
    new_hidden_states = new_hidden_states.float()
    old_mean = old_hidden_states.mean(dim=0, keepdim=True)
    old_centered = _normalize_rows(old_hidden_states - old_mean)
    new_centered = _normalize_rows(new_hidden_states - old_mean)
    return {
        "scores": new_centered @ old_centered.transpose(0, 1),
        "old_gram": old_centered @ old_centered.transpose(0, 1),
    }


def topk_gap_weights(scores, top_k=4):
    """Turn cosine scores into sparse weights using rank k+1 as background."""
    if top_k >= scores.shape[1]:
        raise ValueError("top_k must be smaller than the number of old tools")

    ranked_values, ranked_indices = torch.topk(
        scores.float(),
        k=top_k + 1,
        dim=-1,
    )
    background = ranked_values[:, top_k : top_k + 1]
    gaps = (ranked_values[:, :top_k] - background).clamp_min(0.0)
    gap_sums = gaps.sum(dim=-1, keepdim=True)
    normalized = gaps / gap_sums.clamp_min(1e-12)

    tied_rows = gap_sums.squeeze(-1) == 0
    if tied_rows.any():
        normalized[tied_rows] = 1.0 / top_k

    weights = torch.zeros_like(scores, dtype=torch.float32)
    weights.scatter_(1, ranked_indices[:, :top_k], normalized)
    return {
        "weights": weights,
        "scores": scores.float(),
        "neighbor_indices": ranked_indices[:, :top_k],
        "neighbor_scores": ranked_values[:, :top_k],
        "background_scores": background.squeeze(-1),
    }


def _limit_negative_mass(anchor, candidate, cap):
    negative_mass = candidate.clamp_max(0.0).neg().sum()
    if negative_mass <= cap:
        return candidate

    lower = 0.0
    upper = 1.0
    direction = candidate - anchor
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        blended = anchor + middle * direction
        current = blended.clamp_max(0.0).neg().sum()
        if current <= cap:
            lower = middle
        else:
            upper = middle
    return anchor + lower * direction


def _move_to_negative_mass_cap(anchor, candidate, cap):
    """Follow an affine direction until its negative mass reaches the cap."""
    cap = float(cap)
    direction = candidate - anchor

    def negative_mass(scale):
        moved = anchor + scale * direction
        return moved.clamp_max(0.0).neg().sum()

    upper = 1.0
    while negative_mass(upper) < cap and upper < 1e6:
        upper *= 2.0
    if negative_mass(upper) < cap:
        return candidate, 1.0

    lower = 0.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if negative_mass(middle) <= cap:
            lower = middle
        else:
            upper = middle
    return anchor + lower * direction, lower


def _old_geometry_rbf_kernel(cosine_gram):
    """Build an RBF kernel whose bandwidth uses only old-tool geometry."""
    cosine_gram = cosine_gram.double()
    old_tool_count = cosine_gram.shape[0]
    if old_tool_count <= 4:
        raise ValueError(
            "at least five old tools are required for fourth-neighbor bandwidth"
        )

    neighbor_scores = cosine_gram.clone()
    neighbor_scores.fill_diagonal_(float("-inf"))
    fourth_neighbor_scores = torch.topk(
        neighbor_scores,
        k=4,
        dim=-1,
    ).values[:, -1]
    bandwidth = (
        1.0 - fourth_neighbor_scores
    ).clamp_min(0.0).median().clamp_min(1e-6)
    kernel = torch.exp((cosine_gram - 1.0) / bandwidth)
    return kernel, bandwidth


def _affine_kernel_ridge_weights(
    old_kernel,
    new_kernel,
    ridge_relative,
):
    """Solve kernel ridge weights with an exact sum-to-one constraint."""
    old_kernel = old_kernel.double()
    new_kernel = new_kernel.double()
    old_tool_count = old_kernel.shape[0]
    ridge = (
        float(ridge_relative)
        * old_kernel.diagonal().mean().clamp_min(1e-12)
    )

    system = torch.zeros(
        old_tool_count + 1,
        old_tool_count + 1,
        dtype=torch.float64,
        device=old_kernel.device,
    )
    system[:old_tool_count, :old_tool_count] = (
        old_kernel
        + ridge
        * torch.eye(
            old_tool_count,
            dtype=torch.float64,
            device=old_kernel.device,
        )
    )
    system[:old_tool_count, old_tool_count] = 1.0
    system[old_tool_count, :old_tool_count] = 1.0

    target = torch.cat(
        [
            new_kernel.transpose(0, 1),
            torch.ones(
                1,
                new_kernel.shape[0],
                dtype=torch.float64,
                device=new_kernel.device,
            ),
        ],
        dim=0,
    )
    weights = torch.linalg.solve(system, target)[:old_tool_count]
    return weights.transpose(0, 1), ridge


def query_prototype_bridge_weights(
    document_scores,
    old_document_gram,
    old_query_prototypes,
    convex_result,
    document_ridge_relative=0.1,
    query_ridge_relative=0.1,
    negative_mass_cap=0.1,
    extrapolate_to_cap=False,
):
    """Bridge document similarity to affine parameter-fusion weights.

    New documents first reconstruct a query prototype from old-tool query
    prototypes.  That prediction is then reconstructed in the old query
    prototype geometry.  Both kernel-ridge stages have sum-to-one weights, and
    the final parameter weights are pulled back toward the convex document
    anchor when their total negative mass exceeds the requested cap.
    """
    document_scores = document_scores.double()
    old_document_gram = old_document_gram.double()
    old_query_prototypes = old_query_prototypes.double()

    document_kernel, document_bandwidth = _old_geometry_rbf_kernel(
        old_document_gram
    )
    new_document_kernel = torch.exp(
        (document_scores - 1.0) / document_bandwidth
    )
    document_bridge_weights, document_ridge = (
        _affine_kernel_ridge_weights(
            document_kernel,
            new_document_kernel,
            document_ridge_relative,
        )
    )
    predicted_query_prototypes = (
        document_bridge_weights @ old_query_prototypes
    )

    old_query_mean = old_query_prototypes.mean(dim=0, keepdim=True)
    old_query_centered = functional.normalize(
        old_query_prototypes - old_query_mean,
        p=2,
        dim=-1,
        eps=1e-12,
    )
    predicted_query_centered = functional.normalize(
        predicted_query_prototypes - old_query_mean,
        p=2,
        dim=-1,
        eps=1e-12,
    )
    old_query_gram = (
        old_query_centered @ old_query_centered.transpose(0, 1)
    )
    predicted_query_scores = (
        predicted_query_centered @ old_query_centered.transpose(0, 1)
    )

    query_kernel, query_bandwidth = _old_geometry_rbf_kernel(
        old_query_gram
    )
    predicted_query_kernel = torch.exp(
        (predicted_query_scores - 1.0) / query_bandwidth
    )
    raw_weights, query_ridge = _affine_kernel_ridge_weights(
        query_kernel,
        predicted_query_kernel,
        query_ridge_relative,
    )

    anchor_weights = convex_result["weights"].double()
    final_weights = torch.empty_like(raw_weights)
    ray_scales = []
    for row_index in range(raw_weights.shape[0]):
        if extrapolate_to_cap:
            final_weights[row_index], ray_scale = (
                _move_to_negative_mass_cap(
                    anchor_weights[row_index],
                    raw_weights[row_index],
                    float(negative_mass_cap),
                )
            )
        else:
            final_weights[row_index] = _limit_negative_mass(
                anchor_weights[row_index],
                raw_weights[row_index],
                float(negative_mass_cap),
            )
            direction = raw_weights[row_index] - anchor_weights[row_index]
            ray_scale = float(
                (
                    (final_weights[row_index] - anchor_weights[row_index])
                    * direction
                ).sum()
                / direction.square().sum().clamp_min(1e-12)
            )
        ray_scales.append(ray_scale)

    return {
        "weights": final_weights.float(),
        "raw_weights": raw_weights.float(),
        "convex_anchor_weights": anchor_weights.float(),
        "document_bridge_weights": document_bridge_weights.float(),
        "predicted_query_prototypes": predicted_query_prototypes.float(),
        "old_query_gram": old_query_gram.float(),
        "predicted_query_scores": predicted_query_scores.float(),
        "negative_mass": (
            final_weights.clamp_max(0.0).neg().sum(dim=1)
        ).float(),
        "raw_negative_mass": (
            raw_weights.clamp_max(0.0).neg().sum(dim=1)
        ).float(),
        "ray_scales": torch.tensor(ray_scales, dtype=torch.float32),
        "extrapolate_to_cap": bool(extrapolate_to_cap),
        "document_kernel_bandwidth": document_bandwidth.float(),
        "query_kernel_bandwidth": query_bandwidth.float(),
        "document_ridge_value": document_ridge.float(),
        "query_ridge_value": query_ridge.float(),
        "neighbor_indices": convex_result["neighbor_indices"],
        "neighbor_scores": convex_result["neighbor_scores"],
        "background_scores": convex_result["background_scores"],
        "scores": document_scores.float(),
    }


def local_affine_weights(
    scores,
    old_gram,
    convex_result,
    ridge_relative=0.1,
    negative_mass_cap=0.1,
):
    """Reconstruct each new document locally with bounded affine weights."""
    scores = scores.float()
    old_gram = old_gram.float()
    anchor_weights = convex_result["weights"].float()
    neighbor_indices = convex_result["neighbor_indices"]
    final_weights = torch.zeros_like(anchor_weights)
    raw_weights = torch.zeros_like(anchor_weights)
    ridge_values = []
    reconstruction_errors = []

    for row_index in range(scores.shape[0]):
        indices = neighbor_indices[row_index]
        similarities = scores[row_index, indices]
        gram = old_gram[indices][:, indices]
        covariance = (
            gram
            - similarities[:, None]
            - similarities[None, :]
            + 1.0
        )
        ridge = (
            float(ridge_relative)
            * covariance.diagonal().mean().clamp(min=1e-6)
        )
        anchor = anchor_weights[row_index, indices]
        count = len(indices)
        system = torch.zeros(
            count + 1,
            count + 1,
            dtype=torch.float64,
        )
        system[:count, :count] = (
            covariance + ridge * torch.eye(count)
        ).double()
        system[:count, count] = 1.0
        system[count, :count] = 1.0
        target = torch.cat(
            [
                ridge * anchor,
                torch.ones(1, dtype=anchor.dtype),
            ]
        ).double()
        affine = torch.linalg.solve(system, target)[:count].float()
        limited = _limit_negative_mass(
            anchor,
            affine,
            float(negative_mass_cap),
        )
        raw_weights[row_index, indices] = affine
        final_weights[row_index, indices] = limited
        ridge_values.append(float(ridge.item()))
        reconstruction_errors.append(
            float((limited @ covariance @ limited).item())
        )

    return {
        "weights": final_weights,
        "raw_weights": raw_weights,
        "convex_anchor_weights": anchor_weights,
        "negative_mass": (
            final_weights.clamp_max(0.0).neg().sum(dim=1)
        ),
        "ridge_values": torch.tensor(ridge_values),
        "reconstruction_errors": torch.tensor(reconstruction_errors),
        "neighbor_indices": neighbor_indices,
        "neighbor_scores": convex_result["neighbor_scores"],
        "background_scores": convex_result["background_scores"],
        "scores": scores,
    }


def loo_residual_krr_weights(
    scores,
    old_gram,
    convex_result,
    top_k=4,
    ridge_relative=0.1,
    negative_mass_cap=0.1,
    confidence_gate=True,
):
    """Transfer old leave-one-out residuals with a document-only RBF kernel."""
    scores = scores.float()
    old_gram = old_gram.float()
    old_tool_count = old_gram.shape[0]
    if old_gram.shape != (old_tool_count, old_tool_count):
        raise ValueError("old_gram must be a square matrix")
    if top_k + 1 >= old_tool_count:
        raise ValueError("top_k must leave room for a background old tool")

    loo_scores = old_gram.clone()
    loo_scores.fill_diagonal_(float("-inf"))
    loo_weights = topk_gap_weights(
        loo_scores,
        top_k=top_k,
    )["weights"]
    residual_operator = (
        torch.eye(old_tool_count, dtype=torch.float32) - loo_weights
    )

    fourth_neighbor = min(4, old_tool_count - 1)
    neighbor_scores = torch.topk(
        loo_scores,
        k=fourth_neighbor,
        dim=-1,
    ).values[:, -1]
    bandwidth = (1.0 - neighbor_scores).median().clamp(min=1e-4)
    old_kernel = torch.exp((old_gram - 1.0) / bandwidth)
    new_kernel = torch.exp((scores - 1.0) / bandwidth)
    ridge = (
        float(ridge_relative)
        * old_kernel.diagonal().mean().clamp(min=1e-6)
    )

    system = (
        old_kernel
        + ridge * torch.eye(old_tool_count, dtype=torch.float32)
    ).double()
    cholesky = torch.linalg.cholesky(system)
    residual_solution = torch.cholesky_solve(
        residual_operator.double(),
        cholesky,
    )
    raw_direction = (
        new_kernel.double() @ residual_solution
    ).float()
    raw_direction = (
        raw_direction
        - raw_direction.sum(dim=1, keepdim=True) / old_tool_count
    )

    confidence = torch.ones(scores.shape[0], dtype=torch.float32)
    if confidence_gate:
        kernel_solution = torch.cholesky_solve(
            old_kernel.double(),
            cholesky,
        )
        reference_confidence = (
            old_kernel.double() * kernel_solution.transpose(0, 1)
        ).sum(dim=1).median().clamp(min=1e-12)
        new_solution = torch.cholesky_solve(
            new_kernel.transpose(0, 1).double(),
            cholesky,
        )
        new_confidence = (
            new_kernel.double()
            * new_solution.transpose(0, 1)
        ).sum(dim=1)
        confidence = torch.sqrt(
            (new_confidence / reference_confidence).clamp(min=0.0)
        ).clamp(max=1.0).float()

    anchor_weights = convex_result["weights"].float()
    raw_weights = (
        anchor_weights
        + confidence[:, None] * raw_direction
    )
    final_weights = torch.empty_like(raw_weights)
    for row_index in range(raw_weights.shape[0]):
        final_weights[row_index] = _limit_negative_mass(
            anchor_weights[row_index],
            raw_weights[row_index],
            float(negative_mass_cap),
        )

    return {
        "weights": final_weights,
        "raw_weights": raw_weights,
        "convex_anchor_weights": anchor_weights,
        "loo_anchor_weights": loo_weights,
        "residual_operator": residual_operator,
        "kernel_bandwidth": bandwidth.detach(),
        "ridge_value": ridge.detach(),
        "kernel_confidence": confidence,
        "negative_mass": (
            final_weights.clamp_max(0.0).neg().sum(dim=1)
        ),
        "raw_negative_mass": (
            raw_weights.clamp_max(0.0).neg().sum(dim=1)
        ),
        "neighbor_indices": convex_result["neighbor_indices"],
        "neighbor_scores": convex_result["neighbor_scores"],
        "background_scores": convex_result["background_scores"],
        "scores": scores,
    }
