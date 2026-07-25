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
    old_hidden_states = old_hidden_states.float()
    new_hidden_states = new_hidden_states.float()
    old_mean = old_hidden_states.mean(dim=0, keepdim=True)
    old_centered = _normalize_rows(old_hidden_states - old_mean)
    new_centered = _normalize_rows(new_hidden_states - old_mean)
    return new_centered @ old_centered.transpose(0, 1)


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
