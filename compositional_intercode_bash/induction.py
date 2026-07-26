"""Vocabulary pruning and train/dev capacity selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .unigram import (
    aggregate_weighted_sequences,
    CandidatePiece,
    Piece,
    ProcedureUnigramModel,
    Signature,
    fit_em,
    initialize_model,
    piece_sort_key,
)


@dataclass
class FittedVocabulary:
    initialization: str
    model: ProcedureUnigramModel
    train_log_likelihood: float
    train_map_objective: float
    em_history: list[dict[str, float]]
    pruning_round: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "initialization": self.initialization,
            "K": self.model.size,
            "train_log_likelihood": self.train_log_likelihood,
            "train_map_objective": self.train_map_objective,
            "rho": self.model.rho,
            "lexicon_hash": self.model.lexicon_hash(),
            "pruning_round": self.pruning_round,
            "em_iterations": len(self.em_history),
        }


def weighted_log_likelihood(
    model: ProcedureUnigramModel,
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
) -> float:
    return float(
        sum(
            float(weight) * model.log_probability(sequence)
            for sequence, weight in zip(sequences, weights)
        )
    )


def map_objective(
    model: ProcedureUnigramModel,
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
    *,
    epsilon: float,
) -> tuple[float, float]:
    log_likelihood = weighted_log_likelihood(model, sequences, weights)
    objective = log_likelihood + epsilon * float(np.log(model.theta).sum())
    return log_likelihood, objective


def delete_piece_without_refit(
    model: ProcedureUnigramModel,
    piece_id: int,
) -> ProcedureUnigramModel:
    if model.mandatory[piece_id]:
        raise ValueError("Mandatory singleton pieces cannot be deleted")
    keep = [index for index in range(model.size) if index != piece_id]
    denominator = 1.0 - float(model.theta[piece_id])
    if denominator <= 0:
        raise ValueError("Cannot renormalize after deleting a probability-one piece")
    return ProcedureUnigramModel(
        [model.pieces[index] for index in keep],
        [float(model.theta[index]) / denominator for index in keep],
        model.rho,
        mandatory=[model.mandatory[index] for index in keep],
    )


def one_step_deletion_scores(
    model: ProcedureUnigramModel,
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
) -> list[tuple[int, float]]:
    baseline = weighted_log_likelihood(model, sequences, weights)
    scores: list[tuple[int, float]] = []
    for piece_id in range(model.size):
        if model.mandatory[piece_id]:
            continue
        reduced = delete_piece_without_refit(model, piece_id)
        reduced_likelihood = weighted_log_likelihood(reduced, sequences, weights)
        scores.append((piece_id, baseline - reduced_likelihood))
    scores.sort(
        key=lambda item: (
            item[1],
            piece_sort_key(model.pieces[item[0]]),
        )
    )
    return scores


def _fit_current(
    model: ProcedureUnigramModel,
    initialization: str,
    pruning_round: int,
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
    *,
    epsilon: float,
    em_max_iterations: int,
) -> FittedVocabulary:
    fitted, history = fit_em(
        model,
        sequences,
        weights,
        epsilon=epsilon,
        max_iterations=em_max_iterations,
    )
    likelihood, objective = map_objective(
        fitted,
        sequences,
        weights,
        epsilon=epsilon,
    )
    return FittedVocabulary(
        initialization=initialization,
        model=fitted,
        train_log_likelihood=likelihood,
        train_map_objective=objective,
        em_history=history,
        pruning_round=pruning_round,
    )


def _remove_set(
    model: ProcedureUnigramModel,
    remove_ids: set[int],
) -> ProcedureUnigramModel:
    if any(model.mandatory[piece_id] for piece_id in remove_ids):
        raise ValueError("Pruning attempted to remove a mandatory piece")
    keep = [index for index in range(model.size) if index not in remove_ids]
    retained_mass = float(model.theta[keep].sum())
    if retained_mass <= 0:
        raise ValueError("Pruning removed all probability mass")
    return ProcedureUnigramModel(
        [model.pieces[index] for index in keep],
        [float(model.theta[index]) / retained_mass for index in keep],
        model.rho,
        mandatory=[model.mandatory[index] for index in keep],
    )


def pruning_path(
    initial_model: ProcedureUnigramModel,
    initialization: str,
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
    *,
    minimum_k: int,
    hidden_size: int,
    native_anchor: int = 247,
    batch_fraction: float = 0.20,
    epsilon: float = 1e-8,
    em_max_iterations: int = 100,
) -> dict[int, FittedVocabulary]:
    """Fit a deterministic pruning path.

    The old experiment treated 247 as a hard model capacity.  Here it is only
    an anchor: converged models above 247 remain valid candidates because the
    tokenizer and embedding table can be expanded.  At every size, 20% of the
    currently removable pieces are deleted as one batch, with exact landing
    points at ``hidden_size`` (when needed), 247, and ``minimum_k``.  Each
    converged node is retained for dev selection.  This gives a deterministic
    logarithmic capacity grid instead of evaluating every adjacent K.  Models
    with K greater than ``hidden_size`` are only intermediate pruning states
    because strict row orthogonality would be impossible.
    """

    mandatory_count = sum(initial_model.mandatory)
    if minimum_k < mandatory_count:
        raise ValueError(
            f"minimum_k={minimum_k} is below mandatory singleton count {mandatory_count}"
        )
    if minimum_k > hidden_size:
        raise ValueError(
            f"minimum_k={minimum_k} exceeds hidden_size={hidden_size}; "
            "strict orthogonal memory initialization is impossible"
        )
    if not 0.0 < batch_fraction < 1.0:
        raise ValueError("batch_fraction must be between zero and one")

    current = initial_model
    fitted = _fit_current(
        current,
        initialization,
        0,
        sequences,
        weights,
        epsilon=epsilon,
        em_max_iterations=em_max_iterations,
    )
    current = fitted.model
    retained: dict[int, FittedVocabulary] = {}
    if current.size <= hidden_size:
        retained[current.size] = fitted

    round_index = 0
    while current.size > minimum_k:
        scores = one_step_deletion_scores(current, sequences, weights)
        if not scores:
            break
        removable = len(scores)
        anchors = {minimum_k}
        if minimum_k <= native_anchor <= hidden_size:
            anchors.add(native_anchor)
        if minimum_k <= hidden_size:
            anchors.add(hidden_size)
        lower_anchors = [value for value in anchors if value < current.size]
        target = max(lower_anchors) if lower_anchors else minimum_k
        delete_count = min(
            max(1, int(math.floor(batch_fraction * removable))),
            current.size - target,
        )
        delete_count = min(delete_count, current.size - minimum_k)
        if delete_count <= 0:
            break
        remove_ids = {piece_id for piece_id, _score in scores[:delete_count]}
        current = _remove_set(current, remove_ids)
        round_index += 1
        fitted = _fit_current(
            current,
            initialization,
            round_index,
            sequences,
            weights,
            epsilon=epsilon,
            em_max_iterations=em_max_iterations,
        )
        current = fitted.model
        if current.size <= hidden_size:
            retained[current.size] = fitted

    if minimum_k not in retained:
        raise RuntimeError(
            f"Pruning stopped at K={current.size} before requested minimum K={minimum_k}"
        )
    return retained


def run_three_initializations(
    candidates: Sequence[CandidatePiece],
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
    *,
    hidden_size: int,
    minimum_k: int | None = None,
    native_anchor: int = 247,
    epsilon: float = 1e-8,
    em_max_iterations: int = 100,
) -> tuple[dict[int, FittedVocabulary], dict[str, dict[int, FittedVocabulary]]]:
    # The likelihood and every EM sufficient statistic are additive in the
    # row weight.  NL2Bash contains many rows with the same utility/operator
    # signature sequence, so collapse them once before the expensive pruning
    # path.  Candidate document frequencies were already computed from the
    # uncollapsed, group-aware records and are intentionally unchanged.
    unique_sequences, unique_weights = aggregate_weighted_sequences(
        sequences,
        weights,
    )
    mandatory_count = sum(candidate.mandatory for candidate in candidates)
    if minimum_k is None:
        minimum_k = mandatory_count
    effective_rows = float(unique_weights.sum())
    singleton_piece_total = float(
        sum(
            weight * len(sequence)
            for sequence, weight in zip(unique_sequences, unique_weights)
        )
    )
    initial_rho = effective_rows / (effective_rows + singleton_piece_total)

    all_paths: dict[str, dict[int, FittedVocabulary]] = {}
    for mode in ("df", "uniform", "df_length"):
        initial = initialize_model(
            candidates,
            mode=mode,
            initial_rho=initial_rho,
            epsilon=epsilon,
        )
        all_paths[mode] = pruning_path(
            initial,
            mode,
            unique_sequences,
            unique_weights,
            minimum_k=minimum_k,
            hidden_size=hidden_size,
            native_anchor=native_anchor,
            epsilon=epsilon,
            em_max_iterations=em_max_iterations,
        )

    candidate_sizes = sorted(set().union(*(path.keys() for path in all_paths.values())))
    winners: dict[int, FittedVocabulary] = {}
    for size in candidate_sizes:
        options = [path[size] for path in all_paths.values() if size in path]
        options.sort(
            key=lambda item: (
                -item.train_map_objective,
                item.model.lexicon_hash(),
            )
        )
        winners[size] = options[0]
    return winners, all_paths


def dev_group_scores(
    model: ProcedureUnigramModel,
    sequences: Sequence[Sequence[Signature]],
    group_ids: Sequence[str],
) -> dict[str, tuple[float, float]]:
    if len(sequences) != len(group_ids):
        raise ValueError("sequences and group_ids have unequal length")
    members: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        members.setdefault(group_id, []).append(index)
    scores: dict[str, tuple[float, float]] = {}
    for group_id, indices in members.items():
        row_weight = 1.0 / len(indices)
        numerator = sum(
            row_weight * model.log_probability(sequences[index]) for index in indices
        )
        denominator = sum(row_weight * len(sequences[index]) for index in indices)
        scores[group_id] = (float(numerator), float(denominator))
    return scores


def paired_one_standard_error_select(
    models: dict[int, FittedVocabulary],
    dev_sequences: Sequence[Sequence[Signature]],
    dev_group_ids: Sequence[str],
    *,
    bootstrap_seed: int = 271828,
    bootstrap_replicates: int = 1000,
) -> tuple[int, dict[str, Any]]:
    if not models:
        raise ValueError("No candidate vocabulary models were supplied")
    sizes = sorted(models)
    per_size = {
        size: dev_group_scores(models[size].model, dev_sequences, dev_group_ids)
        for size in sizes
    }
    group_ids = sorted(set(dev_group_ids))
    if not group_ids:
        raise ValueError("Development set has no groups")
    for size in sizes:
        if set(per_size[size]) != set(group_ids):
            raise AssertionError("Candidate K values do not share the same dev groups")

    def aggregate(size: int, sampled_groups: Sequence[str]) -> float:
        numerator = 0.0
        denominator = 0.0
        for group_id in sampled_groups:
            group_numerator, group_denominator = per_size[size][group_id]
            numerator += group_numerator
            denominator += group_denominator
        return -numerator / denominator

    full_scores = {size: aggregate(size, group_ids) for size in sizes}
    best_full = min(sizes, key=lambda size: (full_scores[size], size))
    rng = np.random.default_rng(bootstrap_seed)
    differences = {size: [] for size in sizes}
    group_array = np.asarray(group_ids, dtype=object)
    for _ in range(bootstrap_replicates):
        sampled = rng.choice(group_array, size=len(group_array), replace=True).tolist()
        best_score = aggregate(best_full, sampled)
        for size in sizes:
            differences[size].append(aggregate(size, sampled) - best_score)

    standard_errors = {
        size: float(np.std(differences[size], ddof=1))
        if bootstrap_replicates > 1
        else 0.0
        for size in sizes
    }
    full_differences = {
        size: full_scores[size] - full_scores[best_full] for size in sizes
    }
    eligible = [
        size
        for size in sizes
        if full_differences[size] <= standard_errors[size]
    ]
    selected = min(eligible)
    report = {
        "schema": "paired_one_se_v1",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "best_full_k": best_full,
        "selected_k": selected,
        "curve": [
            {
                "K": size,
                "dev_nll_per_atom": full_scores[size],
                "difference_from_best": full_differences[size],
                "paired_difference_se": standard_errors[size],
                "initialization": models[size].initialization,
                "lexicon_hash": models[size].model.lexicon_hash(),
            }
            for size in sizes
        ],
    }
    return selected, report
