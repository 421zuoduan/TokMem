"""Unigram procedure induction with exact latent-boundary inference.

The parser supplies an atom sequence.  This module learns probabilities for
repeated contiguous atom sequences and sums over every complete segmentation.
No embedding similarity, fixed span length, or clustering is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


Signature = tuple[str, str]
Piece = tuple[Signature, ...]
NEG_INF = float("-inf")


def _logaddexp(left: float, right: float) -> float:
    if left == NEG_INF:
        return right
    if right == NEG_INF:
        return left
    if left < right:
        left, right = right, left
    return left + math.log1p(math.exp(right - left))


def _logsumexp(values: Iterable[float]) -> float:
    total = NEG_INF
    for value in values:
        total = _logaddexp(total, value)
    return total


def piece_sort_key(piece: Piece) -> bytes:
    flat = "\0".join(f"{connector}\x1f{head}" for connector, head in piece)
    return flat.encode("utf-8")


def piece_to_json(piece: Piece) -> list[list[str]]:
    return [[connector, head] for connector, head in piece]


def piece_from_json(value: Sequence[Sequence[str]]) -> Piece:
    return tuple((str(connector), str(head)) for connector, head in value)


@dataclass(frozen=True)
class Segment:
    piece_id: int
    start: int
    end: int

    def to_dict(self) -> dict[str, int]:
        return {
            "piece_id": self.piece_id,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class ForwardBackward:
    log_alpha: tuple[float, ...]
    log_beta: tuple[float, ...]
    log_partition: float


@dataclass(frozen=True)
class CandidatePiece:
    piece: Piece
    group_df: int
    proper_group_df: int
    mandatory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece": piece_to_json(self.piece),
            "length": len(self.piece),
            "group_df": self.group_df,
            "proper_group_df": self.proper_group_df,
            "mandatory": self.mandatory,
        }


def enumerate_candidates(
    grouped_sequences: Iterable[tuple[str, Sequence[Signature]]],
    *,
    minimum_group_df: int = 2,
) -> list[CandidatePiece]:
    """Fold all occurrences into unique sequence types and count group support."""

    groups: dict[Piece, set[str]] = {}
    proper_groups: dict[Piece, set[str]] = {}
    singletons: set[Piece] = set()
    for group_id, sequence_value in grouped_sequences:
        sequence = tuple(sequence_value)
        local_seen: set[Piece] = set()
        local_proper: set[Piece] = set()
        for start in range(len(sequence)):
            singleton = (sequence[start],)
            singletons.add(singleton)
            for end in range(start + 1, len(sequence) + 1):
                piece = sequence[start:end]
                local_seen.add(piece)
                if start > 0 or end < len(sequence):
                    local_proper.add(piece)
        for piece in local_seen:
            groups.setdefault(piece, set()).add(group_id)
        for piece in local_proper:
            proper_groups.setdefault(piece, set()).add(group_id)

    candidates: list[CandidatePiece] = []
    for piece, support_groups in groups.items():
        mandatory = len(piece) == 1
        proper_df = len(proper_groups.get(piece, ()))
        if mandatory or (
            len(support_groups) >= minimum_group_df and proper_df >= 1
        ):
            candidates.append(
                CandidatePiece(
                    piece=piece,
                    group_df=len(support_groups),
                    proper_group_df=proper_df,
                    mandatory=mandatory,
                )
            )
    candidates.sort(key=lambda item: (not item.mandatory, piece_sort_key(item.piece)))
    return candidates


class ProcedureUnigramModel:
    """A normalized unigram model over complete procedure segmentations."""

    def __init__(
        self,
        pieces: Sequence[Piece],
        theta: Sequence[float],
        rho: float,
        *,
        mandatory: Sequence[bool] | None = None,
    ) -> None:
        if len(pieces) == 0:
            raise ValueError("Procedure vocabulary must not be empty")
        if len(pieces) != len(theta):
            raise ValueError("pieces and theta must have equal length")
        if len(set(pieces)) != len(pieces):
            raise ValueError("Procedure vocabulary contains duplicate pieces")
        theta_array = np.asarray(theta, dtype=np.float64)
        if not np.all(np.isfinite(theta_array)) or np.any(theta_array <= 0):
            raise ValueError("All theta values must be finite and positive")
        theta_sum = float(theta_array.sum())
        if not math.isclose(theta_sum, 1.0, rel_tol=1e-10, abs_tol=1e-12):
            theta_array = theta_array / theta_sum
        if not 0.0 < float(rho) < 1.0:
            raise ValueError("rho must be strictly between zero and one")

        self.pieces = tuple(tuple(piece) for piece in pieces)
        self.theta = theta_array
        self.rho = float(rho)
        if mandatory is None:
            self.mandatory = tuple(len(piece) == 1 for piece in self.pieces)
        else:
            if len(mandatory) != len(self.pieces):
                raise ValueError("mandatory mask has the wrong size")
            self.mandatory = tuple(bool(value) for value in mandatory)

        self.piece_to_id = {piece: index for index, piece in enumerate(self.pieces)}
        self._piece_ids_by_first: dict[Signature, list[int]] = {}
        for piece_id, piece in enumerate(self.pieces):
            self._piece_ids_by_first.setdefault(piece[0], []).append(piece_id)
        for piece_ids in self._piece_ids_by_first.values():
            piece_ids.sort(key=lambda index: (len(self.pieces[index]), index))

    @property
    def size(self) -> int:
        return len(self.pieces)

    @property
    def log_weights(self) -> np.ndarray:
        return np.log1p(-self.rho) + np.log(self.theta)

    def inventory_hash(self) -> str:
        """Hash only the ordered procedure inventory.

        This is useful when only the mapping from ``piece_id`` to signatures
        matters.  It deliberately does not claim that two fitted unigram
        models with the same pieces are the same model.
        """

        digest = hashlib.sha256()
        for piece in self.pieces:
            digest.update(piece_sort_key(piece))
            digest.update(b"\n")
        return digest.hexdigest()

    def model_hash(self) -> str:
        """Hash every value that can change segmentation probabilities."""

        payload = {
            "schema": "procedure_unigram_model_hash_v1",
            "rho_hex": float(self.rho).hex(),
            "pieces": [
                {
                    "piece_id": piece_id,
                    "signatures": piece_to_json(piece),
                    "theta_hex": float(self.theta[piece_id]).hex(),
                    "mandatory": bool(self.mandatory[piece_id]),
                }
                for piece_id, piece in enumerate(self.pieces)
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def lexicon_hash(self) -> str:
        """Backward-compatible name for the complete fitted-model hash."""

        return self.model_hash()

    def arcs_from(
        self,
        sequence: Sequence[Signature],
        start: int,
    ) -> Iterator[tuple[int, int]]:
        if start >= len(sequence):
            return
        for piece_id in self._piece_ids_by_first.get(sequence[start], ()):
            piece = self.pieces[piece_id]
            end = start + len(piece)
            if end <= len(sequence) and tuple(sequence[start:end]) == piece:
                yield end, piece_id

    def all_arcs(
        self,
        sequence: Sequence[Signature],
    ) -> list[list[tuple[int, int]]]:
        return [list(self.arcs_from(sequence, start)) for start in range(len(sequence))]

    def _assert_covered(self, sequence: Sequence[Signature], log_partition: float) -> None:
        if log_partition == NEG_INF:
            raise ValueError(
                "No complete segmentation exists; mandatory singleton coverage is broken "
                f"for sequence of length {len(sequence)}"
            )

    def forward_backward(self, sequence: Sequence[Signature]) -> ForwardBackward:
        length = len(sequence)
        if length == 0:
            raise ValueError("Empty atom sequences are not supported")
        arcs = self.all_arcs(sequence)
        log_weights = self.log_weights
        alpha = [NEG_INF] * (length + 1)
        alpha[0] = 0.0
        for start in range(length):
            if alpha[start] == NEG_INF:
                continue
            for end, piece_id in arcs[start]:
                alpha[end] = _logaddexp(
                    alpha[end],
                    alpha[start] + float(log_weights[piece_id]),
                )

        beta = [NEG_INF] * (length + 1)
        beta[length] = 0.0
        for start in range(length - 1, -1, -1):
            beta[start] = _logsumexp(
                float(log_weights[piece_id]) + beta[end]
                for end, piece_id in arcs[start]
                if beta[end] != NEG_INF
            )

        self._assert_covered(sequence, alpha[length])
        if not math.isclose(alpha[length], beta[0], rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(
                f"Forward/backward mismatch: alpha={alpha[length]}, beta={beta[0]}"
            )
        return ForwardBackward(tuple(alpha), tuple(beta), alpha[length])

    def log_probability(self, sequence: Sequence[Signature]) -> float:
        result = self.forward_backward(sequence)
        return math.log(self.rho) + result.log_partition

    def arc_posteriors(
        self,
        sequence: Sequence[Signature],
    ) -> list[tuple[int, int, int, float]]:
        result = self.forward_backward(sequence)
        log_weights = self.log_weights
        values: list[tuple[int, int, int, float]] = []
        for start, arcs in enumerate(self.all_arcs(sequence)):
            for end, piece_id in arcs:
                log_posterior = (
                    result.log_alpha[start]
                    + float(log_weights[piece_id])
                    + result.log_beta[end]
                    - result.log_partition
                )
                values.append((start, end, piece_id, math.exp(log_posterior)))
        return values

    def expected_piece_counts(self, sequence: Sequence[Signature]) -> np.ndarray:
        counts = np.zeros(self.size, dtype=np.float64)
        for _start, _end, piece_id, probability in self.arc_posteriors(sequence):
            counts[piece_id] += probability
        return counts

    @staticmethod
    def _boundary_bits(segments: Sequence[Segment], length: int) -> tuple[int, ...]:
        boundaries = {segment.end for segment in segments[:-1]}
        return tuple(1 if position in boundaries else 0 for position in range(1, length))

    def viterbi(self, sequence: Sequence[Signature]) -> tuple[list[Segment], float]:
        length = len(sequence)
        log_weights = self.log_weights
        states: list[tuple[float, list[Segment]] | None] = [None] * (length + 1)
        states[0] = (0.0, [])

        def better(
            candidate: tuple[float, list[Segment]],
            incumbent: tuple[float, list[Segment]] | None,
        ) -> bool:
            if incumbent is None:
                return True
            candidate_score, candidate_path = candidate
            incumbent_score, incumbent_path = incumbent
            if candidate_score != incumbent_score:
                return candidate_score > incumbent_score
            if len(candidate_path) != len(incumbent_path):
                return len(candidate_path) < len(incumbent_path)
            candidate_bits = self._boundary_bits(candidate_path, length)
            incumbent_bits = self._boundary_bits(incumbent_path, length)
            if candidate_bits != incumbent_bits:
                return candidate_bits < incumbent_bits
            return tuple(x.piece_id for x in candidate_path) < tuple(
                x.piece_id for x in incumbent_path
            )

        for start in range(length):
            state = states[start]
            if state is None:
                continue
            score, path = state
            for end, piece_id in self.arcs_from(sequence, start):
                candidate = (
                    score + float(log_weights[piece_id]),
                    [*path, Segment(piece_id, start, end)],
                )
                if better(candidate, states[end]):
                    states[end] = candidate
        if states[length] is None:
            raise ValueError("No complete segmentation exists")
        path_score, path = states[length]
        return path, math.log(self.rho) + path_score

    def count_forward(
        self,
        sequence: Sequence[Signature],
        piece_count: int,
    ) -> np.ndarray:
        if piece_count <= 0:
            raise ValueError("piece_count must be positive")
        length = len(sequence)
        alpha = np.full((piece_count + 1, length + 1), NEG_INF, dtype=np.float64)
        alpha[0, 0] = 0.0
        log_weights = self.log_weights
        for used in range(1, piece_count + 1):
            for start in range(length):
                if not np.isfinite(alpha[used - 1, start]):
                    continue
                for end, piece_id in self.arcs_from(sequence, start):
                    alpha[used, end] = np.logaddexp(
                        alpha[used, end],
                        alpha[used - 1, start] + log_weights[piece_id],
                    )
        if not np.isfinite(alpha[piece_count, length]):
            raise ValueError(
                f"No segmentation with J={piece_count} for sequence length {length}"
            )
        return alpha

    def count_path_count(self, sequence: Sequence[Signature], piece_count: int) -> int:
        length = len(sequence)
        counts = [[0] * (length + 1) for _ in range(piece_count + 1)]
        counts[0][0] = 1
        for used in range(1, piece_count + 1):
            for start in range(length):
                for end, _piece_id in self.arcs_from(sequence, start):
                    counts[used][end] += counts[used - 1][start]
        return counts[piece_count][length]

    def fixed_count_log_partition(
        self,
        sequence: Sequence[Signature],
        piece_count: int,
    ) -> float:
        return float(self.count_forward(sequence, piece_count)[piece_count, len(sequence)])

    def segmentation_log_weight(self, segments: Sequence[Segment]) -> float:
        return float(sum(self.log_weights[segment.piece_id] for segment in segments))

    def fixed_count_map_mass(
        self,
        sequence: Sequence[Signature],
        map_segments: Sequence[Segment] | None = None,
    ) -> float:
        if map_segments is None:
            map_segments, _ = self.viterbi(sequence)
        log_z = self.fixed_count_log_partition(sequence, len(map_segments))
        return math.exp(self.segmentation_log_weight(map_segments) - log_z)

    def sample_fixed_count(
        self,
        sequence: Sequence[Signature],
        piece_count: int,
        rng: np.random.Generator,
    ) -> list[Segment]:
        alpha = self.count_forward(sequence, piece_count)
        log_weights = self.log_weights
        end = len(sequence)
        path_reversed: list[Segment] = []
        for used in range(piece_count, 0, -1):
            candidates: list[tuple[int, int]] = []
            candidate_log_weights: list[float] = []
            for start in range(end):
                if not np.isfinite(alpha[used - 1, start]):
                    continue
                for candidate_end, piece_id in self.arcs_from(sequence, start):
                    if candidate_end == end:
                        candidates.append((start, piece_id))
                        candidate_log_weights.append(
                            float(alpha[used - 1, start] + log_weights[piece_id])
                        )
            if not candidates:
                raise AssertionError("FFBS reached an impossible backward state")
            maximum = max(candidate_log_weights)
            probabilities = np.exp(np.asarray(candidate_log_weights) - maximum)
            probabilities /= probabilities.sum()
            choice = int(rng.choice(len(candidates), p=probabilities))
            start, piece_id = candidates[choice]
            path_reversed.append(Segment(piece_id, start, end))
            end = start
        if end != 0:
            raise AssertionError("FFBS did not return to sequence start")
        return list(reversed(path_reversed))

    def sample_full(
        self,
        sequence: Sequence[Signature],
        rng: np.random.Generator,
    ) -> list[Segment]:
        result = self.forward_backward(sequence)
        log_weights = self.log_weights
        end = len(sequence)
        path_reversed: list[Segment] = []
        while end > 0:
            candidates: list[tuple[int, int]] = []
            candidate_log_weights: list[float] = []
            for start in range(end):
                if result.log_alpha[start] == NEG_INF:
                    continue
                for candidate_end, piece_id in self.arcs_from(sequence, start):
                    if candidate_end == end:
                        candidates.append((start, piece_id))
                        candidate_log_weights.append(
                            result.log_alpha[start] + float(log_weights[piece_id])
                        )
            maximum = max(candidate_log_weights)
            probabilities = np.exp(np.asarray(candidate_log_weights) - maximum)
            probabilities /= probabilities.sum()
            choice = int(rng.choice(len(candidates), p=probabilities))
            start, piece_id = candidates[choice]
            path_reversed.append(Segment(piece_id, start, end))
            end = start
        return list(reversed(path_reversed))

    def conditional_entropy(
        self,
        sequence: Sequence[Signature],
        piece_count: int,
    ) -> float:
        """Exact entropy of q(B | x, J) via count-conditioned arc marginals."""

        length = len(sequence)
        alpha = self.count_forward(sequence, piece_count)
        log_weights = self.log_weights
        beta = np.full((piece_count + 1, length + 1), NEG_INF, dtype=np.float64)
        beta[0, length] = 0.0
        for remaining in range(1, piece_count + 1):
            for start in range(length - 1, -1, -1):
                beta[remaining, start] = _logsumexp(
                    float(log_weights[piece_id]) + beta[remaining - 1, end]
                    for end, piece_id in self.arcs_from(sequence, start)
                    if np.isfinite(beta[remaining - 1, end])
                )
        log_z = float(alpha[piece_count, length])
        expected_log_weight = 0.0
        for used_before in range(piece_count):
            remaining_after = piece_count - used_before - 1
            for start in range(length):
                if not np.isfinite(alpha[used_before, start]):
                    continue
                for end, piece_id in self.arcs_from(sequence, start):
                    if not np.isfinite(beta[remaining_after, end]):
                        continue
                    log_probability = (
                        float(alpha[used_before, start])
                        + float(log_weights[piece_id])
                        + float(beta[remaining_after, end])
                        - log_z
                    )
                    expected_log_weight += math.exp(log_probability) * float(
                        log_weights[piece_id]
                    )
        return log_z - expected_log_weight

    def to_dict(self) -> dict[str, Any]:
        inventory_hash = self.inventory_hash()
        model_hash = self.model_hash()
        return {
            "schema": "procedure_unigram_v2",
            "rho": self.rho,
            "pieces": [
                {
                    "piece_id": piece_id,
                    "signatures": piece_to_json(piece),
                    "theta": float(self.theta[piece_id]),
                    "mandatory": self.mandatory[piece_id],
                }
                for piece_id, piece in enumerate(self.pieces)
            ],
            "inventory_hash": inventory_hash,
            "model_hash": model_hash,
            # Keep the old field name so training/checkpoint callers do not
            # need a second migration.  In v2 it is the complete model hash.
            "lexicon_hash": model_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcedureUnigramModel":
        records = sorted(value["pieces"], key=lambda item: int(item["piece_id"]))
        model = cls(
            [piece_from_json(item["signatures"]) for item in records],
            [float(item["theta"]) for item in records],
            float(value["rho"]),
            mandatory=[bool(item["mandatory"]) for item in records],
        )
        expected_inventory_hash = value.get("inventory_hash")
        if (
            expected_inventory_hash
            and model.inventory_hash() != expected_inventory_hash
        ):
            raise ValueError("Procedure inventory hash mismatch")
        expected_model_hash = value.get("model_hash")
        if expected_model_hash and model.model_hash() != expected_model_hash:
            raise ValueError("Procedure model hash mismatch")

        # v1 used ``lexicon_hash`` for the ordered pieces only.  New artifacts
        # use the same field as a compatibility alias for the complete model
        # hash, so old downloaded artifacts remain readable but cannot be
        # mistaken for newly verified segmentation artifacts.
        expected_legacy_hash = value.get("lexicon_hash")
        if expected_legacy_hash:
            if value.get("schema") == "procedure_unigram_v1":
                actual_legacy_hash = model.inventory_hash()
            else:
                actual_legacy_hash = model.model_hash()
            if actual_legacy_hash != expected_legacy_hash:
                raise ValueError("Procedure lexicon hash mismatch")
        return model


def aggregate_weighted_sequences(
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
) -> tuple[list[tuple[Signature, ...]], np.ndarray]:
    """Combine identical atom sequences without changing a weighted objective."""

    if len(sequences) != len(weights):
        raise ValueError("sequences and weights have unequal length")
    grouped_weights: dict[tuple[Signature, ...], list[float]] = {}
    for sequence_value, weight_value in zip(sequences, weights):
        sequence = tuple(sequence_value)
        weight = float(weight_value)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Sequence weights must be finite and positive")
        grouped_weights.setdefault(sequence, []).append(weight)

    ordered_sequences = sorted(grouped_weights)
    aggregated_weights = np.asarray(
        [math.fsum(grouped_weights[sequence]) for sequence in ordered_sequences],
        dtype=np.float64,
    )
    return ordered_sequences, aggregated_weights


def group_balanced_weights(group_ids: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for group_id in group_ids:
        counts[group_id] = counts.get(group_id, 0) + 1
    return np.asarray([1.0 / counts[group_id] for group_id in group_ids], dtype=np.float64)


def initialize_model(
    candidates: Sequence[CandidatePiece],
    *,
    mode: str,
    initial_rho: float,
    epsilon: float = 1e-8,
) -> ProcedureUnigramModel:
    if mode not in {"df", "uniform", "df_length"}:
        raise ValueError(f"Unknown initialization mode: {mode}")
    if mode == "uniform":
        raw = np.ones(len(candidates), dtype=np.float64)
    elif mode == "df":
        raw = np.asarray([candidate.group_df for candidate in candidates], dtype=np.float64)
    else:
        raw = np.asarray(
            [candidate.group_df * len(candidate.piece) for candidate in candidates],
            dtype=np.float64,
        )
    raw += epsilon
    raw /= raw.sum()
    return ProcedureUnigramModel(
        [candidate.piece for candidate in candidates],
        raw,
        initial_rho,
        mandatory=[candidate.mandatory for candidate in candidates],
    )


def fit_em(
    model: ProcedureUnigramModel,
    sequences: Sequence[Sequence[Signature]],
    weights: Sequence[float],
    *,
    epsilon: float = 1e-8,
    relative_tolerance: float = 1e-6,
    patience: int = 3,
    max_iterations: int = 100,
) -> tuple[ProcedureUnigramModel, list[dict[str, float]]]:
    if len(sequences) != len(weights):
        raise ValueError("sequences and weights have unequal length")
    weights_array = np.asarray(weights, dtype=np.float64)
    if np.any(weights_array <= 0) or not np.all(np.isfinite(weights_array)):
        raise ValueError("EM weights must be finite and positive")

    history: list[dict[str, float]] = []
    stable_iterations = 0
    previous_objective: float | None = None
    current = model
    for iteration in range(max_iterations):
        expected_counts = np.zeros(current.size, dtype=np.float64)
        log_likelihood = 0.0
        expected_piece_total = 0.0
        for sequence, weight in zip(sequences, weights_array):
            log_likelihood += float(weight) * current.log_probability(sequence)
            row_counts = current.expected_piece_counts(sequence)
            expected_counts += float(weight) * row_counts
            expected_piece_total += float(weight) * float(row_counts.sum())

        theta = (expected_counts + epsilon) / (
            expected_piece_total + current.size * epsilon
        )
        effective_rows = float(weights_array.sum())
        rho = effective_rows / (effective_rows + expected_piece_total)
        updated = ProcedureUnigramModel(
            current.pieces,
            theta,
            rho,
            mandatory=current.mandatory,
        )
        updated_log_likelihood = sum(
            float(weight) * updated.log_probability(sequence)
            for sequence, weight in zip(sequences, weights_array)
        )
        objective = updated_log_likelihood + epsilon * float(np.log(updated.theta).sum())
        history.append(
            {
                "iteration": float(iteration + 1),
                "log_likelihood": float(updated_log_likelihood),
                "map_objective": float(objective),
                "rho": float(rho),
                "expected_pieces": float(expected_piece_total),
            }
        )
        if previous_objective is not None:
            denominator = max(1.0, abs(previous_objective))
            relative_change = (objective - previous_objective) / denominator
            if relative_change < -1e-9:
                raise RuntimeError(
                    "EM MAP objective decreased beyond numerical tolerance: "
                    f"{previous_objective} -> {objective}"
                )
            stable_iterations = (
                stable_iterations + 1
                if relative_change < relative_tolerance
                else 0
            )
        previous_objective = objective
        current = updated
        if stable_iterations >= patience:
            break
    return current, history
