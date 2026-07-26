"""Balanced presentation manifests and fixed-count boundary views."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .io_utils import stable_id
from .unigram import ProcedureUnigramModel, Segment, Signature


MIN_PRIMARY_AMBIGUOUS_GROUPS = 20
MIN_PRIMARY_AMBIGUOUS_LONG_GROUP_FRACTION = 0.05
MIN_PRIMARY_REGULAR_GROUPS = 100


def required_primary_ambiguous_groups(long_group_count: int) -> int:
    return max(
        MIN_PRIMARY_AMBIGUOUS_GROUPS,
        math.ceil(
            MIN_PRIMARY_AMBIGUOUS_LONG_GROUP_FRACTION * long_group_count
        ),
    )


def _hash_order(*parts: object) -> bytes:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).digest()


def segmentation_metadata(
    model: ProcedureUnigramModel,
    sequence: Sequence[Signature],
) -> dict[str, Any]:
    map_segments, map_score = model.viterbi(sequence)
    piece_count = len(map_segments)
    same_count_paths = model.count_path_count(sequence, piece_count)
    map_mass = model.fixed_count_map_mass(sequence, map_segments)
    entropy = model.conditional_entropy(sequence, piece_count)
    effective_paths = math.exp(entropy)
    is_ambiguous = (
        piece_count >= 2
        and same_count_paths >= 2
        and 1.0 - map_mass >= 0.25
    )
    return {
        "map_segments": [segment.to_dict() for segment in map_segments],
        "map_score": map_score,
        "J_map": piece_count,
        "same_count_path_count": same_count_paths,
        "same_count_map_mass": map_mass,
        "same_count_entropy": entropy,
        "effective_path_count": effective_paths,
        "is_ambiguous": is_ambiguous,
    }


def attach_segmentation_metadata(
    atom_records: Sequence[Mapping[str, Any]],
    model: ProcedureUnigramModel,
) -> list[dict[str, Any]]:
    output = []
    inventory_hash = model.inventory_hash()
    model_hash = model.model_hash()
    for record in atom_records:
        sequence = tuple(
            (str(signature[0]), str(signature[1]))
            for signature in record["canonical_signatures"]
        )
        output.append(
            {
                **record,
                # ``lexicon_hash`` remains as a compatibility field, but now
                # identifies the complete fitted model rather than just its
                # ordered pieces.
                "lexicon_hash": model_hash,
                "procedure_inventory_hash": inventory_hash,
                "procedure_model_hash": model_hash,
                "procedure_count": model.size,
                **segmentation_metadata(model, sequence),
            }
        )
    return output


def _choose_group_row(
    group_id: str,
    rows: Sequence[Mapping[str, Any]],
    selection_count: int,
) -> Mapping[str, Any]:
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    return ordered[selection_count % len(ordered)]


def _segments_key(
    segments: Sequence[Segment | Mapping[str, Any]],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            int(segment.piece_id if isinstance(segment, Segment) else segment["piece_id"]),
            int(segment.start if isinstance(segment, Segment) else segment["start"]),
            int(segment.end if isinstance(segment, Segment) else segment["end"]),
        )
        for segment in segments
    )


def _best_fixed_count_path_containing(
    model: ProcedureUnigramModel,
    sequence: Sequence[Signature],
    piece_id: int,
    piece_count: int,
) -> list[Segment] | None:
    """Highest-weight J-piece path constrained to contain ``piece_id``."""

    length = len(sequence)
    states: dict[
        tuple[int, int, bool],
        tuple[float, tuple[Segment, ...]],
    ] = {(0, 0, False): (0.0, ())}
    for used in range(piece_count):
        layer = [
            (key, value)
            for key, value in states.items()
            if key[0] == used
        ]
        for (_used, start, seen), (score, path) in layer:
            for end, candidate_id in model.arcs_from(sequence, start):
                key = (used + 1, end, seen or candidate_id == piece_id)
                candidate_path = (*path, Segment(candidate_id, start, end))
                candidate = (
                    score + float(model.log_weights[candidate_id]),
                    candidate_path,
                )
                incumbent = states.get(key)
                if (
                    incumbent is None
                    or candidate[0] > incumbent[0]
                    or (
                        candidate[0] == incumbent[0]
                        and _segments_key(candidate_path)
                        < _segments_key(incumbent[1])
                    )
                ):
                    states[key] = candidate
    result = states.get((piece_count, length, True))
    return list(result[1]) if result is not None else None


def _singleton_anchor_path(
    model: ProcedureUnigramModel,
    sequence: Sequence[Signature],
    piece_id: int,
) -> list[Segment] | None:
    """Build a legal path containing one target and singleton context."""

    candidates: list[tuple[float, list[Segment]]] = []
    for start in range(len(sequence)):
        for end, candidate_id in model.arcs_from(sequence, start):
            if candidate_id != piece_id:
                continue
            path: list[Segment] = []
            valid = True
            for position in range(start):
                singleton_id = model.piece_to_id.get((sequence[position],))
                if singleton_id is None:
                    valid = False
                    break
                path.append(Segment(singleton_id, position, position + 1))
            if not valid:
                continue
            path.append(Segment(piece_id, start, end))
            for position in range(end, len(sequence)):
                singleton_id = model.piece_to_id.get((sequence[position],))
                if singleton_id is None:
                    valid = False
                    break
                path.append(Segment(singleton_id, position, position + 1))
            if valid:
                candidates.append((model.segmentation_log_weight(path), path))
    if not candidates:
        return None
    candidates.sort(key=lambda value: (-value[0], _segments_key(value[1])))
    return candidates[0][1]


def _coverage_anchor_candidates(
    train_records: Sequence[Mapping[str, Any]],
    model: ProcedureUnigramModel,
) -> list[dict[str, Any]]:
    """Create TRAIN-only legal paths from which deterministic set cover can choose."""

    candidates: dict[tuple[str, tuple[tuple[int, int, int], ...]], dict[str, Any]] = {}
    best_for_piece: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for row in sorted(train_records, key=lambda value: value["sample_id"]):
        sequence = tuple(
            (str(value[0]), str(value[1]))
            for value in row["canonical_signatures"]
        )
        map_segments = [
            Segment(
                int(value["piece_id"]),
                int(value["start"]),
                int(value["end"]),
            )
            for value in row["map_segments"]
        ]
        map_key = _segments_key(map_segments)
        map_candidate = {
            "sample_id": str(row["sample_id"]),
            "template_group_id": str(row["template_group_id"]),
            "segments": map_segments,
            "segment_key": map_key,
            "piece_ids": frozenset(segment.piece_id for segment in map_segments),
            "differs_from_map": False,
            "fixed_J": True,
        }
        candidates[(map_candidate["sample_id"], map_key)] = map_candidate

        occurring_ids = {
            piece_id
            for start in range(len(sequence))
            for _end, piece_id in model.arcs_from(sequence, start)
        }
        for piece_id in occurring_ids:
            path = _best_fixed_count_path_containing(
                model,
                sequence,
                piece_id,
                len(map_segments),
            )
            fixed_j = path is not None
            if path is None:
                path = _singleton_anchor_path(model, sequence, piece_id)
            if path is None:
                continue
            path_key = _segments_key(path)
            candidate = {
                "sample_id": str(row["sample_id"]),
                "template_group_id": str(row["template_group_id"]),
                "segments": path,
                "segment_key": path_key,
                "piece_ids": frozenset(segment.piece_id for segment in path),
                "differs_from_map": path_key != map_key,
                "fixed_J": fixed_j,
            }
            candidates[(candidate["sample_id"], path_key)] = candidate
            preference = (
                int(fixed_j),
                model.segmentation_log_weight(path),
                -len(path),
                str(row["sample_id"]),
                path_key,
            )
            incumbent = best_for_piece.get(piece_id)
            if incumbent is None or preference > incumbent[0]:
                best_for_piece[piece_id] = (preference, candidate)

    missing_occurrence = sorted(set(range(model.size)) - set(best_for_piece))
    if missing_occurrence:
        raise RuntimeError(
            "Procedure inventory contains TRAIN-unobservable pieces: "
            f"{missing_occurrence}"
        )
    for _preference, candidate in best_for_piece.values():
        candidates[(candidate["sample_id"], candidate["segment_key"])] = candidate
    return sorted(
        candidates.values(),
        key=lambda value: (
            value["sample_id"],
            value["segment_key"],
        ),
    )


def _greedy_multicover(
    candidates: Sequence[Mapping[str, Any]],
    procedure_count: int,
    minimum_exposures: int,
    presentation_seed: int,
) -> list[dict[str, Any]]:
    remaining = [minimum_exposures] * procedure_count
    selected: list[dict[str, Any]] = []
    while any(value > 0 for value in remaining):
        candidates_with_gain = []
        for candidate in candidates:
            gain = sum(
                remaining[piece_id] > 0
                for piece_id in candidate["piece_ids"]
            )
            if gain:
                candidates_with_gain.append((candidate, gain))
        if not candidates_with_gain:
            uncovered = [
                piece_id for piece_id, value in enumerate(remaining) if value > 0
            ]
            raise RuntimeError(
                f"TRAIN-only coverage set cover cannot expose procedures {uncovered}"
            )
        best_gain = max(gain for _candidate, gain in candidates_with_gain)
        gain_ties = [
            candidate
            for candidate, gain in candidates_with_gain
            if gain == best_gain
        ]
        best_fixed_j = max(
            int(bool(candidate["fixed_J"])) for candidate in gain_ties
        )
        finalists = [
            candidate
            for candidate in gain_ties
            if int(bool(candidate["fixed_J"])) == best_fixed_j
        ]
        chosen = dict(
            min(
                finalists,
                key=lambda candidate: _hash_order(
                    presentation_seed,
                    candidate["sample_id"],
                    candidate["segment_key"],
                    len(selected),
                ),
            )
        )
        selected.append(chosen)
        for piece_id in chosen["piece_ids"]:
            if remaining[piece_id] > 0:
                remaining[piece_id] -= 1
    return selected


def build_balanced_presentations(
    train_records: Sequence[Mapping[str, Any]],
    model: ProcedureUnigramModel,
    *,
    epochs: int = 100,
    presentation_seed: int = 314159,
    minimum_procedure_exposures: int = 10,
    primary: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minimum_procedure_exposures <= 0:
        raise ValueError("minimum_procedure_exposures must be positive")
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for record in train_records:
        by_group.setdefault(record["template_group_id"], []).append(record)
    ambiguous_rows = {
        record["sample_id"] for record in train_records if record["is_ambiguous"]
    }
    a_groups = {
        group_id
        for group_id, rows in by_group.items()
        if any(row["sample_id"] in ambiguous_rows for row in rows)
    }
    r_groups = set(by_group) - a_groups
    long_groups = {
        group_id
        for group_id, rows in by_group.items()
        if any(len(row["canonical_signatures"]) >= 3 for row in rows)
    }
    required_a = required_primary_ambiguous_groups(len(long_groups))
    gates = {
        "A_groups": len(a_groups),
        "R_groups": len(r_groups),
        "L_ge_3_groups": len(long_groups),
        "required_A_groups": required_a,
        "A_gate_passed": len(a_groups) >= required_a,
        "R_gate_passed": len(r_groups) >= MIN_PRIMARY_REGULAR_GROUPS,
    }
    if primary and (not gates["A_gate_passed"] or not gates["R_gate_passed"]):
        raise RuntimeError(f"Primary A/R group support gate failed: {gates}")
    if len(a_groups) < 2 or len(r_groups) < 2:
        raise RuntimeError(
            "Balanced views require at least two ambiguous and two regular template groups"
        )
    per_epoch_pool_size = 2 * (min(len(a_groups), len(r_groups)) // 2)
    if per_epoch_pool_size <= 0:
        raise RuntimeError("No balanced A/R batches can be formed")

    coverage_candidates = _coverage_anchor_candidates(train_records, model)
    selected_anchors = _greedy_multicover(
        coverage_candidates,
        model.size,
        minimum_procedure_exposures,
        presentation_seed,
    )
    required: dict[str, list[dict[str, Any]]] = {"A": [], "R": []}
    for anchor_index, anchor in enumerate(selected_anchors):
        pool = (
            "A"
            if anchor["template_group_id"] in a_groups
            else "R"
        )
        required[pool].append(
            {
                "pool": pool,
                "template_group_id": anchor["template_group_id"],
                "sample_id": anchor["sample_id"],
                "presentation_kind": "coverage_anchor",
                "anchor_segments": [
                    segment.to_dict() for segment in anchor["segments"]
                ],
                "anchor_fixed_J": bool(anchor["fixed_J"]),
                "anchor_index": anchor_index,
            }
        )
    for group_id in sorted(a_groups):
        row = _choose_group_row(
            group_id,
            [
                row
                for row in by_group[group_id]
                if row["sample_id"] in ambiguous_rows
            ],
            0,
        )
        required["A"].append(
            {
                "pool": "A",
                "template_group_id": group_id,
                "sample_id": str(row["sample_id"]),
                "presentation_kind": "boundary_sample",
                "anchor_segments": None,
                "anchor_fixed_J": True,
                "anchor_index": None,
            }
        )

    def filler(pool: str, group_id: str, index: int) -> dict[str, Any]:
        rows = by_group[group_id]
        if pool == "A":
            rows = [
                row for row in rows if row["sample_id"] in ambiguous_rows
            ]
        row = _choose_group_row(group_id, rows, index)
        anchor_segments = row["map_segments"] if pool == "R" else None
        return {
            "pool": pool,
            "template_group_id": group_id,
            "sample_id": str(row["sample_id"]),
            "presentation_kind": (
                "boundary_sample" if pool == "A" else "reference_anchor"
            ),
            "anchor_segments": anchor_segments,
            "anchor_fixed_J": True,
            "anchor_index": None,
        }

    group_coverage_epoch_floor = math.ceil(
        max(len(a_groups), len(r_groups)) / per_epoch_pool_size
    )
    actual_epochs = max(
        epochs,
        group_coverage_epoch_floor,
        math.ceil(
            max(len(required["A"]), len(required["R"]))
            / per_epoch_pool_size
        ),
    )
    scheduled: dict[str, list[dict[str, Any]]]
    epoch_buckets: list[dict[str, list[dict[str, Any]]]]
    maximum_epochs = (
        actual_epochs + len(required["A"]) + len(required["R"]) + 1
    )
    while True:
        if actual_epochs > maximum_epochs:
            raise RuntimeError(
                "Could not schedule coverage anchors into distinct-group "
                "2A+2R epochs"
            )
        target_per_pool = actual_epochs * per_epoch_pool_size
        if any(len(required[pool]) > target_per_pool for pool in ("A", "R")):
            actual_epochs += 1
            continue
        scheduled = {
            pool: [dict(value) for value in required[pool]]
            for pool in ("A", "R")
        }
        total_group_counts = Counter(
            value["template_group_id"]
            for pool in ("A", "R")
            for value in scheduled[pool]
        )
        fill_counts = Counter()
        fill_groups = {
            "A": sorted(a_groups),
            "R": sorted(r_groups),
        }
        fill_failed = False
        for pool in ("A", "R"):
            while len(scheduled[pool]) < target_per_pool:
                choices = [
                    group_id
                    for group_id in fill_groups[pool]
                    if total_group_counts[group_id] < actual_epochs
                ]
                if not choices:
                    fill_failed = True
                    break
                minimum_group_count = min(
                    total_group_counts[group_id]
                    for group_id in choices
                )
                least_used_groups = [
                    group_id
                    for group_id in choices
                    if total_group_counts[group_id] == minimum_group_count
                ]
                group_id = min(
                    least_used_groups,
                    key=lambda value: (
                        _hash_order(
                            presentation_seed,
                            actual_epochs,
                            pool,
                            len(scheduled[pool]),
                            value,
                        ),
                    ),
                )
                value = filler(pool, group_id, fill_counts[(pool, group_id)])
                fill_counts[(pool, group_id)] += 1
                total_group_counts[group_id] += 1
                scheduled[pool].append(value)
            if fill_failed:
                break
        if fill_failed:
            actual_epochs += 1
            continue

        uncovered_group_ids_by_pool = {
            pool: sorted(
                (a_groups if pool == "A" else r_groups)
                - {
                    value["template_group_id"]
                    for value in scheduled[pool]
                }
            )
            for pool in ("A", "R")
        }
        if any(uncovered_group_ids_by_pool.values()):
            actual_epochs += 1
            continue

        epoch_buckets = [
            {"A": [], "R": [], "groups": set()}
            for _ in range(actual_epochs)
        ]
        assignment_failed = False
        for pool in ("A", "R"):
            by_scheduled_group: dict[str, list[dict[str, Any]]] = {}
            for value in scheduled[pool]:
                by_scheduled_group.setdefault(
                    value["template_group_id"],
                    [],
                ).append(value)
            ordered_groups = sorted(
                by_scheduled_group,
                key=lambda group_id: (
                    -len(by_scheduled_group[group_id]),
                    _hash_order(
                        presentation_seed,
                        actual_epochs,
                        pool,
                        group_id,
                    ),
                ),
            )
            for group_id in ordered_groups:
                values = sorted(
                    by_scheduled_group[group_id],
                    key=lambda value: (
                        value["presentation_kind"] != "coverage_anchor",
                        _hash_order(
                            presentation_seed,
                            pool,
                            group_id,
                            value["sample_id"],
                            value.get("anchor_index"),
                        ),
                    ),
                )
                choices = [
                    epoch
                    for epoch, bucket in enumerate(epoch_buckets)
                    if len(bucket[pool]) < per_epoch_pool_size
                    and group_id not in bucket["groups"]
                ]
                choices.sort(
                    key=lambda epoch: (
                        len(epoch_buckets[epoch][pool]),
                        _hash_order(
                            presentation_seed,
                            epoch,
                            pool,
                            group_id,
                        ),
                    )
                )
                if len(choices) < len(values):
                    assignment_failed = True
                    break
                for value, epoch in zip(values, choices):
                    epoch_buckets[epoch][pool].append(value)
                    epoch_buckets[epoch]["groups"].add(group_id)
            if assignment_failed:
                break
        if not assignment_failed and all(
            len(bucket[pool]) == per_epoch_pool_size
            for bucket in epoch_buckets
            for pool in ("A", "R")
        ):
            break
        actual_epochs += 1

    presentations: list[dict[str, Any]] = []
    for epoch, bucket in enumerate(epoch_buckets):
        for pool in ("A", "R"):
            bucket[pool].sort(
                key=lambda value: _hash_order(
                    presentation_seed,
                    epoch,
                    pool,
                    value["template_group_id"],
                    value["sample_id"],
                    value.get("anchor_index"),
                )
            )
        for batch_id in range(per_epoch_pool_size // 2):
            chosen = [
                *bucket["A"][batch_id * 2 : batch_id * 2 + 2],
                *bucket["R"][batch_id * 2 : batch_id * 2 + 2],
            ]
            chosen.sort(
                key=lambda value: _hash_order(
                    presentation_seed,
                    epoch,
                    batch_id,
                    value["pool"],
                    value["template_group_id"],
                )
            )
            if Counter(value["pool"] for value in chosen) != {"A": 2, "R": 2}:
                raise AssertionError("Effective batch does not contain exactly 2A + 2R")
            for position, value in enumerate(chosen):
                presentation_id = stable_id(
                    "PRESENTATION_V2",
                    presentation_seed,
                    epoch,
                    batch_id,
                    position,
                    value["pool"],
                    value["template_group_id"],
                    value["sample_id"],
                    value["presentation_kind"],
                    value.get("anchor_index"),
                    value.get("anchor_segments"),
                )
                presentations.append(
                    {
                        **value,
                        "presentation_id": presentation_id,
                        "presentation_seed": presentation_seed,
                        "epoch": epoch,
                        "effective_batch_id": batch_id,
                        "position_in_batch": position,
                        "within_epoch_occurrence_index": 0,
                    }
                )
    if len({row["presentation_id"] for row in presentations}) != len(presentations):
        raise AssertionError("Presentation IDs are not globally unique")

    per_epoch_group = Counter(
        (record["epoch"], record["template_group_id"]) for record in presentations
    )
    if max(per_epoch_group.values(), default=0) > 1:
        raise AssertionError("A template group appears more than once in an epoch")
    group_counts = Counter(record["template_group_id"] for record in presentations)
    group_counts_by_pool = {
        pool: Counter(
            record["template_group_id"]
            for record in presentations
            if record["pool"] == pool
        )
        for pool in ("A", "R")
    }
    total = sum(group_counts.values())
    report = {
        "schema": "balanced_presentations_v2",
        **gates,
        "minimum_requested_epochs": epochs,
        "epochs": actual_epochs,
        "group_coverage_epoch_floor": group_coverage_epoch_floor,
        "group_count_per_pool_per_epoch": per_epoch_pool_size,
        "available_group_count_by_pool": {
            "A": len(a_groups),
            "R": len(r_groups),
        },
        "covered_group_count_by_pool": {
            pool: len(group_counts_by_pool[pool])
            for pool in ("A", "R")
        },
        "uncovered_group_ids_by_pool": uncovered_group_ids_by_pool,
        "min_presentations_per_group_by_pool": {
            pool: min(group_counts_by_pool[pool].values())
            for pool in ("A", "R")
        },
        "max_presentations_per_group_by_pool": {
            pool: max(group_counts_by_pool[pool].values())
            for pool in ("A", "R")
        },
        "presentations": len(presentations),
        "effective_batches": len(presentations) // 4,
        "minimum_procedure_exposures": minimum_procedure_exposures,
        "coverage_anchor_count": sum(
            value["presentation_kind"] == "coverage_anchor"
            for value in presentations
        ),
        "coverage_anchor_fixed_J_count": sum(
            value["presentation_kind"] == "coverage_anchor"
            and value["anchor_fixed_J"]
            for value in presentations
        ),
        "coverage_anchor_variable_J_count": sum(
            value["presentation_kind"] == "coverage_anchor"
            and not value["anchor_fixed_J"]
            for value in presentations
        ),
        "maximum_group_share": max(group_counts.values()) / total,
        "group_ess": total**2 / sum(value**2 for value in group_counts.values()),
    }
    return presentations, report


def _segment_tuple(segments: Sequence[Segment | Mapping[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    result = []
    for segment in segments:
        if isinstance(segment, Segment):
            result.append((segment.piece_id, segment.start, segment.end))
        else:
            result.append(
                (
                    int(segment["piece_id"]),
                    int(segment["start"]),
                    int(segment["end"]),
                )
            )
    return tuple(result)


def compute_boundary_view_id(
    presentation_id: str,
    setting: str,
    data_seed: int,
    procedure_model_hash: str,
    segments: Sequence[Segment | Mapping[str, Any]],
) -> str:
    return stable_id(
        "BOUNDARY_VIEW_V1",
        presentation_id,
        setting,
        data_seed,
        procedure_model_hash,
        _segment_tuple(segments),
    )


def build_views(
    presentations: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    model: ProcedureUnigramModel,
    *,
    setting: str,
    data_seed: int,
    minimum_procedure_exposures: int = 10,
    primary: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if setting not in {"map", "sampled"}:
        raise ValueError("setting must be 'map' or 'sampled'")
    inventory_hash = model.inventory_hash()
    model_hash = model.model_hash()
    for record in train_records:
        artifact_model_hash = record.get(
            "procedure_model_hash",
            record.get("lexicon_hash"),
        )
        if artifact_model_hash != model_hash:
            raise ValueError(
                "Segmentation artifact was generated by another procedure model: "
                f"sample={record.get('sample_id')}, "
                f"artifact={artifact_model_hash}, current={model_hash}"
            )
        artifact_inventory_hash = record.get("procedure_inventory_hash")
        if (
            artifact_inventory_hash is not None
            and artifact_inventory_hash != inventory_hash
        ):
            raise ValueError(
                "Segmentation artifact procedure inventory differs from the model: "
                f"sample={record.get('sample_id')}, "
                f"artifact={artifact_inventory_hash}, current={inventory_hash}"
            )
        if int(record.get("procedure_count", -1)) != model.size:
            raise ValueError(
                "Segmentation artifact procedure count differs from the lexicon: "
                f"sample={record.get('sample_id')}, "
                f"artifact={record.get('procedure_count')}, current={model.size}"
            )
    by_sample = {record["sample_id"]: record for record in train_records}
    output: list[dict[str, Any]] = []
    changed = 0
    boundary_sample_presentations = 0
    changed_boundary_samples = 0
    for presentation in presentations:
        row = by_sample[presentation["sample_id"]]
        sequence = tuple(
            (str(value[0]), str(value[1])) for value in row["canonical_signatures"]
        )
        map_segments = [
            Segment(
                int(value["piece_id"]),
                int(value["start"]),
                int(value["end"]),
            )
            for value in row["map_segments"]
        ]
        anchor_segments = presentation.get("anchor_segments")
        if anchor_segments is not None:
            segments = [
                Segment(
                    int(value["piece_id"]),
                    int(value["start"]),
                    int(value["end"]),
                )
                for value in anchor_segments
            ]
        elif setting == "map":
            segments = map_segments
        else:
            seed_payload = hashlib.sha256(
                (
                    f"{data_seed}\0{row['sample_id']}\0{presentation['epoch']}\0"
                    f"{presentation['within_epoch_occurrence_index']}"
                ).encode("utf-8")
            ).digest()
            rng_seed = int.from_bytes(seed_payload[:16], "big")
            rng = np.random.Generator(np.random.PCG64(rng_seed))
            segments = model.sample_fixed_count(
                sequence,
                int(row["J_map"]),
                rng,
            )
        position = 0
        for segment in segments:
            if (
                segment.start != position
                or segment.end <= segment.start
                or segment.end > len(sequence)
                or model.pieces[segment.piece_id]
                != tuple(sequence[segment.start : segment.end])
            ):
                raise ValueError(
                    f"Invalid anchor/sample segmentation for {row['sample_id']}"
                )
            position = segment.end
        if position != len(sequence):
            raise ValueError(
                f"Segmentation does not cover {row['sample_id']}"
            )
        differs = _segment_tuple(segments) != _segment_tuple(map_segments)
        changed += int(differs)
        if presentation.get("presentation_kind") == "boundary_sample":
            boundary_sample_presentations += 1
            changed_boundary_samples += int(differs)
        output.append(
            {
                **presentation,
                "view_id": compute_boundary_view_id(
                    presentation["presentation_id"],
                    setting,
                    data_seed,
                    model_hash,
                    segments,
                ),
                "setting": setting,
                "data_seed": data_seed,
                "lexicon_hash": model_hash,
                "procedure_inventory_hash": inventory_hash,
                "procedure_model_hash": model_hash,
                "procedure_count": model.size,
                "instruction": row["instruction_raw"],
                "command_raw": row["command_raw"],
                "base_chunks": row["base_chunks"],
                "segments": [segment.to_dict() for segment in segments],
                "map_segments": row["map_segments"],
                "J_map": row["J_map"],
                "J_target": len(segments),
                "is_A": str(presentation["pool"]) == "A",
                "source_is_ambiguous": bool(row["is_ambiguous"]),
                "differs_from_map": differs,
            }
        )
    positive_counts = Counter(
        int(segment["piece_id"])
        for record in output
        for segment in record["segments"]
    )
    uncovered = [
        piece_id
        for piece_id in range(model.size)
        if positive_counts[piece_id] < minimum_procedure_exposures
    ]
    if uncovered:
        raise RuntimeError(
            "Serialized training views do not positively expose every procedure "
            f"at least {minimum_procedure_exposures} times; uncovered={uncovered}"
        )
    changed_rate = changed / len(output) if output else 0.0
    boundary_sample_changed_rate = (
        changed_boundary_samples / boundary_sample_presentations
        if boundary_sample_presentations
        else 0.0
    )
    if (
        primary
        and setting == "sampled"
        and boundary_sample_changed_rate < 0.10
    ):
        raise RuntimeError(
            "Sampled boundary-change rate "
            f"{boundary_sample_changed_rate:.4f} is below the preregistered "
            "0.10 gate"
        )
    report = {
        "schema": "boundary_views_v2",
        "setting": setting,
        "data_seed": data_seed,
        "lexicon_hash": model_hash,
        "procedure_inventory_hash": inventory_hash,
        "procedure_model_hash": model_hash,
        "procedure_count": model.size,
        "presentations": len(output),
        "changed_presentations": changed,
        "changed_rate": changed_rate,
        "boundary_sample_presentations": boundary_sample_presentations,
        "changed_boundary_samples": changed_boundary_samples,
        "boundary_sample_changed_rate": boundary_sample_changed_rate,
        "minimum_procedure_exposures": minimum_procedure_exposures,
        "procedure_positive_counts": [
            positive_counts[piece_id] for piece_id in range(model.size)
        ],
        "uncovered_procedure_ids": uncovered,
        "native_reserved_boundary": 247,
        "native_procedure_positive_coverage": sum(
            positive_counts[piece_id] >= minimum_procedure_exposures
            for piece_id in range(min(247, model.size))
        ),
        "native_procedure_count": min(247, model.size),
        "added_procedure_positive_coverage": sum(
            positive_counts[piece_id] >= minimum_procedure_exposures
            for piece_id in range(247, model.size)
        ),
        "added_procedure_count": max(0, model.size - 247),
    }
    return output, report
