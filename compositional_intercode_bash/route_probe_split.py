"""Build a deterministic TRAIN-only, group-disjoint routing-LR probe split."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import (
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    stable_id,
    write_json,
)


ROUTE_PROBE_SPLIT_SCHEMA = "intercode_bash_route_lr_split_v1"
DEFAULT_SPLIT_SEED = 271828
DEFAULT_HOLDOUT_FRACTION = 0.1
DEFAULT_FIT_SOURCE_EPOCHS = 20


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("split_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _ordered_effective_batches(
    views: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    if not views or len(views) % 4:
        raise ValueError("Views must contain complete four-row effective batches")
    batches = []
    seen_keys: set[tuple[int, int]] = set()
    for start in range(0, len(views), 4):
        batch = list(views[start : start + 4])
        keys = {
            (int(row["epoch"]), int(row["effective_batch_id"]))
            for row in batch
        }
        if len(keys) != 1:
            raise ValueError(
                f"Rows {start}:{start + 4} do not share one effective-batch key"
            )
        key = next(iter(keys))
        if key in seen_keys:
            raise ValueError(f"Repeated effective-batch key: {key}")
        seen_keys.add(key)
        if [int(row["position_in_batch"]) for row in batch] != [0, 1, 2, 3]:
            raise ValueError(f"Effective batch {key} is not ordered 0,1,2,3")
        pools = Counter(str(row["pool"]) for row in batch)
        if pools != Counter({"A": 2, "R": 2}):
            raise ValueError(
                f"Effective batch {key} must contain exactly 2A+2R, got {dict(pools)}"
            )
        batches.append(batch)
    return batches


def _pool_groups(
    views: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    group_to_pool: dict[str, str] = {}
    for row in views:
        pool = str(row["pool"])
        if pool not in {"A", "R"}:
            raise ValueError(f"Unknown boundary pool: {pool!r}")
        group_id = str(row["template_group_id"])
        previous = group_to_pool.setdefault(group_id, pool)
        if previous != pool:
            raise ValueError(
                f"Template group {group_id!r} appears in both {previous} and {pool}"
            )
    return {
        pool: sorted(
            group_id
            for group_id, group_pool in group_to_pool.items()
            if group_pool == pool
        )
        for pool in ("A", "R")
    }


def _ranked_groups(
    groups: Sequence[str],
    *,
    pool: str,
    split_seed: int,
) -> list[str]:
    return sorted(
        groups,
        key=lambda group_id: (
            stable_id(
                "intercode-bash-route-probe-group-v1",
                split_seed,
                pool,
                group_id,
            ),
            group_id,
        ),
    )


def _holdout_count(number_of_groups: int, holdout_fraction: float) -> int:
    if number_of_groups < 2:
        raise ValueError("Each pool needs at least two groups for a disjoint split")
    return min(
        number_of_groups - 1,
        max(1, int(math.ceil(number_of_groups * holdout_fraction))),
    )


def _select_supported_holdout_groups(
    groups_by_pool: Mapping[str, Sequence[str]],
    group_piece_ids: Mapping[str, set[int]],
    *,
    split_seed: int,
    holdout_fraction: float,
) -> dict[str, list[str]]:
    """Keep at least one non-holdout template group supporting every piece."""

    piece_group_support = Counter(
        piece_id
        for group_id in (*groups_by_pool["A"], *groups_by_pool["R"])
        for piece_id in group_piece_ids[group_id]
    )
    selected = {"A": [], "R": []}
    for pool in ("A", "R"):
        target_count = _holdout_count(
            len(groups_by_pool[pool]),
            holdout_fraction,
        )
        for group_id in _ranked_groups(
            groups_by_pool[pool],
            pool=pool,
            split_seed=split_seed,
        ):
            pieces = group_piece_ids[group_id]
            if any(piece_group_support[piece_id] <= 1 for piece_id in pieces):
                continue
            selected[pool].append(group_id)
            for piece_id in pieces:
                piece_group_support[piece_id] -= 1
            if len(selected[pool]) == target_count:
                break
        if len(selected[pool]) != target_count:
            raise ValueError(
                f"Cannot hold out {target_count} {pool} groups while retaining "
                "training support for every procedure piece"
            )
        selected[pool].sort()
    return selected


def _piece_ids(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    return {
        int(segment["piece_id"])
        for row in rows
        for segment in row["segments"]
    }


def _piece_positive_counts(
    rows: Sequence[Mapping[str, Any]],
    procedure_count: int,
) -> list[int]:
    counts = Counter(
        int(segment["piece_id"])
        for row in rows
        for segment in row["segments"]
    )
    invalid = sorted(
        piece_id for piece_id in counts if not 0 <= piece_id < procedure_count
    )
    if invalid:
        raise ValueError(f"Views contain invalid procedure piece IDs: {invalid}")
    return [counts[piece_id] for piece_id in range(procedure_count)]


def _coverage_first_fit_batches(
    eligible_batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    requested_fit_updates: int,
    required_piece_ids: set[int],
) -> tuple[list[list[Mapping[str, Any]]], int]:
    """Greedily cover all pieces, then fill and restore source schedule order."""

    if len(eligible_batches) < requested_fit_updates:
        raise ValueError(
            f"Only {len(eligible_batches)} eligible batches exist; "
            f"{requested_fit_updates} are required"
        )
    batch_piece_ids = [
        _piece_ids(batch) for batch in eligible_batches
    ]
    uncovered = set(required_piece_ids)
    selected_indices: set[int] = set()
    while uncovered:
        best_index = max(
            (
                index
                for index in range(len(eligible_batches))
                if index not in selected_indices
            ),
            key=lambda index: (
                len(batch_piece_ids[index] & uncovered),
                -index,
            ),
        )
        newly_covered = batch_piece_ids[best_index] & uncovered
        if not newly_covered:
            raise ValueError(
                f"Eligible batches cannot cover procedure pieces {sorted(uncovered)}"
            )
        selected_indices.add(best_index)
        uncovered -= newly_covered
        if len(selected_indices) > requested_fit_updates:
            raise ValueError(
                "The requested partial-run update budget is too small to cover "
                "every procedure piece"
            )
    coverage_batch_count = len(selected_indices)
    for index in range(len(eligible_batches)):
        if len(selected_indices) == requested_fit_updates:
            break
        selected_indices.add(index)
    return (
        [
            list(eligible_batches[index])
            for index in sorted(selected_indices)
        ],
        coverage_batch_count,
    )


def _validate_formal_view_source(
    views_path: Path,
    views: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[list[Mapping[str, Any]]]]:
    report_path = views_path.with_suffix(".report.json")
    report = read_json(report_path)
    if not (
        report.get("primary_ready") is True
        and report.get("primary_run") is True
        and report.get("primary_requested") is True
        and report.get("deny_policy") == "all_candidates"
    ):
        raise ValueError("Routing-LR manifests require primary formal TRAIN views")
    views_sha256 = sha256_file(views_path)
    if report.get("views_sha256") != views_sha256:
        raise ValueError("Views JSONL differs from its same-name report")
    if int(report.get("presentations", -1)) != len(views):
        raise ValueError("View report presentation count differs from JSONL")
    presentation_report_path = views_path.parent / "presentation_report.json"
    presentation_report = read_json(presentation_report_path)
    if (
        report.get("presentation_report_sha256")
        != sha256_file(presentation_report_path)
    ):
        raise ValueError("Presentation schedule report differs from the view report")
    if presentation_report.get("schema") != "balanced_presentations_v2":
        raise ValueError("Unknown presentation schedule schema")
    epochs = sorted({int(row["epoch"]) for row in views})
    reported_epochs = int(report.get("presentation_epochs", -1))
    if epochs != list(range(reported_epochs)):
        raise ValueError(
            f"View epochs are {epochs}, expected 0 through {reported_epochs - 1}"
        )
    batches = _ordered_effective_batches(views)
    batches_per_epoch = Counter(
        int(batch[0]["epoch"]) for batch in batches
    )
    if set(batches_per_epoch) != set(epochs):
        raise ValueError("At least one source epoch contains no effective batch")
    if len(set(batches_per_epoch.values())) != 1:
        raise ValueError(
            "Source epochs must have equal update counts for an epoch-equivalent probe"
        )
    groups_by_pool = _pool_groups(views)
    expected_group_counts = {
        pool: len(groups_by_pool[pool]) for pool in ("A", "R")
    }
    expected_empty = {"A": [], "R": []}
    for field, expected in (
        ("epochs", reported_epochs),
        ("presentations", len(views)),
        ("available_group_count_by_pool", expected_group_counts),
        ("covered_group_count_by_pool", expected_group_counts),
        ("uncovered_group_ids_by_pool", expected_empty),
    ):
        if presentation_report.get(field) != expected:
            raise ValueError(
                f"Presentation schedule field {field!r} is not formal/full-coverage"
            )
    if int(presentation_report.get("group_coverage_epoch_floor", -1)) > reported_epochs:
        raise ValueError("Presentation schedule is shorter than its group-coverage floor")
    return report, batches


def build_route_probe_manifest(
    views_path: str | Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    fit_source_epochs: int = DEFAULT_FIT_SOURCE_EPOCHS,
) -> dict[str, Any]:
    """Select disjoint holdout groups and complete fit batches from TRAIN views."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between zero and one")
    if fit_source_epochs <= 0:
        raise ValueError("fit_source_epochs must be positive")
    views_path = Path(views_path)
    views = list(read_jsonl(views_path))
    report, batches = _validate_formal_view_source(views_path, views)
    groups_by_pool = _pool_groups(views)
    views_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in views:
        views_by_group[str(row["template_group_id"])].append(row)
    procedure_count = int(report.get("procedure_count", -1))
    if procedure_count <= 0:
        raise ValueError("Formal view report lacks a positive procedure_count")
    source_positive_counts = _piece_positive_counts(views, procedure_count)
    if any(count <= 0 for count in source_positive_counts):
        raise ValueError("Formal source schedule does not cover every procedure piece")
    group_piece_ids = {
        group_id: _piece_ids(views_by_group[group_id])
        for group_id in (*groups_by_pool["A"], *groups_by_pool["R"])
    }
    holdout_by_pool = _select_supported_holdout_groups(
        groups_by_pool,
        group_piece_ids,
        split_seed=split_seed,
        holdout_fraction=holdout_fraction,
    )
    holdout_groups = {
        group_id
        for values in holdout_by_pool.values()
        for group_id in values
    }

    updates_per_source_epoch = len(batches) // int(report["presentation_epochs"])
    requested_fit_updates = fit_source_epochs * updates_per_source_epoch
    eligible_batches = [
        batch
        for batch in batches
        if not any(
            str(row["template_group_id"]) in holdout_groups
            for row in batch
        )
    ]
    eligible_piece_ids = _piece_ids(
        [row for batch in eligible_batches for row in batch]
    )
    required_piece_ids = set(range(procedure_count))
    if eligible_piece_ids != required_piece_ids:
        missing = sorted(required_piece_ids - eligible_piece_ids)
        raise ValueError(
            "Selected holdout groups remove all eligible batches for procedure "
            f"pieces {missing}"
        )
    fit_batches, coverage_batch_count = _coverage_first_fit_batches(
        eligible_batches,
        requested_fit_updates=requested_fit_updates,
        required_piece_ids=required_piece_ids,
    )
    fit_views = [row for batch in fit_batches for row in batch]
    fit_groups_by_pool = {
        pool: sorted(
            {
                str(row["template_group_id"])
                for row in fit_views
                if str(row["pool"]) == pool
            }
        )
        for pool in ("A", "R")
    }

    probe_views = []
    for pool in ("A", "R"):
        for group_id in holdout_by_pool[pool]:
            candidates = views_by_group[group_id]
            selected = min(
                candidates,
                key=lambda row: (
                    stable_id(
                        "intercode-bash-route-probe-view-v1",
                        split_seed,
                        pool,
                        group_id,
                        row["view_id"],
                    ),
                    str(row["view_id"]),
                ),
            )
            probe_views.append(selected)
    probe_views.sort(
        key=lambda row: (
            str(row["pool"]),
            str(row["template_group_id"]),
            str(row["view_id"]),
        )
    )
    fit_positive_counts = _piece_positive_counts(
        fit_views,
        procedure_count,
    )
    if any(count <= 0 for count in fit_positive_counts):
        raise AssertionError("Coverage-first fit selection missed a procedure piece")
    probe_piece_ids = sorted(_piece_ids(probe_views))

    last_fit_epoch = max(int(row["epoch"]) for row in fit_views)
    manifest: dict[str, Any] = {
        "schema": ROUTE_PROBE_SPLIT_SCHEMA,
        "source_views_file": views_path.name,
        "source_views_sha256": sha256_file(views_path),
        "source_view_report_sha256": sha256_file(
            views_path.with_suffix(".report.json")
        ),
        "source_presentation_report_sha256": sha256_file(
            views_path.parent / "presentation_report.json"
        ),
        "source_schedule_epochs": int(report["presentation_epochs"]),
        "source_schedule_updates": len(batches),
        "updates_per_source_epoch": updates_per_source_epoch,
        "split_seed": int(split_seed),
        "holdout_fraction": float(holdout_fraction),
        "fit_source_epoch_equivalent": int(fit_source_epochs),
        "fit_updates": len(fit_batches),
        "fit_presentations": len(fit_views),
        "last_source_epoch_used_by_fit": last_fit_epoch,
        "fit_coverage_batch_count": coverage_batch_count,
        "procedure_count": procedure_count,
        "fit_procedure_positive_counts": fit_positive_counts,
        "fit_procedure_positive_counts_sha256": sha256_bytes(
            canonical_json_bytes(fit_positive_counts)
        ),
        "probe_procedure_piece_ids": probe_piece_ids,
        "probe_procedure_piece_ids_sha256": sha256_bytes(
            canonical_json_bytes(probe_piece_ids)
        ),
        "available_template_group_ids": groups_by_pool,
        "fit_template_group_ids": fit_groups_by_pool,
        "holdout_template_group_ids": holdout_by_pool,
        "fit_view_ids": [str(row["view_id"]) for row in fit_views],
        "probe_view_ids": [str(row["view_id"]) for row in probe_views],
        "probe_presentations_by_pool": {
            pool: sum(str(row["pool"]) == pool for row in probe_views)
            for pool in ("A", "R")
        },
    }
    manifest["split_sha256"] = _manifest_hash(manifest)
    validate_route_probe_manifest(manifest, views_path=views_path, views=views)
    return manifest


def validate_route_probe_manifest(
    manifest: Mapping[str, Any],
    *,
    views_path: str | Path,
    views: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Validate source binding and return ordered fit/probe views."""

    manifest = dict(manifest)
    if manifest.get("schema") != ROUTE_PROBE_SPLIT_SCHEMA:
        raise ValueError("Unknown routing-LR split schema")
    if manifest.get("split_sha256") != _manifest_hash(manifest):
        raise ValueError("Routing-LR split manifest hash is invalid")
    views_path = Path(views_path)
    if manifest.get("source_views_sha256") != sha256_file(views_path):
        raise ValueError("Routing-LR split was built from different views")
    report_path = views_path.with_suffix(".report.json")
    if manifest.get("source_view_report_sha256") != sha256_file(report_path):
        raise ValueError("Routing-LR split was built from a different view report")
    presentation_report_path = views_path.parent / "presentation_report.json"
    if manifest.get("source_presentation_report_sha256") != sha256_file(
        presentation_report_path
    ):
        raise ValueError(
            "Routing-LR split was built from a different presentation report"
        )
    if views is None:
        views = list(read_jsonl(views_path))
    else:
        views = list(views)
    source_report, source_batches = _validate_formal_view_source(
        views_path,
        views,
    )
    source_epochs = int(source_report["presentation_epochs"])
    if int(manifest["source_schedule_epochs"]) != source_epochs:
        raise ValueError("Manifest source epoch count is inconsistent")
    if int(manifest["source_schedule_updates"]) != len(source_batches):
        raise ValueError("Manifest source update count is inconsistent")
    if int(manifest["updates_per_source_epoch"]) != (
        len(source_batches) // source_epochs
    ):
        raise ValueError("Manifest updates-per-source-epoch is inconsistent")
    if int(manifest["procedure_count"]) != int(source_report["procedure_count"]):
        raise ValueError("Manifest procedure count differs from the view report")
    actual_available_groups = _pool_groups(views)
    if {
        pool: sorted(
            str(value)
            for value in manifest["available_template_group_ids"][pool]
        )
        for pool in ("A", "R")
    } != actual_available_groups:
        raise ValueError("Manifest available template groups are inconsistent")

    by_view_id: dict[str, Mapping[str, Any]] = {}
    for row in views:
        view_id = str(row["view_id"])
        if view_id in by_view_id:
            raise ValueError(f"Duplicate source view_id: {view_id}")
        by_view_id[view_id] = row
    fit_ids = [str(value) for value in manifest["fit_view_ids"]]
    probe_ids = [str(value) for value in manifest["probe_view_ids"]]
    if len(set(fit_ids)) != len(fit_ids) or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("Routing-LR split contains duplicate view IDs")
    if set(fit_ids) & set(probe_ids):
        raise ValueError("Fit and probe view IDs overlap")
    try:
        fit_views = [by_view_id[view_id] for view_id in fit_ids]
        probe_views = [by_view_id[view_id] for view_id in probe_ids]
    except KeyError as exc:
        raise ValueError(f"Routing-LR split references an unknown view: {exc}") from exc

    fit_batches = _ordered_effective_batches(fit_views)
    if int(manifest["fit_updates"]) != len(fit_batches):
        raise ValueError("Manifest fit-update count is inconsistent")
    if int(manifest["fit_presentations"]) != len(fit_views):
        raise ValueError("Manifest fit-presentation count is inconsistent")
    if (
        int(manifest["fit_source_epoch_equivalent"])
        * int(manifest["updates_per_source_epoch"])
        != len(fit_batches)
    ):
        raise ValueError("Fit updates are not the recorded source-epoch equivalent")
    procedure_count = int(manifest["procedure_count"])
    fit_positive_counts = _piece_positive_counts(
        fit_views,
        procedure_count,
    )
    if fit_positive_counts != [
        int(value) for value in manifest["fit_procedure_positive_counts"]
    ]:
        raise ValueError("Manifest fit procedure coverage is inconsistent")
    if manifest.get("fit_procedure_positive_counts_sha256") != sha256_bytes(
        canonical_json_bytes(fit_positive_counts)
    ):
        raise ValueError("Manifest fit procedure coverage hash is invalid")
    if any(count <= 0 for count in fit_positive_counts):
        raise ValueError("Partial fit views do not cover every procedure piece")
    coverage_batch_count = int(manifest["fit_coverage_batch_count"])
    if not 0 < coverage_batch_count <= len(fit_batches):
        raise ValueError("Manifest coverage batch count is invalid")

    actual_fit_by_pool = {
        pool: sorted(
            {
                str(row["template_group_id"])
                for row in fit_views
                if str(row["pool"]) == pool
            }
        )
        for pool in ("A", "R")
    }
    recorded_fit_by_pool = {
        pool: sorted(str(value) for value in manifest["fit_template_group_ids"][pool])
        for pool in ("A", "R")
    }
    if actual_fit_by_pool != recorded_fit_by_pool:
        raise ValueError("Manifest fit template groups are inconsistent")
    holdout_by_pool = {
        pool: sorted(
            str(value)
            for value in manifest["holdout_template_group_ids"][pool]
        )
        for pool in ("A", "R")
    }
    fit_groups = {
        group_id
        for values in actual_fit_by_pool.values()
        for group_id in values
    }
    holdout_groups = {
        group_id
        for values in holdout_by_pool.values()
        for group_id in values
    }
    if fit_groups & holdout_groups:
        raise ValueError("Fit and probe template groups overlap")

    actual_probe_by_pool = {
        pool: sorted(
            str(row["template_group_id"])
            for row in probe_views
            if str(row["pool"]) == pool
        )
        for pool in ("A", "R")
    }
    if actual_probe_by_pool != holdout_by_pool:
        raise ValueError(
            "Probe views must contain exactly one view from every holdout group"
        )
    probe_piece_ids = sorted(_piece_ids(probe_views))
    if probe_piece_ids != [
        int(value) for value in manifest["probe_procedure_piece_ids"]
    ]:
        raise ValueError("Manifest probe procedure pieces are inconsistent")
    if manifest.get("probe_procedure_piece_ids_sha256") != sha256_bytes(
        canonical_json_bytes(probe_piece_ids)
    ):
        raise ValueError("Manifest probe procedure-piece hash is invalid")
    if not set(probe_piece_ids).issubset(
        {
            piece_id
            for piece_id, count in enumerate(fit_positive_counts)
            if count > 0
        }
    ):
        raise ValueError("Probe contains a procedure piece absent from fit views")
    for pool in ("A", "R"):
        if int(manifest["probe_presentations_by_pool"][pool]) != len(
            actual_probe_by_pool[pool]
        ):
            raise ValueError("Manifest probe pool count is inconsistent")
    return fit_views, probe_views


def load_route_probe_manifest(
    manifest_path: str | Path,
    *,
    views_path: str | Path,
    views: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    manifest = read_json(manifest_path)
    fit_views, probe_views = validate_route_probe_manifest(
        manifest,
        views_path=views_path,
        views=views,
    )
    return manifest, fit_views, probe_views


def checkpoint_split_record(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact, path-independent split identity stored in partial checkpoints."""

    if manifest.get("split_sha256") != _manifest_hash(manifest):
        raise ValueError("Cannot record an invalid routing-LR split manifest")
    return {
        "schema": manifest["schema"],
        "split_sha256": manifest["split_sha256"],
        "source_views_sha256": manifest["source_views_sha256"],
        "source_presentation_report_sha256": manifest[
            "source_presentation_report_sha256"
        ],
        "split_seed": int(manifest["split_seed"]),
        "holdout_fraction": float(manifest["holdout_fraction"]),
        "fit_source_epoch_equivalent": int(
            manifest["fit_source_epoch_equivalent"]
        ),
        "fit_updates": int(manifest["fit_updates"]),
        "procedure_count": int(manifest["procedure_count"]),
        "fit_coverage_batch_count": int(
            manifest["fit_coverage_batch_count"]
        ),
        "fit_procedure_positive_counts_sha256": manifest[
            "fit_procedure_positive_counts_sha256"
        ],
        "probe_procedure_piece_ids_sha256": manifest[
            "probe_procedure_piece_ids_sha256"
        ],
        "fit_view_ids_sha256": sha256_bytes(
            canonical_json_bytes(manifest["fit_view_ids"])
        ),
        "probe_view_ids_sha256": sha256_bytes(
            canonical_json_bytes(manifest["probe_view_ids"])
        ),
        "fit_template_group_ids": {
            pool: list(manifest["fit_template_group_ids"][pool])
            for pool in ("A", "R")
        },
        "holdout_template_group_ids": {
            pool: list(manifest["holdout_template_group_ids"][pool])
            for pool in ("A", "R")
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
    )
    parser.add_argument(
        "--fit-source-epochs",
        type=int,
        default=DEFAULT_FIT_SOURCE_EPOCHS,
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be strictly between zero and one")
    if args.fit_source_epochs <= 0:
        parser.error("--fit-source-epochs must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_route_probe_manifest(
        args.views,
        split_seed=args.split_seed,
        holdout_fraction=args.holdout_fraction,
        fit_source_epochs=args.fit_source_epochs,
    )
    write_json(args.output, manifest)
    print(
        {
            "split_sha256": manifest["split_sha256"],
            "fit_updates": manifest["fit_updates"],
            "probe_presentations_by_pool": manifest[
                "probe_presentations_by_pool"
            ],
        }
    )


if __name__ == "__main__":
    main()
