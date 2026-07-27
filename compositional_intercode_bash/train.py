"""Train TokMem or TapMem on a frozen boundary-view schedule."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoint import (
    base_model_load_location,
    build_base_model_identity,
    save_checkpoint,
    validate_base_model_identity,
)
from .data_sources import (
    SOURCE_ARTIFACT_FILENAMES,
    validate_materialized_sources,
)
from .io_utils import (
    PACKAGE_ROOT,
    artifact_record,
    ensure_output_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .memory_model import (
    METHODS,
    TCRA_ADDITIVE_ONLY_TRIGGER_POLICY,
    TCRA_TRIGGER_POLICY,
    ProceduralMemoryModel,
)
from .route_probe_split import (
    checkpoint_split_record,
    load_route_probe_manifest,
)
from .training_data import (
    BoundaryViewDataset,
    ROUTING_TARGET_MODES,
    TARGET_SERIALIZATION_POLICY,
    compute_training_loss_sums,
    left_pad_collate,
    normalized_microbatch_loss,
)
from .unigram import ProcedureUnigramModel
from .views import compute_boundary_view_id


LOSS_NORMALIZATION = "effective_batch_token_and_route_site_v1"
TRAINING_EPOCH_REPEAT_POLICY = "source_epoch_modulo_v1"
SOURCE_ARTIFACT_NAMES = tuple(SOURCE_ARTIFACT_FILENAMES)
PROVENANCE_ARTIFACT_NAMES = (
    *SOURCE_ARTIFACT_NAMES,
    "adjudications",
    "groups",
    "provenance",
    "provenance_report",
    "derived_split",
    "derived_split_report",
)
PROCEDURE_EVIDENCE_ARTIFACT_NAMES = (
    "atoms",
    "atomization_report",
    "candidates",
    "capacity_selection",
    "pruning_paths",
    "procedure_grid",
)
VIEW_INPUT_ARTIFACT_NAMES = (
    *PROVENANCE_ARTIFACT_NAMES,
    "procedure_lexicon",
    "segmentations",
    "procedure_report",
    "training_boundary_gate",
    *PROCEDURE_EVIDENCE_ARTIFACT_NAMES,
)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scheduler_lambda(step: int, warmup: int, total: int) -> float:
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    remaining = max(0, total - step - 1)
    decay_steps = max(1, total - warmup)
    return remaining / decay_steps


def _scheduler_geometry(
    *,
    run_total_updates: int,
    source_total_updates: int,
) -> tuple[int, int]:
    if not 0 < run_total_updates <= source_total_updates:
        raise ValueError(
            "Run updates must be positive and cannot exceed the source schedule"
        )
    return source_total_updates, source_total_updates // 10


def _expand_training_schedule(
    source_views: list[dict[str, Any]],
    *,
    source_epochs: int,
    training_epochs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse source epochs in order without changing the serialized artifact."""

    if source_epochs <= 0:
        raise ValueError("Source schedule must contain at least one epoch")
    if training_epochs < source_epochs:
        raise ValueError(
            "--training-epochs must cover the complete source schedule; "
            "use --epoch-limit for a partial exploratory run"
        )
    views_by_epoch = {
        epoch: [
            view for view in source_views if int(view["epoch"]) == epoch
        ]
        for epoch in range(source_epochs)
    }
    if any(not rows for rows in views_by_epoch.values()):
        raise ValueError("Source schedule contains an empty epoch")

    expanded: list[dict[str, Any]] = []
    for training_epoch in range(training_epochs):
        source_epoch = training_epoch % source_epochs
        repeat_cycle = training_epoch // source_epochs
        for source_view in views_by_epoch[source_epoch]:
            view = dict(source_view)
            view["_training_epoch"] = training_epoch
            view["_source_epoch"] = source_epoch
            view["_repeat_cycle"] = repeat_cycle
            expanded.append(view)
    complete_source_passes, prefix_source_epochs = divmod(
        training_epochs,
        source_epochs,
    )
    return expanded, {
        "schema": TRAINING_EPOCH_REPEAT_POLICY,
        "source_artifact_rewritten": False,
        "mapping": "source_epoch = training_epoch % source_schedule_epochs",
        "source_schedule_epochs": source_epochs,
        "trained_epochs": training_epochs,
        "complete_source_passes": complete_source_passes,
        "additional_source_epoch_prefix": prefix_source_epochs,
        "repeated_training_epochs": training_epochs - source_epochs,
    }


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient in {name}")


def _routing_head_norms(
    model: ProceduralMemoryModel,
) -> dict[str, float] | None:
    if model.routing_head is None:
        return None
    row_norms = model.routing_head.weight.detach().float().norm(dim=1)
    return {
        "mean": float(row_norms.mean().item()),
        "max": float(row_norms.max().item()),
    }


def _views_report_path(views_path: str | Path) -> Path:
    views_path = Path(views_path)
    if views_path.suffix != ".jsonl":
        raise ValueError(
            f"Training views must use a .jsonl path; got {views_path}"
        )
    return views_path.with_suffix(".report.json")


def _views_coverage_path(views_path: str | Path) -> Path:
    views_path = Path(views_path)
    if views_path.suffix != ".jsonl":
        raise ValueError(
            f"Training views must use a .jsonl path; got {views_path}"
        )
    return views_path.with_suffix(".coverage.json")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} artifact is missing: {path}")
    return path


def _require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Required {label} directory is missing: {path}")
    return path


def _validate_presentation_group_coverage(
    paths: dict[str, Path],
    view_report: dict[str, Any],
) -> dict[str, Any]:
    presentation_report = read_json(paths["presentation_report"])
    if presentation_report.get("schema") != "balanced_presentations_v2":
        raise ValueError("Unknown training presentation report schema")

    views = list(read_jsonl(paths["views"]))
    if not views:
        raise ValueError("Training views are empty")
    source_group_is_ambiguous: dict[str, bool] = {}
    for record in read_jsonl(paths["segmentations"]):
        if record.get("derived_split") != "TRAIN":
            continue
        group_id = str(record["template_group_id"])
        source_group_is_ambiguous[group_id] = (
            source_group_is_ambiguous.get(group_id, False)
            or bool(record["is_ambiguous"])
        )
    source_groups = {
        "A": {
            group_id
            for group_id, is_ambiguous in source_group_is_ambiguous.items()
            if is_ambiguous
        },
        "R": {
            group_id
            for group_id, is_ambiguous in source_group_is_ambiguous.items()
            if not is_ambiguous
        },
    }
    view_group_counts = {"A": Counter(), "R": Counter()}
    per_epoch_pool_counts: Counter[tuple[int, str]] = Counter()
    for view in views:
        pool = str(view.get("pool"))
        if pool not in source_groups:
            raise ValueError(f"Unknown training presentation pool: {pool!r}")
        group_id = str(view.get("template_group_id"))
        if group_id not in source_groups[pool]:
            raise ValueError(
                f"View group {group_id!r} does not belong to pool {pool}"
            )
        view_group_counts[pool][group_id] += 1
        per_epoch_pool_counts[(int(view["epoch"]), pool)] += 1

    epochs = sorted({int(view["epoch"]) for view in views})
    expected_epoch_pool_keys = {
        (epoch, pool) for epoch in epochs for pool in ("A", "R")
    }
    if set(per_epoch_pool_counts) != expected_epoch_pool_keys:
        raise ValueError("Training views do not contain both pools in every epoch")
    per_epoch_pool_sizes = set(per_epoch_pool_counts.values())
    if len(per_epoch_pool_sizes) != 1:
        raise ValueError("Training views change pool size between epochs")
    per_epoch_pool_size = next(iter(per_epoch_pool_sizes))
    available_counts = {
        pool: len(source_groups[pool]) for pool in ("A", "R")
    }
    covered_counts = {
        pool: len(view_group_counts[pool]) for pool in ("A", "R")
    }
    uncovered = {
        pool: sorted(source_groups[pool] - set(view_group_counts[pool]))
        for pool in ("A", "R")
    }
    group_coverage_epoch_floor = (
        max(available_counts.values()) + per_epoch_pool_size - 1
    ) // per_epoch_pool_size
    expected_fields = {
        "epochs": len(epochs),
        "presentations": len(views),
        "group_coverage_epoch_floor": group_coverage_epoch_floor,
        "group_count_per_pool_per_epoch": per_epoch_pool_size,
        "available_group_count_by_pool": available_counts,
        "covered_group_count_by_pool": covered_counts,
        "uncovered_group_ids_by_pool": uncovered,
        "min_presentations_per_group_by_pool": {
            pool: min(view_group_counts[pool].values())
            for pool in ("A", "R")
        },
        "max_presentations_per_group_by_pool": {
            pool: max(view_group_counts[pool].values())
            for pool in ("A", "R")
        },
    }
    for field, expected in expected_fields.items():
        if (
            field in presentation_report
            and presentation_report[field] != expected
        ):
            raise ValueError(
                f"Presentation report field {field!r} differs from "
                f"the serialized TRAIN views: expected={expected!r}, "
                f"actual={presentation_report[field]!r}"
            )
    if (
        view_report.get("presentation_epochs") != presentation_report["epochs"]
        or view_report.get("minimum_requested_epochs")
        != presentation_report.get("minimum_requested_epochs")
    ):
        raise ValueError(
            "View report epoch metadata differs from the presentation report"
        )
    return {
        **expected_fields,
        "all_eligible_template_groups_covered": not any(uncovered.values()),
    }


def _validate_primary_status(
    report: dict[str, Any],
    *,
    allow_exploratory: bool,
) -> dict[str, Any]:
    for field in ("primary_ready", "primary_run", "deny_policy"):
        if field not in report:
            raise ValueError(f"View report lacks required field {field!r}")
    primary_ready = report["primary_ready"] is True
    primary_run = report["primary_run"] is True
    deny_policy = report["deny_policy"]
    if primary_run and deny_policy != "all_candidates":
        raise ValueError(
            "A primary provenance run must use deny_policy='all_candidates'"
        )
    if not primary_run and deny_policy != "all_candidates":
        raise ValueError(
            "An exploratory provenance run must use deny_policy='all_candidates'"
        )
    formal_ready = (
        primary_ready
        and primary_run
        and deny_policy == "all_candidates"
        and report.get("primary_requested") is True
    )
    if not formal_ready and not allow_exploratory:
        raise RuntimeError(
            "Formal training only accepts a view report with primary_ready=true, "
            "primary_run=true, deny_policy='all_candidates', and "
            "primary_requested=true. Pass --allow-exploratory-views only for "
            "non-primary debugging runs."
        )
    return {
        "primary_ready": primary_ready,
        "primary_run": primary_run,
        "deny_policy": str(deny_policy),
        "primary_requested": report.get("primary_requested") is True,
        "formal_ready": formal_ready,
        "exploratory_override": bool(not formal_ready and allow_exploratory),
    }


def _validate_training_artifacts(
    views_path: str | Path,
    procedure_lexicon_path: str | Path,
    lexicon: ProcedureUnigramModel,
    *,
    allow_exploratory: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the complete provenance-to-view artifact chain before training."""

    views_path = _require_file(Path(views_path), "views")
    report_path = _require_file(_views_report_path(views_path), "view report")
    artifact_dir = report_path.parent
    paths = {
        "raw_pairs_official": _require_file(
            artifact_dir / SOURCE_ARTIFACT_FILENAMES["raw_pairs_official"],
            "materialized NL2Bash source",
        ),
        "intercode_tasks": _require_file(
            artifact_dir / SOURCE_ARTIFACT_FILENAMES["intercode_tasks"],
            "materialized InterCode source",
        ),
        "source_manifest": _require_file(
            artifact_dir / SOURCE_ARTIFACT_FILENAMES["source_manifest"],
            "source manifest",
        ),
        "procedure_lexicon": _require_file(
            Path(procedure_lexicon_path),
            "procedure lexicon",
        ),
        "views": views_path,
        "view_report": report_path,
        "coverage_report": _require_file(
            _views_coverage_path(views_path),
            "procedure positive-coverage report",
        ),
        "segmentations": _require_file(
            artifact_dir / "segmentations.jsonl",
            "segmentations",
        ),
        "adjudications": _require_file(
            artifact_dir / "adjudications.used.jsonl",
            "adjudication snapshot",
        ),
        "groups": _require_file(
            artifact_dir / "groups.json",
            "source grouping report",
        ),
        "provenance": _require_file(
            artifact_dir / "provenance.jsonl",
            "provenance",
        ),
        "provenance_report": _require_file(
            artifact_dir / "provenance_report.json",
            "provenance report",
        ),
        "derived_split": _require_file(
            artifact_dir / "raw_pairs.grouped_and_split.jsonl",
            "derived split",
        ),
        "derived_split_report": _require_file(
            artifact_dir / "derived_split_report.json",
            "derived split report",
        ),
        "procedure_report": _require_file(
            artifact_dir / "procedure_report.json",
            "procedure report",
        ),
        "training_boundary_gate": _require_file(
            artifact_dir / "training_boundary_gate.json",
            "training boundary gate",
        ),
        "atoms": _require_file(
            artifact_dir / "atoms.jsonl",
            "strict atom records",
        ),
        "atomization_report": _require_file(
            artifact_dir / "atomization_report.json",
            "atomization report",
        ),
        "candidates": _require_file(
            artifact_dir / "candidates.jsonl",
            "procedure candidates",
        ),
        "capacity_selection": _require_file(
            artifact_dir / "capacity_selection.json",
            "capacity selection report",
        ),
        "pruning_paths": _require_file(
            artifact_dir / "pruning_paths.json",
            "procedure pruning paths",
        ),
        "procedure_grid": _require_directory(
            artifact_dir / "procedure_grid",
            "procedure grid",
        ),
        "training_presentations": _require_file(
            artifact_dir / "training_presentations.jsonl",
            "training presentations",
        ),
        "presentation_report": _require_file(
            artifact_dir / "presentation_report.json",
            "presentation report",
        ),
    }
    records = {name: artifact_record(path) for name, path in paths.items()}
    hashes = {name: value["sha256"] for name, value in records.items()}
    report = read_json(report_path)
    model_hash = lexicon.model_hash()
    inventory_hash = lexicon.inventory_hash()

    if report.get("views_sha256") != hashes["views"]:
        raise ValueError("views JSONL differs from its same-name view report")
    if report.get("coverage_report_sha256") != hashes["coverage_report"]:
        raise ValueError("Coverage report differs from the view report")
    coverage_report = read_json(paths["coverage_report"])
    for field in (
        "procedure_count",
        "minimum_procedure_exposures",
        "procedure_positive_counts",
        "uncovered_procedure_ids",
        "native_reserved_boundary",
        "native_procedure_positive_coverage",
        "native_procedure_count",
        "added_procedure_positive_coverage",
        "added_procedure_count",
    ):
        if coverage_report.get(field) != report.get(field):
            raise ValueError(
                f"Coverage report field {field!r} differs from the view report"
            )
    for field in ("lexicon_hash", "procedure_model_hash"):
        if report.get(field) != model_hash:
            raise ValueError(
                f"View report {field} does not match the loaded procedure model"
            )
    if report.get("procedure_inventory_hash") != inventory_hash:
        raise ValueError(
            "View report procedure_inventory_hash does not match the lexicon"
        )
    if int(report.get("procedure_count", -1)) != lexicon.size:
        raise ValueError("View report procedure_count does not match the lexicon")

    expected_view_inputs = {
        name: hashes[name]
        for name in VIEW_INPUT_ARTIFACT_NAMES
    }
    if report.get("input_sha256") != expected_view_inputs:
        raise ValueError(
            "View report input_sha256 does not match the current training artifacts"
        )

    provenance_report = read_json(paths["provenance_report"])
    source_integrity = validate_materialized_sources(
        artifact_dir,
        require_canonical=not allow_exploratory,
    )
    for name in SOURCE_ARTIFACT_NAMES:
        if source_integrity[name]["sha256"] != hashes[name]:
            raise ValueError(
                f"Validated source hash differs for {name}"
            )
    expected_source_inputs = {
        name: hashes[name] for name in SOURCE_ARTIFACT_NAMES
    }
    if provenance_report.get("input_sha256") != expected_source_inputs:
        raise ValueError(
            "Provenance report was generated from different source artifacts"
        )
    if provenance_report.get("provenance_sha256") != hashes["provenance"]:
        raise ValueError("provenance.jsonl differs from provenance_report.json")
    if provenance_report.get("derived_split_sha256") != hashes["derived_split"]:
        raise ValueError(
            "Derived split differs from provenance_report.json"
        )
    for field, name in (
        ("adjudications_sha256", "adjudications"),
        ("groups_sha256", "groups"),
        ("derived_split_report_sha256", "derived_split_report"),
    ):
        if provenance_report.get(field) != hashes[name]:
            raise ValueError(
                f"{name} differs from provenance_report.json"
            )
    if provenance_report.get("primary_ready") is True and (
        provenance_report.get("tasks") != 200
        or provenance_report.get("deny_policy") != "all_candidates"
        or provenance_report.get("unresolved_candidates_removed") is not True
        or provenance_report.get("candidate_union_group_count")
        != provenance_report.get("actual_deny_group_count")
    ):
        raise ValueError(
            "Primary provenance does not prove that every retrieved "
            "InterCode candidate source group was removed"
        )

    procedure_report = read_json(paths["procedure_report"])
    if procedure_report.get("selected_lexicon_hash") != model_hash:
        raise ValueError("Procedure report selected model hash differs")
    if procedure_report.get("procedure_model_hash") != model_hash:
        raise ValueError("Procedure report model hash differs")
    if procedure_report.get("procedure_inventory_hash") != inventory_hash:
        raise ValueError("Procedure report inventory hash differs")
    if int(procedure_report.get("selected_k", -1)) != lexicon.size:
        raise ValueError("Procedure report selected_k differs")
    if (
        procedure_report.get("procedure_lexicon_sha256")
        != hashes["procedure_lexicon"]
    ):
        raise ValueError("Procedure lexicon differs from procedure_report.json")
    if (
        procedure_report.get("segmentations_sha256")
        != hashes["segmentations"]
    ):
        raise ValueError("Segmentations differ from procedure_report.json")
    expected_procedure_inputs = {
        name: hashes[name]
        for name in PROVENANCE_ARTIFACT_NAMES
    }
    if procedure_report.get("input_sha256") != expected_procedure_inputs:
        raise ValueError(
            "Procedure report input_sha256 does not match provenance artifacts"
        )
    if (
        procedure_report.get("training_boundary_gate_sha256")
        != hashes["training_boundary_gate"]
    ):
        raise ValueError(
            "Training boundary gate differs from procedure_report.json"
        )
    expected_evidence_hashes = {
        name: hashes[name] for name in PROCEDURE_EVIDENCE_ARTIFACT_NAMES
    }
    if procedure_report.get("evidence_sha256") != expected_evidence_hashes:
        raise ValueError(
            "Procedure induction evidence differs from procedure_report.json"
        )
    actual_grid_files = sorted(
        path.relative_to(paths["procedure_grid"]).as_posix()
        for path in paths["procedure_grid"].rglob("*")
        if path.is_file()
    )
    if procedure_report.get("procedure_grid_files") != actual_grid_files:
        raise ValueError("Procedure grid file list differs from procedure report")
    if (
        report.get("training_presentations_sha256")
        != hashes["training_presentations"]
    ):
        raise ValueError(
            "Training presentations differ from the view report"
        )
    if (
        report.get("presentation_report_sha256")
        != hashes["presentation_report"]
    ):
        raise ValueError("Presentation report differs from the view report")
    template_group_coverage = _validate_presentation_group_coverage(
        paths,
        report,
    )

    status = _validate_primary_status(
        report,
        allow_exploratory=allow_exploratory,
    )
    for source_name, source_report in (
        ("provenance report", provenance_report),
        ("procedure report", procedure_report),
    ):
        for field in ("primary_ready", "primary_run", "deny_policy"):
            if source_report.get(field) != report.get(field):
                raise ValueError(
                    f"{source_name} {field} disagrees with the view report"
                )
    return report, {
        **status,
        "canonical_source": source_integrity["canonical_ready"],
        "view_report_path": str(report_path.resolve()),
        "template_group_coverage": template_group_coverage,
        "artifact_sha256": hashes,
        "artifacts": records,
    }


def _view_metadata(
    views: list[dict[str, Any]],
    lexicon: ProcedureUnigramModel,
) -> dict[str, Any]:
    if not views:
        raise ValueError("Training views are empty")
    required = (
        "setting",
        "data_seed",
        "presentation_seed",
        "lexicon_hash",
        "procedure_inventory_hash",
        "procedure_model_hash",
        "procedure_count",
    )
    metadata: dict[str, Any] = {}
    for field in required:
        missing = [
            view.get("view_id", view.get("presentation_id", "<unknown>"))
            for view in views
            if field not in view
        ]
        if missing:
            raise ValueError(
                f"Training views lack required field {field!r}; "
                f"first missing view={missing[0]}"
            )
        values = {view[field] for view in views}
        if len(values) != 1:
            raise ValueError(
                f"Training views mix multiple {field} values: "
                f"{sorted(values, key=repr)}"
            )
        metadata[field] = next(iter(values))
    if str(metadata["lexicon_hash"]) != lexicon.lexicon_hash():
        raise ValueError(
            "Training views were generated with a different procedure lexicon: "
            f"views={metadata['lexicon_hash']}, current={lexicon.lexicon_hash()}"
        )
    if int(metadata["procedure_count"]) != lexicon.size:
        raise ValueError(
            f"Training views expect K={metadata['procedure_count']}, "
            f"but the lexicon has K={lexicon.size}"
        )
    if str(metadata["procedure_model_hash"]) != lexicon.model_hash():
        raise ValueError(
            "Training views carry a different procedure_model_hash"
        )
    if str(metadata["procedure_inventory_hash"]) != lexicon.inventory_hash():
        raise ValueError(
            "Training views carry a different procedure_inventory_hash"
        )
    return {
        "setting": str(metadata["setting"]),
        "data_seed": int(metadata["data_seed"]),
        "presentation_seed": int(metadata["presentation_seed"]),
        "lexicon_hash": str(metadata["lexicon_hash"]),
        "procedure_inventory_hash": str(
            metadata["procedure_inventory_hash"]
        ),
        "procedure_model_hash": str(metadata["procedure_model_hash"]),
        "procedure_count": int(metadata["procedure_count"]),
    }


def _validate_view_schedule(
    views: list[dict[str, Any]],
    lexicon: ProcedureUnigramModel,
    report: dict[str, Any],
) -> dict[str, Any]:
    metadata = _view_metadata(views, lexicon)
    for field in (
        "setting",
        "data_seed",
        "lexicon_hash",
        "procedure_inventory_hash",
        "procedure_model_hash",
        "procedure_count",
    ):
        if report.get(field) != metadata[field]:
            raise ValueError(
                f"View report {field} differs from serialized views"
            )
    if int(report.get("presentations", -1)) != len(views):
        raise ValueError("View report presentation count differs from views JSONL")
    if len(views) % 4:
        raise ValueError(
            f"View schedule has {len(views)} rows; complete 2A+2R groups need a multiple of 4"
        )
    positive_counts = Counter(
        int(segment["piece_id"])
        for view in views
        for segment in view["segments"]
    )
    serialized_positive_counts = [
        positive_counts[piece_id] for piece_id in range(lexicon.size)
    ]
    if report.get("procedure_positive_counts") != serialized_positive_counts:
        raise ValueError(
            "View report procedure_positive_counts differs from serialized views"
        )
    minimum_exposures = int(report.get("minimum_procedure_exposures", 0))
    if minimum_exposures <= 0:
        raise ValueError("View report lacks a positive procedure exposure threshold")
    uncovered = [
        piece_id
        for piece_id, count in enumerate(serialized_positive_counts)
        if count < minimum_exposures
    ]
    if uncovered or report.get("uncovered_procedure_ids") != []:
        raise ValueError(
            f"Training views contain procedures without positive coverage: {uncovered}"
        )

    seen_view_ids: set[str] = set()
    seen_presentation_ids: set[str] = set()
    seen_group_keys: set[tuple[int, int]] = set()
    for index, view in enumerate(views):
        for field in (
            "view_id",
            "presentation_id",
            "epoch",
            "effective_batch_id",
            "position_in_batch",
            "pool",
            "segments",
        ):
            if field not in view:
                raise ValueError(f"View row {index} lacks required field {field!r}")
        expected_view_id = compute_boundary_view_id(
            str(view["presentation_id"]),
            metadata["setting"],
            metadata["data_seed"],
            metadata["procedure_model_hash"],
            view["segments"],
        )
        if view["view_id"] != expected_view_id:
            raise ValueError(
                f"View row {index} has an invalid view_id; expected {expected_view_id}"
            )
        view_id = str(view["view_id"])
        presentation_id = str(view["presentation_id"])
        if view_id in seen_view_ids:
            raise ValueError(f"Duplicate view_id in training schedule: {view_id}")
        if presentation_id in seen_presentation_ids:
            raise ValueError(
                f"Duplicate presentation_id in training schedule: {presentation_id}"
            )
        seen_view_ids.add(view_id)
        seen_presentation_ids.add(presentation_id)

    for start in range(0, len(views), 4):
        group = views[start : start + 4]
        keys = {
            (int(row["epoch"]), int(row["effective_batch_id"]))
            for row in group
        }
        if len(keys) != 1:
            raise ValueError(
                f"Rows {start}:{start + 4} do not share epoch/effective_batch_id"
            )
        key = next(iter(keys))
        if key in seen_group_keys:
            raise ValueError(f"Effective batch key is repeated: {key}")
        seen_group_keys.add(key)
        positions = [int(row["position_in_batch"]) for row in group]
        if positions != [0, 1, 2, 3]:
            raise ValueError(
                f"Effective batch {key} positions must be ordered 0,1,2,3; got {positions}"
            )
        pools = Counter(str(row["pool"]) for row in group)
        if pools != Counter({"A": 2, "R": 2}):
            raise ValueError(
                f"Effective batch {key} must contain exactly 2A+2R; got {dict(pools)}"
            )
        for row in group:
            if bool(row.get("is_A")) != (str(row["pool"]) == "A"):
                raise ValueError(
                    f"View {row['view_id']} has inconsistent pool/is_A labels"
                )
    return {
        **metadata,
        "effective_batches": len(views) // 4,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    set_seed(args.model_seed)
    output_dir = ensure_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lexicon_data = read_json(args.procedure_lexicon)
    lexicon = ProcedureUnigramModel.from_dict(lexicon_data)
    view_report, artifact_integrity = _validate_training_artifacts(
        args.views,
        args.procedure_lexicon,
        lexicon,
        allow_exploratory=bool(
            getattr(args, "allow_exploratory_views", False)
        ),
    )
    source_views = list(read_jsonl(args.views))
    view_metadata = _validate_view_schedule(
        source_views,
        lexicon,
        view_report,
    )
    base_model_identity = build_base_model_identity(
        args.model_name,
        args.model_revision,
        require_reproducible=bool(artifact_integrity["formal_ready"]),
    )
    model_load_name, model_load_revision = base_model_load_location(
        base_model_identity
    )
    epoch_values = sorted({int(view["epoch"]) for view in source_views})
    reported_epochs = int(view_report.get("presentation_epochs", -1))
    if epoch_values != list(range(reported_epochs)):
        raise ValueError(
            f"View schedule epochs are {epoch_values}, expected "
            f"{list(range(reported_epochs))}"
        )
    if (
        args.expected_epochs is not None
        and reported_epochs != args.expected_epochs
    ):
        raise ValueError(
            f"View schedule uses {reported_epochs} epochs, but "
            f"--expected-epochs={args.expected_epochs}"
        )
    source_schedule_updates = int(view_metadata["effective_batches"])
    epoch_limit = getattr(args, "epoch_limit", None)
    training_epochs = getattr(args, "training_epochs", None)
    route_probe_manifest = None
    route_probe_record = None
    repeat_policy = None
    formal_training_epochs = training_epochs
    route_probe_split = getattr(args, "route_probe_split", None)
    if training_epochs is not None and (
        epoch_limit is not None or route_probe_split is not None
    ):
        raise ValueError(
            "--training-epochs cannot be combined with --epoch-limit or "
            "--route-probe-split"
        )
    if route_probe_split is not None:
        if args.method != "tapmem":
            raise ValueError("Routing-LR split training is TapMem-only")
        if epoch_limit is not None:
            raise ValueError("--route-probe-split and --epoch-limit cannot be combined")
        route_probe_manifest, fit_views, _probe_views = load_route_probe_manifest(
            route_probe_split,
            views_path=args.views,
            views=source_views,
        )
        views = [
            {
                **dict(view),
                "_source_epoch": int(view["epoch"]),
                "_training_epoch": None,
                "_repeat_cycle": 0,
            }
            for view in fit_views
        ]
        view_metadata = {
            **view_metadata,
            "effective_batches": len(views) // 4,
        }
        route_probe_record = checkpoint_split_record(route_probe_manifest)
        formal_training_epochs = None
        scheduler_horizon_updates = source_schedule_updates
    elif epoch_limit is not None:
        if epoch_limit <= 0 or epoch_limit > reported_epochs:
            raise ValueError(
                f"--epoch-limit must be in [1, {reported_epochs}], got {epoch_limit}"
            )
        views = [
            {
                **view,
                "_source_epoch": int(view["epoch"]),
                "_training_epoch": int(view["epoch"]),
                "_repeat_cycle": 0,
            }
            for view in source_views
            if int(view["epoch"]) < epoch_limit
        ]
        if not views or len(views) % 4:
            raise AssertionError("Epoch-limited schedule broke complete 2A+2R batches")
        view_metadata = {
            **view_metadata,
            "effective_batches": len(views) // 4,
        }
        formal_training_epochs = None
        scheduler_horizon_updates = source_schedule_updates
    else:
        trained_epochs = (
            training_epochs
            if training_epochs is not None
            else reported_epochs
        )
        views, repeat_policy = _expand_training_schedule(
            source_views,
            source_epochs=reported_epochs,
            training_epochs=trained_epochs,
        )
        view_metadata = {
            **view_metadata,
            "effective_batches": len(views) // 4,
        }
        formal_training_epochs = trained_epochs
        scheduler_horizon_updates = int(view_metadata["effective_batches"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_name,
        revision=model_load_revision,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model = ProceduralMemoryModel.from_pretrained(
        model_load_name,
        tokenizer,
        lexicon.size,
        method=args.method,
        initialization_seed=args.model_seed,
        revision=model_load_revision,
        device=args.device,
        dtype=_dtype(args.dtype),
        local_files_only=not args.allow_download,
        logit_bias_scale=args.logit_bias_scale,
        memory_bank_probability_threshold=(
            args.memory_bank_probability_threshold
        ),
        enable_memory_bank_constraint=args.enable_memory_bank_constraint,
    )
    if model.registry.num_procedures != lexicon.size:
        raise AssertionError("Procedure inventory was silently changed")
    initial_orthogonality = model.orthogonality_report()
    tolerance = 1e-3 if _dtype(args.dtype) != torch.float32 else 1e-5
    if initial_orthogonality["max_absolute_gram_error"] > tolerance:
        raise RuntimeError(
            f"Initial procedure embeddings are not orthogonal: {initial_orthogonality}"
        )

    dataset = BoundaryViewDataset(
        views,
        tokenizer,
        model.registry,
        method=args.method,
        max_length=args.max_length,
        procedure_model=lexicon,
        routing_target_mode=args.routing_target_mode,
        routing_candidate_mass=args.routing_candidate_mass,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda rows: left_pad_collate(
            rows,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )
    memory_parameters = [model.procedure_embeddings]
    if model.eoc_embedding is not None:
        memory_parameters.append(model.eoc_embedding)
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": memory_parameters,
            "lr": args.learning_rate,
            "group_name": "memory",
        }
    ]
    resolved_routing_learning_rate = (
        args.routing_learning_rate
        if args.routing_learning_rate is not None
        else args.learning_rate
    )
    if model.routing_head is not None:
        parameter_groups.append(
            {
                "params": list(model.routing_head.parameters()),
                "lr": resolved_routing_learning_rate,
                "group_name": "routing_head",
            }
        )
    optimizer = AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    total_micro_batches = len(loader)
    if total_micro_batches % args.gradient_accumulation_steps:
        raise AssertionError(
            "The serialized view schedule does not contain complete effective batches"
        )
    run_total_updates = (
        total_micro_batches // args.gradient_accumulation_steps
    )
    if run_total_updates != view_metadata["effective_batches"]:
        raise AssertionError(
            f"Expected {view_metadata['effective_batches']} effective batches, "
            f"but loader geometry gives {run_total_updates}"
        )
    scheduler_total_updates, warmup_steps = _scheduler_geometry(
        run_total_updates=run_total_updates,
        source_total_updates=scheduler_horizon_updates,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_lambda(
            step,
            warmup_steps,
            scheduler_total_updates,
        ),
    )

    model.train()
    model.base_model.eval()
    if model.base_model.training:
        raise AssertionError("Frozen base model must remain in eval mode during training")
    training_modes = {
        "memory_wrapper_training": bool(model.training),
        "base_model_training": bool(model.base_model.training),
        "base_model_forced_eval": True,
    }
    optimizer.zero_grad(set_to_none=True)
    logs: list[dict[str, Any]] = []
    update_step = 0
    route_sites_total = 0
    pending_micro_batches: list[int] = []
    pending_presentation_ids: list[str] = []
    pending_view_ids: list[str] = []
    pending_ar_loss_sum = 0.0
    pending_route_loss_sum = 0.0
    pending_supervised_tokens = 0
    pending_route_sites = 0
    aggregate_ar_loss_sum = 0.0
    aggregate_route_loss_sum = 0.0
    aggregate_supervised_tokens = 0
    aggregate_route_sites = 0
    effective_batch_counts = []
    for start in range(0, len(dataset.examples), 4):
        group_examples = dataset.examples[start : start + 4]
        group_supervised_tokens = sum(
            example.target_length for example in group_examples
        )
        group_route_sites = sum(
            example.route_site_count for example in group_examples
        )
        if group_supervised_tokens <= 0:
            raise AssertionError("An effective batch has no supervised tokens")
        effective_batch_counts.append(
            (group_supervised_tokens, group_route_sites)
        )
    for micro_index, batch in enumerate(loader):
        group_index = micro_index // args.gradient_accumulation_steps
        group_supervised_tokens, group_route_sites = effective_batch_counts[
            group_index
        ]
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        components = compute_training_loss_sums(
            model,
            input_ids,
            attention_mask,
            labels,
            (
                batch["routing_target_probabilities"].to(args.device)
                if batch["routing_target_probabilities"] is not None
                else None
            ),
            (
                batch["routing_valid_mask"].to(args.device)
                if batch["routing_valid_mask"] is not None
                else None
            ),
            args.routing_margin,
        )
        micro_loss = normalized_microbatch_loss(
            components,
            effective_batch_supervised_tokens=group_supervised_tokens,
            effective_batch_route_sites=group_route_sites,
            route_loss_weight=args.route_loss_weight,
        )
        # Backpropagate immediately so gradient accumulation retains only one
        # frozen-backbone graph at a time.
        micro_loss.backward()
        pending_micro_batches.append(micro_index)
        pending_presentation_ids.extend(
            str(value) for value in batch["presentation_ids"]
        )
        pending_view_ids.extend(str(value) for value in batch["view_ids"])
        pending_ar_loss_sum += float(components.ar_loss_sum.detach().item())
        pending_route_loss_sum += float(
            components.route_loss_sum.detach().item()
        )
        pending_supervised_tokens += components.supervised_tokens
        pending_route_sites += components.route_sites
        should_step = (
            (micro_index + 1) % args.gradient_accumulation_steps == 0
        )
        if should_step:
            if pending_supervised_tokens != group_supervised_tokens:
                raise AssertionError(
                    "Observed supervised-token count differs from serialized group"
                )
            if pending_route_sites != group_route_sites:
                raise AssertionError(
                    "Observed route-site count differs from serialized group"
                )
            ar_loss = pending_ar_loss_sum / group_supervised_tokens
            route_loss = (
                pending_route_loss_sum / group_route_sites
                if group_route_sites
                else 0.0
            )
            total_loss = ar_loss + args.route_loss_weight * route_loss
            metrics: dict[str, float | int] = {
                "total_loss": total_loss,
                "ar_loss": ar_loss,
                "route_loss": route_loss,
                "ar_loss_sum": pending_ar_loss_sum,
                "route_loss_sum": pending_route_loss_sum,
                "supervised_tokens": pending_supervised_tokens,
                "route_sites": pending_route_sites,
            }
            _assert_finite_gradients(model)
            learning_rate = optimizer.param_groups[0]["lr"]
            routing_learning_rate = (
                optimizer.param_groups[1]["lr"]
                if model.routing_head is not None
                else None
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_step += 1
            route_sites_total += int(metrics["route_sites"])
            aggregate_ar_loss_sum += float(metrics["ar_loss_sum"])
            aggregate_route_loss_sum += float(metrics["route_loss_sum"])
            aggregate_supervised_tokens += int(metrics["supervised_tokens"])
            aggregate_route_sites += int(metrics["route_sites"])
            group_start = (update_step - 1) * 4
            group_views = views[group_start : group_start + 4]
            source_epoch = int(group_views[0]["_source_epoch"])
            training_epoch = group_views[0]["_training_epoch"]
            repeat_cycle = int(group_views[0]["_repeat_cycle"])
            group_key = (
                source_epoch,
                int(group_views[0]["effective_batch_id"]),
            )
            if pending_view_ids != [
                str(view["view_id"]) for view in group_views
            ]:
                raise AssertionError(
                    "DataLoader order differs from the validated view schedule"
                )
            logs.append(
                {
                    "effective_batch_index": update_step - 1,
                    "optimizer_step": update_step,
                    "epoch": (
                        int(training_epoch)
                        if training_epoch is not None
                        else source_epoch
                    ),
                    "training_epoch": (
                        int(training_epoch)
                        if training_epoch is not None
                        else None
                    ),
                    "source_epoch": source_epoch,
                    "repeat_cycle": repeat_cycle,
                    "repeat_policy": (
                        repeat_policy["schema"]
                        if repeat_policy is not None
                        else None
                    ),
                    "effective_batch_id": group_key[1],
                    "source_effective_batch_id": group_key[1],
                    "micro_batches": list(pending_micro_batches),
                    "microbatch_geometry": {
                        "batch_size": args.batch_size,
                        "gradient_accumulation_steps": (
                            args.gradient_accumulation_steps
                        ),
                    },
                    "learning_rate": learning_rate,
                    "routing_learning_rate": routing_learning_rate,
                    "routing_head_row_norms": _routing_head_norms(model),
                    **metrics,
                    "presentation_ids": list(pending_presentation_ids),
                    "view_ids": list(pending_view_ids),
                    "pools": [str(view["pool"]) for view in group_views],
                }
            )
            pending_micro_batches.clear()
            pending_presentation_ids.clear()
            pending_view_ids.clear()
            pending_ar_loss_sum = 0.0
            pending_route_loss_sum = 0.0
            pending_supervised_tokens = 0
            pending_route_sites = 0
    if pending_micro_batches:
        raise AssertionError("Training ended with an incomplete effective batch")
    if update_step != run_total_updates:
        raise AssertionError(
            f"Expected {run_total_updates} updates, performed {update_step}"
        )

    exposure = dataset.exposure_report()
    if args.method == "tapmem" and route_sites_total != exposure["route_sites"]:
        raise AssertionError(
            f"Observed route sites {route_sites_total} != serialized {exposure['route_sites']}"
        )
    overall_ar_loss = aggregate_ar_loss_sum / aggregate_supervised_tokens
    overall_route_loss = (
        aggregate_route_loss_sum / aggregate_route_sites
        if aggregate_route_sites
        else 0.0
    )
    overall_loss = (
        overall_ar_loss + args.route_loss_weight * overall_route_loss
    )
    checkpoint_formal_ready = bool(
        artifact_integrity["formal_ready"]
        and base_model_identity["reproducible"] is True
        and epoch_limit is None
        and route_probe_manifest is None
    )
    summary = {
        "method": args.method,
        "K": lexicon.size,
        "lexicon_hash": lexicon.lexicon_hash(),
        "presentations": len(dataset),
        "micro_batches": total_micro_batches,
        "optimizer_steps": update_step,
        "run_total_updates": run_total_updates,
        "scheduler_total_updates": scheduler_total_updates,
        "warmup_steps": warmup_steps,
        "source_schedule_epochs": reported_epochs,
        "source_schedule_updates": source_schedule_updates,
        "formal_training_epochs": formal_training_epochs,
        "training_epochs_requested": training_epochs,
        "repeat_policy": repeat_policy,
        "trained_epochs": (
            epoch_limit
            if epoch_limit is not None
            else (
                formal_training_epochs
                if route_probe_manifest is None
                else None
            )
        ),
        "trained_source_epoch_equivalent": (
            route_probe_manifest["fit_source_epoch_equivalent"]
            if route_probe_manifest is not None
            else (
                epoch_limit
                if epoch_limit is not None
                else formal_training_epochs
            )
        ),
        "epoch_limit": epoch_limit,
        "route_probe_split": route_probe_record,
        "memory_learning_rate": args.learning_rate,
        "routing_learning_rate": (
            resolved_routing_learning_rate
            if model.routing_head is not None
            else None
        ),
        "routing_head_row_norms": _routing_head_norms(model),
        "initial_orthogonality": initial_orthogonality,
        "view_metadata": view_metadata,
        "artifact_integrity": artifact_integrity,
        "artifact_sha256": artifact_integrity["artifact_sha256"],
        "primary_ready": artifact_integrity["primary_ready"],
        "primary_run": artifact_integrity["primary_run"],
        "deny_policy": artifact_integrity["deny_policy"],
        "artifact_formal_ready": artifact_integrity["formal_ready"],
        "base_model_identity": base_model_identity,
        "formal_ready": checkpoint_formal_ready,
        "training_modes": training_modes,
        "loss_normalization": LOSS_NORMALIZATION,
        "target_serialization_policy": TARGET_SERIALIZATION_POLICY,
        "routing_target_mode": args.routing_target_mode,
        "routing_candidate_mass": args.routing_candidate_mass,
        "routing_margin": args.routing_margin,
        "tcra_trigger_policy": (
            (
                TCRA_TRIGGER_POLICY
                if model.use_memory_bank_constraint
                else TCRA_ADDITIVE_ONLY_TRIGGER_POLICY
            )
            if args.method == "tapmem"
            else None
        ),
        "memory_bank_constraint_enabled": (
            model.use_memory_bank_constraint
            if args.method == "tapmem"
            else False
        ),
        "memory_bank_probability_threshold": (
            args.memory_bank_probability_threshold
            if args.method == "tapmem"
            else None
        ),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "exposure": exposure,
        "final_loss": logs[-1]["total_loss"] if logs else None,
        "overall_loss": overall_loss,
        "overall_ar_loss": overall_ar_loss,
        "overall_route_loss": overall_route_loss,
    }
    write_jsonl(output_dir / "training_log.jsonl", logs)
    write_json(output_dir / "training_summary.json", summary)
    training_config = {
        **vars(args),
        "routing_learning_rate": (
            resolved_routing_learning_rate
            if model.routing_head is not None
            else None
        ),
        "data_seed": view_metadata["data_seed"],
        "setting": view_metadata["setting"],
        "presentation_seed": view_metadata["presentation_seed"],
        "views_lexicon_hash": view_metadata["lexicon_hash"],
        "loss_normalization": LOSS_NORMALIZATION,
        "source_schedule_epochs": reported_epochs,
        "source_schedule_updates": source_schedule_updates,
        "formal_training_epochs": formal_training_epochs,
        "repeat_policy": repeat_policy,
    }
    validate_base_model_identity(base_model_identity)
    save_checkpoint(
        output_dir,
        model,
        tokenizer,
        base_model_name=model_load_name,
        base_model_revision=model_load_revision,
        base_model_identity=base_model_identity,
        lexicon_hash=lexicon.lexicon_hash(),
        training_config=training_config,
        training_summary=summary,
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--procedure-lexicon",
        default=str(PACKAGE_ROOT / "artifacts" / "default" / "procedure_lexicon.json"),
    )
    parser.add_argument("--views", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--expected-epochs",
        type=int,
        help="Optional exact check of epochs stored in the source views artifact",
    )
    parser.add_argument(
        "--training-epochs",
        type=int,
        help=(
            "Formal training epochs; after one complete source pass, reuse "
            "source epochs from epoch 0 in order"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument(
        "--routing-learning-rate",
        type=float,
        help="TapMem routing-head LR; defaults to --learning-rate",
    )
    parser.add_argument(
        "--epoch-limit",
        type=int,
        help="Train only the first N schedule epochs for an exploratory probe",
    )
    parser.add_argument(
        "--route-probe-split",
        help=(
            "TRAIN-only group-disjoint manifest for an exploratory TapMem "
            "routing-head LR run"
        ),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--route-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--routing-target-mode",
        choices=sorted(ROUTING_TARGET_MODES),
        default="one_hot",
        help=(
            "TapMem routing supervision: the selected procedure only, or the "
            "fixed-count conditional posterior at the current boundary"
        ),
    )
    parser.add_argument(
        "--routing-candidate-mass",
        type=float,
        default=0.9,
        help="Cumulative fixed-count probability retained as a valid set",
    )
    parser.add_argument(
        "--routing-margin",
        type=float,
        default=0.5,
        help="Required valid-vs-invalid calibrated memory-logit margin",
    )
    parser.add_argument("--logit-bias-scale", type=float, default=1.0)
    parser.add_argument(
        "--memory-bank-probability-threshold",
        type=float,
        default=0.5,
    )
    memory_bank_group = parser.add_mutually_exclusive_group()
    memory_bank_group.add_argument(
        "--enable-memory-bank-constraint",
        action="store_true",
        help=(
            "Enable the legacy hard memory-bank probability gate; the "
            "default TapMem path uses additive raw-top-1 TCRA only"
        ),
    )
    memory_bank_group.add_argument(
        "--disable-memory-bank-constraint",
        dest="enable_memory_bank_constraint",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(enable_memory_bank_constraint=False)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-exploratory-views",
        action="store_true",
        help=(
            "Allow non-primary provenance/view artifacts for debugging only; "
            "formal training rejects them by default"
        ),
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch sizes must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.training_epochs is not None and args.training_epochs <= 0:
        parser.error("--training-epochs must be positive")
    if args.routing_learning_rate is not None and args.routing_learning_rate <= 0:
        parser.error("--routing-learning-rate must be positive")
    if not 0.0 < args.routing_candidate_mass <= 1.0:
        parser.error("--routing-candidate-mass must be in (0, 1]")
    if args.routing_margin < 0.0:
        parser.error("--routing-margin must be nonnegative")
    if not 0.0 <= args.memory_bank_probability_threshold <= 1.0:
        parser.error("--memory-bank-probability-threshold must be in [0, 1]")
    if args.method != "tapmem" and args.routing_learning_rate is not None:
        parser.error("--routing-learning-rate applies only to tapmem")
    if args.method != "tapmem" and args.enable_memory_bank_constraint:
        parser.error("--enable-memory-bank-constraint applies only to tapmem")
    if args.method != "tapmem" and args.routing_target_mode != "one_hot":
        parser.error("--routing-target-mode applies only to tapmem")
    if args.route_probe_split is not None and args.method != "tapmem":
        parser.error("--route-probe-split applies only to tapmem")
    if args.route_probe_split is not None and args.epoch_limit is not None:
        parser.error("--route-probe-split and --epoch-limit cannot be combined")
    if args.training_epochs is not None and (
        args.route_probe_split is not None or args.epoch_limit is not None
    ):
        parser.error(
            "--training-epochs cannot be combined with --route-probe-split "
            "or --epoch-limit"
        )
    if (
        args.route_probe_split is not None
        and args.memory_bank_probability_threshold != 0.5
    ):
        parser.error("Routing-LR probes fix --memory-bank-probability-threshold=0.5")
    if args.batch_size * args.gradient_accumulation_steps != 4:
        parser.error(
            "batch_size * gradient_accumulation_steps must equal 4 so each "
            "optimizer update contains one fixed 2A+2R effective batch"
        )
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train(args)
    print(summary)


if __name__ == "__main__":
    main()
