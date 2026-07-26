from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from compositional_intercode_bash.io_utils import sha256_file
from compositional_intercode_bash.probe_checkpoint import (
    _batch_probe_metrics,
    _validate_checkpoint_split,
)
from compositional_intercode_bash.route_probe_split import (
    build_route_probe_manifest,
    checkpoint_split_record,
    validate_route_probe_manifest,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _formal_views(root: Path) -> tuple[Path, list[dict]]:
    views_path = root / "views.sampled.1729.jsonl"
    rows = []
    effective_batch_id = 0
    for epoch in range(2):
        for a_index in range(4):
            for r_index in range(4):
                groups = [
                    (f"A-{a_index}", "A"),
                    (f"A-{a_index}", "A"),
                    (f"R-{r_index}", "R"),
                    (f"R-{r_index}", "R"),
                ]
                for position, (group_id, pool) in enumerate(groups):
                    piece_id = (
                        a_index % 2
                        if pool == "A"
                        else 2 + r_index % 2
                    )
                    rows.append(
                        {
                            "view_id": (
                                f"v-{epoch}-{effective_batch_id}-{position}"
                            ),
                            "presentation_id": (
                                f"p-{epoch}-{effective_batch_id}-{position}"
                            ),
                            "template_group_id": group_id,
                            "pool": pool,
                            "is_A": pool == "A",
                            "epoch": epoch,
                            "effective_batch_id": effective_batch_id,
                            "position_in_batch": position,
                            "segments": [
                                {"piece_id": piece_id, "start": 0, "end": 1}
                            ],
                        }
                    )
                effective_batch_id += 1
    _write_jsonl(views_path, rows)
    presentation_report_path = root / "presentation_report.json"
    _write_json(
        presentation_report_path,
        {
            "schema": "balanced_presentations_v2",
            "minimum_requested_epochs": 2,
            "epochs": 2,
            "presentations": len(rows),
            "group_coverage_epoch_floor": 1,
            "group_count_per_pool_per_epoch": 32,
            "available_group_count_by_pool": {"A": 4, "R": 4},
            "covered_group_count_by_pool": {"A": 4, "R": 4},
            "uncovered_group_ids_by_pool": {"A": [], "R": []},
            "min_presentations_per_group_by_pool": {"A": 16, "R": 16},
            "max_presentations_per_group_by_pool": {"A": 16, "R": 16},
        },
    )
    _write_json(
        views_path.with_suffix(".report.json"),
        {
            "schema": "boundary_views_v2",
            "primary_ready": True,
            "primary_run": True,
            "primary_requested": True,
            "deny_policy": "all_candidates",
            "presentation_epochs": 2,
            "presentations": len(rows),
            "procedure_count": 4,
            "views_sha256": sha256_file(views_path),
            "presentation_report_sha256": sha256_file(
                presentation_report_path
            ),
        },
    )
    return views_path, rows


class RouteProbeSplitTest(unittest.TestCase):
    def test_manifest_is_deterministic_complete_and_group_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            views_path, rows = _formal_views(Path(temporary))
            first = build_route_probe_manifest(
                views_path,
                split_seed=17,
                holdout_fraction=0.25,
                fit_source_epochs=1,
            )
            second = build_route_probe_manifest(
                views_path,
                split_seed=17,
                holdout_fraction=0.25,
                fit_source_epochs=1,
            )
            self.assertEqual(first, second)
            fit_views, probe_views = validate_route_probe_manifest(
                first,
                views_path=views_path,
                views=rows,
            )
            self.assertEqual(len(fit_views), 16 * 4)
            self.assertEqual(len(probe_views), 2)
            self.assertEqual(
                first["probe_presentations_by_pool"],
                {"A": 1, "R": 1},
            )
            self.assertTrue(
                all(
                    count > 0
                    for count in first["fit_procedure_positive_counts"]
                )
            )
            self.assertTrue(
                set(first["probe_procedure_piece_ids"]).issubset(
                    {
                        piece_id
                        for piece_id, count in enumerate(
                            first["fit_procedure_positive_counts"]
                        )
                        if count > 0
                    }
                )
            )
            fit_groups = {
                str(row["template_group_id"]) for row in fit_views
            }
            probe_groups = {
                str(row["template_group_id"]) for row in probe_views
            }
            self.assertFalse(fit_groups & probe_groups)
            for start in range(0, len(fit_views), 4):
                batch = fit_views[start : start + 4]
                self.assertEqual(
                    [row["pool"] for row in batch].count("A"),
                    2,
                )
                self.assertEqual(
                    [row["pool"] for row in batch].count("R"),
                    2,
                )

    def test_checkpoint_must_record_exact_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            views_path, _rows = _formal_views(Path(temporary))
            manifest = build_route_probe_manifest(
                views_path,
                split_seed=19,
                holdout_fraction=0.25,
                fit_source_epochs=1,
            )
            metadata = {
                "formal_ready": False,
                "training_summary": {
                    "route_probe_split": checkpoint_split_record(manifest),
                    "run_total_updates": manifest["fit_updates"],
                    "scheduler_total_updates": manifest[
                        "source_schedule_updates"
                    ],
                    "warmup_steps": (
                        manifest["source_schedule_updates"] // 10
                    ),
                },
            }
            _validate_checkpoint_split(metadata, manifest)
            metadata["training_summary"]["route_probe_split"][
                "split_sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs"):
                _validate_checkpoint_split(metadata, manifest)


class _ProbeModel:
    def __init__(self):
        self.registry = SimpleNamespace(
            procedure_token_ids=(1, 2),
            eoc_token_id=3,
            num_procedures=2,
        )
        self._procedure_token_ids = torch.tensor([1, 2], dtype=torch.long)
        self.memory_bank_probability_threshold = 0.5
        self.logit_bias_scale = 1.0

    def routing_scores(self, hidden):
        return hidden

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
    ):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 6), -10.0)
        hidden = torch.zeros((batch, length, 2))

        logits[0, 0, 1:3] = 0.0
        hidden[0, 0] = torch.tensor([10.0, 0.0])
        logits[0, 3, 1:3] = 0.0
        logits[0, 3, 4] = 0.2

        logits[1, 0, 1:3] = -1.0
        logits[1, 0, 4] = 0.0
        hidden[1, 0] = torch.tensor([0.0, 10.0])
        logits[1, 3, 1:3] = -3.0
        logits[1, 3, 4] = 0.2
        return SimpleNamespace(
            logits=logits,
            final_hidden_state=hidden,
        )


class RouteProbeMetricTest(unittest.TestCase):
    def test_reports_threshold_metrics_separately_by_pool(self):
        labels = torch.tensor(
            [
                [-100, 1, 4, 3, 4],
                [-100, 2, 4, 3, 4],
            ],
            dtype=torch.long,
        )
        batch = {
            "input_ids": torch.zeros_like(labels),
            "attention_mask": torch.ones_like(labels),
            "labels": labels,
            "pools": ["A", "R"],
        }
        metrics = _batch_probe_metrics(
            _ProbeModel(),
            batch,
            device="cpu",
        )
        self.assertEqual(metrics["A"].outside_memory_sites, 1)
        self.assertEqual(metrics["A"].outside_memory_triggers, 1)
        self.assertEqual(metrics["A"].outside_ordinary_sites, 1)
        self.assertEqual(metrics["A"].outside_ordinary_triggers, 1)
        self.assertEqual(
            metrics["A"].triggered_memory_bank_top1_correct,
            1,
        )
        self.assertEqual(metrics["R"].outside_memory_sites, 1)
        self.assertEqual(metrics["R"].outside_memory_triggers, 0)
        self.assertEqual(metrics["R"].outside_ordinary_sites, 1)
        self.assertEqual(metrics["R"].outside_ordinary_triggers, 0)
        self.assertEqual(metrics["A"].route_sites, 1)
        self.assertEqual(metrics["R"].route_sites, 1)


if __name__ == "__main__":
    unittest.main()
