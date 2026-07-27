from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from compositional_intercode_bash.io_utils import sha256_directory, sha256_file
from compositional_intercode_bash.memory_model import (
    MemoryTokenRegistry,
    ProceduralMemoryModel,
)
from compositional_intercode_bash.tests.helpers import (
    ByteTokenizer,
    TinyCausalLM,
    write_valid_source_artifacts,
)
from compositional_intercode_bash.train import (
    PROCEDURE_EVIDENCE_ARTIFACT_NAMES,
    PROVENANCE_ARTIFACT_NAMES,
    VIEW_INPUT_ARTIFACT_NAMES,
    _expand_training_schedule,
    _scheduler_geometry,
    _scheduler_lambda,
    _validate_training_artifacts,
    _validate_view_schedule,
)
from compositional_intercode_bash.training_data import (
    BoundaryViewDataset,
    compute_training_loss_sums,
    left_pad_collate,
    normalized_microbatch_loss,
)
from compositional_intercode_bash.unigram import ProcedureUnigramModel
from compositional_intercode_bash.views import compute_boundary_view_id


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class PositionAwareTinyCausalLM(TinyCausalLM):
    """Expose batching-dependent padding mistakes through explicit positions."""

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)
        positioned = inputs_embeds + (
            position_ids.to(inputs_embeds.dtype).unsqueeze(-1) * 0.01
        )
        hidden = self.transform(positioned)
        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            past_key_values=("cache",) if use_cache else None,
        )


class TrainingArtifactIntegrityTest(unittest.TestCase):
    def test_151_runtime_epochs_reuse_the_100_epoch_source_prefix(self):
        source_views = [
            {
                "epoch": epoch,
                "source_position": position,
            }
            for epoch in range(100)
            for position in range(52)
        ]
        expanded, repeat_policy = _expand_training_schedule(
            source_views,
            source_epochs=100,
            training_epochs=151,
        )

        self.assertEqual(len(source_views), 5200)
        self.assertEqual(len(expanded), 7852)
        self.assertEqual(len(expanded) // 4, 1963)
        self.assertEqual(
            _scheduler_geometry(
                run_total_updates=len(expanded) // 4,
                source_total_updates=len(expanded) // 4,
            ),
            (1963, 196),
        )
        self.assertEqual(
            [
                (row["epoch"], row["source_position"])
                for row in expanded[:5200]
            ],
            [
                (row["epoch"], row["source_position"])
                for row in source_views
            ],
        )
        self.assertEqual(
            [
                (row["epoch"], row["source_position"])
                for row in expanded[5200:]
            ],
            [
                (row["epoch"], row["source_position"])
                for row in source_views[: 51 * 52]
            ],
        )
        self.assertEqual(expanded[-1]["_training_epoch"], 150)
        self.assertEqual(expanded[-1]["_source_epoch"], 50)
        self.assertEqual(expanded[-1]["_repeat_cycle"], 1)
        self.assertNotIn("_training_epoch", source_views[0])
        self.assertEqual(
            repeat_policy,
            {
                "schema": "source_epoch_modulo_v1",
                "source_artifact_rewritten": False,
                "mapping": (
                    "source_epoch = training_epoch % "
                    "source_schedule_epochs"
                ),
                "source_schedule_epochs": 100,
                "trained_epochs": 151,
                "complete_source_passes": 1,
                "additional_source_epoch_prefix": 51,
                "repeated_training_epochs": 51,
            },
        )

    def test_partial_run_keeps_full_schedule_learning_rate_horizon(self):
        scheduler_updates, warmup_steps = _scheduler_geometry(
            run_total_updates=260,
            source_total_updates=1963,
        )
        self.assertEqual(scheduler_updates, 1963)
        self.assertEqual(warmup_steps, 196)
        self.assertGreater(
            _scheduler_lambda(259, warmup_steps, scheduler_updates),
            0.9,
        )
        self.assertEqual(_scheduler_lambda(259, 26, 260), 0.0)

    def _model(self) -> ProcedureUnigramModel:
        signature = ("START", "find")
        return ProcedureUnigramModel(
            [(signature,)],
            [1.0],
            0.2,
            mandatory=[True],
        )

    def _materialize(
        self,
        root: Path,
        *,
        primary: bool,
    ) -> tuple[ProcedureUnigramModel, Path, Path, list[dict]]:
        model = self._model()
        write_valid_source_artifacts(root)
        lexicon_path = root / "procedure_lexicon.json"
        segmentations_path = root / "segmentations.jsonl"
        provenance_path = root / "provenance.jsonl"
        provenance_report_path = root / "provenance_report.json"
        adjudications_path = root / "adjudications.used.jsonl"
        groups_path = root / "groups.json"
        derived_path = root / "raw_pairs.grouped_and_split.jsonl"
        derived_report_path = root / "derived_split_report.json"
        procedure_report_path = root / "procedure_report.json"
        boundary_gate_path = root / "training_boundary_gate.json"
        presentations_path = root / "training_presentations.jsonl"
        presentation_report_path = root / "presentation_report.json"
        views_path = root / "views.sampled.1729.jsonl"
        coverage_path = views_path.with_suffix(".coverage.json")
        evidence_paths = {
            "atoms": root / "atoms.jsonl",
            "atomization_report": root / "atomization_report.json",
            "candidates": root / "candidates.jsonl",
            "capacity_selection": root / "capacity_selection.json",
            "pruning_paths": root / "pruning_paths.json",
        }
        grid_path = root / "procedure_grid"
        grid_path.mkdir()
        fixture_pools = ("A", "R", "A", "R")
        fixture_groups = [
            f"{pool}-{position // 2}"
            for position, pool in enumerate(fixture_pools)
        ]

        _write_json(lexicon_path, model.to_dict())
        _write_jsonl(
            segmentations_path,
            [
                {
                    "derived_split": "TRAIN",
                    "template_group_id": group_id,
                    "is_ambiguous": pool == "A",
                }
                for pool, group_id in zip(fixture_pools, fixture_groups)
            ],
        )
        _write_jsonl(adjudications_path, [])
        _write_json(groups_path, {"schema": "test_groups"})
        _write_jsonl(provenance_path, [])
        _write_jsonl(derived_path, [])
        _write_json(derived_report_path, {"schema": "test"})
        _write_json(boundary_gate_path, {"schema": "test_gate"})
        _write_jsonl(
            presentations_path,
            [
                {
                    "presentation_id": f"presentation-{position}",
                    "template_group_id": group_id,
                    "pool": pool,
                    "epoch": 0,
                }
                for position, (pool, group_id) in enumerate(
                    zip(fixture_pools, fixture_groups)
                )
            ],
        )
        _write_json(
            presentation_report_path,
            {
                "schema": "balanced_presentations_v2",
                "minimum_requested_epochs": 1,
                "epochs": 1,
                "presentations": 4,
                "group_coverage_epoch_floor": 1,
                "group_count_per_pool_per_epoch": 2,
                "available_group_count_by_pool": {"A": 2, "R": 2},
                "covered_group_count_by_pool": {"A": 2, "R": 2},
                "uncovered_group_ids_by_pool": {"A": [], "R": []},
                "min_presentations_per_group_by_pool": {"A": 1, "R": 1},
                "max_presentations_per_group_by_pool": {"A": 1, "R": 1},
            },
        )
        for name, path in evidence_paths.items():
            if path.suffix == ".jsonl":
                _write_jsonl(path, [{"evidence": name}])
            else:
                _write_json(path, {"evidence": name})
        _write_json(grid_path / "K_0001.json", model.to_dict())

        primary_ready = bool(primary)
        primary_run = bool(primary)
        deny_policy = "all_candidates"
        _write_json(
            provenance_report_path,
            {
                "primary_ready": primary_ready,
                "primary_run": primary_run,
                "deny_policy": deny_policy,
                "tasks": 200 if primary else 200,
                "resolved": 200 if primary else 0,
                "unresolved": 0 if primary else 200,
                "candidate_union_group_count": 0,
                "actual_deny_group_count": 0,
                "unresolved_candidates_removed": True,
                "input_sha256": {
                    "raw_pairs_official": sha256_file(
                        root / "raw_pairs.official.jsonl"
                    ),
                    "intercode_tasks": sha256_file(
                        root / "intercode_tasks.jsonl"
                    ),
                    "source_manifest": sha256_file(
                        root / "source_manifest.json"
                    ),
                },
                "adjudications_sha256": sha256_file(adjudications_path),
                "groups_sha256": sha256_file(groups_path),
                "provenance_sha256": sha256_file(provenance_path),
                "derived_split_sha256": sha256_file(derived_path),
                "derived_split_report_sha256": sha256_file(
                    derived_report_path
                ),
            },
        )
        artifact_paths = {
            "raw_pairs_official": root / "raw_pairs.official.jsonl",
            "intercode_tasks": root / "intercode_tasks.jsonl",
            "source_manifest": root / "source_manifest.json",
            "adjudications": adjudications_path,
            "groups": groups_path,
            "provenance": provenance_path,
            "provenance_report": provenance_report_path,
            "derived_split": derived_path,
            "derived_split_report": derived_report_path,
        }
        provenance_inputs = {
            name: sha256_file(artifact_paths[name])
            for name in PROVENANCE_ARTIFACT_NAMES
        }
        _write_json(
            procedure_report_path,
            {
                "primary_ready": primary_ready,
                "primary_run": primary_run,
                "deny_policy": deny_policy,
                "selected_k": model.size,
                "selected_lexicon_hash": model.model_hash(),
                "procedure_model_hash": model.model_hash(),
                "procedure_inventory_hash": model.inventory_hash(),
                "procedure_lexicon_sha256": sha256_file(lexicon_path),
                "segmentations_sha256": sha256_file(segmentations_path),
                "training_boundary_gate_sha256": sha256_file(
                    boundary_gate_path
                ),
                "evidence_sha256": {
                    **{
                        name: sha256_file(path)
                        for name, path in evidence_paths.items()
                    },
                    "procedure_grid": sha256_directory(grid_path),
                },
                "procedure_grid_files": ["K_0001.json"],
                "input_sha256": provenance_inputs,
            },
        )

        views = []
        for position, (pool, group_id) in enumerate(
            zip(fixture_pools, fixture_groups)
        ):
            presentation_id = f"presentation-{position}"
            segments = [{"piece_id": 0, "start": 0, "end": 1}]
            views.append(
                {
                    "presentation_id": presentation_id,
                    "view_id": compute_boundary_view_id(
                        presentation_id,
                        "sampled",
                        1729,
                        model.model_hash(),
                        segments,
                    ),
                    "setting": "sampled",
                    "data_seed": 1729,
                    "presentation_seed": 314159,
                    "lexicon_hash": model.model_hash(),
                    "procedure_inventory_hash": model.inventory_hash(),
                    "procedure_model_hash": model.model_hash(),
                    "procedure_count": model.size,
                    "epoch": 0,
                    "effective_batch_id": 0,
                    "position_in_batch": position,
                    "pool": pool,
                    "template_group_id": group_id,
                    "is_A": pool == "A",
                    "segments": segments,
                }
            )
        _write_jsonl(views_path, views)
        coverage = {
            "schema": "procedure_positive_coverage_v1",
            "procedure_count": model.size,
            "minimum_procedure_exposures": 1,
            "procedure_positive_counts": [4],
            "uncovered_procedure_ids": [],
            "native_reserved_boundary": 247,
            "native_procedure_positive_coverage": 1,
            "native_procedure_count": 1,
            "added_procedure_positive_coverage": 0,
            "added_procedure_count": 0,
        }
        _write_json(coverage_path, coverage)
        procedure_artifact_paths = {
            **artifact_paths,
            "procedure_lexicon": lexicon_path,
            "segmentations": segmentations_path,
            "procedure_report": procedure_report_path,
            "training_boundary_gate": boundary_gate_path,
            **evidence_paths,
            "procedure_grid": grid_path,
        }
        _write_json(
            views_path.with_suffix(".report.json"),
            {
                "setting": "sampled",
                "data_seed": 1729,
                "lexicon_hash": model.model_hash(),
                "procedure_inventory_hash": model.inventory_hash(),
                "procedure_model_hash": model.model_hash(),
                "procedure_count": model.size,
                "presentations": len(views),
                "presentation_epochs": 1,
                "minimum_requested_epochs": 1,
                **{
                    field: value
                    for field, value in coverage.items()
                    if field != "schema"
                },
                "primary_ready": primary_ready,
                "primary_run": primary_run,
                "deny_policy": deny_policy,
                "primary_requested": bool(primary),
                "input_sha256": {
                    name: (
                        sha256_directory(procedure_artifact_paths[name])
                        if procedure_artifact_paths[name].is_dir()
                        else sha256_file(procedure_artifact_paths[name])
                    )
                    for name in VIEW_INPUT_ARTIFACT_NAMES
                },
                "views_sha256": sha256_file(views_path),
                "coverage_report_sha256": sha256_file(coverage_path),
                "training_presentations_sha256": sha256_file(
                    presentations_path
                ),
                "presentation_report_sha256": sha256_file(
                    presentation_report_path
                ),
            },
        )
        return model, lexicon_path, views_path, views

    def test_formal_chain_and_every_effective_batch_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, lexicon_path, views_path, views = self._materialize(
                Path(temporary),
                primary=True,
            )
            report, integrity = _validate_training_artifacts(
                views_path,
                lexicon_path,
                model,
                allow_exploratory=False,
            )
            metadata = _validate_view_schedule(views, model, report)
            self.assertTrue(integrity["formal_ready"])
            self.assertEqual(metadata["effective_batches"], 1)
            self.assertEqual(
                set(integrity["artifact_sha256"]),
                set(VIEW_INPUT_ARTIFACT_NAMES)
                | {
                    "views",
                    "view_report",
                    "coverage_report",
                    "training_presentations",
                    "presentation_report",
                },
            )

            invalid = [dict(row) for row in views]
            invalid[0]["view_id"] = "tampered"
            with self.assertRaisesRegex(ValueError, "invalid view_id"):
                _validate_view_schedule(invalid, model, report)

            invalid = [dict(row) for row in views]
            invalid[0]["pool"] = "R"
            invalid[0]["is_A"] = False
            with self.assertRaisesRegex(ValueError, "exactly 2A\\+2R"):
                _validate_view_schedule(invalid, model, report)

    def test_exploratory_views_require_an_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, lexicon_path, views_path, _views = self._materialize(
                Path(temporary),
                primary=False,
            )
            with self.assertRaisesRegex(RuntimeError, "Formal training"):
                _validate_training_artifacts(
                    views_path,
                    lexicon_path,
                    model,
                    allow_exploratory=False,
                )
            _report, integrity = _validate_training_artifacts(
                views_path,
                lexicon_path,
                model,
                allow_exploratory=True,
            )
            self.assertTrue(integrity["exploratory_override"])
            self.assertFalse(integrity["formal_ready"])

    def test_modified_views_fail_the_same_name_report_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, lexicon_path, views_path, _views = self._materialize(
                Path(temporary),
                primary=True,
            )
            with views_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "same-name view report"):
                _validate_training_artifacts(
                    views_path,
                    lexicon_path,
                    model,
                    allow_exploratory=False,
                )

    def test_partial_template_group_coverage_is_reported_not_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, lexicon_path, views_path, _views = self._materialize(
                root,
                primary=True,
            )
            segmentations_path = root / "segmentations.jsonl"
            with segmentations_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "derived_split": "TRAIN",
                            "template_group_id": "R-unseen",
                            "is_ambiguous": False,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            procedure_report_path = root / "procedure_report.json"
            procedure_report = json.loads(
                procedure_report_path.read_text(encoding="utf-8")
            )
            procedure_report["segmentations_sha256"] = sha256_file(
                segmentations_path
            )
            _write_json(procedure_report_path, procedure_report)

            presentation_report_path = root / "presentation_report.json"
            presentation_report = json.loads(
                presentation_report_path.read_text(encoding="utf-8")
            )
            for field in (
                "group_coverage_epoch_floor",
                "available_group_count_by_pool",
                "covered_group_count_by_pool",
                "uncovered_group_ids_by_pool",
                "min_presentations_per_group_by_pool",
                "max_presentations_per_group_by_pool",
            ):
                presentation_report.pop(field)
            _write_json(presentation_report_path, presentation_report)
            view_report_path = views_path.with_suffix(".report.json")
            view_report = json.loads(
                view_report_path.read_text(encoding="utf-8")
            )
            view_report["input_sha256"]["segmentations"] = sha256_file(
                segmentations_path
            )
            view_report["input_sha256"]["procedure_report"] = sha256_file(
                procedure_report_path
            )
            view_report["presentation_report_sha256"] = sha256_file(
                presentation_report_path
            )
            _write_json(view_report_path, view_report)

            _report, integrity = _validate_training_artifacts(
                views_path,
                lexicon_path,
                model,
                allow_exploratory=False,
            )
            coverage = integrity["template_group_coverage"]
            self.assertFalse(
                coverage["all_eligible_template_groups_covered"]
            )
            self.assertEqual(
                coverage["available_group_count_by_pool"],
                {"A": 2, "R": 3},
            )
            self.assertEqual(
                coverage["covered_group_count_by_pool"],
                {"A": 2, "R": 2},
            )
            self.assertEqual(
                coverage["uncovered_group_ids_by_pool"],
                {"A": [], "R": ["R-unseen"]},
            )


class EffectiveBatchLossTest(unittest.TestCase):
    @staticmethod
    def _views() -> list[dict]:
        return [
            {
                "presentation_id": "p0",
                "instruction": "one",
                "command_raw": "a",
                "base_chunks": ["a"],
                "segments": [{"piece_id": 0, "start": 0, "end": 1}],
            },
            {
                "presentation_id": "p1",
                "instruction": "two",
                "command_raw": "b|c",
                "base_chunks": ["b|", "c"],
                "segments": [{"piece_id": 1, "start": 0, "end": 2}],
            },
            {
                "presentation_id": "p2",
                "instruction": "three",
                "command_raw": "d|e",
                "base_chunks": ["d|", "e"],
                "segments": [
                    {"piece_id": 2, "start": 0, "end": 1},
                    {"piece_id": 0, "start": 1, "end": 2},
                ],
            },
            {
                "presentation_id": "p3",
                "instruction": "four",
                "command_raw": "f|g|h",
                "base_chunks": ["f|", "g|", "h"],
                "segments": [
                    {"piece_id": 0, "start": 0, "end": 1},
                    {"piece_id": 1, "start": 1, "end": 3},
                ],
            },
        ]

    def _run_geometry(self, batch_size: int):
        torch.manual_seed(1937)
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        model = ProceduralMemoryModel(
            PositionAwareTinyCausalLM(len(tokenizer), hidden_size=8),
            registry,
            method="tapmem",
            initialization_seed=73,
            logit_bias_scale=1.25,
        )
        model.train()
        model.base_model.eval()
        self.assertFalse(model.base_model.training)
        dataset = BoundaryViewDataset(
            self._views(),
            tokenizer,
            registry,
            method="tapmem",
            max_length=10000,
        )
        records = [dataset[index] for index in range(len(dataset))]
        supervised_tokens = sum(
            example.target_length for example in dataset.examples
        )
        route_sites = sum(
            example.route_site_count for example in dataset.examples
        )
        ar_loss_sum = 0.0
        route_loss_sum = 0.0
        observed_supervised_tokens = 0
        observed_route_sites = 0
        for start in range(0, len(records), batch_size):
            batch = left_pad_collate(
                records[start : start + batch_size],
                pad_token_id=tokenizer.pad_token_id,
            )
            component = compute_training_loss_sums(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            micro_loss = normalized_microbatch_loss(
                component,
                effective_batch_supervised_tokens=supervised_tokens,
                effective_batch_route_sites=route_sites,
                route_loss_weight=0.3,
            )
            micro_loss.backward()
            ar_loss_sum += float(component.ar_loss_sum.detach().item())
            route_loss_sum += float(component.route_loss_sum.detach().item())
            observed_supervised_tokens += component.supervised_tokens
            observed_route_sites += component.route_sites
        self.assertEqual(observed_supervised_tokens, supervised_tokens)
        self.assertEqual(observed_route_sites, route_sites)
        ar_loss = ar_loss_sum / supervised_tokens
        route_loss = route_loss_sum / route_sites
        total_loss = ar_loss + 0.3 * route_loss
        metrics = {
            "total_loss": total_loss,
            "ar_loss": ar_loss,
            "route_loss": route_loss,
            "supervised_tokens": supervised_tokens,
            "route_sites": route_sites,
        }
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        return torch.tensor(total_loss), metrics, gradients

    def test_tapmem_4x1_2x2_and_1x4_have_equal_loss_and_gradients(self):
        results = {
            "4x1": self._run_geometry(4),
            "2x2": self._run_geometry(2),
            "1x4": self._run_geometry(1),
        }
        reference_loss, reference_metrics, reference_gradients = results["4x1"]
        for name in ("2x2", "1x4"):
            loss, metrics, gradients = results[name]
            torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-6)
            self.assertEqual(
                metrics["supervised_tokens"],
                reference_metrics["supervised_tokens"],
            )
            self.assertEqual(metrics["route_sites"], reference_metrics["route_sites"])
            self.assertAlmostEqual(
                metrics["total_loss"],
                reference_metrics["total_loss"],
                delta=1e-6,
            )
            self.assertEqual(set(gradients), set(reference_gradients))
            for parameter_name, reference_gradient in reference_gradients.items():
                torch.testing.assert_close(
                    gradients[parameter_name],
                    reference_gradient,
                    atol=2e-6,
                    rtol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()
