from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compositional_intercode_bash.io_utils import (
    PACKAGE_ROOT,
    read_json,
    sha256_directory,
    sha256_file,
    write_json,
    write_jsonl,
)
from compositional_intercode_bash.prepare_data import (
    PROVENANCE_ARTIFACTS,
    PROCEDURE_EVIDENCE_ARTIFACTS,
    SOURCE_ARTIFACTS,
    VIEW_INPUT_ARTIFACTS,
    filter_models_by_training_boundary_gate,
    load_procedure_integrity,
    load_provenance_integrity,
    load_view_integrity,
    main,
    training_boundary_gate,
)
from compositional_intercode_bash.tests.helpers import write_valid_source_artifacts
from compositional_intercode_bash.unigram import ProcedureUnigramModel
from compositional_intercode_bash.views import (
    attach_segmentation_metadata,
    build_balanced_presentations,
)


class PrepareIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=PACKAGE_ROOT)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _directory(self, name: str) -> Path:
        path = self.root / name
        path.mkdir()
        return path

    def _write_provenance(
        self,
        directory: Path,
        *,
        primary: bool = True,
    ) -> dict:
        write_valid_source_artifacts(directory)
        adjudications_path = directory / "adjudications.used.jsonl"
        groups_path = directory / "groups.json"
        provenance_path = directory / "provenance.jsonl"
        derived_path = directory / "raw_pairs.grouped_and_split.jsonl"
        derived_report_path = directory / "derived_split_report.json"
        write_jsonl(adjudications_path, [])
        write_json(groups_path, {"schema": "groups-test"})
        write_jsonl(provenance_path, [{"task_id": "task-0"}])
        write_jsonl(
            derived_path,
            [{"sample_id": "sample-0", "derived_split": "TRAIN"}],
        )
        write_json(derived_report_path, {"schema": "derived-test"})
        write_json(
            directory / "provenance_report.json",
            {
                "schema": "intercode_provenance_v3",
                "primary_ready": bool(primary),
                "primary_run": primary,
                "deny_policy": "all_candidates",
                "tasks": 200,
                "resolved": 200,
                "unresolved": 0,
                "candidate_union_group_count": 0,
                "actual_deny_group_count": 0,
                "unresolved_candidates_removed": True,
                "input_sha256": {
                    name: sha256_file(
                        directory
                        / {
                            "raw_pairs_official": "raw_pairs.official.jsonl",
                            "intercode_tasks": "intercode_tasks.jsonl",
                            "source_manifest": "source_manifest.json",
                        }[name]
                    )
                    for name in SOURCE_ARTIFACTS
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
        return load_provenance_integrity(directory, require_primary=primary)

    def _model(self) -> ProcedureUnigramModel:
        signature = ("<START>", "find")
        return ProcedureUnigramModel(
            [(signature,)],
            [1.0],
            0.25,
            mandatory=[True],
        )

    def _write_procedure(self, directory: Path, provenance_integrity: dict) -> dict:
        model = self._model()
        lexicon_path = directory / "procedure_lexicon.json"
        segmentations_path = directory / "segmentations.jsonl"
        gate_path = directory / "training_boundary_gate.json"
        evidence_paths = {
            "atoms": directory / "atoms.jsonl",
            "atomization_report": directory / "atomization_report.json",
            "candidates": directory / "candidates.jsonl",
            "capacity_selection": directory / "capacity_selection.json",
            "pruning_paths": directory / "pruning_paths.json",
        }
        grid_path = directory / "procedure_grid"
        grid_path.mkdir()
        for name, path in evidence_paths.items():
            if path.suffix == ".jsonl":
                write_jsonl(path, [{"evidence": name}])
            else:
                write_json(path, {"evidence": name})
        write_json(grid_path / "K_0001.json", model.to_dict())
        write_json(lexicon_path, model.to_dict())
        write_jsonl(
            segmentations_path,
            [
                {
                    "sample_id": "sample-0",
                    "procedure_count": model.size,
                    "lexicon_hash": model.model_hash(),
                    "procedure_model_hash": model.model_hash(),
                    "procedure_inventory_hash": model.inventory_hash(),
                }
            ],
        )
        write_json(
            gate_path,
            {
                "schema": "training_boundary_gate_v1",
                "gate_mode": "primary",
                "per_k": [],
            },
        )
        write_json(
            directory / "procedure_report.json",
            {
                "schema": "procedure_induction_v1",
                "selected_k": model.size,
                "selected_lexicon_hash": model.model_hash(),
                "procedure_model_hash": model.model_hash(),
                "procedure_inventory_hash": model.inventory_hash(),
                "input_sha256": {
                    name: provenance_integrity[name]["sha256"]
                    for name in PROVENANCE_ARTIFACTS
                },
                "primary_ready": provenance_integrity["primary_ready"],
                "primary_run": provenance_integrity["primary_run"],
                "deny_policy": provenance_integrity["deny_policy"],
                "training_boundary_gate_sha256": sha256_file(gate_path),
                "procedure_lexicon_sha256": sha256_file(lexicon_path),
                "segmentations_sha256": sha256_file(segmentations_path),
                "evidence_sha256": {
                    **{
                        name: sha256_file(path)
                        for name, path in evidence_paths.items()
                    },
                    "procedure_grid": sha256_directory(grid_path),
                },
                "procedure_grid_files": ["K_0001.json"],
            },
        )
        return load_procedure_integrity(
            directory,
            provenance_integrity,
            require_primary=True,
        )

    def _write_views(self, directory: Path, procedure_integrity: dict) -> None:
        presentations_path = directory / "training_presentations.jsonl"
        presentation_report_path = directory / "presentation_report.json"
        views_path = directory / "views.map.7.jsonl"
        coverage_path = views_path.with_suffix(".coverage.json")
        write_jsonl(presentations_path, [{"presentation_id": "p-0"}])
        write_json(presentation_report_path, {"schema": "presentations-test"})
        write_jsonl(
            views_path,
            [
                {
                    "view_id": "v-0",
                    "setting": "map",
                    "data_seed": 7,
                    "lexicon_hash": procedure_integrity[
                        "procedure_model_hash"
                    ],
                    "procedure_model_hash": procedure_integrity[
                        "procedure_model_hash"
                    ],
                    "procedure_inventory_hash": procedure_integrity[
                        "procedure_inventory_hash"
                    ],
                    "procedure_count": procedure_integrity[
                        "procedure_count"
                    ],
                }
            ],
        )
        coverage = {
            "schema": "procedure_positive_coverage_v1",
            "procedure_count": procedure_integrity["procedure_count"],
            "minimum_procedure_exposures": 1,
            "procedure_positive_counts": [1],
            "uncovered_procedure_ids": [],
            "native_reserved_boundary": 247,
            "native_procedure_positive_coverage": 1,
            "native_procedure_count": 1,
            "added_procedure_positive_coverage": 0,
            "added_procedure_count": 0,
        }
        write_json(coverage_path, coverage)
        write_json(
            directory / "views.map.7.report.json",
            {
                "schema": "boundary_views_v1",
                "setting": "map",
                "data_seed": 7,
                "primary_requested": True,
                "primary_ready": procedure_integrity["primary_ready"],
                "primary_run": procedure_integrity["primary_run"],
                "deny_policy": procedure_integrity["deny_policy"],
                "lexicon_hash": procedure_integrity["procedure_model_hash"],
                "procedure_model_hash": procedure_integrity[
                    "procedure_model_hash"
                ],
                "procedure_inventory_hash": procedure_integrity[
                    "procedure_inventory_hash"
                ],
                "procedure_count": procedure_integrity["procedure_count"],
                "presentations": 1,
                "presentation_epochs": 1,
                "minimum_requested_epochs": 1,
                **{
                    field: value
                    for field, value in coverage.items()
                    if field != "schema"
                },
                "input_sha256": {
                    name: procedure_integrity[name]["sha256"]
                    for name in VIEW_INPUT_ARTIFACTS
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

    def test_primary_procedure_and_view_cli_reject_exploratory_provenance(self):
        directory = self._directory("exploratory")
        self._write_provenance(directory, primary=False)
        for stage in ("procedures", "views"):
            with self.subTest(stage=stage), self.assertRaisesRegex(
                RuntimeError,
                "cannot be promoted",
            ):
                main(
                    [
                        stage,
                        "--artifact-dir",
                        str(directory),
                        "--primary",
                    ]
                )

    def test_procedure_chain_rejects_changed_provenance_and_derived_split(self):
        for artifact_name in ("provenance", "derived_split"):
            with self.subTest(artifact_name=artifact_name):
                directory = self._directory(f"changed-{artifact_name}")
                provenance_integrity = self._write_provenance(directory)
                self._write_procedure(directory, provenance_integrity)
                if artifact_name == "provenance":
                    changed_path = directory / "provenance.jsonl"
                    report_field = "provenance_sha256"
                else:
                    changed_path = (
                        directory / "raw_pairs.grouped_and_split.jsonl"
                    )
                    report_field = "derived_split_sha256"
                write_jsonl(changed_path, [{"changed": artifact_name}])
                provenance_report = read_json(
                    directory / "provenance_report.json"
                )
                provenance_report[report_field] = sha256_file(changed_path)
                write_json(
                    directory / "provenance_report.json",
                    provenance_report,
                )
                current_provenance = load_provenance_integrity(
                    directory,
                    require_primary=True,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "different provenance",
                ):
                    load_procedure_integrity(
                        directory,
                        current_provenance,
                        require_primary=True,
                    )

    def test_procedure_chain_rejects_each_bound_output_and_model_mismatch(self):
        path_by_name = {
            "procedure_lexicon": "procedure_lexicon.json",
            "segmentations": "segmentations.jsonl",
            "training_boundary_gate": "training_boundary_gate.json",
        }
        for artifact_name, relative_path in path_by_name.items():
            with self.subTest(artifact_name=artifact_name):
                directory = self._directory(f"output-{artifact_name}")
                provenance_integrity = self._write_provenance(directory)
                self._write_procedure(directory, provenance_integrity)
                path = directory / relative_path
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    load_procedure_integrity(
                        directory,
                        provenance_integrity,
                        require_primary=True,
                    )

        directory = self._directory("model-field")
        provenance_integrity = self._write_provenance(directory)
        self._write_procedure(directory, provenance_integrity)
        report_path = directory / "procedure_report.json"
        report = read_json(report_path)
        report["procedure_model_hash"] = "0" * 64
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "procedure_model_hash"):
            load_procedure_integrity(
                directory,
                provenance_integrity,
                require_primary=True,
            )

    def test_view_chain_binds_every_upstream_and_output(self):
        directory = self._directory("view-chain")
        provenance_integrity = self._write_provenance(directory)
        procedure_integrity = self._write_procedure(
            directory,
            provenance_integrity,
        )
        self._write_views(directory, procedure_integrity)
        load_view_integrity(
            directory,
            procedure_integrity,
            setting="map",
            data_seed=7,
            require_primary=True,
        )

        for artifact_name in VIEW_INPUT_ARTIFACTS:
            with self.subTest(upstream=artifact_name):
                changed_integrity = copy.deepcopy(procedure_integrity)
                changed_integrity[artifact_name]["sha256"] = "f" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "different provenance/procedure",
                ):
                    load_view_integrity(
                        directory,
                        changed_integrity,
                        setting="map",
                        data_seed=7,
                        require_primary=True,
                    )

        output_by_name = {
            "views": "views.map.7.jsonl",
            "presentations": "training_presentations.jsonl",
            "presentation_report": "presentation_report.json",
        }
        for artifact_name, relative_path in output_by_name.items():
            with self.subTest(output=artifact_name):
                separate = self._directory(f"view-output-{artifact_name}")
                current_provenance = self._write_provenance(separate)
                current_procedure = self._write_procedure(
                    separate,
                    current_provenance,
                )
                self._write_views(separate, current_procedure)
                path = separate / relative_path
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "differs"):
                    load_view_integrity(
                        separate,
                        current_procedure,
                        setting="map",
                        data_seed=7,
                        require_primary=True,
                    )

    def test_physical_tampering_of_each_view_upstream_is_rejected(self):
        path_by_name = {
            "raw_pairs_official": "raw_pairs.official.jsonl",
            "intercode_tasks": "intercode_tasks.jsonl",
            "source_manifest": "source_manifest.json",
            "adjudications": "adjudications.used.jsonl",
            "groups": "groups.json",
            "provenance": "provenance.jsonl",
            "provenance_report": "provenance_report.json",
            "derived_split": "raw_pairs.grouped_and_split.jsonl",
            "derived_split_report": "derived_split_report.json",
            "procedure_lexicon": "procedure_lexicon.json",
            "segmentations": "segmentations.jsonl",
            "procedure_report": "procedure_report.json",
            "training_boundary_gate": "training_boundary_gate.json",
            "atoms": "atoms.jsonl",
            "atomization_report": "atomization_report.json",
            "candidates": "candidates.jsonl",
            "capacity_selection": "capacity_selection.json",
            "pruning_paths": "pruning_paths.json",
            "procedure_grid": "procedure_grid",
        }
        self.assertEqual(set(path_by_name), set(VIEW_INPUT_ARTIFACTS))
        for artifact_name, relative_path in path_by_name.items():
            with self.subTest(artifact_name=artifact_name):
                directory = self._directory(f"physical-{artifact_name}")
                provenance_integrity = self._write_provenance(directory)
                procedure_integrity = self._write_procedure(
                    directory,
                    provenance_integrity,
                )
                self._write_views(directory, procedure_integrity)
                path = directory / relative_path
                if path.is_dir():
                    member = next(path.iterdir())
                    member.write_bytes(member.read_bytes() + b" ")
                else:
                    path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(
                    (ValueError, RuntimeError),
                ):
                    current_provenance = load_provenance_integrity(
                        directory,
                        require_primary=True,
                    )
                    current_procedure = load_procedure_integrity(
                        directory,
                        current_provenance,
                        require_primary=True,
                    )
                    load_view_integrity(
                        directory,
                        current_procedure,
                        setting="map",
                        data_seed=7,
                        require_primary=True,
                    )

    def test_view_chain_rejects_changed_procedure_report_and_model_hash(self):
        directory = self._directory("view-procedure-report")
        provenance_integrity = self._write_provenance(directory)
        procedure_integrity = self._write_procedure(
            directory,
            provenance_integrity,
        )
        self._write_views(directory, procedure_integrity)

        procedure_report_path = directory / "procedure_report.json"
        procedure_report = read_json(procedure_report_path)
        procedure_report["audit_note"] = "changed after views were built"
        write_json(procedure_report_path, procedure_report)
        current_procedure = load_procedure_integrity(
            directory,
            provenance_integrity,
            require_primary=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "different provenance/procedure",
        ):
            load_view_integrity(
                directory,
                current_procedure,
                setting="map",
                data_seed=7,
                require_primary=True,
            )

        separate = self._directory("view-model-field")
        current_provenance = self._write_provenance(separate)
        current_procedure = self._write_procedure(
            separate,
            current_provenance,
        )
        self._write_views(separate, current_procedure)
        view_report_path = separate / "views.map.7.report.json"
        view_report = read_json(view_report_path)
        view_report["lexicon_hash"] = "0" * 64
        write_json(view_report_path, view_report)
        with self.assertRaisesRegex(ValueError, "lexicon_hash"):
            load_view_integrity(
                separate,
                current_procedure,
                setting="map",
                data_seed=7,
                require_primary=True,
            )

    def test_training_boundary_gate_uses_same_primary_a_r_counts(self):
        a = ("<START>", "a")
        b = ("PIPE", "b")
        c = ("PIPE", "c")
        model = ProcedureUnigramModel(
            [(a,), (b,), (c,), (a, b), (b, c)],
            [0.2] * 5,
            0.5,
            mandatory=[True, True, True, False, False],
        )
        records = []
        for pool in ("A", "R"):
            for index in range(100):
                records.append(
                    {
                        "sample_id": f"{pool}-{index}",
                        "template_group_id": f"{pool}-{index}",
                        "canonical_signatures": [
                            list(value)
                            for value in ((a, b, c) if pool == "A" else (a, c))
                        ],
                    }
                )
        gate = training_boundary_gate(records, model)
        segmented = attach_segmentation_metadata(records, model)
        _presentations, view_gate = build_balanced_presentations(
            segmented,
            model,
            epochs=1,
            minimum_procedure_exposures=1,
            primary=True,
        )
        self.assertEqual(gate["A_groups"], 100)
        self.assertEqual(gate["R_groups"], 100)
        self.assertTrue(gate["primary_gate_passed"])
        for field in (
            "A_groups",
            "R_groups",
            "L_ge_3_groups",
            "required_A_groups",
            "A_gate_passed",
            "R_gate_passed",
        ):
            self.assertEqual(gate[field], view_gate[field])

    def test_only_gate_eligible_k_values_reach_one_se_pool(self):
        winners = {
            size: SimpleNamespace(model=SimpleNamespace(size=size))
            for size in (100, 200, 300)
        }

        def fake_gate(_records, model):
            passed = model.size != 200
            return {
                "K": model.size,
                "primary_gate_passed": passed,
                "exploratory_gate_passed": True,
            }

        with mock.patch(
            "compositional_intercode_bash.prepare_data.training_boundary_gate",
            side_effect=fake_gate,
        ):
            eligible, report = filter_models_by_training_boundary_gate(
                winners,
                [],
                require_primary=True,
            )
        self.assertEqual(set(eligible), {100, 300})
        self.assertEqual(report["eligible_k_count"], 2)
        self.assertEqual(
            [
                item["K"]
                for item in report["per_k"]
                if item["eligible_for_one_se"]
            ],
            [100, 300],
        )


if __name__ == "__main__":
    unittest.main()
