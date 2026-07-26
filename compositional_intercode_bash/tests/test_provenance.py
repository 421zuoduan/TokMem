from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from compositional_intercode_bash import provenance
from compositional_intercode_bash.templates import command_template_keys


def _raw_record(index: int, instruction: str, command: str) -> dict:
    return {
        "sample_id": f"sample-{index:04d}",
        "source_line": index + 1,
        "instruction_raw": instruction,
        "command_raw": command,
        "official_split": "TRAIN",
    }


class ProvenanceGroupingTest(unittest.TestCase):
    def test_empty_bashlint_template_is_not_a_grouping_key(self):
        with mock.patch(
            "compositional_intercode_bash.templates.bashlex_fallback_fingerprint",
            return_value=None,
        ):
            self.assertEqual(
                command_template_keys("echo x", bashlint_template=lambda _command: ""),
                (),
            )
            self.assertEqual(
                command_template_keys(
                    "echo x",
                    bashlint_template=lambda _command: "  \t ",
                ),
                (),
            )

    def _build_with_signatures(self, records, signature_by_command):
        tokenizer = lambda value: ([value.lower()], [])
        with (
            mock.patch.object(
                provenance,
                "load_official_basic_tokenizer",
                return_value=tokenizer,
            ),
            mock.patch.object(
                provenance,
                "load_bashlint_template_function",
                return_value=object(),
            ),
            mock.patch.object(
                provenance,
                "command_template_keys",
                side_effect=lambda command, bashlint_template=None: signature_by_command[
                    command
                ],
            ),
            mock.patch.object(
                provenance,
                "bashlex_utility_operator_sequence",
                return_value=("UTILITY:test",),
            ),
        ):
            return provenance.build_source_groups(records, "/unused")

    def test_template_group_requires_the_full_conjunction(self):
        records = [
            _raw_record(0, "instruction-a", "a"),
            _raw_record(1, "instruction-b", "b"),
            _raw_record(2, "instruction-c", "c"),
            _raw_record(3, "instruction-d", "d"),
        ]
        signatures = {
            "a": ("BASHLINT:x", "BASHLEX:a"),
            "b": ("BASHLINT:x", "BASHLEX:b"),
            "c": ("BASHLINT:y", "BASHLEX:b"),
            "d": ("BASHLINT:x", "BASHLEX:a"),
        }
        grouped, report = self._build_with_signatures(records, signatures)
        ids = {record["command_raw"]: record["template_group_id"] for record in grouped}
        self.assertEqual(ids["a"], ids["d"])
        self.assertNotEqual(ids["a"], ids["b"])
        self.assertNotEqual(ids["b"], ids["c"])
        self.assertEqual(report["schema"], "nl2bash_groups_v2")

        shuffled, _ = self._build_with_signatures(list(reversed(records)), signatures)
        source_by_sample = {
            record["sample_id"]: record["source_group_id"] for record in grouped
        }
        self.assertEqual(
            source_by_sample,
            {
                record["sample_id"]: record["source_group_id"]
                for record in shuffled
            },
        )

    def test_template_hub_is_suppressed_above_64_not_at_64(self):
        signature = ("BASHLINT:shared", "BASHLEX:shared")
        records_64 = [
            _raw_record(index, f"instruction-{index}", f"command-{index}")
            for index in range(64)
        ]
        signatures_64 = {
            record["command_raw"]: signature for record in records_64
        }
        _grouped, report = self._build_with_signatures(records_64, signatures_64)
        self.assertEqual(report["template_graph"]["largest_component"], 64)
        self.assertEqual(report["suppressed_signature_count"], 0)

        records_65 = [
            _raw_record(index, f"instruction-{index}", f"command-{index}")
            for index in range(65)
        ]
        signatures_65 = {
            record["command_raw"]: signature for record in records_65
        }
        grouped, report = self._build_with_signatures(records_65, signatures_65)
        self.assertEqual(report["template_graph"]["largest_component"], 1)
        self.assertEqual(report["source_graph"]["largest_component"], 1)
        self.assertEqual(report["suppressed_signature_count"], 1)
        self.assertTrue(
            all(record["template_signature_suppressed"] for record in grouped)
        )

    def test_sequence_edit_one_rejects_atomic_and_requires_text_overlap(self):
        self.assertFalse(
            provenance._is_sequence_exact_candidate(
                ("UTILITY:find",),
                ("UTILITY:find",),
                1.0,
            )
        )
        self.assertFalse(
            provenance._is_sequence_edit_one_candidate(
                ("UTILITY:find",),
                ("UTILITY:sort",),
                1.0,
            )
        )
        task = ("UTILITY:find", "PIPE:|", "UTILITY:sort")
        changed = ("UTILITY:find", "PIPE:|", "UTILITY:uniq")
        self.assertTrue(
            provenance._is_sequence_edit_one_candidate(task, changed, 0.25)
        )
        self.assertFalse(
            provenance._is_sequence_edit_one_candidate(task, changed, 0.249)
        )
        self.assertFalse(
            provenance._is_sequence_edit_one_candidate(task, task, 1.0)
        )

    def test_primary_denies_every_candidate_group(self):
        raw = [
            {
                **_raw_record(0, "candidate one", "command one"),
                "source_group_id": "g1",
                "normalized_instruction": "candidate one",
            },
            {
                **_raw_record(1, "candidate two", "command two"),
                "source_group_id": "g2",
                "normalized_instruction": "candidate two",
            },
            {
                **_raw_record(2, "candidate three", "command three"),
                "source_group_id": "g3",
                "normalized_instruction": "candidate three",
            },
        ]
        tasks = [{"task_id": "fs1:000", "query": "query", "gold": "gold"}]
        channels = {
            "exact_pair": set(),
            "exact_instruction": set(),
            "exact_command": set(),
            "template": set(),
            "sequence_exact": {0, 1, 2},
            "sequence_edit_one": set(),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            adjudications = Path(temporary_dir) / "adjudications.jsonl"
            adjudications.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "fs1:000",
                                "annotator": "A",
                                "accepted_source_group_ids": ["g1"],
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "fs1:000",
                                "annotator": "B",
                                "accepted_source_group_ids": ["g1"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    provenance,
                    "load_official_basic_tokenizer",
                    return_value=object(),
                ),
                mock.patch.object(
                    provenance,
                    "load_bashlint_template_function",
                    return_value=object(),
                ),
                mock.patch.object(
                    provenance,
                    "_candidate_groups",
                    return_value=(channels, ("UTILITY:x",), "query"),
                ),
            ):
                _rows, primary_deny, primary_report = (
                    provenance.retrieve_intercode_provenance(
                        tasks,
                        raw,
                        "/unused",
                        adjudications_path=adjudications,
                        primary=True,
                        required_task_ids=frozenset({"fs1:000"}),
                    )
                )
                _rows, exploratory_deny, exploratory_report = (
                    provenance.retrieve_intercode_provenance(
                        tasks,
                        raw,
                        "/unused",
                        adjudications_path=adjudications,
                        primary=False,
                    )
                )
        self.assertEqual(primary_deny, {"g1", "g2", "g3"})
        self.assertEqual(primary_report["deny_policy"], "all_candidates")
        self.assertTrue(primary_report["primary_ready"])
        self.assertTrue(primary_report["unresolved_candidates_removed"])
        self.assertEqual(exploratory_deny, {"g1", "g2", "g3"})
        self.assertEqual(exploratory_report["deny_policy"], "all_candidates")

    def test_unresolved_candidates_are_removed_without_adjudication(self):
        raw = [
            {
                **_raw_record(index, f"candidate {index}", f"command {index}"),
                "source_group_id": f"g{index + 1}",
                "normalized_instruction": f"candidate {index}",
            }
            for index in range(2)
        ]
        tasks = [{"task_id": "fs1:000", "query": "query", "gold": "gold"}]
        channels = {
            "exact_pair": set(),
            "exact_instruction": set(),
            "exact_command": set(),
            "template": set(),
            "sequence_exact": {0, 1},
            "sequence_edit_one": set(),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            adjudications = Path(temporary_dir) / "adjudications.jsonl"

            def write_labels(include_adjudicator: bool) -> None:
                labels = [
                    {
                        "task_id": "fs1:000",
                        "annotator": "A",
                        "accepted_source_group_ids": ["g1", "g2"],
                    },
                    {
                        "task_id": "fs1:000",
                        "annotator": "B",
                        "accepted_source_group_ids": ["g1"],
                    },
                ]
                if include_adjudicator:
                    labels.append(
                        {
                            "task_id": "fs1:000",
                            "annotator": "ADJUDICATOR",
                            "accepted_source_group_ids": ["g1"],
                        }
                    )
                adjudications.write_text(
                    "".join(json.dumps(label) + "\n" for label in labels),
                    encoding="utf-8",
                )

            with (
                mock.patch.object(
                    provenance,
                    "load_official_basic_tokenizer",
                    return_value=object(),
                ),
                mock.patch.object(
                    provenance,
                    "load_bashlint_template_function",
                    return_value=object(),
                ),
                mock.patch.object(
                    provenance,
                    "_candidate_groups",
                    return_value=(channels, ("UTILITY:x",), "query"),
                ),
            ):
                write_labels(include_adjudicator=False)
                unresolved_rows, unresolved_deny, unresolved_report = (
                    provenance.retrieve_intercode_provenance(
                        tasks,
                        raw,
                        "/unused",
                        adjudications_path=adjudications,
                        primary=True,
                        required_task_ids=frozenset({"fs1:000"}),
                    )
                )
                write_labels(include_adjudicator=True)
                rows, deny, report = provenance.retrieve_intercode_provenance(
                    tasks,
                    raw,
                    "/unused",
                    adjudications_path=adjudications,
                    primary=True,
                    required_task_ids=frozenset({"fs1:000"}),
                )
                adjudications.write_text(
                    json.dumps(
                        {
                            "task_id": "fs1:000",
                            "annotator": "ADJUDICATOR",
                            "accepted_source_group_ids": ["g1"],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "only after two independent",
                ):
                    provenance.retrieve_intercode_provenance(
                        tasks,
                        raw,
                        "/unused",
                        adjudications_path=adjudications,
                        primary=True,
                        required_task_ids=frozenset({"fs1:000"}),
                    )
        self.assertEqual(unresolved_deny, {"g1", "g2"})
        self.assertEqual(unresolved_report["unresolved"], 1)
        self.assertTrue(unresolved_report["primary_ready"])
        self.assertEqual(
            unresolved_rows[0]["denied_source_group_ids"],
            ["g1", "g2"],
        )
        self.assertEqual(deny, {"g1", "g2"})
        self.assertEqual(report["unresolved"], 0)
        self.assertEqual(rows[0]["resolution_mode"], "explicit_adjudication")


if __name__ == "__main__":
    unittest.main()
