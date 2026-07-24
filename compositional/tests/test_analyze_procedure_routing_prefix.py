#!/usr/bin/env python3
"""Direct unit checks for procedure-routing correct-prefix analysis."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = COMPOSITIONAL_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_procedure_routing_prefix import (  # noqa: E402
    aggregate_trials,
    analyze_entries,
    analyze_records,
    build_provenance,
    correct_prefix_length,
    label_prefix_record,
    resolve_run_dir,
    select_generation_entries,
    update_run_index,
    validate_prediction_artifact,
    write_manifest,
)


def record(index, expected_tools, predicted_tools, tool_f1=0.0):
    return {
        "index": index,
        "user_input": f"sample-{index}",
        "expected_tools": expected_tools,
        "expected_calls": [json.dumps({"index": index}) for _ in expected_tools],
        "predicted_tools": predicted_tools,
        "predicted_calls": [json.dumps({"index": index}) for _ in predicted_tools],
        "tool_f1": tool_f1,
    }


class ProcedureRoutingPrefixTest(unittest.TestCase):
    def test_correct_prefix_stops_at_first_wrong_or_missing_tool(self):
        self.assertEqual(correct_prefix_length(["a", "b"], ["x", "b"]), 0)
        self.assertEqual(correct_prefix_length(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(correct_prefix_length(["a", "b", "c"], ["a", "b"]), 2)
        self.assertEqual(correct_prefix_length(["a", "b"], ["a", "b"]), 2)

    def test_extra_prediction_preserves_full_gold_prefix_but_is_labeled(self):
        labeled = label_prefix_record(
            record(0, ["a", "b"], ["a", "b", "extra"])
        )

        self.assertEqual(labeled["correct_prefix_length"], 2)
        self.assertTrue(labeled["all_gold_routing_correct"])
        self.assertFalse(labeled["routing_sequence_exact"])
        self.assertTrue(labeled["over_generated_after_gold"])
        self.assertEqual(labeled["first_routing_error_position"], 3)
        self.assertEqual(labeled["first_routing_error_kind"], "extra_prediction")

    def test_analysis_stratifies_by_gold_length_and_fills_zero_counts(self):
        records = [
            record(0, ["a", "b"], ["x", "b"]),
            record(1, ["a", "b"], ["a", "x"]),
            record(2, ["a", "b"], ["a", "b"]),
            record(3, ["a", "b", "c"], ["a", "b", "x"]),
            record(4, ["a", "b", "c"], ["a", "b", "c"]),
        ]

        _labeled, metrics = analyze_records(records)

        two = metrics["by_gold_count"]["2"]
        self.assertEqual(two["samples"], 3)
        self.assertEqual(two["prefix_counts"], {"0": 1, "1": 1, "2": 1})
        self.assertEqual(two["average_prefix_length"], 1.0)

        three = metrics["by_gold_count"]["3"]
        self.assertEqual(three["samples"], 2)
        self.assertEqual(
            three["prefix_counts"],
            {"0": 0, "1": 0, "2": 1, "3": 1},
        )
        self.assertEqual(sum(three["prefix_counts"].values()), 2)

    def test_trial_aggregation_uses_mean_counts_instead_of_pooling(self):
        trial_records = [
            [
                record(0, ["a", "b"], ["a", "b"]),
                record(1, ["a", "b"], ["a", "x"]),
            ],
            [
                record(0, ["a", "b"], ["a", "b"]),
                record(1, ["a", "b"], ["a", "b"]),
            ],
        ]
        trials = []
        for trial, records in enumerate(trial_records, start=1):
            _labeled, metrics = analyze_records(records)
            trials.append({"trial": trial, "metrics": metrics})

        aggregate = aggregate_trials(trials)
        two = aggregate["by_gold_count"]["2"]

        self.assertEqual(aggregate["trials"], 2)
        self.assertEqual(two["samples_per_trial"], 2)
        self.assertEqual(two["prefix_counts"]["1"]["mean"], 0.5)
        self.assertEqual(two["prefix_counts"]["2"]["mean"], 1.5)
        self.assertAlmostEqual(two["prefix_rates"]["2"]["mean"], 0.75)

    def test_entry_analysis_writes_json_jsonl_and_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = []
            for method in ("tokmem", "tapmem"):
                prediction_path = root / f"{method}.jsonl"
                records = [
                    record(0, ["a", "b"], ["a", "b"], tool_f1=1.0),
                    record(1, ["a", "b", "c"], ["a", "x"], tool_f1=0.5),
                ]
                with open(prediction_path, "w", encoding="utf-8") as handle:
                    for item in records:
                        handle.write(json.dumps(item) + "\n")
                entries.append(
                    {
                        "model": "llama1b",
                        "method": method,
                        "trial": 1,
                        "run_name": method,
                        "checkpoint_path": f"/{method}.pt",
                        "prediction_path": str(prediction_path),
                        "archived_eval_metrics": {"avg_tool_f1_score": 0.75},
                    }
                )

            summary_json, per_trial_path, summary_md = analyze_entries(
                entries,
                root,
            )

            self.assertTrue(summary_json.exists())
            self.assertTrue(per_trial_path.exists())
            self.assertTrue(summary_md.exists())
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertIn("comparison", summary)
            self.assertEqual(summary["metric_audit_failures"], [])
            markdown = summary_md.read_text(encoding="utf-8")
            self.assertIn("Gold procedure count = 2", markdown)
            self.assertIn("count=0", markdown)
            self.assertIn("Overall averages", markdown)

    def test_provenance_changes_run_directory_without_expanding_all_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "test.json"
            data = [{"tools": ["a", "b"]}]
            data_path.write_text(json.dumps(data), encoding="utf-8")
            entries = [
                {
                    "method": "tokmem",
                    "trial": 1,
                    "run_name": "tokmem-trial1",
                    "base_model_name": "/models/Llama-3.2-1B-Instruct",
                    "run_config_sha256": "config-a",
                    "checkpoint_archive_fingerprint": {
                        "kind": "test",
                        "digest": "checkpoint-a",
                    },
                }
            ]
            args = SimpleNamespace(
                limit=None,
                device="cuda",
                dtype="bfloat16",
                max_new_tokens=512,
                eval_batch_size=16,
                metric_tolerance=0.005,
                strict_metric_audit=False,
                run_label=None,
            )

            first = build_provenance(args, entries, data_path, data)
            first_dir = resolve_run_dir(root, args, ["tokmem"], first)
            args.max_new_tokens = 256
            second = build_provenance(args, entries, data_path, data)
            second_dir = resolve_run_dir(root, args, ["tokmem"], second)

            self.assertNotEqual(first["id"], second["id"])
            self.assertNotEqual(first_dir, second_dir)
            self.assertRegex(first_dir.name, r"^figure4_full_tokmem_p[0-9a-f]{10}$")
            self.assertNotIn("bfloat16", first_dir.name)
            self.assertNotIn("bs16", first_dir.name)
            self.assertNotIn("gen512", first_dir.name)

    def test_generation_tasks_select_worker_shard_without_changing_full_entries(self):
        entries = [
            {"method": "tokmem", "trial": 1},
            {"method": "tokmem", "trial": 2},
            {"method": "tapmem", "trial": 1},
        ]

        selected = select_generation_entries(
            "tokmem:2,tapmem:1",
            entries,
        )

        self.assertEqual(
            [(entry["method"], entry["trial"]) for entry in selected],
            [("tokmem", 2), ("tapmem", 1)],
        )
        self.assertEqual(len(entries), 3)
        with self.assertRaisesRegex(ValueError, "Unknown or unselected"):
            select_generation_entries("tapmem:2", entries)

    def test_prediction_artifact_rejects_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prediction.jsonl"
            entry = {
                "method": "tokmem",
                "trial": 1,
                "run_name": "tokmem-trial1",
                "checkpoint_archive_fingerprint": {"digest": "checkpoint-a"},
            }
            provenance = {
                "id": "provenance-a",
                "payload": {"data": {"sha256": "data-a"}},
            }
            item = record(0, ["a", "b"], ["a", "b"])
            item.update(
                {
                    "method": "tokmem",
                    "trial": 1,
                    "run_name": "tokmem-trial1",
                    "provenance_id": "wrong-provenance",
                    "data_sha256": "data-a",
                    "checkpoint_fingerprint": "checkpoint-a",
                }
            )
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_prediction_artifact(path, 1, entry, provenance)

    def test_existing_manifest_rejects_different_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "test.json"
            data = [{"tools": ["a", "b"]}]
            data_path.write_text(json.dumps(data), encoding="utf-8")
            args = SimpleNamespace(
                limit=None,
                dtype="bfloat16",
                max_new_tokens=512,
                eval_batch_size=16,
                metric_tolerance=0.005,
            )
            entries = [
                {
                    "method": "tokmem",
                    "trial": 1,
                    "checkpoint_archive_fingerprint": {
                        "kind": "test",
                        "digest": "checkpoint-a",
                    },
                }
            ]
            first = {"id": "first", "payload": {"value": 1}}
            second = {"id": "second", "payload": {"value": 2}}

            _manifest_path, manifest = write_manifest(
                args,
                entries,
                data_path,
                root,
                data,
                first,
            )
            index_path = update_run_index(
                root / "output-root",
                root,
                manifest,
                status="manifested",
            )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["runs"][0]["provenance_id"], "first")
            self.assertEqual(index["runs"][0]["status"], "manifested")
            with self.assertRaisesRegex(ValueError, "different provenance"):
                write_manifest(args, entries, data_path, root, data, second)


if __name__ == "__main__":
    unittest.main()
