#!/usr/bin/env python3
"""Direct unit checks for the rebuttal error-type and transition analyzer."""

import json
import tempfile
import unittest
from pathlib import Path

from analyze_error_type_transitions import (
    CATEGORY_KEYS,
    analyze_manifest,
    analyze_records,
    classify_error_type,
    label_record,
)


def record(index, expected_tools, predicted_tools, call_exact=True):
    return {
        "index": index,
        "user_input": f"sample-{index}",
        "expected_tools": expected_tools,
        "expected_calls": [json.dumps({"value": index})],
        "predicted_tools": predicted_tools,
        "predicted_calls": [json.dumps({"value": index})],
        "call_exact": call_exact,
    }


class ErrorTypeAnalysisTest(unittest.TestCase):
    def test_all_six_categories_are_mutually_exclusive(self):
        records = [
            record(0, ["a", "b"], ["a", "b"], call_exact=True),
            record(1, ["a", "b"], ["a", "b"], call_exact=False),
            record(2, ["a", "b"], ["a"], call_exact=False),
            record(3, ["a", "b"], ["b", "a"], call_exact=False),
            record(4, ["a", "b"], ["x", "b"], call_exact=False),
            record(5, ["a", "b"], ["a", "x"], call_exact=False),
        ]

        labeled, metrics = analyze_records(records, recompute_exact=False)

        self.assertEqual([item["error_type"] for item in labeled], list(CATEGORY_KEYS))
        self.assertEqual(metrics["samples"], 6)
        self.assertEqual(sum(metrics["category_counts"].values()), 6)
        self.assertTrue(all(count == 1 for count in metrics["category_counts"].values()))

    def test_later_step_rates_use_gold_positions_and_first_correct_filter(self):
        records = [
            record(0, ["a", "b"], ["a", "b"]),
            record(1, ["a", "b"], ["a", "b"], call_exact=False),
            record(2, ["a", "b"], ["a"], call_exact=False),
            record(3, ["a", "b"], ["b", "a"], call_exact=False),
            record(4, ["a", "b"], ["x", "b"], call_exact=False),
            record(5, ["a", "b"], ["a", "x"], call_exact=False),
        ]

        _labeled, metrics = analyze_records(records, recompute_exact=False)

        self.assertEqual(metrics["later_step_mismatch_all_numerator"], 3)
        self.assertEqual(metrics["later_step_mismatch_all_denominator"], 6)
        self.assertEqual(metrics["later_step_mismatch_all"], 0.5)
        self.assertEqual(metrics["first_correct_samples"], 4)
        self.assertEqual(metrics["later_step_mismatch_first_correct_numerator"], 2)
        self.assertEqual(metrics["later_step_mismatch_first_correct_denominator"], 4)
        self.assertEqual(metrics["later_step_mismatch_first_correct"], 0.5)

    def test_extra_predictions_do_not_expand_later_step_denominator(self):
        records = [record(0, ["a", "b"], ["a", "b", "extra"], call_exact=False)]

        _labeled, metrics = analyze_records(records, recompute_exact=False)

        self.assertEqual(metrics["category_counts"]["length_error"], 1)
        self.assertEqual(metrics["later_step_mismatch_all_numerator"], 0)
        self.assertEqual(metrics["later_step_mismatch_all_denominator"], 1)

    def test_order_only_uses_multiset_with_duplicate_counts(self):
        self.assertEqual(
            classify_error_type(["a", "a", "b"], ["a", "b", "a"], call_exact=False),
            "order_only_error",
        )
        self.assertEqual(
            classify_error_type(["a", "a", "b"], ["a", "b", "b"], call_exact=False),
            "later_only_routing_error",
        )

    def test_call_exact_is_recomputed_with_existing_evaluator(self):
        item = record(0, ["a"], ["a"], call_exact=False)
        item["expected_calls"] = ['{"x": 1, "y": 2}']
        item["predicted_calls"] = ['{ "y": 2, "x": 1 }']

        labeled = label_record(item, recompute_exact=True)

        self.assertTrue(labeled["call_exact"])
        self.assertEqual(labeled["error_type"], "correct")

    def test_manifest_analysis_checks_alignment_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = []
            for method in ("tokmem", "tapmem", "eoc_only"):
                prediction_path = root / f"{method}.jsonl"
                with open(prediction_path, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(record(0, ["a", "b"], ["a", "b"])) + "\n")
                entries.append(
                    {
                        "model": "llama1b",
                        "method": method,
                        "trial": 1,
                        "run_name": method,
                        "checkpoint_path": f"/{method}.pt",
                        "prediction_path": str(prediction_path),
                    }
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "output_dir": str(root / "output"),
                        "data_path": "/test.json",
                        "limit": 1,
                        "max_new_tokens": 512,
                        "entries": entries,
                    }
                ),
                encoding="utf-8",
            )

            summary_json, per_trial, summary_md = analyze_manifest(manifest_path)

            self.assertTrue(summary_json.exists())
            self.assertTrue(per_trial.exists())
            self.assertTrue(summary_md.exists())
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertIn("tokmem_vs_tapmem", summary["comparisons"])
            self.assertIn("tokmem_vs_eoc_only", summary["comparisons"])


if __name__ == "__main__":
    unittest.main()
