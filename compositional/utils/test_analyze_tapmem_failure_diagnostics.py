#!/usr/bin/env python3
"""Focused tests for the TapMem failure taxonomy."""

import sys
import unittest
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_tapmem_failure_diagnostics import (  # noqa: E402
    ambiguity_key,
    classify_first_procedure_failure,
    summarize_oracle,
)


def base_record(predicted_tools, raw_ids, tool_positions, eoc_positions):
    return {
        "index": 7,
        "expected_tools": ["tool_a", "tool_b"],
        "predicted_tools": predicted_tools,
        "raw_generated_token_ids": raw_ids,
        "tool_positions": tool_positions,
        "eoc_positions": eoc_positions,
        "eoc_token_id": 99,
    }


class FailureClassificationTest(unittest.TestCase):
    def test_premature_stop_after_valid_eoc(self):
        record = base_record(["tool_a"], [10, 1, 99], [0], [2])
        failure = classify_first_procedure_failure(record, {}, {})
        self.assertEqual(failure["cause"], "premature_sequence_stop")
        self.assertEqual(failure["position"], 1)

    def test_missing_eoc_boundary(self):
        record = base_record(["tool_a"], [10, 1], [0], [])
        failure = classify_first_procedure_failure(record, {}, {})
        self.assertEqual(failure["cause"], "missing_eoc_boundary")

    def test_intrinsic_routing_at_valid_boundary(self):
        record = base_record(
            ["tool_a", "tool_c"],
            [10, 1, 99, 12],
            [0, 3],
            [2],
        )
        oracle = {(1, 7, 1): {"selected_matches_expected": False}}
        failure = classify_first_procedure_failure(record, oracle, {})
        self.assertEqual(failure["cause"], "intrinsic_routing")

    def test_context_propagation_when_oracle_recovers(self):
        record = base_record(
            ["tool_a", "tool_c"],
            [10, 1, 99, 12],
            [0, 3],
            [2],
        )
        oracle = {(1, 7, 1): {"selected_matches_expected": True}}
        failure = classify_first_procedure_failure(record, oracle, {})
        self.assertEqual(failure["cause"], "context_propagation")

    def test_ambiguity_overrides_routing_cause(self):
        record = base_record(
            ["tool_a", "tool_c"],
            [10, 1, 99, 12],
            [0, 3],
            [2],
        )
        key = ambiguity_key(7, 1, "tool_b", "tool_c")
        annotations = {key: {"key": key, "label": "ambiguous"}}
        failure = classify_first_procedure_failure(record, {}, annotations)
        self.assertEqual(failure["cause"], "query_schema_ambiguity")

    def test_malformed_boundary_precedes_wrong_tool(self):
        record = base_record(
            ["tool_a", "tool_c"],
            [10, 1, 12],
            [0, 2],
            [],
        )
        failure = classify_first_procedure_failure(record, {}, {})
        self.assertEqual(failure["cause"], "malformed_boundary_before_tool")

    def test_over_generation_after_terminal_eoc(self):
        record = {
            "index": 7,
            "expected_tools": ["tool_a"],
            "predicted_tools": ["tool_a", "tool_c"],
            "raw_generated_token_ids": [10, 1, 99, 12],
            "tool_positions": [0, 3],
            "eoc_positions": [2],
            "eoc_token_id": 99,
        }
        failure = classify_first_procedure_failure(record, {}, {})
        self.assertEqual(failure["cause"], "over_generation_after_terminal_eoc")


class OracleSummaryTest(unittest.TestCase):
    def test_transition_accuracy_and_tcra_flips(self):
        records = [
            {
                "boundary_type": "transition",
                "selected_matches_expected": True,
                "base": {"gold_rank": 2},
                "fused": {"gold_rank": 1},
            },
            {
                "boundary_type": "transition",
                "selected_matches_expected": False,
                "base": {"gold_rank": 1},
                "fused": {"gold_rank": 3},
            },
            {
                "boundary_type": "initial",
                "selected_matches_expected": True,
                "base": {"gold_rank": 1},
                "fused": {"gold_rank": 1},
            },
            {
                "boundary_type": "terminal",
                "selected_matches_expected": True,
                "base": {"gold_rank": None},
                "fused": {"gold_rank": None},
            },
        ]
        summary = summarize_oracle(records)
        self.assertEqual(
            summary["transition"]["full_vocab_top1_accuracy"]["numerator"],
            1,
        )
        self.assertEqual(summary["transition"]["tcra_wrong_to_correct"], 1)
        self.assertEqual(summary["transition"]["tcra_correct_to_wrong"], 1)
        self.assertEqual(
            summary["terminal"]["full_vocab_top1_accuracy"]["rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
