#!/usr/bin/env python3
"""CPU checks for memory-bank constrained decoding."""

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import torch
import torch.nn as nn


COMPOSITIONAL_DIR = Path(__file__).resolve().parent
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from main_sequential import build_parser, validate_args  # noqa: E402
from model import FunctionCallingModel  # noqa: E402
from utils.run_memory_bank_constraint_eval import (  # noqa: E402
    summarize_prediction_file,
    validate_prediction_file,
)


class TinyConstraintHelpers:
    tool_reserved_token_ids = [1, 3]

    def _get_tool_reserved_token_ids_tensor(self, device):
        return torch.tensor(self.tool_reserved_token_ids, dtype=torch.long, device=device)

    def _sample_next_tokens(self, logits, temperature=0.6, top_p=0.9, do_sample=False):
        if do_sample:
            raise AssertionError("These deterministic tests use greedy decoding")
        return torch.argmax(logits, dim=-1)


class MemoryBankHelperTest(unittest.TestCase):
    def setUp(self):
        self.model = TinyConstraintHelpers()

    def test_memory_mass_uses_full_vocabulary_normalization(self):
        probabilities = torch.tensor([[0.05, 0.20, 0.10, 0.30, 0.15, 0.20]])
        logits = probabilities.log()

        memory_mass, normalized_entropy = FunctionCallingModel._compute_memory_bank_statistics(
            self.model,
            logits,
        )

        self.assertAlmostEqual(memory_mass.item(), 0.5, places=6)
        conditional = torch.tensor([0.4, 0.6])
        expected_entropy = -(conditional * conditional.log()).sum().item() / math.log(2)
        self.assertAlmostEqual(normalized_entropy.item(), expected_entropy, places=6)

    def test_constraint_selects_memory_or_ending_token_without_full_mask(self):
        logits = torch.tensor(
            [
                [0.0, 2.0, 9.0, 1.0, 0.0, 3.0],
                [0.0, 4.0, 8.0, 2.0, 0.0, 3.0],
            ]
        )
        active_rows = torch.tensor([True, False])

        next_tokens, changed_rows = FunctionCallingModel._select_with_memory_bank_constraint(
            self.model,
            logits,
            active_rows,
            end_token_id=5,
            do_sample=False,
        )

        self.assertEqual(next_tokens.tolist(), [5, 2])
        self.assertEqual(changed_rows.tolist(), [True, False])


class FakeTokenizer:
    eos_token_id = 5


class FakeGenerationModel(nn.Module, TinyConstraintHelpers):
    _compute_memory_bank_statistics = FunctionCallingModel._compute_memory_bank_statistics
    _select_with_memory_bank_constraint = FunctionCallingModel._select_with_memory_bank_constraint
    _build_decision_context = FunctionCallingModel._build_decision_context
    generate_with_tool_prediction = FunctionCallingModel.generate_with_tool_prediction

    def __init__(self, step_logits, use_eoc, use_logit_bias=False, tool_bias=0.0):
        super().__init__()
        self.step_logits = [torch.tensor(values, dtype=torch.float32).unsqueeze(0) for values in step_logits]
        self.step_index = 0
        self.use_eoc = use_eoc
        self.eoc_token_id = 4 if use_eoc else None
        self.use_logit_bias = use_logit_bias
        self.tool_bias = float(tool_bias)
        self.logit_bias_steps = []
        self.use_tool_head_replacement = False
        self.use_memory_bank_constraint = True
        self.memory_bank_probability_threshold = 0.5

    def _generation_forward_step(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        return_last_hidden_state=False,
    ):
        logits = self.step_logits[self.step_index].to(input_ids.device)
        self.step_index += 1
        hidden_states = None
        if return_last_hidden_state:
            hidden_states = torch.zeros(input_ids.size(0), 1, device=input_ids.device)
        return logits.expand(input_ids.size(0), -1), hidden_states, None

    def _apply_logit_bias_to_logits(self, logits, hidden_states, active_decision_rows):
        biased_logits = logits.clone()
        active_indices = active_decision_rows.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() > 0:
            biased_logits[active_indices, 3] += self.tool_bias
            self.logit_bias_steps.append(self.step_index - 1)
        return biased_logits

    def _parse_generated_sequences(self, generated_tokens, input_tokens, tokenizer):
        generated = generated_tokens[:, input_tokens.size(1) :]
        return [{"generated_token_ids": row.tolist()} for row in generated]


class MemoryBankGenerationTest(unittest.TestCase):
    def _generate(self, step_logits, use_eoc, use_logit_bias=False, tool_bias=0.0):
        model = FakeGenerationModel(
            step_logits,
            use_eoc=use_eoc,
            use_logit_bias=use_logit_bias,
            tool_bias=tool_bias,
        )
        result = model.generate_with_tool_prediction(
            user_tokens=torch.tensor([[0]]),
            user_mask=torch.tensor([[1]]),
            tokenizer=FakeTokenizer(),
            max_new_tokens=len(step_logits),
            do_sample=False,
            use_memory_bank_constraint=True,
            memory_bank_probability_threshold=0.5,
            use_eoc=use_eoc,
        )[0]
        return model, result

    def test_tokmem_uses_probability_gate_after_response_token(self):
        _model, result = self._generate(
            [
                [0.0, 5.0, 8.0, 1.0, 0.0, 2.0],
                [0.0, -2.0, 6.0, -2.0, 0.0, 1.0],
                [0.0, 4.0, 4.5, 4.0, 0.0, -1.0],
                [0.0, -2.0, 0.0, -2.0, 0.0, 6.0],
            ],
            use_eoc=False,
        )

        self.assertEqual(result["generated_token_ids"], [1, 2, 1, 5])
        diagnostics = result["memory_bank_constraint"]
        self.assertEqual(diagnostics["mode"], "probability_threshold")
        self.assertEqual(diagnostics["initial_trigger_count"], 1)
        self.assertEqual(diagnostics["transition_trigger_count"], 1)
        self.assertEqual(diagnostics["changed_token_count"], 2)

    def test_eoc_boundary_constraint_can_select_ending_token(self):
        _model, result = self._generate(
            [
                [0.0, 5.0, 8.0, 1.0, 0.0, 2.0],
                [0.0, -2.0, 6.0, -2.0, 0.0, 1.0],
                [0.0, -2.0, 0.0, -2.0, 6.0, 1.0],
                [0.0, 2.0, 9.0, 1.0, 0.0, 4.0],
            ],
            use_eoc=True,
        )

        self.assertEqual(result["generated_token_ids"], [1, 2, 4, 5])
        diagnostics = result["memory_bank_constraint"]
        self.assertEqual(diagnostics["mode"], "eoc_boundary")
        self.assertEqual(diagnostics["initial_trigger_count"], 1)
        self.assertEqual(diagnostics["transition_trigger_count"], 1)
        self.assertEqual(diagnostics["changed_token_count"], 2)

    def test_tapmem_applies_logit_bias_before_eoc_boundary_constraint(self):
        model, result = self._generate(
            [
                [0.0, 2.0, 9.0, 1.0, 0.0, 0.0],
                [0.0, -2.0, 6.0, -2.0, 0.0, 1.0],
                [0.0, -2.0, 0.0, -2.0, 6.0, 1.0],
                [0.0, 2.0, 9.0, 1.0, 0.0, 4.0],
            ],
            use_eoc=True,
            use_logit_bias=True,
            tool_bias=7.0,
        )

        self.assertEqual(result["generated_token_ids"], [3, 2, 4, 3])
        self.assertEqual(model.logit_bias_steps, [0, 3])
        diagnostics = result["memory_bank_constraint"]
        self.assertEqual(diagnostics["mode"], "eoc_boundary")
        self.assertEqual(diagnostics["trigger_count"], 2)
        self.assertEqual(diagnostics["changed_token_count"], 2)
        self.assertGreater(diagnostics["mean_trigger_memory_mass"], 0.1)
        self.assertLess(diagnostics["mean_trigger_memory_mass"], 0.5)

    def test_tapmem_without_constraint_keeps_full_vocabulary_selection(self):
        model = FakeGenerationModel(
            [
                [0.0, 2.0, 9.0, 1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, -2.0, 0.0, 6.0],
            ],
            use_eoc=True,
            use_logit_bias=True,
            tool_bias=7.0,
        )
        result = model.generate_with_tool_prediction(
            user_tokens=torch.tensor([[0]]),
            user_mask=torch.tensor([[1]]),
            tokenizer=FakeTokenizer(),
            max_new_tokens=2,
            do_sample=False,
            use_memory_bank_constraint=False,
            use_eoc=True,
        )[0]

        self.assertEqual(result["generated_token_ids"], [2, 5])
        self.assertEqual(model.logit_bias_steps, [0])
        self.assertNotIn("memory_bank_constraint", result)


class MemoryBankCliTest(unittest.TestCase):
    BASE_ARGS = ["--model_name", "unused", "--training_rounds", "51-100:1"]

    def _parse_and_validate(self, extra_args):
        parser = build_parser()
        argv = self.BASE_ARGS + extra_args
        args = parser.parse_args(argv)
        validate_args(args, parser, argv)
        return args

    def _assert_parser_error(self, extra_args, expected_message):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self._parse_and_validate(extra_args)
        self.assertIn(expected_message, stderr.getvalue())

    def test_default_threshold_is_half(self):
        args = self._parse_and_validate(["--use_memory_bank_constraint"])
        self.assertTrue(args.use_memory_bank_constraint)
        self.assertEqual(args.memory_bank_probability_threshold, 0.5)

    def test_eoc_constraint_is_valid_without_routing_head(self):
        args = self._parse_and_validate(["--use_eoc", "--use_memory_bank_constraint"])
        self.assertTrue(args.use_eoc)
        self.assertTrue(args.use_memory_bank_constraint)

    def test_constraint_allows_logit_bias_with_eoc(self):
        args = self._parse_and_validate(
            ["--use_eoc", "--use_logit_bias", "--use_memory_bank_constraint"]
        )
        self.assertTrue(args.use_eoc)
        self.assertTrue(args.use_logit_bias)
        self.assertTrue(args.use_memory_bank_constraint)

    def test_constraint_rejects_tool_head_replacement(self):
        self._assert_parser_error(
            ["--use_eoc", "--use_tool_head_replacement", "--use_memory_bank_constraint"],
            "--use_memory_bank_constraint cannot be combined with --use_tool_head_replacement",
        )

    def test_threshold_range_is_validated(self):
        self._assert_parser_error(
            ["--use_memory_bank_constraint", "--memory_bank_probability_threshold", "1.1"],
            "--memory_bank_probability_threshold must be between 0 and 1",
        )


class MemoryBankSummaryTest(unittest.TestCase):
    def test_trigger_diagnostics_are_weighted_by_trigger_count(self):
        records = [
            {
                "index": 0,
                "tool_f1": 1.0,
                "f1": 0.5,
                "tool_sequence_exact": True,
                "call_exact": False,
                "memory_bank_constraint": {
                    "trigger_count": 1,
                    "transition_trigger_count": 0,
                    "changed_token_count": 1,
                    "mean_trigger_memory_mass": 0.6,
                    "mean_trigger_normalized_entropy": 0.2,
                },
            },
            {
                "index": 1,
                "tool_f1": 0.0,
                "f1": 1.0,
                "tool_sequence_exact": False,
                "call_exact": True,
                "memory_bank_constraint": {
                    "trigger_count": 3,
                    "transition_trigger_count": 2,
                    "changed_token_count": 1,
                    "mean_trigger_memory_mass": 0.8,
                    "mean_trigger_normalized_entropy": 0.4,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "predictions.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            metrics = summarize_prediction_file(path)

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["trigger_count"], 4)
        self.assertEqual(metrics["transition_trigger_count"], 2)
        self.assertAlmostEqual(metrics["triggers_per_sample"], 2.0)
        self.assertAlmostEqual(metrics["changed_trigger_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_trigger_memory_mass"], 0.75)
        self.assertAlmostEqual(metrics["mean_trigger_normalized_entropy"], 0.35)

    def test_prediction_validation_rejects_stale_threshold(self):
        record = {
            "index": 0,
            "method": "tokmem_bank_constraint",
            "memory_bank_constraint": {
                "enabled": True,
                "mode": "probability_threshold",
                "probability_threshold": 0.4,
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "predictions.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Constraint threshold"):
                validate_prediction_file(
                    path,
                    expected_count=1,
                    expected_method="tokmem_bank_constraint",
                    expected_mode="probability_threshold",
                    expected_threshold=0.5,
                )


if __name__ == "__main__":
    unittest.main()
