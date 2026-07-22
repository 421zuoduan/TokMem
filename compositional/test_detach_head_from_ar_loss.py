#!/usr/bin/env python3
"""CPU tests for detaching the tool head from the train-add AR-loss path."""

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


COMPOSITIONAL_DIR = Path(__file__).resolve().parent
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from main_sequential import build_parser, validate_args  # noqa: E402
from run_layout import build_run_config, build_training_summary_payload  # noqa: E402
from training import apply_logit_train_add, compute_logit_bias_loss  # noqa: E402


class TinyToolHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_bias_head = nn.Linear(4, 3)
        self.tool_reserved_token_ids = [1, 3, 5]
        self.logit_bias_scale = 1.7
        with torch.no_grad():
            self.logit_bias_head.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.3, 0.4],
                        [-0.4, 0.5, 0.1, -0.2],
                        [0.3, 0.2, -0.5, 0.1],
                    ]
                )
            )
            self.logit_bias_head.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    def _get_logit_bias_scores(self, hidden_states):
        return self.logit_bias_head(hidden_states)

    def _get_tool_reserved_token_ids_tensor(self, device):
        return torch.tensor(self.tool_reserved_token_ids, dtype=torch.long, device=device)


class DetachHeadGradientTest(unittest.TestCase):
    def _run_ar_backward(self, detach_head_from_ar_loss=None):
        model = TinyToolHeadModel()
        base_logits = torch.linspace(-0.4, 0.4, steps=8).reshape(1, 1, 8)
        base_logits = base_logits.clone().requires_grad_(True)
        shift_logits = base_logits * 1.0
        boundary_hidden_states = torch.tensor(
            [[0.6, -0.3, 0.2, 0.8]],
            requires_grad=True,
        )
        kwargs = {
            "model": model,
            "shift_logits": shift_logits,
            "boundary_hidden_states": boundary_hidden_states,
            "batch_indices": torch.tensor([0]),
            "time_indices": torch.tensor([0]),
            "detach": True,
        }
        if detach_head_from_ar_loss is not None:
            kwargs["detach_head_from_ar_loss"] = detach_head_from_ar_loss

        output_logits = apply_logit_train_add(**kwargs)
        output_snapshot = output_logits.detach().clone()
        ar_loss = F.cross_entropy(output_logits.reshape(-1, 8), torch.tensor([1]))
        ar_loss_snapshot = ar_loss.detach().clone()
        ar_loss.backward()

        head_grad = model.logit_bias_head.weight.grad
        return {
            "output": output_snapshot,
            "ar_loss": ar_loss_snapshot,
            "head_grad": None if head_grad is None else head_grad.detach().clone(),
            "base_logits_grad": base_logits.grad.detach().clone(),
        }

    def test_default_matches_explicit_false_and_ar_updates_head(self):
        default_result = self._run_ar_backward()
        false_result = self._run_ar_backward(detach_head_from_ar_loss=False)

        self.assertTrue(torch.equal(default_result["output"], false_result["output"]))
        self.assertTrue(torch.equal(default_result["ar_loss"], false_result["ar_loss"]))
        self.assertIsNotNone(default_result["head_grad"])
        self.assertGreater(default_result["head_grad"].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(default_result["head_grad"], false_result["head_grad"]))

    def test_true_preserves_forward_and_blocks_ar_gradient_to_head(self):
        false_result = self._run_ar_backward(detach_head_from_ar_loss=False)
        true_result = self._run_ar_backward(detach_head_from_ar_loss=True)

        self.assertTrue(torch.equal(false_result["output"], true_result["output"]))
        self.assertTrue(torch.equal(false_result["ar_loss"], true_result["ar_loss"]))
        self.assertIsNone(true_result["head_grad"])
        self.assertGreater(true_result["base_logits_grad"].abs().sum().item(), 0.0)

    def test_classification_loss_still_trains_head_with_detached_boundary(self):
        model = TinyToolHeadModel()
        tool_embedding = nn.Parameter(
            torch.tensor(
                [
                    [0.2, 0.1, -0.3, 0.5],
                    [-0.4, 0.6, 0.7, -0.2],
                ]
            )
        )
        lora_output = nn.Parameter(
            torch.tensor(
                [
                    [0.05, -0.02, 0.03, 0.01],
                    [-0.01, 0.04, -0.05, 0.02],
                ]
            )
        )
        boundary_hidden_states = tool_embedding + lora_output
        boundary_hidden_states.retain_grad()

        _, classification_loss = compute_logit_bias_loss(
            model,
            boundary_hidden_states,
            torch.tensor([0, 2]),
            detach=True,
        )
        classification_loss.backward()

        self.assertIsNotNone(model.logit_bias_head.weight.grad)
        self.assertGreater(model.logit_bias_head.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(boundary_hidden_states.grad)
        self.assertIsNone(tool_embedding.grad)
        self.assertIsNone(lora_output.grad)


class DetachHeadCliTest(unittest.TestCase):
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

    def test_flag_requires_logit_bias(self):
        self._assert_parser_error(
            ["--detach_head_from_ar_loss"],
            "--detach_head_from_ar_loss requires --use_logit_bias",
        )

    def test_flag_requires_logit_train_add(self):
        self._assert_parser_error(
            [
                "--use_eoc",
                "--use_logit_bias",
                "--no-use_logit_train_add",
                "--detach_head_from_ar_loss",
            ],
            "--detach_head_from_ar_loss requires --use_logit_train_add",
        )

    def test_flag_does_not_apply_to_tool_head_replacement(self):
        self._assert_parser_error(
            ["--use_eoc", "--use_tool_head_replacement", "--detach_head_from_ar_loss"],
            "--detach_head_from_ar_loss does not apply to --use_tool_head_replacement",
        )

    def test_complete_combination_is_valid(self):
        args = self._parse_and_validate(
            [
                "--use_eoc",
                "--use_logit_bias",
                "--use_logit_train_add",
                "--detach",
                "--detach_head_from_ar_loss",
            ]
        )
        self.assertTrue(args.detach_head_from_ar_loss)
        self.assertTrue(args.use_logit_train_add)

    def test_default_is_false(self):
        args = self._parse_and_validate([])
        self.assertFalse(args.detach_head_from_ar_loss)


class DetachHeadResultRecordingTest(unittest.TestCase):
    def test_run_config_records_flag(self):
        payload = build_run_config(
            {"detach_head_from_ar_loss": True},
            {
                "run_name": "test-run",
                "run_dir": "/tmp/test-run",
                "timestamp": "20260717_000000",
            },
        )
        self.assertTrue(payload["args"]["detach_head_from_ar_loss"])

    def test_training_summary_records_flag(self):
        payload = build_training_summary_payload(
            run_name="test-run",
            all_results=[
                {
                    "round": 1,
                    "tools": "51-100",
                    "epochs": 1,
                    "results": {
                        "avg_total_loss": 1.0,
                        "avg_ar_loss": 0.8,
                        "avg_logit_bias_loss": 0.2,
                        "use_logit_bias": True,
                        "use_logit_train_add": True,
                        "detach_head_from_ar_loss": True,
                        "detach": True,
                    },
                }
            ],
        )
        self.assertTrue(payload["rounds"][0]["detach_head_from_ar_loss"])


if __name__ == "__main__":
    unittest.main()
