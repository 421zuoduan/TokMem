#!/usr/bin/env python3
"""Small CPU checks for the cold-start tensor and token bookkeeping."""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from cold_start.runtime import append_cold_start_tools  # noqa: E402
from cold_start.similarity import (  # noqa: E402
    centered_cosine_scores,
    topk_gap_weights,
)


class _Tokenizer:
    def get_vocab(self):
        return {
            "<|reserved_special_token_0|>": 10,
            "<|reserved_special_token_1|>": 11,
            "<|reserved_special_token_2|>": 12,
            "<|reserved_special_token_3|>": 13,
            "<|reserved_special_token_4|>": 14,
        }


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.decouple_embeddings = False
        self.use_eoc = True
        self.num_tools = 2
        self.num_reserved_slots = 3
        self.tool_names = ["old_a", "old_b"]
        self.tool_name_to_id = {"old_a": 0, "old_b": 1}
        self.tool_id_to_name = {0: "old_a", 1: "old_b"}
        self.tool_reserved_token_names = [
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
        ]
        self.tool_reserved_token_ids = [10, 11]
        self.reserved_token_names = list(self.tool_reserved_token_names)
        self.reserved_token_ids = list(self.tool_reserved_token_ids)
        self.eoc_token_name = "<|reserved_special_token_2|>"
        self.eoc_token_id = 12
        self.trainable_reserved_token_names = (
            self.tool_reserved_token_names + [self.eoc_token_name]
        )
        self.trainable_reserved_token_ids = [10, 11, 12]
        memory = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            )
        )
        self.trainable_tool_embeddings = memory
        self.trainable_tool_input_embeddings = memory
        self.trainable_tool_output_embeddings = memory
        self.logit_bias_head = nn.Linear(3, 2)
        self.logit_bias_scale = 1.0
        self.tool_id_to_token_id = {0: 10, 1: 11}
        self.token_id_to_tool_id = {10: 0, 11: 1}
        self.register_buffer(
            "_trainable_reserved_token_id_tensor",
            torch.tensor([10, 11, 12]),
            persistent=False,
        )
        self.register_buffer(
            "_tool_reserved_token_id_tensor",
            torch.tensor([10, 11]),
            persistent=False,
        )
        lookup = torch.full((13,), -1, dtype=torch.long)
        lookup[torch.tensor([10, 11, 12])] = torch.arange(3)
        self.register_buffer(
            "_trainable_reserved_index_lookup",
            lookup,
            persistent=False,
        )
        self._min_trainable_reserved_token_id = 10
        self._max_trainable_reserved_token_id = 12

    def _get_logit_bias_scores(self, hidden_states):
        return self.logit_bias_head(hidden_states)

    def _get_tool_reserved_token_ids_tensor(self, device):
        return self._tool_reserved_token_id_tensor.to(device)


class ColdStartTest(unittest.TestCase):
    def test_document_cosine_uses_only_old_tool_mean(self):
        old_hidden_states = torch.tensor(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
            ]
        )
        new_hidden_states = torch.tensor([[1.5, 0.5, 1.0]])
        old_mean = old_hidden_states.mean(dim=0, keepdim=True)
        expected = torch.nn.functional.normalize(
            new_hidden_states - old_mean,
            dim=-1,
        ) @ torch.nn.functional.normalize(
            old_hidden_states - old_mean,
            dim=-1,
        ).transpose(0, 1)

        actual = centered_cosine_scores(
            old_hidden_states,
            new_hidden_states,
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_topk_gap_weights_are_sparse_and_normalized(self):
        scores = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.1]]
        )
        result = topk_gap_weights(scores, top_k=4)
        weights = result["weights"]
        self.assertEqual(torch.count_nonzero(weights).item(), 4)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1)))
        self.assertEqual(result["neighbor_indices"][0, 0].item(), 0)

    def test_append_preserves_old_rows_and_eoc_id(self):
        model = _TinyModel()
        old_memory = model.trainable_tool_embeddings.detach().clone()
        old_weight = model.logit_bias_head.weight.detach().clone()
        old_bias = model.logit_bias_head.bias.detach().clone()

        registry = append_cold_start_tools(
            model,
            ["new_c"],
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.2, 0.3, 0.4]]),
            torch.tensor([0.1]),
        )

        self.assertEqual(model.eoc_token_id, 12)
        self.assertEqual(model.tool_reserved_token_ids, [10, 11, 13])
        self.assertEqual(registry["eoc_token_id"], 12)
        self.assertTrue(
            torch.equal(model.trainable_tool_embeddings[:2], old_memory[:2])
        )
        self.assertTrue(
            torch.equal(model.trainable_tool_embeddings[-1], old_memory[-1])
        )
        self.assertIs(
            model.trainable_tool_embeddings,
            model.trainable_tool_input_embeddings,
        )
        self.assertIs(
            model.trainable_tool_embeddings,
            model.trainable_tool_output_embeddings,
        )
        self.assertTrue(
            torch.equal(model.logit_bias_head.weight[:2], old_weight)
        )
        self.assertTrue(torch.equal(model.logit_bias_head.bias[:2], old_bias))

        hidden = torch.tensor([[0.3, -0.2, 0.7]])
        scores = model.logit_bias_head(hidden).float()
        expected_old = torch.log_softmax(scores[:, :2], dim=-1) + math.log(2)
        logits = torch.zeros(1, 20)
        updated = model._apply_logit_bias_to_logits(
            logits,
            hidden,
            torch.tensor([True]),
        )
        self.assertTrue(
            torch.allclose(updated[0, torch.tensor([10, 11])], expected_old[0])
        )


if __name__ == "__main__":
    unittest.main()
