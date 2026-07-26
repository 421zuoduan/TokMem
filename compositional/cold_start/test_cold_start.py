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

from cold_start.runtime import (  # noqa: E402
    append_cold_start_tools,
    coherence_partial_renorm_new_embeddings,
    make_convex_new_embeddings,
    partial_renorm_new_embeddings,
    renorm_new_embeddings_to_old_mean,
)
from cold_start.calibration import (  # noqa: E402
    equalized_old_query_offsets,
    preserve_gate_with_identity_offsets,
    select_per_tool_safe_strengths,
)
from cold_start.similarity import (  # noqa: E402
    centered_cosine_geometry,
    centered_cosine_scores,
    local_affine_weights,
    loo_residual_krr_weights,
    query_prototype_bridge_weights,
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
    def test_convex_embedding_uses_document_weights(self):
        model = _TinyModel()
        weights = torch.tensor([[0.25, 0.75]])

        actual = make_convex_new_embeddings(model, weights)
        expected = (
            weights
            @ model.trainable_tool_embeddings[: model.num_tools].detach()
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_shared_convex_rows_cannot_beat_best_donor_logit(self):
        model = _TinyModel()
        weights = torch.tensor([[0.25, 0.75]])
        new_embedding = make_convex_new_embeddings(model, weights)
        old_tcra_weight = model.logit_bias_head.weight.detach().clone()
        old_tcra_bias = model.logit_bias_head.bias.detach().clone()
        append_cold_start_tools(
            model,
            ["new_c"],
            new_embedding,
            weights @ old_tcra_weight,
            weights @ old_tcra_bias,
        )

        hidden = torch.tensor([[0.3, -0.2, 0.7]])
        logits = torch.zeros(1, 20)
        tool_token_ids = torch.tensor(model.tool_reserved_token_ids)
        logits[:, tool_token_ids] = (
            hidden @ model.trainable_tool_output_embeddings[:3].transpose(0, 1)
        )
        updated = model._apply_logit_bias_to_logits(
            logits,
            hidden,
            torch.tensor([True]),
        )
        donor_logits = updated[:, tool_token_ids[:2]]
        new_logit = updated[:, tool_token_ids[2]]

        self.assertTrue(torch.allclose(new_logit, weights @ donor_logits.T))
        self.assertLessEqual(new_logit.item(), donor_logits.max().item())

    def test_convex_embedding_renorm_matches_old_mean_norm(self):
        model = _TinyModel()
        weights = torch.tensor([[0.25, 0.75]])
        new_embedding = make_convex_new_embeddings(model, weights)

        renormalized, target_norm = renorm_new_embeddings_to_old_mean(
            model,
            new_embedding,
        )

        old_mean_norm = (
            model.trainable_tool_embeddings[: model.num_tools]
            .detach()
            .norm(dim=1)
            .mean()
        )
        self.assertTrue(torch.allclose(target_norm, old_mean_norm))
        self.assertTrue(
            torch.allclose(renormalized.norm(dim=1), old_mean_norm.unsqueeze(0))
        )

    def test_partial_renorm_interpolates_embedding_norm(self):
        model = _TinyModel()
        weights = torch.tensor([[0.25, 0.75]])
        new_embedding = make_convex_new_embeddings(model, weights)

        partial, target_norm = partial_renorm_new_embeddings(
            model,
            new_embedding,
            strength=0.25,
        )
        expected_norm = (
            0.75 * new_embedding.float().norm(dim=1)
            + 0.25 * target_norm
        )

        self.assertTrue(torch.allclose(partial.float().norm(dim=1), expected_norm))

    def test_coherence_renorm_preserves_donor_disagreement(self):
        model = _TinyModel()
        weights = torch.tensor([[0.25, 0.75]])
        new_embedding = make_convex_new_embeddings(model, weights)

        renormalized, donor_norms, coherence = (
            coherence_partial_renorm_new_embeddings(
                model,
                new_embedding,
                weights,
                strength=1.0,
            )
        )
        expected_scale = 2.0 - coherence
        expected_norm = new_embedding.float().norm(dim=1) * expected_scale

        self.assertTrue(
            torch.allclose(renormalized.float().norm(dim=1), expected_norm)
        )
        self.assertTrue(torch.all(coherence >= 0.0))
        self.assertTrue(torch.all(coherence <= 1.0))
        self.assertTrue(
            torch.all(renormalized.float().norm(dim=1) < donor_norms)
        )

    def test_local_affine_weights_are_bounded_and_sum_to_one(self):
        scores = torch.tensor([[-0.9, -0.9, -0.9, -0.9, 0.5]])
        old_gram = torch.eye(5)
        convex = topk_gap_weights(scores, top_k=3)
        result = local_affine_weights(
            scores,
            old_gram,
            convex,
            ridge_relative=0.001,
            negative_mass_cap=0.05,
        )
        weights = result["weights"]

        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(1)))
        self.assertGreater(
            result["raw_weights"].clamp_max(0.0).neg().sum().item(),
            0.05,
        )
        self.assertLessEqual(result["negative_mass"].item(), 0.050001)
        self.assertEqual(torch.count_nonzero(weights).item(), 3)

    def test_affine_negative_mass_bounds_score_overshoot(self):
        weights = torch.tensor([[-0.05, 0.25, 0.8]])
        donor_scores = torch.tensor([[-2.0, 1.0, 3.0]])
        combined = (weights * donor_scores).sum(dim=1)
        score_range = (
            donor_scores.max(dim=1).values
            - donor_scores.min(dim=1).values
        )
        upper_bound = (
            donor_scores.max(dim=1).values + 0.05 * score_range
        )

        self.assertLessEqual(combined.item(), upper_bound.item())

    def test_loo_residual_krr_is_affine_and_bounded(self):
        old_hidden_states = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.2],
                [0.2, 0.0, 0.8],
                [0.0, 0.2, 0.8],
            ]
        )
        new_hidden_states = torch.tensor([[0.9, 0.1, 0.1]])
        geometry = centered_cosine_geometry(
            old_hidden_states,
            new_hidden_states,
        )
        convex = topk_gap_weights(geometry["scores"], top_k=2)

        result = loo_residual_krr_weights(
            geometry["scores"],
            geometry["old_gram"],
            convex,
            top_k=2,
            negative_mass_cap=0.05,
        )

        self.assertTrue(
            torch.allclose(
                result["weights"].sum(dim=1),
                torch.ones(1),
                atol=1e-5,
            )
        )
        self.assertLessEqual(result["negative_mass"].item(), 0.050001)
        self.assertTrue(
            torch.allclose(
                result["residual_operator"].sum(dim=1),
                torch.zeros(6),
                atol=1e-5,
            )
        )
        self.assertGreaterEqual(result["kernel_confidence"].item(), 0.0)
        self.assertLessEqual(result["kernel_confidence"].item(), 1.0)

    def test_query_prototype_bridge_uses_affine_cpu_weights(self):
        old_hidden_states = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.2],
                [0.2, 0.0, 0.8],
                [0.0, 0.2, 0.8],
            ]
        )
        new_hidden_states = torch.tensor([[0.9, 0.1, 0.1]])
        old_query_prototypes = torch.tensor(
            [
                [2.0, 0.0, 0.0, 1.0],
                [1.7, 0.3, 0.1, 1.0],
                [0.0, 2.0, 0.0, 1.0],
                [0.2, 1.7, 0.2, 1.0],
                [0.0, 0.1, 2.0, 1.0],
                [0.3, 0.0, 1.7, 1.0],
            ]
        )
        geometry = centered_cosine_geometry(
            old_hidden_states,
            new_hidden_states,
        )
        convex = topk_gap_weights(geometry["scores"], top_k=2)

        result = query_prototype_bridge_weights(
            geometry["scores"],
            geometry["old_gram"],
            old_query_prototypes,
            convex,
            negative_mass_cap=0.05,
        )

        self.assertTrue(
            torch.allclose(
                result["document_bridge_weights"].sum(dim=1),
                torch.ones(1),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                result["raw_weights"].sum(dim=1),
                torch.ones(1),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                result["weights"].sum(dim=1),
                torch.ones(1),
                atol=1e-5,
            )
        )
        self.assertEqual(
            result["predicted_query_prototypes"].device.type,
            "cpu",
        )
        self.assertEqual(result["weights"].device.type, "cpu")

    def test_query_prototype_bridge_triggers_negative_mass_cap(self):
        old_document_gram = torch.eye(6)
        document_scores = torch.tensor(
            [[0.95, 0.8, 0.1, -0.2, -0.5, -0.8]]
        )
        old_query_prototypes = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ]
        )
        convex = topk_gap_weights(document_scores, top_k=2)

        result = query_prototype_bridge_weights(
            document_scores,
            old_document_gram,
            old_query_prototypes,
            convex,
            document_ridge_relative=0.001,
            query_ridge_relative=0.001,
            negative_mass_cap=0.02,
        )

        self.assertGreater(result["raw_negative_mass"].item(), 0.02)
        self.assertLessEqual(result["negative_mass"].item(), 0.020001)
        self.assertTrue(
            torch.allclose(
                result["weights"].sum(dim=1),
                torch.ones(1),
                atol=1e-5,
            )
        )

        extrapolation_cap = float(
            result["raw_negative_mass"].item() * 1.5
        )
        extrapolated = query_prototype_bridge_weights(
            document_scores,
            old_document_gram,
            old_query_prototypes,
            convex,
            document_ridge_relative=0.001,
            query_ridge_relative=0.001,
            negative_mass_cap=extrapolation_cap,
            extrapolate_to_cap=True,
        )
        self.assertTrue(
            torch.allclose(
                extrapolated["negative_mass"],
                torch.tensor([extrapolation_cap]),
                atol=1e-5,
            )
        )
        self.assertGreater(extrapolated["ray_scales"].item(), 1.0)

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
        geometry = centered_cosine_geometry(
            old_hidden_states,
            new_hidden_states,
        )
        self.assertTrue(torch.allclose(geometry["scores"], expected))

    def test_topk_gap_weights_are_sparse_and_normalized(self):
        scores = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.1]]
        )
        result = topk_gap_weights(scores, top_k=4)
        weights = result["weights"]
        self.assertEqual(torch.count_nonzero(weights).item(), 4)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1)))
        self.assertEqual(result["neighbor_indices"][0, 0].item(), 0)

    def test_old_query_offsets_control_any_new_activation(self):
        shape_margins = torch.tensor(
            [
                [2.0, -2.0],
                [1.0, -1.0],
                [0.5, -0.5],
                [0.0, -0.2],
            ]
        )
        threshold_margins = torch.tensor(
            [
                [1.5, -1.0],
                [1.0, -0.8],
                [0.4, -0.4],
                [0.2, -0.1],
                [-0.2, 0.1],
            ]
        )
        result = equalized_old_query_offsets(
            shape_margins,
            threshold_margins,
            tool_tail_quantile=0.5,
            target_old_fpr=0.4,
        )
        calibrated = threshold_margins - result["offsets"]
        actual_fpr = (
            (calibrated.max(dim=1).values > 0).float().mean().item()
        )
        self.assertLessEqual(actual_fpr, 0.4)
        self.assertGreater(
            result["class_offsets"][0].item(),
            result["class_offsets"][1].item(),
        )

    def test_safe_ray_selects_strength_per_tool(self):
        margins = {
            0.0: torch.tensor(
                [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
            ),
            0.5: torch.tensor(
                [[0.2, -0.4], [-0.1, -0.3], [-0.2, -0.2]]
            ),
            1.0: torch.tensor(
                [[0.5, -0.2], [0.4, -0.1], [-0.1, -0.3]]
            ),
        }
        result = select_per_tool_safe_strengths(
            margins,
            target_fpr=0.0,
        )
        self.assertEqual(result["selected_strengths"], [0.0, 1.0])
        self.assertEqual(result["selected_fprs"], [0.0, 0.0])

    def test_identity_offsets_preserve_new_tool_gate(self):
        logits = torch.tensor(
            [[5.0, 4.0, 1.0], [2.0, 3.0, 2.5]]
        )
        offsets = torch.tensor([4.0, 0.0, -2.0])
        calibrated = preserve_gate_with_identity_offsets(
            logits,
            offsets,
        )
        self.assertTrue(
            torch.equal(
                calibrated.max(dim=1).values,
                logits.max(dim=1).values,
            )
        )
        self.assertEqual(calibrated.argmax(dim=1).tolist(), [1, 2])

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

    def test_runtime_identity_offsets_preserve_new_gate(self):
        model = _TinyModel()
        append_cold_start_tools(
            model,
            ["new_c", "new_d"],
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.tensor([1.0, 0.0]),
        )
        hidden = torch.zeros(1, 3)
        active = torch.tensor([True])
        logits = torch.zeros(1, 20)
        original = model._apply_logit_bias_to_logits(
            logits,
            hidden,
            active,
        )
        model._cold_start_new_identity_offsets = torch.tensor(
            [10.0, 0.0]
        )
        calibrated = model._apply_logit_bias_to_logits(
            logits,
            hidden,
            active,
        )
        new_ids = torch.tensor([13, 14])
        self.assertTrue(
            torch.equal(
                original[:, new_ids].max(dim=1).values,
                calibrated[:, new_ids].max(dim=1).values,
            )
        )
        self.assertEqual(original[:, new_ids].argmax(dim=1).item(), 0)
        self.assertEqual(calibrated[:, new_ids].argmax(dim=1).item(), 1)

    def test_tokmem_runtime_can_apply_identity_offsets(self):
        model = _TinyModel()
        model.logit_bias_head = None
        append_cold_start_tools(
            model,
            ["new_c", "new_d"],
            torch.zeros(2, 3),
        )
        model._cold_start_new_identity_offsets = torch.tensor(
            [10.0, 0.0]
        )
        logits = torch.zeros(1, 20)
        logits[0, 13] = 2.0
        logits[0, 14] = 1.0
        calibrated = model._apply_logit_bias_to_logits(
            logits,
            None,
            torch.tensor([True]),
        )
        new_ids = torch.tensor([13, 14])
        self.assertEqual(
            calibrated[:, new_ids].max(dim=1).values.item(),
            2.0,
        )
        self.assertEqual(calibrated[:, new_ids].argmax(dim=1).item(), 1)


if __name__ == "__main__":
    unittest.main()
