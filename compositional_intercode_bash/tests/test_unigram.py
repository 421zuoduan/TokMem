from __future__ import annotations

import math
import unittest

import numpy as np

from compositional_intercode_bash.induction import (
    one_step_deletion_scores,
    weighted_log_likelihood,
)
from compositional_intercode_bash.unigram import (
    ProcedureUnigramModel,
    aggregate_weighted_sequences,
    fit_em,
)


def sig(name):
    return ("X", name)


class UnigramTest(unittest.TestCase):
    def setUp(self):
        self.A, self.B, self.C = sig("A"), sig("B"), sig("C")
        self.pieces = [
            (self.A,),
            (self.B,),
            (self.C,),
            (self.A, self.B),
            (self.B, self.C),
            (self.A, self.B, self.C),
        ]
        self.model = ProcedureUnigramModel(
            self.pieces,
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.3],
            0.2,
        )
        self.sequence = (self.A, self.B, self.C)

    def test_forward_backward_matches_manual_enumeration(self):
        result = self.model.forward_backward(self.sequence)
        self.assertAlmostEqual(math.exp(result.log_partition), 0.266112, places=12)
        self.assertAlmostEqual(
            math.exp(self.model.log_probability(self.sequence)),
            0.0532224,
            places=12,
        )
        counts = self.model.expected_piece_counts(self.sequence)
        expected = np.asarray(
            [
                0.050024049,
                0.001924002,
                0.050024049,
                0.048100048,
                0.048100048,
                0.901875902,
            ]
        )
        np.testing.assert_allclose(counts, expected, atol=2e-9, rtol=0)
        boundary_one = sum(
            probability
            for _start, end, _piece_id, probability in self.model.arc_posteriors(
                self.sequence
            )
            if end == 1
        )
        self.assertAlmostEqual(boundary_one, 0.050024049, places=8)

    def test_fixed_count_ffbs(self):
        rng = np.random.default_rng(123)
        seen = {}
        for _ in range(20000):
            path = self.model.sample_fixed_count(self.sequence, 2, rng)
            key = tuple((segment.start, segment.end) for segment in path)
            seen[key] = seen.get(key, 0) + 1
            self.assertEqual(path[0].start, 0)
            self.assertEqual(path[-1].end, 3)
        self.assertEqual(set(seen), {((0, 1), (1, 3)), ((0, 2), (2, 3))})
        for count in seen.values():
            self.assertLess(abs(count / 20000 - 0.5), 0.02)
        with self.assertRaises(ValueError):
            self.model.sample_fixed_count(self.sequence, 4, rng)

    def test_fixed_count_next_piece_distribution_conditions_on_current_state(self):
        first = dict(
            self.model.fixed_count_next_piece_distribution(
                self.sequence,
                start=0,
                remaining_piece_count=2,
            )
        )
        self.assertEqual(set(first), {0, 3})
        self.assertAlmostEqual(first[0], 0.5, places=12)
        self.assertAlmostEqual(first[3], 0.5, places=12)

        second = dict(
            self.model.fixed_count_next_piece_distribution(
                self.sequence,
                start=1,
                remaining_piece_count=1,
            )
        )
        self.assertEqual(set(second), {4})
        self.assertAlmostEqual(second[4], 1.0, places=12)

        with self.assertRaises(ValueError):
            self.model.fixed_count_next_piece_distribution(
                self.sequence,
                start=1,
                remaining_piece_count=3,
            )

    def test_viterbi_tie_break_does_not_depend_on_arc_order(self):
        model = ProcedureUnigramModel(
            [
                (self.A,),
                (self.B,),
                (self.C,),
                (self.A, self.B),
                (self.B, self.C),
            ],
            [0.1, 0.1, 0.1, 0.35, 0.35],
            0.2,
        )
        path, _score = model.viterbi(self.sequence)
        self.assertEqual([(segment.start, segment.end) for segment in path], [(0, 2), (2, 3)])

    def test_em_objective_is_non_decreasing(self):
        fitted, history = fit_em(
            self.model,
            [self.sequence, self.sequence],
            [0.5, 0.5],
            max_iterations=20,
        )
        objectives = [row["map_objective"] for row in history]
        self.assertTrue(all(right + 1e-9 >= left for left, right in zip(objectives, objectives[1:])))
        self.assertAlmostEqual(float(fitted.theta.sum()), 1.0)

    def test_log_space_does_not_underflow(self):
        atoms = tuple(sig(str(index)) for index in range(12))
        pieces = [(atom,) for atom in atoms]
        tiny = ProcedureUnigramModel(
            pieces,
            np.full(len(pieces), 1.0 / len(pieces)),
            1e-200,
        )
        self.assertTrue(math.isfinite(tiny.log_probability(atoms)))

    def test_duplicate_sequence_aggregation_is_objective_equivalent(self):
        sequences = [
            self.sequence,
            (self.A, self.B),
            self.sequence,
            (self.A, self.B),
        ]
        weights = [0.2, 0.4, 0.3, 0.6]
        unique_sequences, unique_weights = aggregate_weighted_sequences(
            sequences,
            weights,
        )
        self.assertEqual(
            unique_sequences,
            [(self.A, self.B), self.sequence],
        )
        np.testing.assert_allclose(unique_weights, [1.0, 0.5], rtol=0, atol=0)
        self.assertAlmostEqual(
            weighted_log_likelihood(self.model, sequences, weights),
            weighted_log_likelihood(
                self.model,
                unique_sequences,
                unique_weights,
            ),
            places=12,
        )

        expanded_model, expanded_history = fit_em(
            self.model,
            sequences,
            weights,
            max_iterations=20,
        )
        compact_model, compact_history = fit_em(
            self.model,
            unique_sequences,
            unique_weights,
            max_iterations=20,
        )
        np.testing.assert_allclose(
            expanded_model.theta,
            compact_model.theta,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertAlmostEqual(expanded_model.rho, compact_model.rho, places=12)
        self.assertAlmostEqual(
            expanded_history[-1]["map_objective"],
            compact_history[-1]["map_objective"],
            places=10,
        )

        expanded_scores = one_step_deletion_scores(
            expanded_model,
            sequences,
            weights,
        )
        compact_scores = one_step_deletion_scores(
            compact_model,
            unique_sequences,
            unique_weights,
        )
        self.assertEqual(
            [piece_id for piece_id, _score in expanded_scores],
            [piece_id for piece_id, _score in compact_scores],
        )
        np.testing.assert_allclose(
            [score for _piece_id, score in expanded_scores],
            [score for _piece_id, score in compact_scores],
            rtol=1e-11,
            atol=1e-11,
        )

    def test_model_hash_covers_probabilities_rho_and_mandatory_mask(self):
        changed_theta = ProcedureUnigramModel(
            self.pieces,
            [0.2, 0.1, 0.1, 0.1, 0.2, 0.3],
            self.model.rho,
            mandatory=self.model.mandatory,
        )
        changed_rho = ProcedureUnigramModel(
            self.pieces,
            self.model.theta,
            0.3,
            mandatory=self.model.mandatory,
        )
        changed_mandatory = ProcedureUnigramModel(
            self.pieces,
            self.model.theta,
            self.model.rho,
            mandatory=[True, True, True, True, False, False],
        )
        variants = (changed_theta, changed_rho, changed_mandatory)
        for variant in variants:
            self.assertEqual(self.model.inventory_hash(), variant.inventory_hash())
            self.assertNotEqual(self.model.model_hash(), variant.model_hash())
            self.assertEqual(variant.lexicon_hash(), variant.model_hash())

        payload = self.model.to_dict()
        self.assertEqual(payload["schema"], "procedure_unigram_v2")
        self.assertEqual(payload["inventory_hash"], self.model.inventory_hash())
        self.assertEqual(payload["model_hash"], self.model.model_hash())
        restored = ProcedureUnigramModel.from_dict(payload)
        self.assertEqual(restored.model_hash(), self.model.model_hash())

        payload["pieces"][0]["theta"] = 0.2
        payload["pieces"][1]["theta"] = 0.05
        payload["pieces"][5]["theta"] = 0.25
        with self.assertRaisesRegex(ValueError, "model hash"):
            ProcedureUnigramModel.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
