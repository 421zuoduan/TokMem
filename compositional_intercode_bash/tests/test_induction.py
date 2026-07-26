from __future__ import annotations

import unittest
from unittest import mock

from compositional_intercode_bash import induction
from compositional_intercode_bash.induction import FittedVocabulary
from compositional_intercode_bash.unigram import CandidatePiece, ProcedureUnigramModel


def _toy_model(size: int = 10, mandatory: int = 2) -> ProcedureUnigramModel:
    pieces = [(("<START>", f"utility-{index}"),) for index in range(size)]
    return ProcedureUnigramModel(
        pieces,
        [1.0 / size] * size,
        0.5,
        mandatory=[index < mandatory for index in range(size)],
    )


def _identity_fit(
    model,
    initialization,
    pruning_round,
    sequences,
    weights,
    *,
    epsilon,
    em_max_iterations,
):
    return FittedVocabulary(
        initialization=initialization,
        model=model,
        train_log_likelihood=0.0,
        train_map_objective=0.0,
        em_history=[],
        pruning_round=pruning_round,
    )


def _ordered_scores(model, sequences, weights):
    return [
        (piece_id, float(piece_id))
        for piece_id, is_mandatory in enumerate(model.mandatory)
        if not is_mandatory
    ]


class PruningGridTest(unittest.TestCase):
    def test_three_initializations_receive_aggregated_sequences(self):
        a = ("START", "a")
        b = ("PIPE", "b")
        candidates = [
            CandidatePiece((a,), 2, 1, True),
            CandidatePiece((b,), 2, 1, True),
        ]
        observed = []

        def capture_path(
            model,
            initialization,
            sequences,
            weights,
            **_kwargs,
        ):
            observed.append((list(sequences), list(weights)))
            fitted = _identity_fit(
                model,
                initialization,
                0,
                sequences,
                weights,
                epsilon=1e-8,
                em_max_iterations=1,
            )
            return {model.size: fitted}

        with mock.patch.object(
            induction,
            "pruning_path",
            side_effect=capture_path,
        ):
            induction.run_three_initializations(
                candidates,
                [(a, b), (a,), (a, b)],
                [0.2, 0.5, 0.3],
                hidden_size=4,
            )

        self.assertEqual(len(observed), 3)
        for sequences, weights in observed:
            self.assertEqual(sequences, [(a,), (a, b)])
            self.assertEqual(weights, [0.5, 0.5])

    def test_batch_grid_lands_on_native_anchor_and_minimum(self):
        with (
            mock.patch.object(induction, "_fit_current", side_effect=_identity_fit),
            mock.patch.object(
                induction,
                "one_step_deletion_scores",
                side_effect=_ordered_scores,
            ),
        ):
            path = induction.pruning_path(
                _toy_model(),
                "uniform",
                [],
                [],
                minimum_k=2,
                hidden_size=10,
                native_anchor=7,
                batch_fraction=0.5,
            )
        self.assertEqual(set(path), {10, 7, 5, 4, 3, 2})
        self.assertIn(10, path)
        self.assertIn(7, path)
        self.assertIn(2, path)

    def test_models_above_hidden_size_are_pruning_only(self):
        with (
            mock.patch.object(induction, "_fit_current", side_effect=_identity_fit),
            mock.patch.object(
                induction,
                "one_step_deletion_scores",
                side_effect=_ordered_scores,
            ),
        ):
            path = induction.pruning_path(
                _toy_model(),
                "uniform",
                [],
                [],
                minimum_k=2,
                hidden_size=8,
                native_anchor=7,
                batch_fraction=0.5,
            )
        self.assertNotIn(10, path)
        self.assertIn(8, path)
        self.assertIn(7, path)


if __name__ == "__main__":
    unittest.main()
