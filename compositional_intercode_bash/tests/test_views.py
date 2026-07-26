from __future__ import annotations

import unittest

from compositional_intercode_bash.unigram import ProcedureUnigramModel
from compositional_intercode_bash.views import (
    _greedy_multicover,
    _hash_order,
    attach_segmentation_metadata,
    build_balanced_presentations,
    build_views,
)


class ViewModelHashTest(unittest.TestCase):
    def test_multicover_fast_selection_matches_original_full_sort(self):
        candidates = [
            {
                "sample_id": f"sample-{index}",
                "segment_key": f"segments-{index}",
                "piece_ids": piece_ids,
                "fixed_J": fixed_j,
            }
            for index, (piece_ids, fixed_j) in enumerate(
                [
                    ([0, 1], False),
                    ([1, 2], True),
                    ([0, 2], True),
                    ([0], True),
                    ([1], False),
                    ([2], True),
                ]
            )
        ]

        remaining = [3, 3, 3]
        expected = []
        while any(value > 0 for value in remaining):
            ranked = []
            for candidate in candidates:
                gain = sum(
                    remaining[piece_id] > 0
                    for piece_id in candidate["piece_ids"]
                )
                if gain:
                    ranked.append(
                        (
                            -gain,
                            -int(bool(candidate["fixed_J"])),
                            _hash_order(
                                123,
                                candidate["sample_id"],
                                candidate["segment_key"],
                                len(expected),
                            ),
                            candidate,
                        )
                    )
            ranked.sort(key=lambda value: value[:3])
            chosen = dict(ranked[0][3])
            expected.append(chosen)
            for piece_id in chosen["piece_ids"]:
                if remaining[piece_id] > 0:
                    remaining[piece_id] -= 1

        self.assertEqual(
            _greedy_multicover(candidates, 3, 3, 123),
            expected,
        )

    def test_group_coverage_floor_handles_extreme_pool_imbalance(self):
        signature = ("START", "x")
        model = ProcedureUnigramModel(
            [(signature,)],
            [1.0],
            0.5,
            mandatory=[True],
        )
        records = []
        for pool, group_count in (("A", 2), ("R", 100)):
            for index in range(group_count):
                records.append(
                    {
                        "sample_id": f"{pool}-{index}",
                        "template_group_id": f"{pool}-{index}",
                        "canonical_signatures": [list(signature)],
                        "map_segments": [
                            {"piece_id": 0, "start": 0, "end": 1}
                        ],
                        "is_ambiguous": pool == "A",
                    }
                )

        presentations, report = build_balanced_presentations(
            records,
            model,
            epochs=1,
            minimum_procedure_exposures=1,
        )

        self.assertEqual(report["group_coverage_epoch_floor"], 50)
        self.assertEqual(report["epochs"], 50)
        self.assertEqual(
            report["available_group_count_by_pool"],
            {"A": 2, "R": 100},
        )
        self.assertEqual(
            report["covered_group_count_by_pool"],
            {"A": 2, "R": 100},
        )
        self.assertEqual(
            report["uncovered_group_ids_by_pool"],
            {"A": [], "R": []},
        )
        self.assertEqual(len(presentations), 200)
        epoch_groups = [
            (record["epoch"], record["template_group_id"])
            for record in presentations
        ]
        self.assertEqual(len(epoch_groups), len(set(epoch_groups)))

    def test_coverage_anchors_give_every_procedure_positive_targets(self):
        a = ("START", "a")
        b = ("PIPE", "b")
        c = ("PIPE", "c")
        model = ProcedureUnigramModel(
            [(a,), (b,), (c,), (a, b), (b, c)],
            [0.2] * 5,
            0.5,
            mandatory=[True, True, True, False, False],
        )
        records = []
        for pool in ("A", "R"):
            for index in range(2):
                signatures = (a, b, c) if pool == "A" else (a, c)
                chunks = ["a", "|b", "|c"] if pool == "A" else ["a", "|c"]
                records.append(
                    {
                        "sample_id": f"{pool}-{index}",
                        "template_group_id": f"{pool}-{index}",
                        "instruction_raw": "run pipeline",
                        "command_raw": "".join(chunks),
                        "base_chunks": chunks,
                        "canonical_signatures": [
                            list(signature) for signature in signatures
                        ],
                    }
                )
        segmented = attach_segmentation_metadata(records, model)
        presentations, presentation_report = build_balanced_presentations(
            segmented,
            model,
            epochs=1,
            minimum_procedure_exposures=2,
        )
        views, report = build_views(
            presentations,
            segmented,
            model,
            setting="sampled",
            data_seed=5,
            minimum_procedure_exposures=2,
        )
        self.assertTrue(views)
        self.assertEqual(report["uncovered_procedure_ids"], [])
        self.assertTrue(
            all(value >= 2 for value in report["procedure_positive_counts"])
        )
        self.assertGreater(
            presentation_report["coverage_anchor_variable_J_count"],
            0,
        )

    def setUp(self):
        self.signature = ("START", "find")
        self.record = {
            "sample_id": "sample-0",
            "instruction_raw": "list files",
            "command_raw": "find",
            "base_chunks": ["find"],
            "canonical_signatures": [list(self.signature)],
        }
        self.presentation = {
            "presentation_id": "presentation-0",
            "presentation_seed": 7,
            "epoch": 0,
            "effective_batch_id": 0,
            "position_in_batch": 0,
            "pool": "R",
            "template_group_id": "group-0",
            "sample_id": "sample-0",
            "within_epoch_occurrence_index": 0,
        }

    def _model(self, rho: float) -> ProcedureUnigramModel:
        return ProcedureUnigramModel(
            [(self.signature,)],
            [1.0],
            rho,
            mandatory=[True],
        )

    def test_segmentation_and_view_ids_depend_on_complete_model_hash(self):
        first = self._model(0.2)
        second = self._model(0.3)
        self.assertEqual(first.inventory_hash(), second.inventory_hash())
        self.assertNotEqual(first.model_hash(), second.model_hash())

        first_records = attach_segmentation_metadata([self.record], first)
        second_records = attach_segmentation_metadata([self.record], second)
        self.assertEqual(
            first_records[0]["procedure_inventory_hash"],
            second_records[0]["procedure_inventory_hash"],
        )
        self.assertNotEqual(
            first_records[0]["procedure_model_hash"],
            second_records[0]["procedure_model_hash"],
        )

        first_views, _report = build_views(
            [self.presentation],
            first_records,
            first,
            setting="map",
            data_seed=11,
            minimum_procedure_exposures=1,
        )
        second_views, _report = build_views(
            [self.presentation],
            second_records,
            second,
            setting="map",
            data_seed=11,
            minimum_procedure_exposures=1,
        )
        self.assertNotEqual(first_views[0]["view_id"], second_views[0]["view_id"])
        with self.assertRaisesRegex(ValueError, "another procedure model"):
            build_views(
                [self.presentation],
                first_records,
                second,
                setting="map",
                data_seed=11,
                minimum_procedure_exposures=1,
            )


if __name__ == "__main__":
    unittest.main()
