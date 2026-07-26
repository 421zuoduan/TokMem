from __future__ import annotations

import copy
import unittest
from collections import Counter

from compositional_toolathlon.build_smoke_dataset import (
    CLAIM_TOOL_NAME,
    TARGET_TOOL_NAMES,
    _scenario_fingerprint,
    build_scenarios,
)
from compositional_toolathlon.target_policy import EXCLUDED_TOOL_REASONS


def _manifest() -> dict:
    names = [*TARGET_TOOL_NAMES, *EXCLUDED_TOOL_REASONS]
    return {
        "manifest_hash": "a" * 64,
        "benchmark_revision": "test-revision",
        "tools": [
            {
                "tool_name": name,
                "stable_id": f"test::{name}",
                "dispatch_kind": (
                    "terminal" if name == CLAIM_TOOL_NAME else "mcp"
                ),
            }
            for name in names
        ],
    }


class SmokeBuilderTests(unittest.TestCase):
    def test_v5_split_coverage_and_full_distractor_menu(self) -> None:
        manifest = _manifest()
        scenarios = build_scenarios(manifest)
        self.assertEqual(
            Counter(item.task["split"] for item in scenarios),
            Counter({"train": 14, "synthetic_test": 2}),
        )
        manifest_ids = {
            record["stable_id"] for record in manifest["tools"]
        }
        for scenario in scenarios:
            self.assertEqual(
                set(scenario.task["available_tools"]),
                manifest_ids,
            )
            intended = set(scenario.task["intended_required_tools"])
            distractors = set(scenario.task["distractor_tools"])
            self.assertFalse(intended & distractors)
            self.assertEqual(
                intended | distractors | {f"test::{CLAIM_TOOL_NAME}"},
                manifest_ids,
            )

        train_episode_counts: dict[str, set[str]] = {
            name: set() for name in TARGET_TOOL_NAMES
        }
        for scenario in scenarios:
            if scenario.task["split"] != "train":
                continue
            for action in scenario.actions:
                if action.tool_name in train_episode_counts:
                    train_episode_counts[action.tool_name].add(
                        scenario.task["task_id"]
                    )
        self.assertTrue(
            all(len(task_ids) >= 3 for task_ids in train_episode_counts.values())
        )

    def test_templates_and_plan_signatures_do_not_cross_splits(self) -> None:
        scenarios = build_scenarios(_manifest())
        template_splits: dict[str, set[str]] = {}
        plan_splits: dict[str, set[str]] = {}
        for scenario in scenarios:
            split = scenario.task["split"]
            template_splits.setdefault(
                scenario.task["template_id"],
                set(),
            ).add(split)
            plan_splits.setdefault(
                scenario.task["generation_provenance"]["plan_signature"],
                set(),
            ).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in template_splits.values()))
        self.assertTrue(all(len(splits) == 1 for splits in plan_splits.values()))

    def test_checkpoint_fingerprint_binds_task_and_plan(self) -> None:
        manifest = _manifest()
        scenario = build_scenarios(manifest)[0]
        original = _scenario_fingerprint(scenario, manifest)
        changed = copy.deepcopy(scenario)
        changed.task["instruction"] += " changed"
        self.assertNotEqual(
            original,
            _scenario_fingerprint(changed, manifest),
        )


if __name__ == "__main__":
    unittest.main()
