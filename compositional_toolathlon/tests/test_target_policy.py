import copy
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from compositional_toolathlon.audit_dataset import (
    audit_dataset,
    episode_id_hash,
    main as audit_main,
)
from compositional_toolathlon.episode_to_steps import record_content_hash
from compositional_toolathlon.main_train import (
    build_parser as build_training_parser,
    validate_data_audit_for_training,
    validate_step_records_for_split,
)
from compositional_toolathlon.target_policy import (
    EXCLUDED_TOOL_REASONS,
    TARGET_POLICY_HASH,
    TARGET_POLICY_LABEL,
    TARGET_POLICY_NAME,
    TARGET_POLICY_VERSION,
    TARGET_TOOL_NAMES,
    expected_target_tools_report,
    validate_target_tools_report,
)


class FrozenTargetPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tool_names = [*TARGET_TOOL_NAMES, *EXCLUDED_TOOL_REASONS]
        cls.manifest = {
            "manifest_hash": "fixture-manifest-hash",
            "tools": [
                {
                    "tool_name": tool_name,
                    "stable_id": f"fixture::{index:02d}::{tool_name}",
                    "dispatch_kind": "mcp",
                }
                for index, tool_name in enumerate(tool_names)
            ],
        }
        cls.target_report = expected_target_tools_report(cls.manifest)

    def test_policy_has_exact_frozen_denominator(self):
        self.assertEqual(TARGET_POLICY_NAME, "curated_local_office_smoke")
        self.assertEqual(TARGET_POLICY_VERSION, 1)
        self.assertEqual(TARGET_POLICY_LABEL, "curated_local_office_smoke_v1")
        self.assertEqual(len(TARGET_TOOL_NAMES), 38)
        self.assertEqual(len(set(TARGET_TOOL_NAMES)), 38)
        self.assertEqual(len(EXCLUDED_TOOL_REASONS), 9)
        self.assertEqual(
            TARGET_POLICY_HASH,
            "617fc57bbcaf85eab1134bc7482dee18367399d7309745ee1596cfca893f9e3d",
        )

    def test_expected_target_report_exactly_matches_policy(self):
        self.assertEqual(
            self.target_report,
            expected_target_tools_report(self.manifest),
        )
        self.assertEqual(
            len(validate_target_tools_report(self.target_report, self.manifest)),
            38,
        )

    def test_target_report_tampering_is_rejected(self):
        mutations = []

        missing_target = copy.deepcopy(self.target_report)
        missing_target["usable_tool_ids"].pop()
        mutations.append(missing_target)

        renamed_policy = copy.deepcopy(self.target_report)
        renamed_policy["selection_policy"] = "ad_hoc_v1"
        mutations.append(renamed_policy)

        changed_reason = copy.deepcopy(self.target_report)
        changed_reason["excluded_tool_reasons"]["local-claim_done"] = "optional"
        mutations.append(changed_reason)

        reordered_targets = copy.deepcopy(self.target_report)
        reordered_targets["target_tool_names"] = list(
            reversed(reordered_targets["target_tool_names"])
        )
        mutations.append(reordered_targets)

        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "frozen target policy"):
                    validate_target_tools_report(payload, self.manifest)

    def test_audit_cli_rejects_every_non_positive_coverage_threshold(self):
        required_arguments = [
            "audit_dataset",
            "--episodes",
            "unused-episodes.jsonl",
            "--verified-tasks",
            "unused-tasks.jsonl",
            "--manifest",
            "unused-manifest.json",
            "--target-tools",
            "unused-targets.json",
            "--semantic-leakage-audit",
            "unused-semantic.json",
            "--output",
            "unused-output.json",
        ]
        threshold_flags = (
            "--min-successful-episodes",
            "--min-argument-shapes",
            "--min-templates",
        )
        for threshold_flag in threshold_flags:
            for value in ("0", "-1"):
                with self.subTest(threshold_flag=threshold_flag, value=value):
                    with patch.object(
                        sys,
                        "argv",
                        required_arguments + [threshold_flag, value],
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "coverage thresholds must be positive integers",
                        ):
                            audit_main()

    def test_audit_cli_rejects_successful_episode_threshold_below_three(self):
        required_arguments = [
            "audit_dataset",
            "--episodes",
            "unused-episodes.jsonl",
            "--verified-tasks",
            "unused-tasks.jsonl",
            "--manifest",
            "unused-manifest.json",
            "--target-tools",
            "unused-targets.json",
            "--semantic-leakage-audit",
            "unused-semantic.json",
            "--output",
            "unused-output.json",
            "--min-successful-episodes",
            "1",
        ]
        with patch.object(sys, "argv", required_arguments):
            with self.assertRaisesRegex(ValueError, "must be at least 3"):
                audit_main()

    def test_audit_cli_requires_distractor_role_gate(self):
        arguments = [
            "audit_dataset",
            "--episodes",
            "unused-episodes.jsonl",
            "--verified-tasks",
            "unused-tasks.jsonl",
            "--manifest",
            "unused-manifest.json",
            "--target-tools",
            "unused-targets.json",
            "--semantic-leakage-audit",
            "unused-semantic.json",
            "--output",
            "unused-output.json",
        ]
        with patch.object(sys, "argv", arguments):
            with self.assertRaisesRegex(ValueError, "mandatory"):
                audit_main()

    def test_library_audit_still_accepts_custom_target_set(self):
        report = audit_dataset(
            episodes=[],
            task_specs=[],
            manifest=self.manifest,
            target_tool_ids={self.manifest["tools"][0]["stable_id"]},
            min_successful_episodes=1,
            min_argument_shapes=1,
            min_templates=1,
            require_distractor_role=False,
        )
        self.assertEqual(report["target_tool_count"], 1)
        self.assertNotIn("target_policy_hash", report)

    def test_training_gate_requires_exact_policy_metadata_and_denominator(self):
        target_ids = validate_target_tools_report(
            self.target_report,
            self.manifest,
        )
        audit = {
            "schema_version": 2,
            "passed": True,
            "tool_manifest_hash": self.manifest["manifest_hash"],
            "thresholds": {
                "min_successful_episodes": 3,
                "successful_episode_scope": "train",
                "min_argument_shapes": 1,
                "min_templates": 1,
                "require_distractor_role": True,
            },
            "execution_requirements": {"real_execution": True},
            "target_policy_name": TARGET_POLICY_NAME,
            "target_policy_version": TARGET_POLICY_VERSION,
            "target_policy_hash": TARGET_POLICY_HASH,
            "target_tool_count": 38,
            "per_tool": {tool_id: {"passed": True} for tool_id in target_ids},
        }
        validate_data_audit_for_training(audit, self.manifest)

        for field in (
            "target_policy_name",
            "target_policy_version",
            "target_policy_hash",
            "target_tool_count",
        ):
            tampered = copy.deepcopy(audit)
            del tampered[field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "frozen target policy"):
                    validate_data_audit_for_training(tampered, self.manifest)

        tampered = copy.deepcopy(audit)
        tampered["per_tool"].pop(next(iter(tampered["per_tool"])))
        with self.assertRaisesRegex(ValueError, "denominator differs"):
            validate_data_audit_for_training(tampered, self.manifest)

        threshold_tampering = (
            ("min_successful_episodes", 1, "at least 3"),
            ("require_distractor_role", False, "train distractor"),
        )
        for field, value, message in threshold_tampering:
            tampered = copy.deepcopy(audit)
            tampered["thresholds"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    validate_data_audit_for_training(tampered, self.manifest)

        tampered = copy.deepcopy(audit)
        tampered["execution_requirements"]["real_execution"] = False
        with self.assertRaisesRegex(ValueError, "real execution evidence"):
            validate_data_audit_for_training(tampered, self.manifest)

    def test_training_step_gate_binds_exact_count_order_and_content(self):
        records = [
            {"episode_id": "episode-a", "step_index": 0, "argument": "one"},
            {"episode_id": "episode-a", "step_index": 1, "argument": "two"},
        ]
        audit = {
            "split_episode_id_hashes": {
                "train": episode_id_hash({"episode-a"}),
            },
            "split_step_counts": {"train": len(records)},
            "split_step_content_hashes": {
                "train": record_content_hash(records),
            },
        }
        validate_step_records_for_split(audit, "train", records)

        with self.assertRaisesRegex(ValueError, "step count"):
            validate_step_records_for_split(audit, "train", records[:-1])

        tampered = copy.deepcopy(records)
        tampered[1]["argument"] = "changed"
        with self.assertRaisesRegex(ValueError, "step content"):
            validate_step_records_for_split(audit, "train", tampered)

        with self.assertRaisesRegex(ValueError, "step content"):
            validate_step_records_for_split(audit, "train", list(reversed(records)))

    def test_training_cli_has_no_validation_split(self):
        args = build_training_parser().parse_args(
            [
                "--method",
                "tokmem",
                "--model-name",
                "fixture-model",
                "--manifest",
                "manifest.json",
                "--train-steps",
                "train.jsonl",
                "--data-audit",
                "audit.json",
                "--run-dir",
                "run",
            ]
        )
        self.assertFalse(hasattr(args, "validation_steps"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_training_parser().parse_args(
                    [
                        "--method",
                        "tokmem",
                        "--model-name",
                        "fixture-model",
                        "--manifest",
                        "manifest.json",
                        "--train-steps",
                        "train.jsonl",
                        "--validation-steps",
                        "validation.jsonl",
                        "--data-audit",
                        "audit.json",
                        "--run-dir",
                        "run",
                    ]
                )

    def test_empty_validation_split_is_content_bound(self):
        audit = {
            "split_episode_id_hashes": {
                "validation": episode_id_hash(set()),
            },
            "split_step_counts": {"validation": 0},
            "split_step_content_hashes": {
                "validation": record_content_hash([]),
            },
        }
        validate_step_records_for_split(audit, "validation", [])


if __name__ == "__main__":
    unittest.main()
