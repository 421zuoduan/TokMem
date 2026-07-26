import hashlib
import json
import unittest

from compositional_toolathlon.audit_dataset import audit_synthetic_task_splits
from compositional_toolathlon.episode_to_steps import (
    flatten_episode,
    validate_episode,
    validate_group_splits,
)
from compositional_toolathlon.manifest import canonical_json


def _real_episode() -> dict:
    observation = {"type": "text", "text": "completed"}
    return {
        "episode_id": "real_episode_001",
        "task_id": "real_task_001",
        "template_id": "real_template_001",
        "split": "train",
        "instruction": "Complete the local task.",
        "available_tool_ids": ["tool::one"],
        "messages": [
            {
                "role": "assistant",
                "tool_id": "tool::one",
                "arguments": {"path": "/workspace/input.txt"},
            },
            {
                "role": "tool",
                "tool_id": "tool::one",
                "observation": observation,
                "success": True,
                "runtime_metadata": {
                    "is_error": False,
                    "semantic_error": False,
                },
                "observation_sha256": hashlib.sha256(
                    canonical_json(observation).encode("utf-8")
                ).hexdigest(),
            },
        ],
        "evaluator": {"passed": True},
        "tool_manifest_hash": "fixture-manifest",
        "teacher": {
            "termination_reason": "claim_done",
            "visible_assistant_characters": 0,
            "real_execution": True,
            "fresh_environment_id": "fresh-environment-001",
        },
        "workspace_root": "/tmp/toolathlon-real-episode-001",
        "accepted": True,
        "rejection_reasons": [],
    }


def _split_task(
    *,
    task_id: str,
    split: str,
    task_family: str,
    template_id: str,
    asset_seed: int,
    suffix: str = ".txt",
    plan_signature: str | None = None,
) -> dict:
    provenance = {
        "session_id": f"session-{task_id}",
        "seed": asset_seed,
    }
    if plan_signature is not None:
        provenance["plan_signature"] = plan_signature
    return {
        "task_id": task_id,
        "task_family": task_family,
        "template_id": template_id,
        "asset_seed": asset_seed,
        "split": split,
        "instruction": f"Perform isolated operation {task_id}.",
        "initial_workspace": {
            "directories": ["incoming"],
            "files": [
                {
                    "path": f"incoming/input{suffix}",
                    "format": "text",
                }
            ],
        },
        "evaluator": {
            "assertions": [
                {
                    "op": "text_equals",
                    "path": f"out/result{suffix}",
                }
            ]
        },
        "intended_required_tools": ["tool::one"],
        "generation_provenance": provenance,
    }


class RealExecutionIntegrityTests(unittest.TestCase):
    def test_valid_mcp_evidence_is_accepted(self):
        validate_episode(_real_episode())

    def test_tampered_success_cannot_hide_runtime_failure(self):
        episode = _real_episode()
        observation = episode["messages"][1]
        observation["runtime_metadata"]["is_error"] = True
        with self.assertRaisesRegex(ValueError, "marks an error as success"):
            validate_episode(episode)

        episode = _real_episode()
        observation = episode["messages"][1]
        observation["runtime_metadata"] = {"return_code": 7}
        with self.assertRaisesRegex(ValueError, "nonzero return_code"):
            validate_episode(episode)

    def test_error_text_and_observation_tampering_are_rejected(self):
        episode = _real_episode()
        observation = episode["messages"][1]
        observation["observation"] = {
            "type": "text",
            "text": "Failed to update the workbook",
        }
        observation["observation_sha256"] = hashlib.sha256(
            canonical_json(observation["observation"]).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "Error:/Failed to"):
            validate_episode(episode)

        episode = _real_episode()
        episode["messages"][1]["observation"]["text"] = "tampered after execution"
        with self.assertRaisesRegex(ValueError, "observation_sha256 mismatch"):
            validate_episode(episode)

    def test_real_execution_requires_environment_and_workspace_evidence(self):
        episode = _real_episode()
        del episode["teacher"]["fresh_environment_id"]
        with self.assertRaisesRegex(ValueError, "fresh_environment_id"):
            validate_episode(episode)

        episode = _real_episode()
        del episode["workspace_root"]
        with self.assertRaisesRegex(ValueError, "workspace_root"):
            validate_episode(episode)

    def test_real_execution_environments_and_workspaces_cannot_be_reused(self):
        first = _real_episode()
        second = json.loads(json.dumps(first))
        second["episode_id"] = "real_episode_002"
        second["template_id"] = "real_template_002"
        with self.assertRaisesRegex(ValueError, "reuse fresh_environment_id"):
            validate_group_splits([first, second])

        second["teacher"]["fresh_environment_id"] = "fresh-environment-002"
        with self.assertRaisesRegex(ValueError, "reuse workspace_root"):
            validate_group_splits([first, second])

    def test_rejected_episode_requires_explicit_training_policy(self):
        episode = _real_episode()
        episode["accepted"] = False
        episode["rejection_reasons"] = ["deterministic evaluator failed"]
        episode["evaluator"]["passed"] = False
        observation = episode["messages"][1]
        observation["success"] = False
        observation["runtime_metadata"]["semantic_error"] = True

        with self.assertRaisesRegex(ValueError, "accepted=true"):
            validate_episode(episode)
        with self.assertRaisesRegex(ValueError, "clean successful calls"):
            validate_episode(episode, include_rejected=True)

        steps = flatten_episode(
            episode,
            include_rejected=True,
            require_clean=False,
        )
        self.assertEqual(len(steps), 1)
        self.assertFalse(steps[0]["episode_accepted"])
        self.assertFalse(steps[0]["episode_evaluator_passed"])
        self.assertFalse(steps[0]["target_call_success"])
        self.assertEqual(
            steps[0]["episode_rejection_reasons"],
            ["deterministic evaluator failed"],
        )


class SyntheticSplitIntegrityTests(unittest.TestCase):
    def test_renamed_family_and_unique_template_do_not_hide_layout_leakage(self):
        tasks = [
            _split_task(
                task_id="train-task",
                split="train",
                task_family="renamed-family-a",
                template_id="unique-template-a",
                asset_seed=1,
            ),
            _split_task(
                task_id="test-task",
                split="synthetic_test",
                task_family="renamed-family-b",
                template_id="unique-template-b",
                asset_seed=2,
            ),
        ]
        report = audit_synthetic_task_splits(tasks)
        failure_kinds = {failure["kind"] for failure in report["failures"]}
        self.assertFalse(report["passed"])
        self.assertIn("asset_layout_cross_split", failure_kinds)
        self.assertNotIn("template_cross_split", failure_kinds)

    def test_generation_plan_signature_cannot_cross_splits(self):
        tasks = [
            _split_task(
                task_id="train-task",
                split="train",
                task_family="family-a",
                template_id="template-a",
                asset_seed=1,
                suffix=".txt",
                plan_signature="same-generation-plan",
            ),
            _split_task(
                task_id="validation-task",
                split="validation",
                task_family="family-b",
                template_id="template-b",
                asset_seed=2,
                suffix=".csv",
                plan_signature="same-generation-plan",
            ),
        ]
        report = audit_synthetic_task_splits(tasks)
        failures = [
            failure
            for failure in report["failures"]
            if failure["kind"] == "plan_signature_cross_split"
        ]
        self.assertEqual(
            failures,
            [
                {
                    "kind": "plan_signature_cross_split",
                    "signature": "same-generation-plan",
                    "splits": ["train", "validation"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
