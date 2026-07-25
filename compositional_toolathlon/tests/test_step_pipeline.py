import json
import tempfile
import unittest
from pathlib import Path

import torch

from compositional_toolathlon.audit_dataset import audit_dataset
from compositional_toolathlon.config import load_experiment_config
from compositional_toolathlon.context import (
    ObservationPolicy,
    compact_observation,
    normalize_workspace_paths,
    render_step_context,
)
from compositional_toolathlon.decode_one_call import decode_one_call
from compositional_toolathlon.dataset import (
    EpisodeBalancedSampler,
    truncate_preserving_supervised_target,
)
from compositional_toolathlon.diagnostics import aggregate_call_records
from compositional_toolathlon.episode_to_steps import (
    flatten_episode,
    prepare_steps,
    validate_episode,
    validate_group_splits,
)
from compositional_toolathlon.environment import benchmark_provenance
from compositional_toolathlon.generate_tasks import load_generation_config
from compositional_toolathlon.manifest import (
    build_manifest,
    schema_hash,
    validate_tool_arguments,
)
from compositional_toolathlon.mcp_adapter import assert_runtime_tool_set
from compositional_toolathlon.local_tools import (
    PythonExecuteTool,
    augment_default_decoupled_manifest,
)
from compositional_toolathlon.masked_routing import (
    apply_masked_logit_bias,
    build_available_tool_mask,
    mask_backbone_tool_logits,
    mask_tool_head_logits,
    masked_routing_cross_entropy,
)
from compositional_toolathlon.model_client import ProposedAction
from compositional_toolathlon.official_metrics import (
    attach_official_evaluator,
    summarize_official_records,
)
from compositional_toolathlon.rollout import Action, run_closed_loop_rollout
from compositional_toolathlon.run_official_agent import (
    build_official_trajectory_envelope,
    load_official_agent_bundle,
    rollout_to_official_messages,
)
from compositional_toolathlon.teacher import TeacherSettings, collect_teacher_candidate
from compositional_toolathlon.synthetic_workspace import (
    apply_workspace_recipe,
    evaluate_workspace,
    safe_relative_path,
    verify_task_assets,
)
from compositional_toolathlon.training import (
    compute_stepwise_loss,
    gather_assistant_start_boundaries,
)


def make_manifest():
    return build_manifest(
        {
            "filesystem": {
                "tools": [
                    {
                        "name": "list_directory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                    {
                        "name": "move_file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "destination": {"type": "string"},
                            },
                            "required": ["source", "destination"],
                        },
                    },
                ]
            },
            "excel": {
                "tools": [
                    {
                        "name": "read_sheet",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ]
            },
            "local": {
                "tools": [
                    {
                        "name": "claim_done",
                        "dispatch_kind": "terminal",
                        "description": "Finish the task.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ]
            },
        },
        "2aed2468858f15818acafa178518390cc4b0f5cb",
    )


def make_episode(manifest):
    tool_ids = {
        (record["server"], record["tool_name"]): record["stable_id"]
        for record in manifest["tools"]
    }
    list_id = tool_ids[("filesystem", "list_directory")]
    move_id = tool_ids[("filesystem", "move_file")]
    distractor_id = tool_ids[("excel", "read_sheet")]
    return {
        "episode_id": "synthetic_sort_0001",
        "task_id": "synthetic_sort_0001",
        "template_id": "sort_by_extension_v1",
        "asset_seed": 11,
        "split": "train",
        "instruction": "Move the only CSV file into the reports directory.",
        "available_tool_ids": [list_id, move_id, distractor_id],
        "messages": [
            {
                "role": "assistant",
                "tool_id": list_id,
                "arguments": {"path": "/tmp/workspace"},
            },
            {
                "role": "tool",
                "tool_id": list_id,
                "observation": {"entries": ["notes.txt", "sales.csv", "reports"]},
                "success": True,
            },
            {
                "role": "assistant",
                "tool_id": move_id,
                "arguments": {
                    "source": "/tmp/workspace/sales.csv",
                    "destination": "/tmp/workspace/reports/sales.csv",
                },
            },
            {
                "role": "tool",
                "tool_id": move_id,
                "observation": {"moved": True},
                "success": True,
            },
        ],
        "evaluator": {"passed": True, "version": "fixture-v1"},
        "tool_manifest_hash": manifest["manifest_hash"],
        "teacher": {
            "model": "fixture",
            "prompt_version": "fixture-v1",
            "visible_assistant_characters": 0,
            "termination_reason": "claim_done",
        },
        "accepted": True,
        "rejection_reasons": [],
    }


def make_coverage_fixture(manifest, index, split):
    episode = json.loads(json.dumps(make_episode(manifest)))
    task_id = f"coverage_task_{index}"
    template_id = f"coverage_template_{index}"
    instruction = f"coveragefixture{index}"
    episode.update(
        {
            "episode_id": f"coverage_episode_{index}",
            "task_id": task_id,
            "template_id": template_id,
            "asset_seed": 1000 + index,
            "split": split,
            "instruction": instruction,
            "available_tool_ids": [
                record["stable_id"] for record in manifest["tools"]
            ],
        }
    )

    target_tool_id = next(
        record["stable_id"]
        for record in manifest["tools"]
        if record["tool_name"] == "list_directory"
    )
    task = {
        "task_id": task_id,
        "task_family": f"coverage_family_{index}",
        "template_id": template_id,
        "asset_seed": 1000 + index,
        "split": split,
        "instruction": instruction,
        "available_tools": [
            record["stable_id"] for record in manifest["tools"]
        ],
        "intended_required_tools": [target_tool_id],
        "distractor_tools": [
            record["stable_id"]
            for record in manifest["tools"]
            if record["stable_id"] != target_tool_id
        ],
        "initial_workspace": {
            "directories": [],
            "files": [
                {
                    "path": f"input_{index}.txt",
                    "format": "text",
                    "content": "pending\n",
                }
            ],
            "remove": [],
        },
        "oracle_final_state": {
            "directories": [],
            "files": [
                {
                    "path": f"output_{index}.txt",
                    "format": "text",
                    "content": "done\n",
                }
            ],
            "remove": [f"input_{index}.txt"],
        },
        "evaluator": {
            "type": "workspace_assertions_v1",
            "assertions": [
                {
                    "op": "text_equals",
                    "path": f"output_{index}.txt",
                    "value": "done\n",
                }
            ],
        },
        "generation_provenance": {
            "session_id": f"coverage_session_{index}",
            "seed": 1000 + index,
            "tool_manifest_hash": manifest["manifest_hash"],
        },
        "verified": True,
    }
    return episode, task, target_tool_id


class ConfigTests(unittest.TestCase):
    def test_experiment_config_is_frozen(self):
        config = load_experiment_config()
        self.assertEqual(len(config.tasks), 10)
        self.assertEqual(
            config.benchmark.revision,
            "2aed2468858f15818acafa178518390cc4b0f5cb",
        )
        self.assertFalse(config.interface.student_receives_tool_docs)

    def test_source_marker_is_not_shadowed_by_parent_git_repository(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            snapshot = Path(directory)
            marker = {
                "source_kind": "github_codeload",
                "revision": "fixed-toolathlon-revision",
                "archive_sha256": "fixture",
            }
            (snapshot / ".toolathlon-source.json").write_text(
                json.dumps(marker),
                encoding="utf-8",
            )
            self.assertEqual(
                benchmark_provenance(snapshot)["revision"],
                "fixed-toolathlon-revision",
            )


class CoverageAuditTests(unittest.TestCase):
    def run_coverage_audit(self, splits):
        manifest = make_manifest()
        episodes = []
        tasks = []
        target_tool_id = None
        for index, split in enumerate(splits):
            episode, task, target_tool_id = make_coverage_fixture(
                manifest,
                index,
                split,
            )
            episodes.append(episode)
            tasks.append(task)
        report = audit_dataset(
            episodes=episodes,
            task_specs=tasks,
            manifest=manifest,
            target_tool_ids={target_tool_id},
            min_successful_episodes=3,
            min_argument_shapes=1,
            min_templates=1,
            require_distractor_role=False,
        )
        return report, target_tool_id

    def test_held_out_episodes_do_not_satisfy_train_coverage(self):
        report, target_tool_id = self.run_coverage_audit(
            ["train", "validation", "synthetic_test"]
        )
        metrics = report["per_tool"][target_tool_id]
        self.assertFalse(report["passed"])
        self.assertEqual(metrics["successful_episode_count"], 3)
        self.assertEqual(metrics["train_successful_episode_count"], 1)
        self.assertEqual(
            metrics["successful_episode_count_by_split"],
            {"train": 1, "validation": 1, "synthetic_test": 1},
        )
        self.assertIn(
            "train_successful_episode_count",
            metrics["failed_thresholds"],
        )

    def test_three_train_episodes_satisfy_train_coverage(self):
        report, target_tool_id = self.run_coverage_audit(
            ["train", "train", "train"]
        )
        metrics = report["per_tool"][target_tool_id]
        self.assertTrue(report["passed"])
        self.assertEqual(metrics["train_successful_episode_count"], 3)
        self.assertEqual(
            report["thresholds"]["successful_episode_scope"],
            "train",
        )


class ManifestTests(unittest.TestCase):
    def test_schema_hash_is_key_order_invariant(self):
        left = {"type": "object", "properties": {"a": {"type": "string"}, "b": {}}}
        right = {"properties": {"b": {}, "a": {"type": "string"}}, "type": "object"}
        self.assertEqual(schema_hash("demo", left), schema_hash("demo", right))

    def test_manifest_is_stable_and_sorted(self):
        first = make_manifest()
        second = make_manifest()
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        stable_ids = [record["stable_id"] for record in first["tools"]]
        self.assertEqual(stable_ids, sorted(stable_ids))
        self.assertEqual(
            [record["memory_slot"] for record in first["tools"]],
            list(range(len(first["tools"]))),
        )
        self.assertTrue(all(record["wire_name"] for record in first["tools"]))

    def test_runtime_wire_names_are_opaque_and_schema_checked(self):
        manifest = build_manifest(
            {
                "toolathlon-gateway": {
                    "tools": [
                        {
                            "name": "server-with-hyphens-read-file-2",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            },
            "2aed2468858f15818acafa178518390cc4b0f5cb",
        )
        assert_runtime_tool_set(
            manifest,
            [
                {
                    "name": "server-with-hyphens-read-file-2",
                    "inputSchema": {"type": "object"},
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "schema_changed"):
            assert_runtime_tool_set(
                manifest,
                [
                    {
                        "name": "server-with-hyphens-read-file-2",
                        "inputSchema": {"type": "string"},
                    }
                ],
            )

    def test_host_local_python_is_explicit_and_not_checked_as_mcp_wire(self):
        base = make_manifest()
        augmented = augment_default_decoupled_manifest(base)
        local_records = [
            record
            for record in augmented["tools"]
            if record["origin"] == "host_local"
        ]
        self.assertEqual(len(local_records), 1)
        self.assertEqual(local_records[0]["dispatch_kind"], "local_python")
        self.assertIsNone(local_records[0]["wire_name"])
        runtime_tools = [
            {
                "name": record["wire_name"],
                "inputSchema": record["input_schema"],
            }
            for record in augmented["tools"]
            if record["origin"] == "gateway_mcp"
        ]
        assert_runtime_tool_set(augmented, runtime_tools)

    def test_tool_arguments_are_checked_before_execution_or_training(self):
        manifest = make_manifest()
        record = next(
            item
            for item in manifest["tools"]
            if item["tool_name"] == "move_file"
        )
        validate_tool_arguments(
            record,
            {"source": "a.txt", "destination": "b.txt"},
        )
        with self.assertRaisesRegex(ValueError, "required property"):
            validate_tool_arguments(record, {"source": "a.txt"})

    def test_provider_normalized_tool_name_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "provider normalization"):
            build_manifest(
                {
                    "gateway": {
                        "tools": [
                            {"name": "server-a-b", "inputSchema": {}},
                            {"name": "server-a_b", "inputSchema": {}},
                        ]
                    }
                },
                "2aed2468858f15818acafa178518390cc4b0f5cb",
            )


class DiagnosticTests(unittest.TestCase):
    def test_tool_and_complete_call_f1_are_separate(self):
        result = aggregate_call_records(
            [
                {
                    "sample_id": "s1",
                    "available_tool_ids": ["tool-a", "tool-b"],
                    "predicted_calls": [
                        {"tool_id": "tool-a", "arguments": {"path": "wrong"}}
                    ],
                    "target_calls": [
                        {"tool_id": "tool-a", "arguments": {"path": "right"}}
                    ],
                }
            ]
        )
        self.assertEqual(result["macro"]["tool_f1"], 1.0)
        self.assertEqual(result["macro"]["arguments_f1"], 0.0)
        self.assertIn("not argument-field F1", result["arguments_f1_definition"])

    def test_official_pass_is_attached_without_replacing_raw_evaluator(self):
        attached = attach_official_evaluator(
            {"method": "tokmem"},
            {"scores": {"passed": True}, "detail": "official"},
            pass_pointer="/scores/passed",
        )
        self.assertTrue(attached["official_evaluator"]["passed"])
        self.assertEqual(
            attached["official_evaluator"]["raw"]["detail"],
            "official",
        )

    def test_repeated_runs_are_summarized_by_task_and_seed(self):
        records = []
        for trial, passed in enumerate((True, False, True)):
            records.append(
                {
                    "method": "tapmem",
                    "task_id": "task-a",
                    "seed": 42,
                    "trial": trial,
                    "tool_call_count": 4,
                    "official_evaluator": {"passed": passed},
                }
            )
        summary = summarize_official_records(records)["methods"]["tapmem"]
        self.assertAlmostEqual(summary["pass_at_1"], 2 / 3)
        self.assertEqual(summary["pass_at_3"], 1.0)
        self.assertEqual(summary["pass_power_3"], 0.0)


class OfficialAgentBridgeTests(unittest.TestCase):
    def test_trusted_bundle_may_live_outside_dump_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "dump" / "agent_workspace"
            workspace.mkdir(parents=True)
            trusted_stash = root / "trusted"
            trusted_stash.mkdir()
            bundle_path = trusted_stash / "agent_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "task_str": "Create the requested artifact.",
                        "host_paths": {
                            "task_root": str(root / "dump"),
                            "agent_workspace": str(workspace),
                            "log_file": str(root / "dump" / "traj_log.json"),
                        },
                        "needed_local_tools": ["claim_done"],
                        "max_steps_under_single_turn_mode": 20,
                        "resolved_task_config": {"task_name": "fixture"},
                    }
                ),
                encoding="utf-8",
            )
            bundle = load_official_agent_bundle(bundle_path)
            self.assertEqual(bundle["_resolved_workspace"], str(workspace.resolve()))
            self.assertEqual(
                bundle["_resolved_output_root"],
                str((root / "dump").resolve()),
            )

    def test_bundle_rejects_workspace_outside_dump_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            bundle_path = root / "bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "task_str": "Do the task.",
                        "host_paths": {
                            "task_root": str(root / "dump"),
                            "agent_workspace": str(outside),
                            "log_file": str(root / "dump" / "traj_log.json"),
                        },
                        "needed_local_tools": [],
                        "max_steps_under_single_turn_mode": 10,
                        "resolved_task_config": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inside the dump root"):
                load_official_agent_bundle(bundle_path)

    def test_rollout_is_serialized_for_original_evaluator(self):
        manifest = make_manifest()
        list_record = next(
            record
            for record in manifest["tools"]
            if record["tool_name"] == "list_directory"
        )
        events = [
            {
                "kind": "tool_call",
                "step_index": 0,
                "action": {
                    "tool_id": list_record["stable_id"],
                    "arguments": {"path": "/workspace"},
                },
                "observation": {"entries": ["a.txt"]},
            }
        ]
        messages = rollout_to_official_messages(
            instruction="Inspect the workspace.",
            events=events,
            manifest=manifest,
        )
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "tool"])
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"],
            "gw_list_directory",
        )
        bundle = {
            "task_str": "Inspect the workspace.",
            "resolved_task_config": {"task_name": "fixture"},
        }
        envelope = build_official_trajectory_envelope(
            bundle=bundle,
            rollout={
                "termination_reason": "claim_done",
                "events": events,
                "tool_call_count": 1,
                "available_tool_ids": [list_record["stable_id"]],
            },
            manifest=manifest,
            fresh_environment_id="fixture-container-1",
            started_at="2026-07-25T00:00:00+00:00",
        )
        self.assertEqual(envelope["status"], "success")
        self.assertEqual(
            envelope["tool_calls"]["tools"][0]["function"]["name"],
            "gw_list_directory",
        )
        self.assertEqual(envelope["key_stats"]["tool_calls"], 1)
        self.assertIsNone(envelope["history_file"])


class SyntheticWorkspaceTests(unittest.TestCase):
    def make_task(self):
        return {
            "initial_workspace": {
                "directories": ["incoming"],
                "files": [
                    {
                        "path": "incoming/note.txt",
                        "format": "text",
                        "content": "draft\n",
                    }
                ],
                "remove": [],
            },
            "oracle_final_state": {
                "directories": ["done"],
                "files": [
                    {
                        "path": "done/note.txt",
                        "format": "text",
                        "content": "approved\n",
                    }
                ],
                "remove": ["incoming/note.txt"],
            },
            "evaluator": {
                "type": "workspace_assertions_v1",
                "assertions": [
                    {
                        "op": "text_equals",
                        "path": "done/note.txt",
                        "value": "approved\n",
                    },
                    {"op": "absent", "path": "incoming/note.txt"},
                ],
            },
        }

    def test_declarative_oracle_passes_and_initial_state_fails(self):
        verification = verify_task_assets(self.make_task())
        self.assertTrue(verification["passed"])
        self.assertFalse(verification["initial_evaluator"]["passed"])
        self.assertTrue(verification["oracle_evaluator"]["passed"])

    def test_workspace_recipe_and_evaluator_do_not_execute_generated_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = self.make_task()
            apply_workspace_recipe(temporary, task["initial_workspace"], require_empty=True)
            self.assertFalse(evaluate_workspace(temporary, task["evaluator"])["passed"])

    def test_parent_and_absolute_paths_are_rejected(self):
        for unsafe in ("../escape", "/absolute", "a/../b"):
            with self.assertRaises(ValueError):
                safe_relative_path(unsafe)


class PythonExecuteIsolationTests(unittest.TestCase):
    def test_python_tool_has_no_unsafe_fallback_without_bubblewrap(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "compositional_toolathlon.local_tools.shutil.which",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "unsafe host execution"):
                    PythonExecuteTool(temporary)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest()
        self.episode = make_episode(self.manifest)

    def test_two_calls_become_two_prefix_steps_without_future_leakage(self):
        steps = flatten_episode(self.episode, workspace_root="/tmp/workspace")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["history"], [])
        self.assertEqual(len(steps[1]["history"]), 1)
        self.assertIn("sales.csv", json.dumps(steps[1]["history"]))
        self.assertNotIn('"moved": true', json.dumps(steps[1]["history"]).lower())
        self.assertIn("<WORKSPACE>", json.dumps(steps[1]))
        self.assertEqual(steps[0]["episode_step_count"], 2)

    def test_rejects_teacher_prose_and_failed_calls(self):
        with_prose = json.loads(json.dumps(self.episode))
        with_prose["messages"][0]["content"] = "I will inspect the directory first."
        with self.assertRaisesRegex(ValueError, "teacher prose"):
            validate_episode(with_prose)

        with_error = json.loads(json.dumps(self.episode))
        with_error["messages"][1]["success"] = False
        with self.assertRaisesRegex(ValueError, "clean successful"):
            validate_episode(with_error)

    def test_rejected_episode_cannot_enter_step_data(self):
        self.episode["accepted"] = False
        with self.assertRaisesRegex(ValueError, "accepted=true"):
            flatten_episode(self.episode)

    def test_template_cannot_cross_splits(self):
        validation = json.loads(json.dumps(self.episode))
        validation["episode_id"] = "synthetic_sort_0002"
        validation["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "cross data splits"):
            validate_group_splits([self.episode, validation])

    def test_prepare_steps_writes_rich_and_legacy_outputs(self):
        validation = json.loads(json.dumps(self.episode))
        validation["episode_id"] = "synthetic_sort_0002"
        validation["template_id"] = "sort_by_suffix_v2"
        validation["split"] = "validation"
        synthetic_test = json.loads(json.dumps(self.episode))
        synthetic_test["episode_id"] = "synthetic_sort_0003"
        synthetic_test["template_id"] = "sort_by_manifest_v3"
        synthetic_test["split"] = "synthetic_test"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metadata = prepare_steps(
                [self.episode, validation, synthetic_test],
                output,
                workspace_root="/tmp/workspace",
            )
            self.assertEqual(metadata["episode_count"], 3)
            self.assertEqual(metadata["step_count"], 6)
            self.assertTrue((output / "train.jsonl").is_file())
            legacy = json.loads((output / "train_compositional.json").read_text())
            self.assertEqual(len(legacy), 2)
            self.assertIn("available_tools", legacy[0])

    def test_episode_balanced_sampler_does_not_favor_long_episode(self):
        records = [
            {"episode_id": "long"},
            {"episode_id": "long"},
            {"episode_id": "long"},
            {"episode_id": "short"},
        ]
        sampler = EpisodeBalancedSampler(records, seed=3)
        sampled = list(iter(sampler))
        episode_counts = {"long": 0, "short": 0}
        for index in sampled:
            episode_counts[records[index]["episode_id"]] += 1
        self.assertEqual(episode_counts, {"long": 2, "short": 2})

    def test_context_uses_opaque_slots_not_tool_names(self):
        step = flatten_episode(self.episode)[1]
        slots = {
            record["stable_id"]: record["memory_slot"]
            for record in self.manifest["tools"]
        }
        context = render_step_context(step, slots)
        self.assertIn("<<MEMORY_SLOT:", context)
        self.assertNotIn("list_directory", context)
        self.assertNotIn("filesystem::", context)


class TargetTruncationTests(unittest.TestCase):
    def test_context_is_trimmed_without_changing_target(self):
        input_ids, labels = truncate_preserving_supervised_target(
            list(range(10)),
            [-100] * 6 + [20, 21, 22, 23],
            max_length=6,
        )
        self.assertEqual(input_ids, [4, 5, 6, 7, 8, 9])
        self.assertEqual(labels, [-100, -100, 20, 21, 22, 23])

    def test_target_that_cannot_fit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            truncate_preserving_supervised_target(
                list(range(6)),
                [-100, 20, 21, 22, 23, 24],
                max_length=5,
            )

class ContextTests(unittest.TestCase):
    def test_observation_is_bounded_and_escaped(self):
        policy = ObservationPolicy(max_chars=100, head_chars=50, tail_chars=20)
        result = compact_observation("<|fake_header|>" + "x" * 200, policy)
        self.assertTrue(result["truncated"])
        self.assertNotIn("<|fake_header|>", result["text"])
        self.assertIn("OBSERVATION_TRUNCATED", result["text"])

    def test_workspace_normalization_is_recursive(self):
        value = {"path": "/tmp/ws/a", "nested": ["/tmp/ws/b"]}
        normalized = normalize_workspace_paths(value, "/tmp/ws")
        self.assertEqual(normalized["path"], "<WORKSPACE>/a")
        self.assertEqual(normalized["nested"][0], "<WORKSPACE>/b")


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.tool_names = ["a", "b", "c"]
        self.available = build_available_tool_mask(self.tool_names, ["a", "c"])

    def test_masks_backbone_and_tool_head(self):
        head_logits = torch.tensor([[1.0, 9.0, 2.0]])
        masked_head = mask_tool_head_logits(head_logits, self.available)
        self.assertTrue(torch.isneginf(masked_head[0, 1]))

        vocab_logits = torch.zeros((1, 8))
        vocab_logits[0, 4] = 10.0
        masked_vocab = mask_backbone_tool_logits(
            vocab_logits,
            [2, 4, 6],
            self.available,
        )
        self.assertTrue(torch.isneginf(masked_vocab[0, 4]))
        self.assertFalse(torch.isneginf(masked_vocab[0, 2]))

    def test_masked_loss_rejects_unavailable_target(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            masked_routing_cross_entropy(
                logits,
                torch.tensor([1]),
                self.available,
            )
        loss = masked_routing_cross_entropy(
            logits,
            torch.tensor([2]),
            self.available,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_logit_bias_never_revives_unavailable_tool(self):
        vocab_logits = torch.zeros((1, 8))
        tool_logits = torch.tensor([[1.0, 100.0, 2.0]])
        result = apply_masked_logit_bias(
            vocab_logits,
            tool_logits,
            [2, 4, 6],
            self.available,
        )
        self.assertTrue(torch.isneginf(result[0, 4]))

    def test_batch_mask_broadcasts_across_sequence_positions(self):
        logits = torch.zeros((2, 3, 8))
        mask = torch.tensor([[True, False, True], [False, True, True]])
        result = mask_backbone_tool_logits(logits, [2, 4, 6], mask)
        self.assertTrue(torch.isneginf(result[0, :, 4]).all())
        self.assertTrue(torch.isneginf(result[1, :, 2]).all())
        self.assertFalse(torch.isneginf(result[0, :, 2]).any())
        self.assertFalse(torch.isneginf(result[1, :, 4]).any())


class GenerationConfigTests(unittest.TestCase):
    def test_generator_sessions_use_distinct_prompts(self):
        config = load_generation_config(
            Path(__file__).resolve().parents[1] / "configs" / "generation.json"
        )
        prompts = [session["prompt"] for session in config["generator_sessions"]]
        self.assertGreaterEqual(len(prompts), 3)
        self.assertEqual(len(prompts), len(set(prompts)))


class FakeActionClient:
    def __init__(self, actions):
        self.actions = list(actions)

    async def next_tool_action(self, **kwargs):
        _ = kwargs
        return self.actions.pop(0)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def call_tool(self, stable_tool_id, arguments):
        self.calls.append((stable_tool_id, arguments))
        return {"success": True, "observation": {"call_number": len(self.calls)}}


class FailingTerminalExecutor(FakeExecutor):
    async def call_tool(self, stable_tool_id, arguments):
        self.calls.append((stable_tool_id, arguments))
        return {"success": False, "observation": {"error": "not accepted"}}


class FakeEvaluator:
    async def evaluate(self):
        return {"passed": True, "version": "fixture-v1"}


class TeacherTests(unittest.IsolatedAsyncioTestCase):
    async def test_teacher_stores_only_calls_and_real_observations(self):
        manifest = make_manifest()
        by_name = {record["tool_name"]: record for record in manifest["tools"]}
        list_id = by_name["list_directory"]["stable_id"]
        done_id = by_name["claim_done"]["stable_id"]
        client = FakeActionClient(
            [
                ProposedAction("call-1", "tool_0000", {"path": "/work"}, ""),
                ProposedAction("call-2", "tool_0001", {}, ""),
            ]
        )
        executor = FakeExecutor()
        task_spec = {
            "task_id": "synthetic_teacher_1",
            "template_id": "teacher_fixture_v1",
            "asset_seed": 7,
            "split": "train",
            "instruction": "Inspect the workspace, then finish.",
            "available_tools": [list_id, done_id],
        }
        episode = await collect_teacher_candidate(
            client=client,
            executor=executor,
            evaluator=FakeEvaluator(),
            task_spec=task_spec,
            manifest=manifest,
            settings=TeacherSettings(
                model="gpt-5.6",
                prompt_version="fixture",
                system_prompt="Return one tool call.",
                max_tool_calls=3,
            ),
            candidate_index=0,
        )
        self.assertTrue(episode["accepted"])
        self.assertEqual(len(episode["messages"]), 4)
        self.assertTrue(all("content" not in message for message in episode["messages"]))
        self.assertEqual(
            [message["role"] for message in episode["messages"]],
            ["assistant", "tool", "assistant", "tool"],
        )
        self.assertEqual(executor.calls[0][0], list_id)
        self.assertEqual(episode["teacher"]["visible_assistant_characters"], 0)

    async def test_visible_teacher_prose_rejects_candidate(self):
        manifest = make_manifest()
        done_id = next(
            record["stable_id"]
            for record in manifest["tools"]
            if record["tool_name"] == "claim_done"
        )
        client = FakeActionClient(
            [ProposedAction("call-1", "tool_0000", {}, "I am done.")]
        )
        episode = await collect_teacher_candidate(
            client=client,
            executor=FakeExecutor(),
            evaluator=FakeEvaluator(),
            task_spec={
                "task_id": "synthetic_teacher_2",
                "template_id": "teacher_fixture_v2",
                "split": "train",
                "instruction": "Finish.",
                "available_tools": [done_id],
            },
            manifest=manifest,
            settings=TeacherSettings(
                model="gpt-5.6",
                prompt_version="fixture",
                system_prompt="Return one tool call.",
                max_tool_calls=1,
                max_visible_assistant_characters=0,
            ),
            candidate_index=0,
        )
        self.assertFalse(episode["accepted"])
        self.assertTrue(
            any("visible prose" in reason for reason in episode["rejection_reasons"])
        )


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        _ = kwargs
        token_text = {20: "{", 21: "}", 22: '"x"', 23: ":", 24: "1"}
        return "".join(token_text.get(int(token_id), "") for token_id in token_ids)


class FakeDecodeModel:
    def __init__(self, sequence, use_eoc):
        self.tool_names = ["tool-a", "tool-b"]
        self.tool_reserved_token_ids = [10, 11]
        self.eoc_token_id = 12 if use_eoc else None
        self.logit_bias_scale = 1.0
        self.sequence = list(sequence)
        self.position = 0

    def _generation_forward_step(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        return_last_hidden_state=False,
    ):
        _ = input_ids, attention_mask, past_key_values, return_last_hidden_state
        logits = torch.full((1, 32), -10.0)
        next_token = self.sequence[self.position]
        logits[0, next_token] = 10.0
        if self.position == 0:
            logits[0, 11] = 100.0
        self.position += 1
        return logits, None, self.position


class DecodeTests(unittest.TestCase):
    def test_eoc_decoder_masks_unavailable_tool_and_stops_at_first_eoc(self):
        model = FakeDecodeModel([10, 20, 21, 12, 11], use_eoc=True)
        action = decode_one_call(
            model=model,
            tokenizer=FakeTokenizer(),
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            available_tool_mask=torch.tensor([True, False]),
            use_eoc=True,
            use_logit_bias=False,
            max_new_tokens=8,
            native_response_end_ids=[2],
        )
        self.assertEqual(action.tool_id, "tool-a")
        self.assertEqual(action.arguments, {})
        self.assertEqual(action.stop_reason, "eoc")
        self.assertEqual(action.generated_token_ids, (10, 20, 21, 12))

    def test_tokmem_decoder_stops_after_json_and_native_response_end(self):
        model = FakeDecodeModel([10, 20, 21, 2, 11], use_eoc=False)
        action = decode_one_call(
            model=model,
            tokenizer=FakeTokenizer(),
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            available_tool_mask=torch.tensor([True, False]),
            use_eoc=False,
            use_logit_bias=False,
            max_new_tokens=8,
            native_response_end_ids=[2],
        )
        self.assertEqual(action.tool_id, "tool-a")
        self.assertEqual(action.arguments, {})
        self.assertEqual(action.stop_reason, "native_response_end")


class FakeRolloutPolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.history_lengths = []

    async def next_action(self, **kwargs):
        self.history_lengths.append(len(kwargs["history"]))
        return self.actions.pop(0)


class RolloutTests(unittest.IsolatedAsyncioTestCase):
    async def test_rollout_waits_for_observation_before_next_action(self):
        policy = FakeRolloutPolicy(
            [
                Action("inspect", {"path": "<WORKSPACE>"}),
                Action("claim_done", {}),
            ]
        )
        executor = FakeExecutor()
        result = await run_closed_loop_rollout(
            instruction="Inspect, then finish.",
            available_tool_ids=["inspect", "claim_done", "distractor"],
            terminal_tool_ids={"claim_done"},
            policy=policy,
            executor=executor,
            evaluator=FakeEvaluator(),
            max_tool_calls=4,
        )
        self.assertEqual(policy.history_lengths, [0, 1])
        self.assertEqual(result["tool_call_count"], 2)
        self.assertEqual(result["termination_reason"], "claim_done")
        self.assertTrue(result["evaluator"]["passed"])

    async def test_failed_terminal_call_is_not_reported_as_claim_done(self):
        result = await run_closed_loop_rollout(
            instruction="Finish.",
            available_tool_ids=["claim_done"],
            terminal_tool_ids={"claim_done"},
            policy=FakeRolloutPolicy([Action("claim_done", {})] * 3),
            executor=FailingTerminalExecutor(),
            evaluator=None,
            max_tool_calls=3,
        )
        self.assertEqual(result["termination_reason"], "consecutive_tool_errors")
        self.assertEqual(result["execution_error_count"], 3)


class FakeTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(32))
        self.tool_names = ["tool-a", "tool-b"]
        self.tool_reserved_token_ids = [10, 11]
        self.token_id_to_tool_id = {10: 0, 11: 1}
        self.logit_bias_head = torch.nn.Linear(4, 2, bias=False)
        self.logit_bias_scale = 1.0

    def forward(self, input_ids, attention_mask, return_hidden_states=False):
        _ = attention_mask
        logits = self.bias.view(1, 1, -1).expand(
            input_ids.shape[0],
            input_ids.shape[1],
            -1,
        )
        hidden = torch.arange(
            input_ids.shape[0] * input_ids.shape[1] * 4,
            dtype=torch.float32,
        ).reshape(input_ids.shape[0], input_ids.shape[1], 4)
        return (logits, hidden) if return_hidden_states else logits

    def _get_logit_bias_scores(self, hidden_states):
        return self.logit_bias_head(hidden_states)


class TrainingLossTests(unittest.TestCase):
    def make_batch(self):
        return {
            "input_ids": torch.tensor(
                [
                    [0, 5, 10, 20, 2],
                    [0, 0, 6, 11, 2],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ]
            ),
            "labels": torch.tensor(
                [
                    [-100, -100, 10, 20, 2],
                    [-100, -100, -100, 11, 2],
                ]
            ),
            "available_tool_mask": torch.tensor(
                [
                    [True, False],
                    [False, True],
                ]
            ),
            "episode_weight": torch.tensor([0.5, 1.0]),
        }

    def test_each_step_contributes_one_assistant_start_site(self):
        model = FakeTrainingModel()
        batch = self.make_batch()
        _, hidden = model(
            batch["input_ids"],
            batch["attention_mask"],
            return_hidden_states=True,
        )
        _, targets, batch_indices, time_indices = gather_assistant_start_boundaries(
            hidden,
            batch["labels"],
            model,
        )
        self.assertEqual(targets.tolist(), [0, 1])
        self.assertEqual(batch_indices.tolist(), [0, 1])
        self.assertEqual(time_indices.tolist(), [1, 2])

    def test_tapmem_step_loss_is_finite_and_has_no_eoc_transition_site(self):
        losses = compute_stepwise_loss(
            model=FakeTrainingModel(),
            batch=self.make_batch(),
            use_logit_bias=True,
            use_logit_train_add=True,
            detach=True,
            logit_bias_loss_weight=0.1,
        )
        self.assertTrue(torch.isfinite(losses.total_loss))
        self.assertEqual(losses.routing_site_count, 2)
        self.assertEqual(losses.assistant_start_site_count, 2)
        self.assertEqual(losses.eoc_transition_site_count, 0)


if __name__ == "__main__":
    unittest.main()
