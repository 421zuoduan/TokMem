from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch.nn as nn

from compositional_intercode_bash import evaluate as evaluate_module
from compositional_intercode_bash.checkpoint import MEMORY_CHECKPOINT_SCHEMA
from compositional_intercode_bash.data_sources import EXPECTED_INTERCODE_COUNTS
from compositional_intercode_bash.intercode_runner import (
    INTERCODE_BASH_EPISODE_SCHEMA,
)
from compositional_intercode_bash.io_utils import (
    PACKAGE_ROOT,
    sha256_text,
    write_json,
)


def _tasks() -> list[dict]:
    source_hashes = {
        fs_id: f"{offset:064x}"
        for offset, fs_id in enumerate(EXPECTED_INTERCODE_COUNTS, start=1)
    }
    return [
        {
            "task_id": f"{fs_id}:{local_index:03d}",
            "fs_id": fs_id,
            "local_index": local_index,
            "query": f"query {fs_id} {local_index}",
            "gold": f"echo {fs_id}-{local_index}",
            "source_sha256": source_hashes[fs_id],
        }
        for fs_id, count in EXPECTED_INTERCODE_COUNTS.items()
        for local_index in range(count)
    ]


def _episode(task: dict, episode_inputs: dict, episode_key: str) -> dict:
    feedback = "ok"
    return {
        "schema": INTERCODE_BASH_EPISODE_SCHEMA,
        "task_id": task["task_id"],
        "fs_id": task["fs_id"],
        "local_index": task["local_index"],
        "query": task["query"],
        "max_turns": 10,
        "turns_taken": 1,
        "max_reward": 1.0,
        "max_released_reward": 1.0,
        "released_success": True,
        "success": True,
        "termination_reason": "success",
        "context_overflow": None,
        "reset_info": {},
        "turns": [
            {
                "turn": 1,
                "reward": 1.0,
                "released_reward": 1.0,
                "reward_info": {
                    "reward": {
                        "file_diff": 0.33,
                        "file_changes": 0.33,
                        "answer_similarity": 0.33,
                    },
                },
                "observation": feedback,
                "observation_record": {
                    "raw_utf8_bytes": 2,
                    "raw_characters": 2,
                    "sha256": sha256_text(feedback),
                    "truncated": False,
                    "feedback_utf8_bytes": 2,
                    "kept_head_utf8_bytes": 2,
                    "kept_tail_utf8_bytes": 0,
                    "feedback": feedback,
                },
                "dropped_history_turns": 0,
                "missing_terminator": False,
                "memory_bank_constraint_trigger_count": 0,
                "memory_bank_constraint_changed_token_count": 0,
            }
        ],
        "run_key": episode_key,
        "run_inputs": episode_inputs,
    }


def _parse_error_episode(
    task: dict,
    episode_inputs: dict,
    episode_key: str,
) -> dict:
    episode = _episode(task, episode_inputs, episode_key)
    status = "orphan"
    raw_status = status.encode("utf-8")
    failure = {
        "policy": evaluate_module.STATUS_PARSE_FAILURE_POLICY["name"],
        "exception_type": "IndexError",
        "exception_message": "list index out of range",
        "parser": evaluate_module.GIT_STATUS_POLICY["parser"],
        "failed_turn_reward": 0.0,
        "task_max_reward": "preserve_prior_official_max",
        "recreate_required": True,
        "status_side": "agent",
        "status_utf8_bytes": len(raw_status),
        "status_sha256": hashlib.sha256(raw_status).hexdigest(),
        "status_preview": status,
        "status_split_token_count": 1,
        "submit_origin": "runner_explicit_submit_after_model_action",
        "traceback_functions": [
            "submit",
            "step",
            "get_reward",
            "parse_status",
        ],
    }
    episode["turns"][0]["reward_info"] = {
        "official_status_parse_error": failure,
    }
    episode["turns"][0]["reward"] = 0.0
    episode["turns"][0]["released_reward"] = 0.0
    episode["max_reward"] = 0.0
    episode["max_released_reward"] = 0.0
    episode["released_success"] = False
    episode["success"] = False
    episode["termination_reason"] = "official_status_parse_error"
    episode["official_status_parse_error"] = dict(failure)
    episode["observed_max_reward_before_parse_error"] = 0.0
    episode[
        "observed_max_released_reward_before_parse_error"
    ] = 0.0
    return episode


class ShardSelectionTest(unittest.TestCase):
    def test_official_source_is_pinned_to_intercode_v1_0_1(self):
        identity = evaluate_module._official_source_identity(
            PACKAGE_ROOT.parent / "datasets" / "intercode"
        )
        self.assertEqual(
            identity["release"],
            evaluate_module.OFFICIAL_INTERCODE_RELEASE,
        )
        self.assertEqual(
            identity["verified_release_hashes"],
            evaluate_module.EXPECTED_OFFICIAL_INTERCODE_HASHES,
        )

    def test_tampered_official_source_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            release_root = Path(temporary)
            for relative_path in (
                evaluate_module.EXPECTED_OFFICIAL_INTERCODE_HASHES
            ):
                path = release_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "differs from the pinned official",
            ):
                evaluate_module._official_source_identity(release_root)

    def test_shared_prompt_creation_is_atomic_and_immutable(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            path = Path(temporary) / "prompt.txt"
            errors = []

            def create_prompt():
                try:
                    evaluate_module._ensure_shared_text_file(
                        path,
                        "fixed prompt\n",
                    )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=create_prompt) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "fixed prompt\n")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                evaluate_module._ensure_shared_text_file(
                    path,
                    "different prompt\n",
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "fixed prompt\n")

    def test_modulo_shards_cover_every_filesystem_exactly_once(self):
        tasks = _tasks()
        shards = [
            evaluate_module.select_shard_tasks(
                tasks,
                shard_count=4,
                shard_index=index,
            )
            for index in range(4)
        ]

        task_ids = [
            str(task["task_id"])
            for shard in shards
            for task in shard
        ]
        self.assertEqual(len(task_ids), 200)
        self.assertEqual(len(set(task_ids)), 200)
        self.assertEqual(set(task_ids), {str(task["task_id"]) for task in tasks})
        for index, shard in enumerate(shards):
            self.assertTrue(shard)
            self.assertTrue(
                all(int(task["local_index"]) % 4 == index for task in shard)
            )

        self.assertEqual(
            [
                sum(task["fs_id"] == "fs2" for task in shard)
                for shard in shards
            ],
            [14, 13, 13, 13],
        )
        self.assertEqual(
            [
                sum(task["fs_id"] == "fs4" for task in shard)
                for shard in shards
            ],
            [7, 7, 7, 6],
        )

    def test_cli_requires_a_complete_valid_shard_pair(self):
        base = [
            "--checkpoint",
            "checkpoint",
            "--output-dir",
            str(PACKAGE_ROOT / "test-output"),
        ]
        args = evaluate_module.parse_args(base)
        self.assertIsNone(args.shard_count)
        self.assertIsNone(args.shard_index)

        with self.assertRaises(SystemExit):
            evaluate_module.parse_args([*base, "--shard-count", "4"])
        with self.assertRaises(SystemExit):
            evaluate_module.parse_args(
                [*base, "--shard-count", "0", "--shard-index", "0"]
            )
        with self.assertRaises(SystemExit):
            evaluate_module.parse_args(
                [*base, "--shard-count", "4", "--shard-index", "4"]
            )
        with self.assertRaises(SystemExit):
            evaluate_module.parse_args(
                [
                    *base,
                    "--shard-count",
                    "4",
                    "--shard-index",
                    "0",
                    "--limit",
                    "1",
                ]
            )
        with self.assertRaises(SystemExit):
            evaluate_module.parse_args(
                [
                    *base,
                    "--shard-count",
                    "4",
                    "--shard-index",
                    "0",
                    "--task-id",
                    "fs1:000",
                ]
            )


class CompleteEpisodeMergeTest(unittest.TestCase):
    def _write_complete_set(
        self,
        episode_dir: Path,
    ) -> tuple[list[dict], dict[str, tuple[dict, str]]]:
        tasks = _tasks()
        expected = {}
        episode_dir.mkdir(parents=True)
        for task in tasks:
            inputs = {
                "task_id": task["task_id"],
                "shared_identity": "fixed",
                "max_turns": 10,
            }
            key = evaluate_module._episode_key(inputs)
            expected[str(task["task_id"])] = (inputs, key)
            write_json(
                episode_dir
                / evaluate_module._episode_filename(str(task["task_id"])),
                _episode(task, inputs, key),
            )
        return tasks, expected

    def test_complete_merge_loads_in_canonical_order(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            episodes = evaluate_module.load_complete_episode_set(
                episode_dir,
                tasks,
                expected,
            )

        self.assertEqual(len(episodes), 200)
        self.assertEqual(
            [episode["task_id"] for episode in episodes],
            [task["task_id"] for task in tasks],
        )

    def test_complete_merge_rejects_missing_extra_and_stale_identity(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            missing_path = episode_dir / "fs4_026.json"
            missing_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "exact episode file set"):
                evaluate_module.load_complete_episode_set(
                    episode_dir,
                    tasks,
                    expected,
                )

        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            write_json(episode_dir / "unexpected.json", {})
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                evaluate_module.load_complete_episode_set(
                    episode_dir,
                    tasks,
                    expected,
                )

        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            stale_path = episode_dir / "fs1_000.json"
            stale = json.loads(stale_path.read_text(encoding="utf-8"))
            stale["run_inputs"]["shared_identity"] = "stale"
            write_json(stale_path, stale)
            with self.assertRaisesRegex(RuntimeError, "run_inputs mismatch"):
                evaluate_module.load_complete_episode_set(
                    episode_dir,
                    tasks,
                    expected,
                )

    def test_complete_merge_rejects_missing_or_inconsistent_released_scores(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            episode_path = episode_dir / "fs1_000.json"
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            del episode["turns"][0]["released_reward"]
            write_json(episode_path, episode)
            with self.assertRaisesRegex(
                RuntimeError,
                "official turn reward",
            ):
                evaluate_module.load_complete_episode_set(
                    episode_dir,
                    tasks,
                    expected,
                )

        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            episode_path = episode_dir / "fs1_000.json"
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["max_released_reward"] = 0.5
            write_json(episode_path, episode)
            with self.assertRaisesRegex(
                RuntimeError,
                "max_released_reward mismatch",
            ):
                evaluate_module.load_complete_episode_set(
                    episode_dir,
                    tasks,
                    expected,
                )

        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            episode_dir = Path(temporary) / "episodes"
            tasks, expected = self._write_complete_set(episode_dir)
            episode_path = episode_dir / "fs1_000.json"
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["released_success"] = False
            write_json(episode_path, episode)
            with self.assertRaisesRegex(
                RuntimeError,
                "released_success mismatch",
            ):
                evaluate_module.load_complete_episode_set(
                    episode_dir,
                    tasks,
                    expected,
                )

    def test_episode_validation_rejects_inconsistent_memory_gate_counts(self):
        task = _tasks()[0]
        episode = _episode(task, {"max_turns": 10}, "key")
        episode["turns"][0][
            "memory_bank_constraint_changed_token_count"
        ] = 1

        with self.assertRaisesRegex(RuntimeError, "turn diagnostics"):
            evaluate_module._validate_episode_result(
                episode,
                episode_path=Path("episode.json"),
            )

    def test_episode_validation_accepts_official_status_parse_failure(self):
        task = _tasks()[0]
        episode = _parse_error_episode(task, {"max_turns": 10}, "key")
        parse_error_turn = episode["turns"][0]
        parse_error_turn["turn"] = 2
        first_turn = _episode(task, {"max_turns": 10}, "key")["turns"][0]
        first_turn["reward"] = 0.67
        first_turn["released_reward"] = 0.67
        first_turn["reward_info"]["reward"] = {
            "file_diff": 0.33,
            "file_changes": 0.0,
            "answer_similarity": 0.33,
        }
        episode["turns"] = [first_turn, parse_error_turn]
        episode["turns_taken"] = 2
        episode["max_reward"] = 0.67
        episode["max_released_reward"] = 0.67
        episode["observed_max_reward_before_parse_error"] = 0.67
        episode[
            "observed_max_released_reward_before_parse_error"
        ] = 0.67

        evaluate_module._validate_episode_result(
            episode,
            episode_path=Path("episode.json"),
        )
        diagnostics = evaluate_module.episode_diagnostics([episode])
        self.assertEqual(
            diagnostics["official_status_parse_error_episodes"],
            1,
        )
        self.assertEqual(
            diagnostics["termination_reasons"][
                "official_status_parse_error"
            ],
            1,
        )

        nonzero_failed_turn = json.loads(json.dumps(episode))
        nonzero_failed_turn["turns"][-1]["reward"] = 0.2
        nonzero_failed_turn["turns"][-1]["released_reward"] = 0.2
        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid official status-parser failure",
        ):
            evaluate_module._validate_episode_result(
                nonzero_failed_turn,
                episode_path=Path("episode.json"),
            )

        cleared_prior_max = json.loads(json.dumps(episode))
        cleared_prior_max["max_reward"] = 0.0
        cleared_prior_max["max_released_reward"] = 0.0
        with self.assertRaisesRegex(
            RuntimeError,
            "max_reward mismatch",
        ):
            evaluate_module._validate_episode_result(
                cleared_prior_max,
                episode_path=Path("episode.json"),
            )

        impossible = json.loads(json.dumps(episode))
        impossible["turns"][0]["reward"] = 1.0
        impossible["turns"][0]["released_reward"] = 1.0
        impossible["turns"][0]["reward_info"]["reward"][
            "file_changes"
        ] = 0.33
        impossible[
            "observed_max_reward_before_parse_error"
        ] = 1.0
        with self.assertRaisesRegex(
            RuntimeError,
            "status-parser failure",
        ):
            evaluate_module._validate_episode_result(
                impossible,
                episode_path=Path("episode.json"),
            )

        bad_policy = json.loads(json.dumps(episode))
        bad_policy["turns"][-1]["reward_info"][
            "official_status_parse_error"
        ]["policy"] = "different-policy"
        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid official status-parser failure",
        ):
            evaluate_module._validate_episode_result(
                bad_policy,
                episode_path=Path("episode.json"),
            )

        bad_type = json.loads(json.dumps(episode))
        bad_type["official_status_parse_error"]["exception_type"] = (
            "ValueError"
        )
        bad_type["turns"][-1]["reward_info"][
            "official_status_parse_error"
        ]["exception_type"] = "ValueError"
        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid official status-parser failure",
        ):
            evaluate_module._validate_episode_result(
                bad_type,
                episode_path=Path("episode.json"),
            )

        non_mapping = json.loads(json.dumps(episode))
        non_mapping["turns"][-1]["reward_info"][
            "official_status_parse_error"
        ] = "not-a-mapping"
        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid official status-parser failure",
        ):
            evaluate_module._validate_episode_result(
                non_mapping,
                episode_path=Path("episode.json"),
            )

        episode["official_status_parse_error"]["exception_type"] = (
            "ValueError"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Inconsistent official status-parser failure",
        ):
            evaluate_module._validate_episode_result(
                episode,
                episode_path=Path("episode.json"),
            )

    def test_episode_diagnostics_exposes_overflow_and_truncation(self):
        task = _tasks()[0]
        episode = _episode(task, {"max_turns": 10}, "key")
        episode["success"] = False
        episode["max_reward"] = 0.4
        episode["termination_reason"] = "context_overflow"
        episode["context_overflow"] = {
            "required_tokens": 129,
            "context_limit": 128,
            "retained_turns": 1,
            "next_turn": 2,
        }
        episode["turns"][0]["reward"] = 0.4
        episode["turns"][0]["observation_record"]["truncated"] = True
        episode["turns"][0]["observation_record"]["raw_utf8_bytes"] = 5000
        episode["turns"][0]["dropped_history_turns"] = 2
        diagnostics = evaluate_module.episode_diagnostics([episode])

        self.assertEqual(diagnostics["context_overflow_episodes"], 1)
        self.assertEqual(diagnostics["context_overflow_rate"], 1.0)
        self.assertEqual(diagnostics["observation_truncated_turns"], 1)
        self.assertEqual(diagnostics["observation_truncated_episodes"], 1)
        self.assertEqual(diagnostics["max_raw_observation_utf8_bytes"], 5000)
        self.assertEqual(diagnostics["dropped_history_pairs"], 2)


class EvaluateShardIntegrationTest(unittest.TestCase):
    def test_shard_recreates_after_status_parse_error_then_resume_merges(self):
        tasks = _tasks()
        metadata = {
            "schema": MEMORY_CHECKPOINT_SCHEMA,
            "method": "tapmem",
            "formal_ready": True,
            "base_model_name": "/fixed/base",
            "base_model_revision": None,
            "base_model_identity": {"identity": "fixed-base"},
            "view_metadata": {"data_seed": 1729},
            "training_config": {"model_seed": 42},
        }
        image_details = {
            fs_id: {
                "requested_name": f"image-{fs_id}",
                "image_id": f"sha256:image-{fs_id}",
                "filesystem_label": fs_id,
                "benchmark_label": "nl2bash",
            }
            for fs_id in EXPECTED_INTERCODE_COUNTS
        }

        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            output = root / "evaluation"
            tokenizer_dir = checkpoint / "tokenizer"
            tokenizer_dir.mkdir(parents=True)
            write_json(checkpoint / "checkpoint.json", metadata)
            (checkpoint / "trainable.safetensors").write_bytes(b"weights")
            (tokenizer_dir / "tokenizer.json").write_text(
                "{}",
                encoding="utf-8",
            )

            model = nn.Linear(1, 1)
            tokenizer = object()
            environment_events = []
            environments = []

            class FakeReusableEnvironment:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.reset_indices = []
                    self.close_calls = 0
                    environments.append(self)
                    environment_events.append(
                        ("construct", kwargs["image_name"])
                    )

                def __enter__(self):
                    return self

                def __exit__(self, _exc_type, _exc_value, _traceback):
                    self.close()
                    return False

                def reset(self, index):
                    self.reset_indices.append(index)
                    environment_events.append(
                        ("reset", self.kwargs["image_name"], index)
                    )
                    return f"query-{index}", {}

                def close(self):
                    self.close_calls += 1
                    environment_events.append(
                        ("close", self.kwargs["image_name"])
                    )

            def fake_episode(**kwargs):
                task = kwargs["task"]
                kwargs["environment"].reset(task["local_index"])
                if task["task_id"] == "fs1:000":
                    episode = _parse_error_episode(task, {}, "")
                else:
                    episode = _episode(task, {}, "")
                episode.pop("run_key")
                episode.pop("run_inputs")
                episode["max_turns"] = kwargs["max_turns"]
                return episode

            common_args = {
                "checkpoint": str(checkpoint),
                "intercode_root": str(root / "intercode"),
                "output_dir": str(output),
                "max_turns": 10,
                "max_new_tokens": 16,
                "context_limit": 128,
                "image_map": None,
                "fs_id": None,
                "limit": None,
                "task_id": None,
                "device": "cpu",
                "dtype": "float32",
                "allow_download": False,
                "allow_exploratory_checkpoint": False,
            }

            with (
                mock.patch.object(
                    evaluate_module,
                    "load_checkpoint",
                    return_value=(model, tokenizer, metadata),
                ),
                mock.patch.object(
                    evaluate_module,
                    "load_intercode_tasks",
                    return_value=tasks,
                ),
                mock.patch.object(
                    evaluate_module,
                    "resolve_docker_images",
                    return_value=image_details,
                ),
                mock.patch.object(
                    evaluate_module,
                    "docker_runtime_identity",
                    return_value={"formal_isolation": True},
                ),
                mock.patch.object(
                    evaluate_module,
                    "_official_source_identity",
                    return_value={"identity": "official-source"},
                ),
                mock.patch.object(
                    evaluate_module,
                    "_runtime_identity",
                    return_value={"device_argument": "cpu"},
                ),
                mock.patch.object(
                    evaluate_module,
                    "ReusableOfficialBashEnv",
                    side_effect=FakeReusableEnvironment,
                ) as environment_factory,
                mock.patch.object(
                    evaluate_module,
                    "run_try_again_episode",
                    side_effect=fake_episode,
                ) as runner,
            ):
                shard_args = SimpleNamespace(
                    **common_args,
                    shard_count=1,
                    shard_index=0,
                    resume=False,
                )
                shard_summary = evaluate_module.evaluate(shard_args)

                self.assertEqual(
                    shard_summary["schema"],
                    evaluate_module.SHARD_SUMMARY_SCHEMA,
                )
                self.assertEqual(shard_summary["tasks"], 200)
                self.assertTrue(shard_summary["evaluation_complete"])
                self.assertTrue(shard_summary["global_input_paper_ready"])
                self.assertIsNone(
                    shard_summary["global_result_paper_ready"]
                )
                self.assertFalse(shard_summary["paper_ready"])
                self.assertFalse(
                    (output / "summary_10_turn.json").exists()
                )
                self.assertTrue(
                    (
                        output
                        / "shard_summary_10_turn_000_of_001.json"
                    ).is_file()
                )
                self.assertEqual(runner.call_count, 200)
                self.assertEqual(environment_factory.call_count, 5)
                self.assertEqual(len(environments), 5)
                self.assertEqual(
                    [environment.close_calls for environment in environments],
                    [1, 1, 1, 1, 1],
                )
                self.assertEqual(
                    [environment.reset_indices for environment in environments],
                    [
                        [0],
                        list(range(1, EXPECTED_INTERCODE_COUNTS["fs1"])),
                        list(range(EXPECTED_INTERCODE_COUNTS["fs2"])),
                        list(range(EXPECTED_INTERCODE_COUNTS["fs3"])),
                        list(range(EXPECTED_INTERCODE_COUNTS["fs4"])),
                    ],
                )
                self.assertEqual(
                    shard_summary["official_status_parse_error_episodes"],
                    1,
                )
                self.assertEqual(
                    [
                        event
                        for event in environment_events
                        if event[0] != "reset"
                    ],
                    [
                        ("construct", "sha256:image-fs1"),
                        ("close", "sha256:image-fs1"),
                        ("construct", "sha256:image-fs1"),
                        ("close", "sha256:image-fs1"),
                        ("construct", "sha256:image-fs2"),
                        ("close", "sha256:image-fs2"),
                        ("construct", "sha256:image-fs3"),
                        ("close", "sha256:image-fs3"),
                        ("construct", "sha256:image-fs4"),
                        ("close", "sha256:image-fs4"),
                    ],
                )

                first_episode = json.loads(
                    (
                        output
                        / "episodes_10_turn"
                        / "fs1_000.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertTrue(
                    first_episode["run_inputs"]["evaluation_complete"]
                )
                self.assertTrue(first_episode["run_inputs"]["paper_ready"])
                self.assertEqual(
                    first_episode["run_inputs"][
                        "environment_lifecycle_policy"
                    ],
                    evaluate_module.ENVIRONMENT_LIFECYCLE_POLICY,
                )

                runner.reset_mock()
                merge_args = SimpleNamespace(
                    **common_args,
                    shard_count=None,
                    shard_index=None,
                    resume=True,
                )
                formal_summary = evaluate_module.evaluate(merge_args)
                self.assertEqual(runner.call_count, 0)
                self.assertEqual(environment_factory.call_count, 5)

                (
                    output
                    / "episodes_10_turn"
                    / "fs2_010.json"
                ).unlink()
                runner.reset_mock()
                resumed_summary = evaluate_module.evaluate(merge_args)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(
                    runner.call_args.kwargs["task"]["task_id"],
                    "fs2:010",
                )
                self.assertEqual(environment_factory.call_count, 6)
                self.assertEqual(environments[-1].reset_indices, [10])
                self.assertEqual(environments[-1].close_calls, 1)

            self.assertEqual(
                formal_summary["schema"],
                evaluate_module.FORMAL_SUMMARY_SCHEMA,
            )
            self.assertEqual(formal_summary["tasks"], 200)
            self.assertEqual(formal_summary["successes"], 199)
            self.assertEqual(
                formal_summary["official_status_parse_error_episodes"],
                1,
            )
            self.assertTrue(formal_summary["evaluation_complete"])
            self.assertTrue(formal_summary["paper_ready"])
            self.assertTrue(resumed_summary["paper_ready"])
            self.assertTrue((output / "summary_10_turn.json").is_file())


if __name__ == "__main__":
    unittest.main()
