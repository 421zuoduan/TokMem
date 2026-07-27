"""Evaluate a memory checkpoint on all four InterCode-Bash file systems."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

from .base_text import (
    BASE_TEXT_ARTIFACT_SCHEMA,
    TOOL_CATALOG_FILENAME,
    TOOL_CATALOG_REPORT_FILENAME,
    compose_base_text_system_prompt,
    load_base_text_artifact,
)
from .checkpoint import MEMORY_CHECKPOINT_SCHEMA, load_checkpoint
from .data_sources import EXPECTED_INTERCODE_COUNTS, load_intercode_tasks
from .intercode_runner import (
    INTERCODE_BASH_EPISODE_SCHEMA,
    OBSERVATION_FEEDBACK_POLICY,
    OBSERVATION_HEAD_UTF8_BYTES,
    OBSERVATION_MAX_UTF8_BYTES,
    OBSERVATION_TAIL_UTF8_BYTES,
    run_try_again_episode,
)
from .io_utils import (
    PACKAGE_ROOT,
    ensure_output_path,
    read_json,
    sha256_directory,
    sha256_file,
    sha256_text,
    stable_id,
    write_json,
)
from .official_env import (
    CONTAINER_READY_COMMAND,
    CONTAINER_READY_POLL_INTERVAL_SECONDS,
    CONTAINER_READY_TIMEOUT_SECONDS,
    DOCKER_API_TIMEOUT_SECONDS,
    GIT_RESET_COMMAND,
    GIT_STATUS_POLICY,
    SCORER_POLICY,
    STATUS_PARSE_FAILURE_POLICY,
    ReusableOfficialBashEnv,
    docker_runtime_identity,
    resolve_docker_images,
)
from .training_data import SYSTEM_PROMPT


DEFAULT_IMAGES = {
    "fs1": "intercode-nl2bash-fs1",
    "fs2": "intercode-nl2bash-fs2",
    "fs3": "intercode-nl2bash-fs3",
    "fs4": "intercode-nl2bash-fs4",
}
RUNNER_PROTOCOL = "intercode_bash_try_again_official_reward_v13"
FORMAL_SUMMARY_SCHEMA = "intercode_bash_eval_summary_v11"
SHARD_SUMMARY_SCHEMA = "intercode_bash_eval_shard_summary_v8"
SHARD_ASSIGNMENT = "per_filesystem_local_index_modulo"
OFFICIAL_INTERCODE_RELEASE = {
    "version": "v1.0.1",
    "commit": "7c311cce135bd306ff47549def69f3fb0944d35d",
    "archive_sha256": (
        "c3f93e58fd0d76a18baf647a8f342346d2b67b527e503cd0452f5ebc8cf51790"
    ),
}
EXPECTED_OFFICIAL_INTERCODE_HASHES = {
    "intercode/envs/ic_env.py": (
        "4cca584431396965faa6652bff8ae7aa33b1c3d7a47c11dbc0a47f467c60eda4"
    ),
    "intercode/envs/bash/bash_env.py": (
        "a2b6ac7ace6298b7e5ec45813332fbaeb5f3f69dd4cac0d4f7f39bbfb1667dea"
    ),
    "intercode/utils/utils.py": (
        "0d709adbdde1690699e8be03aa41118bc71967ee65b3c3f8d607cc4dd43c3fa0"
    ),
    "experiments/eval_n_turn.py": (
        "20a871d2f5458b73d14169627ccc88773cd884365d7fc45f320df0a444af77eb"
    ),
}
OBSERVATION_POLICY = {
    "name": OBSERVATION_FEEDBACK_POLICY,
    "max_utf8_bytes": OBSERVATION_MAX_UTF8_BYTES,
    "head_window_utf8_bytes": OBSERVATION_HEAD_UTF8_BYTES,
    "tail_window_utf8_bytes": OBSERVATION_TAIL_UTF8_BYTES,
}
CONTAINER_STARTUP_POLICY = {
    "name": "active_running_exec_probe_v1",
    "command": CONTAINER_READY_COMMAND,
    "poll_deadline_seconds": CONTAINER_READY_TIMEOUT_SECONDS,
    "poll_interval_seconds": CONTAINER_READY_POLL_INTERVAL_SECONDS,
    "docker_api_timeout_seconds": DOCKER_API_TIMEOUT_SECONDS,
}
ENVIRONMENT_LIFECYCLE_POLICY = {
    "name": "official_reset_with_parse_failure_recreate_v1",
    "reuse_scope": "one_agent_eval_pair_per_filesystem_per_process",
    "episode_reset": "BashEnv.reset(local_index)",
    "agent_reset_command": GIT_RESET_COMMAND,
    "evaluation_reset": "BashEnv.get_reward_before_gold_execution",
    "parse_failure_recovery": "close_pair_and_recreate_from_source_image",
    "final_cleanup": "close_and_remove_pair_after_filesystem_batch",
}


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _official_source_identity(intercode_root: str | Path) -> dict[str, Any]:
    release_root = Path(intercode_root).resolve()
    source_root = release_root / "intercode"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Official InterCode Python source is missing: {source_root}"
        )
    hashes = {
        str(path.relative_to(source_root)): sha256_file(path)
        for path in sorted(source_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if not hashes:
        raise ValueError(f"No official InterCode Python files found under {source_root}")
    verified_hashes = {}
    for relative_path, expected_sha256 in (
        EXPECTED_OFFICIAL_INTERCODE_HASHES.items()
    ):
        path = release_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Required InterCode v1.0.1 source is missing: {path}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "InterCode source differs from the pinned official "
                f"v1.0.1 release: {relative_path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        verified_hashes[relative_path] = actual_sha256
    return {
        "release": OFFICIAL_INTERCODE_RELEASE,
        "source_root": str(source_root),
        "python_file_hashes": hashes,
        "verified_release_hashes": verified_hashes,
        "identity": stable_id(
            "INTERCODE_OFFICIAL_SOURCE_V2",
            json.dumps(
                OFFICIAL_INTERCODE_RELEASE,
                sort_keys=True,
                separators=(",", ":"),
            ),
            json.dumps(hashes, sort_keys=True, separators=(",", ":")),
            json.dumps(
                verified_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    }


def _runtime_identity(device: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "docker": _package_version("docker"),
        "safetensors": _package_version("safetensors"),
        "device_argument": device,
        "cuda_runtime": torch.version.cuda,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        value["cuda_device_name"] = torch.cuda.get_device_name(index)
        value["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(index)
        )
    return value


def evaluation_selection_status(
    all_tasks: Iterable[dict[str, Any]],
    selected_tasks: Iterable[dict[str, Any]],
) -> tuple[dict[str, int], bool]:
    all_tasks = list(all_tasks)
    selected_tasks = list(selected_tasks)
    counts = {
        fs_id: sum(task["fs_id"] == fs_id for task in selected_tasks)
        for fs_id in EXPECTED_INTERCODE_COUNTS
    }
    complete = (
        {str(task["task_id"]) for task in selected_tasks}
        == {str(task["task_id"]) for task in all_tasks}
        and len(selected_tasks) == len(all_tasks) == 200
        and counts == EXPECTED_INTERCODE_COUNTS
    )
    return counts, complete


def shard_configuration(
    shard_count: int | None,
    shard_index: int | None,
) -> tuple[int, int] | None:
    """Validate an optional zero-based formal shard specification."""

    if shard_count is None and shard_index is None:
        return None
    if shard_count is None or shard_index is None:
        raise ValueError("--shard-count and --shard-index must be provided together")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "--shard-index must satisfy 0 <= index < shard-count"
        )
    return int(shard_count), int(shard_index)


def select_shard_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    """Assign every filesystem locally by ``local_index % shard_count``."""

    specification = shard_configuration(shard_count, shard_index)
    if specification is None:
        raise AssertionError("A shard specification was expected")
    count, index = specification
    return [
        task
        for task in tasks
        if int(task["local_index"]) % count == index
    ]


def group_tasks_by_filesystem(
    tasks: Iterable[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group tasks stably so one official environment can serve each filesystem."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        fs_id = str(task["fs_id"])
        groups.setdefault(fs_id, []).append(task)
    return list(groups.items())


def _episode_filename(task_id: str) -> str:
    return f"{task_id.replace(':', '_')}.json"


def _ensure_shared_text_file(path: Path, contents: str) -> None:
    """Create an immutable shared file without exposing partial shard writes."""

    payload = contents.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    if path.read_bytes() != payload:
        raise RuntimeError(
            f"Existing evaluation prompt differs from this run: {path}"
        )


def _episode_key(episode_inputs: dict[str, Any]) -> str:
    return stable_id(
        "INTERCODE_BASH_EPISODE_V9",
        json.dumps(episode_inputs, sort_keys=True, separators=(",", ":")),
    )


def _validate_episode_result(
    episode: dict[str, Any],
    *,
    episode_path: Path,
) -> None:
    if episode.get("schema") != INTERCODE_BASH_EPISODE_SCHEMA:
        raise RuntimeError(
            f"Episode schema mismatch in {episode_path}: "
            f"{episode.get('schema')!r} != {INTERCODE_BASH_EPISODE_SCHEMA!r}"
        )
    turns = episode.get("turns")
    if not isinstance(turns, list):
        raise RuntimeError(f"Episode turns are not a list in {episode_path}")
    turns_taken = episode.get("turns_taken")
    if not isinstance(turns_taken, int) or turns_taken != len(turns):
        raise RuntimeError(
            f"Episode turns_taken mismatch in {episode_path}: "
            f"{turns_taken!r} != {len(turns)}"
        )
    max_turns = episode.get("max_turns")
    if not isinstance(max_turns, int) or max_turns <= 0:
        raise RuntimeError(f"Invalid max_turns in {episode_path}: {max_turns!r}")

    termination_reason = episode.get("termination_reason")
    if termination_reason not in {
        "success",
        "max_turns",
        "context_overflow",
        "official_status_parse_error",
    }:
        raise RuntimeError(
            f"Invalid termination_reason in {episode_path}: "
            f"{termination_reason!r}"
        )
    rewards: list[float] = []
    released_rewards: list[float] = []
    status_parse_error_turns: list[int] = []
    for turn_index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict) or turn.get("turn") != turn_index:
            raise RuntimeError(
                f"Episode turn numbering mismatch in {episode_path} at "
                f"position {turn_index}"
            )
        try:
            reward = float(turn["reward"])
            released_reward = float(turn["released_reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid official turn reward in {episode_path} "
                f"at turn {turn_index}"
            ) from exc
        if (
            not math.isfinite(reward)
            or not math.isfinite(released_reward)
            or not 0.0 <= reward <= 1.0
            or not 0.0 <= released_reward <= 1.0
        ):
            raise RuntimeError(
                f"Out-of-range official turn reward in {episode_path} "
                f"at turn {turn_index}"
            )
        rewards.append(reward)
        released_rewards.append(released_reward)
        reward_info = turn.get("reward_info")
        if not isinstance(reward_info, dict):
            raise RuntimeError(
                f"Missing reward_info in {episode_path} at turn {turn_index}"
            )
        status_parse_error = reward_info.get(
            "official_status_parse_error"
        )
        if status_parse_error is not None:
            if not isinstance(status_parse_error, dict):
                raise RuntimeError(
                    f"Invalid official status-parser failure in "
                    f"{episode_path} at turn {turn_index}"
                )
            status_sha256 = status_parse_error.get("status_sha256")
            traceback_functions = status_parse_error.get(
                "traceback_functions"
            )
            if (
                status_parse_error.get("policy")
                != STATUS_PARSE_FAILURE_POLICY["name"]
                or status_parse_error.get("exception_type") != "IndexError"
                or status_parse_error.get("parser")
                != GIT_STATUS_POLICY["parser"]
                or status_parse_error.get("failed_turn_reward") != 0.0
                or status_parse_error.get("task_max_reward")
                != "preserve_prior_official_max"
                or status_parse_error.get("recreate_required") is not True
                or status_parse_error.get("status_side") != "agent"
                or not isinstance(
                    status_parse_error.get("status_utf8_bytes"),
                    int,
                )
                or int(status_parse_error["status_utf8_bytes"]) <= 0
                or not isinstance(status_sha256, str)
                or len(status_sha256) != 64
                or not isinstance(
                    status_parse_error.get("status_preview"),
                    str,
                )
                or len(status_parse_error["status_preview"])
                > 203
                or not isinstance(
                    status_parse_error.get("status_split_token_count"),
                    int,
                )
                or int(status_parse_error["status_split_token_count"]) <= 0
                or int(status_parse_error["status_split_token_count"]) % 2
                != 1
                or status_parse_error.get("submit_origin")
                != "runner_explicit_submit_after_model_action"
                or not isinstance(traceback_functions, list)
                or traceback_functions[-3:]
                != ["step", "get_reward", "parse_status"]
                or reward != 0.0
                or released_reward != 0.0
            ):
                raise RuntimeError(
                    f"Invalid official status-parser failure in "
                    f"{episode_path} at turn {turn_index}"
                )
            try:
                int(status_sha256, 16)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid status-parser SHA-256 in {episode_path} "
                    f"at turn {turn_index}"
                ) from exc
            status_parse_error_turns.append(turn_index)
        else:
            try:
                official_components = reward_info["reward"]
                file_diff = float(official_components["file_diff"])
                file_changes = float(official_components["file_changes"])
                answer_similarity = float(
                    official_components["answer_similarity"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Incomplete official scorer provenance in {episode_path} "
                    f"at turn {turn_index}"
                ) from exc
            component_values = (
                file_diff,
                file_changes,
                answer_similarity,
            )
            if (
                reward != released_reward
                or any(
                    not math.isfinite(value) or not 0.0 <= value <= 0.33
                    for value in component_values
                )
                or not math.isclose(
                    reward,
                    0.01 + file_diff + file_changes + answer_similarity,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    f"Official scorer provenance mismatch in {episode_path} "
                    f"at turn {turn_index}"
                )
        record = turn.get("observation_record")
        required_record_fields = {
            "raw_utf8_bytes",
            "raw_characters",
            "sha256",
            "truncated",
            "feedback_utf8_bytes",
            "kept_head_utf8_bytes",
            "kept_tail_utf8_bytes",
            "feedback",
        }
        if (
            not isinstance(record, dict)
            or set(record) != required_record_fields
        ):
            raise RuntimeError(
                f"Invalid observation_record in {episode_path} at "
                f"turn {turn_index}"
            )
        feedback = record["feedback"]
        if not isinstance(feedback, str) or turn.get("observation") != feedback:
            raise RuntimeError(
                f"Observation feedback mismatch in {episode_path} at "
                f"turn {turn_index}"
            )
        feedback_utf8_bytes = len(feedback.encode("utf-8"))
        integer_record_fields = (
            "raw_utf8_bytes",
            "raw_characters",
            "feedback_utf8_bytes",
            "kept_head_utf8_bytes",
            "kept_tail_utf8_bytes",
        )
        if any(
            not isinstance(record[field], int) or record[field] < 0
            for field in integer_record_fields
        ):
            raise RuntimeError(
                f"Invalid observation byte counts in {episode_path} at "
                f"turn {turn_index}"
            )
        if record["feedback_utf8_bytes"] != feedback_utf8_bytes:
            raise RuntimeError(
                f"Observation feedback byte count mismatch in {episode_path} "
                f"at turn {turn_index}"
            )
        if (
            not isinstance(record["truncated"], bool)
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
        ):
            raise RuntimeError(
                f"Invalid observation provenance in {episode_path} at "
                f"turn {turn_index}"
            )
        try:
            int(record["sha256"], 16)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid observation SHA-256 in {episode_path} at "
                f"turn {turn_index}"
            ) from exc
        if record["truncated"]:
            if (
                record["raw_utf8_bytes"] <= OBSERVATION_MAX_UTF8_BYTES
                or feedback_utf8_bytes > OBSERVATION_MAX_UTF8_BYTES
                or record["kept_head_utf8_bytes"]
                > OBSERVATION_HEAD_UTF8_BYTES
                or record["kept_tail_utf8_bytes"]
                > OBSERVATION_TAIL_UTF8_BYTES
            ):
                raise RuntimeError(
                    f"Invalid truncated observation bounds in {episode_path} "
                    f"at turn {turn_index}"
                )
        elif (
            record["raw_utf8_bytes"] != feedback_utf8_bytes
            or record["raw_characters"] != len(feedback)
            or record["sha256"] != sha256_text(feedback)
            or record["kept_head_utf8_bytes"] != feedback_utf8_bytes
            or record["kept_tail_utf8_bytes"] != 0
        ):
            raise RuntimeError(
                f"Invalid untruncated observation provenance in {episode_path} "
                f"at turn {turn_index}"
            )
        if (
            not isinstance(turn.get("dropped_history_turns"), int)
            or int(turn["dropped_history_turns"]) < 0
            or not isinstance(turn.get("missing_terminator"), bool)
            or not isinstance(
                turn.get("memory_bank_constraint_trigger_count"),
                int,
            )
            or int(turn["memory_bank_constraint_trigger_count"]) < 0
            or not isinstance(
                turn.get("memory_bank_constraint_changed_token_count"),
                int,
            )
            or int(turn["memory_bank_constraint_changed_token_count"]) < 0
            or int(turn["memory_bank_constraint_changed_token_count"])
            > int(turn["memory_bank_constraint_trigger_count"])
        ):
            raise RuntimeError(
                f"Invalid turn diagnostics in {episode_path} at "
                f"turn {turn_index}"
            )

    has_status_parse_error = bool(status_parse_error_turns)
    if has_status_parse_error:
        if (
            status_parse_error_turns != [turns_taken]
            or termination_reason != "official_status_parse_error"
            or episode.get("official_status_parse_error")
            != turns[-1]["reward_info"]["official_status_parse_error"]
            or any(reward == 1.0 for reward in rewards)
        ):
            raise RuntimeError(
                f"Inconsistent official status-parser failure in {episode_path}"
            )
        expected_success = False
        prior_rewards = rewards[:-1]
        prior_released_rewards = released_rewards[:-1]
        expected_max_reward = max(prior_rewards, default=0.0)
        expected_max_released_reward = max(
            prior_released_rewards,
            default=0.0,
        )
        if (
            episode.get("observed_max_reward_before_parse_error")
            != max(prior_rewards, default=0.0)
            or episode.get(
                "observed_max_released_reward_before_parse_error"
            )
            != max(prior_released_rewards, default=0.0)
        ):
            raise RuntimeError(
                f"Incorrect pre-error reward audit in {episode_path}"
            )
    else:
        if termination_reason == "official_status_parse_error":
            raise RuntimeError(
                f"Missing official status-parser failure in {episode_path}"
            )
        expected_success = any(reward == 1.0 for reward in rewards)
        expected_max_reward = max(rewards, default=0.0)
        expected_max_released_reward = max(
            released_rewards,
            default=0.0,
        )
    if (
        not isinstance(episode.get("success"), bool)
        or episode["success"] != expected_success
    ):
        raise RuntimeError(f"Episode success mismatch in {episode_path}")
    try:
        max_reward = float(episode["max_reward"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid max_reward in {episode_path}") from exc
    if max_reward != expected_max_reward:
        raise RuntimeError(
            f"Episode max_reward mismatch in {episode_path}: "
            f"{max_reward} != {expected_max_reward}"
        )
    try:
        max_released_reward = float(episode["max_released_reward"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid max_released_reward in {episode_path}"
        ) from exc
    if (
        not math.isfinite(max_released_reward)
        or max_released_reward != expected_max_released_reward
    ):
        raise RuntimeError(
            f"Episode max_released_reward mismatch in {episode_path}: "
            f"{max_released_reward} != {expected_max_released_reward}"
        )
    expected_released_success = (
        False
        if has_status_parse_error
        else any(reward == 1.0 for reward in released_rewards)
    )
    if (
        not isinstance(episode.get("released_success"), bool)
        or episode["released_success"] != expected_released_success
    ):
        raise RuntimeError(
            f"Episode released_success mismatch in {episode_path}"
        )

    overflow = episode.get("context_overflow")
    if termination_reason == "success":
        if not expected_success or turns_taken > max_turns or overflow is not None:
            raise RuntimeError(f"Invalid success termination in {episode_path}")
    elif termination_reason == "max_turns":
        if expected_success or turns_taken != max_turns or overflow is not None:
            raise RuntimeError(f"Invalid max-turn termination in {episode_path}")
    elif termination_reason == "context_overflow":
        if (
            expected_success
            or not 0 < turns_taken < max_turns
            or not isinstance(overflow, dict)
            or int(overflow.get("required_tokens", 0))
            <= int(overflow.get("context_limit", 0))
        ):
            raise RuntimeError(
                f"Invalid context-overflow termination in {episode_path}"
            )
    else:
        if (
            not has_status_parse_error
            or not 0 < turns_taken <= max_turns
            or overflow is not None
        ):
            raise RuntimeError(
                f"Invalid status-parser-error termination in {episode_path}"
            )


def _validate_episode_identity(
    episode: dict[str, Any],
    *,
    task: dict[str, Any],
    episode_inputs: dict[str, Any],
    episode_key: str,
    episode_path: Path,
) -> None:
    _validate_episode_result(episode, episode_path=episode_path)
    if episode.get("run_key") != episode_key:
        raise RuntimeError(
            f"Episode identity mismatch in {episode_path}: "
            f"expected run_key={episode_key}, "
            f"found={episode.get('run_key')}"
        )
    if episode.get("run_inputs") != episode_inputs:
        raise RuntimeError(
            f"Episode run_inputs mismatch in {episode_path}"
        )
    if episode.get("max_turns") != episode_inputs["max_turns"]:
        raise RuntimeError(
            f"Episode max_turns mismatch in {episode_path}: "
            f"{episode.get('max_turns')!r} != {episode_inputs['max_turns']!r}"
        )
    if episode_inputs.get("method") != "tapmem" and any(
        int(turn["memory_bank_constraint_trigger_count"]) != 0
        or int(turn["memory_bank_constraint_changed_token_count"]) != 0
        for turn in episode["turns"]
    ):
        raise RuntimeError(
            f"Non-TapMem episode contains memory-bank constraint events in "
            f"{episode_path}"
        )
    expected_fields = {
        "task_id": task["task_id"],
        "fs_id": task["fs_id"],
        "local_index": int(task["local_index"]),
        "query": task["query"],
    }
    for field, expected in expected_fields.items():
        if episode.get(field) != expected:
            raise RuntimeError(
                f"Episode {field} mismatch in {episode_path}: "
                f"{episode.get(field)!r} != {expected!r}"
            )


def load_complete_episode_set(
    episode_dir: str | Path,
    tasks: Iterable[dict[str, Any]],
    expected_identities: dict[str, tuple[dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Load an exact, unique, identity-consistent formal episode set."""

    episode_dir = Path(episode_dir)
    tasks = list(tasks)
    expected_names = {
        _episode_filename(str(task["task_id"])) for task in tasks
    }
    actual_paths = sorted(
        path for path in episode_dir.glob("*.json") if path.is_file()
    )
    actual_names = {path.name for path in actual_paths}
    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    if missing_names or unexpected_names or len(actual_paths) != len(expected_names):
        raise RuntimeError(
            "Formal merge requires the exact episode file set: "
            f"expected={len(expected_names)}, actual={len(actual_paths)}, "
            f"missing={missing_names[:5]}, unexpected={unexpected_names[:5]}"
        )

    episodes: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        episode_path = episode_dir / _episode_filename(task_id)
        episode = read_json(episode_path)
        if task_id in seen_task_ids:
            raise RuntimeError(f"Duplicate expected task ID during merge: {task_id}")
        seen_task_ids.add(task_id)
        episode_inputs, episode_key = expected_identities[task_id]
        _validate_episode_identity(
            episode,
            task=task,
            episode_inputs=episode_inputs,
            episode_key=episode_key,
            episode_path=episode_path,
        )
        episodes.append(episode)

    actual_task_ids = [str(episode.get("task_id")) for episode in episodes]
    if len(set(actual_task_ids)) != len(actual_task_ids):
        raise RuntimeError("Formal merge found duplicate task IDs")
    expected_task_ids = [str(task["task_id"]) for task in tasks]
    if actual_task_ids != expected_task_ids:
        raise RuntimeError("Formal merge episode order or task IDs are inconsistent")
    return episodes


def episode_diagnostics(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episodes = list(episodes)
    turns = [
        turn
        for episode in episodes
        for turn in episode["turns"]
    ]
    termination_reasons = {
        reason: sum(
            episode["termination_reason"] == reason
            for episode in episodes
        )
        for reason in (
            "success",
            "max_turns",
            "context_overflow",
            "official_status_parse_error",
        )
    }
    truncated_turns = sum(
        bool(turn["observation_record"]["truncated"])
        for turn in turns
    )
    truncated_episodes = sum(
        any(
            bool(turn["observation_record"]["truncated"])
            for turn in episode["turns"]
        )
        for episode in episodes
    )
    dropped_history_pairs = sum(
        int(turn["dropped_history_turns"])
        for turn in turns
    )
    episodes_with_history_drop = sum(
        any(int(turn["dropped_history_turns"]) > 0 for turn in episode["turns"])
        for episode in episodes
    )
    max_raw_observation_utf8_bytes = max(
        (
            int(turn["observation_record"]["raw_utf8_bytes"])
            for turn in turns
        ),
        default=0,
    )
    return {
        "termination_reasons": termination_reasons,
        "context_overflow_episodes": termination_reasons["context_overflow"],
        "context_overflow_rate": (
            termination_reasons["context_overflow"] / len(episodes)
            if episodes
            else 0.0
        ),
        "official_status_parse_error_episodes": termination_reasons[
            "official_status_parse_error"
        ],
        "official_status_parse_error_rate": (
            termination_reasons["official_status_parse_error"] / len(episodes)
            if episodes
            else 0.0
        ),
        "observation_truncated_turns": truncated_turns,
        "observation_truncated_episodes": truncated_episodes,
        "max_raw_observation_utf8_bytes": max_raw_observation_utf8_bytes,
        "dropped_history_pairs": dropped_history_pairs,
        "episodes_with_history_drop": episodes_with_history_drop,
        "missing_terminator_turns": sum(
            bool(turn["missing_terminator"]) for turn in turns
        ),
        "memory_bank_constraint_triggers": sum(
            int(turn["memory_bank_constraint_trigger_count"])
            for turn in turns
        ),
        "memory_bank_constraint_changed_tokens": sum(
            int(turn["memory_bank_constraint_changed_token_count"])
            for turn in turns
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint)
    preflight_metadata = read_json(checkpoint_dir / "checkpoint.json")
    artifact_schema = str(preflight_metadata.get("schema", ""))
    checkpoint_formal_ready = preflight_metadata.get("formal_ready") is True
    if not checkpoint_formal_ready and not getattr(
        args,
        "allow_exploratory_checkpoint",
        False,
    ):
        raise RuntimeError(
            "Formal evaluation requires a checkpoint with formal_ready=true. "
            "Pass --allow-exploratory-checkpoint only for debugging."
        )
    if artifact_schema == BASE_TEXT_ARTIFACT_SCHEMA:
        model, tokenizer, checkpoint_metadata, catalog_text = (
            load_base_text_artifact(
                args.checkpoint,
                device=args.device,
                dtype=_dtype(args.dtype),
                local_files_only=not args.allow_download,
            )
        )
        system_prompt = compose_base_text_system_prompt(catalog_text)
    elif artifact_schema == MEMORY_CHECKPOINT_SCHEMA:
        model, tokenizer, checkpoint_metadata = load_checkpoint(
            args.checkpoint,
            device=args.device,
            dtype=_dtype(args.dtype),
            local_files_only=not args.allow_download,
        )
        catalog_text = None
        system_prompt = SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown evaluation artifact schema: {artifact_schema!r}")
    if checkpoint_metadata != preflight_metadata:
        raise RuntimeError("Checkpoint metadata changed while evaluation was loading")
    model.eval()
    all_tasks = load_intercode_tasks(args.intercode_root)
    logical_tasks = list(all_tasks)
    task_id_arguments = getattr(args, "task_id", None)
    if task_id_arguments:
        requested_task_ids = list(dict.fromkeys(task_id_arguments))
        if len(requested_task_ids) != len(task_id_arguments):
            raise ValueError("--task-id contains a duplicate task ID")
        known_task_ids = {str(task["task_id"]) for task in all_tasks}
        missing_task_ids = sorted(set(requested_task_ids) - known_task_ids)
        if missing_task_ids:
            raise ValueError(f"Unknown --task-id values: {missing_task_ids}")
        requested_task_id_set = set(requested_task_ids)
        logical_tasks = [
            task
            for task in all_tasks
            if str(task["task_id"]) in requested_task_id_set
        ]
    if args.fs_id:
        logical_tasks = [
            task for task in logical_tasks if task["fs_id"] == args.fs_id
        ]
    if args.limit is not None:
        logical_tasks = logical_tasks[: args.limit]

    shard_specification = shard_configuration(
        getattr(args, "shard_count", None),
        getattr(args, "shard_index", None),
    )
    if shard_specification is not None and (
        task_id_arguments or args.fs_id or args.limit is not None
    ):
        raise ValueError(
            "Formal sharding cannot be combined with --task-id, --fs-id, or --limit"
        )
    if shard_specification is None:
        execution_tasks = list(logical_tasks)
    else:
        shard_count, shard_index = shard_specification
        execution_tasks = select_shard_tasks(
            logical_tasks,
            shard_count=shard_count,
            shard_index=shard_index,
        )

    image_map = dict(DEFAULT_IMAGES)
    if args.image_map:
        override = json.loads(args.image_map)
        if not isinstance(override, dict):
            raise TypeError("--image-map must decode to a JSON object")
        unknown = set(override) - set(DEFAULT_IMAGES)
        if unknown:
            raise ValueError(f"--image-map contains unknown keys: {sorted(unknown)}")
        if any(not isinstance(value, str) or not value for value in override.values()):
            raise TypeError("--image-map values must be non-empty strings")
        image_map.update(override)
    required_image_map = {
        fs_id: image_map[fs_id]
        for fs_id in sorted({task["fs_id"] for task in logical_tasks})
    }
    image_details = resolve_docker_images(required_image_map)
    image_ids = {
        fs_id: details["image_id"]
        for fs_id, details in image_details.items()
    }
    docker_runtime = docker_runtime_identity()
    checkpoint_hashes = {
        "metadata_sha256": sha256_file(checkpoint_dir / "checkpoint.json"),
        "tokenizer_sha256": sha256_directory(checkpoint_dir / "tokenizer"),
    }
    if artifact_schema == BASE_TEXT_ARTIFACT_SCHEMA:
        checkpoint_hashes.update(
            {
                "catalog_sha256": sha256_file(
                    checkpoint_dir / TOOL_CATALOG_FILENAME
                ),
                "catalog_report_sha256": sha256_file(
                    checkpoint_dir / TOOL_CATALOG_REPORT_FILENAME
                ),
            }
        )
    else:
        checkpoint_hashes["trainable_sha256"] = sha256_file(
            checkpoint_dir / "trainable.safetensors"
        )
    checkpoint_identity = stable_id(
        "INTERCODE_BASH_MODEL_ARTIFACT_V4",
        artifact_schema,
        json.dumps(checkpoint_hashes, sort_keys=True, separators=(",", ":")),
    )
    prompt_sha256 = sha256_text(system_prompt)
    runner_code_hashes = {
        name: sha256_file(PACKAGE_ROOT / name)
        for name in (
            "evaluate.py",
            "checkpoint.py",
            "intercode_runner.py",
            "memory_model.py",
            "official_env.py",
            "training_data.py",
            "base_text.py",
        )
    }
    runner_identity = stable_id(
        "INTERCODE_BASH_RUNNER_CODE_V1",
        json.dumps(runner_code_hashes, sort_keys=True, separators=(",", ":")),
    )
    official_source_identity = _official_source_identity(args.intercode_root)
    runtime_identity = _runtime_identity(args.device)
    selected_counts, evaluation_complete = evaluation_selection_status(
        all_tasks,
        logical_tasks,
    )
    input_paper_ready = bool(
        checkpoint_formal_ready
        and evaluation_complete
        and set(image_details) == set(EXPECTED_INTERCODE_COUNTS)
        and docker_runtime["formal_isolation"] is True
    )

    prompt_path = output_dir / "prompt.txt"
    prompt_contents = system_prompt + "\n"
    _ensure_shared_text_file(prompt_path, prompt_contents)
    episode_dir = output_dir / f"episodes_{args.max_turns}_turn"
    episode_dir.mkdir(parents=True, exist_ok=True)

    expected_identities: dict[str, tuple[dict[str, Any], str]] = {}
    for task in logical_tasks:
        episode_inputs = {
            "task_id": task["task_id"],
            "checkpoint_identity": checkpoint_identity,
            "artifact_schema": artifact_schema,
            "base_model_name": checkpoint_metadata["base_model_name"],
            "base_model_revision": checkpoint_metadata.get("base_model_revision"),
            "base_model_identity": checkpoint_metadata["base_model_identity"],
            "method": checkpoint_metadata["method"],
            "data_seed": checkpoint_metadata.get("view_metadata", {}).get(
                "data_seed"
            ),
            "model_seed": checkpoint_metadata.get("training_config", {}).get(
                "model_seed"
            ),
            "image": image_map[task["fs_id"]],
            "image_id": image_ids[task["fs_id"]],
            "task_source_sha256": task["source_sha256"],
            "max_turns": args.max_turns,
            "context_limit": args.context_limit,
            "max_new_tokens": args.max_new_tokens,
            "observation_policy": OBSERVATION_POLICY,
            "container_startup_policy": CONTAINER_STARTUP_POLICY,
            "environment_lifecycle_policy": ENVIRONMENT_LIFECYCLE_POLICY,
            "git_status_policy": GIT_STATUS_POLICY,
            "scorer_policy": SCORER_POLICY,
            "status_parse_failure_policy": STATUS_PARSE_FAILURE_POLICY,
            "dtype": args.dtype,
            "prompt_sha256": prompt_sha256,
            "runner_protocol": RUNNER_PROTOCOL,
            "runner_identity": runner_identity,
            "official_source_identity": official_source_identity["identity"],
            "runtime_identity": runtime_identity,
            "docker_runtime_identity": docker_runtime,
            "checkpoint_formal_ready": checkpoint_formal_ready,
            "evaluation_complete": evaluation_complete,
            "paper_ready": input_paper_ready,
        }
        task_id = str(task["task_id"])
        if task_id in expected_identities:
            raise RuntimeError(f"Duplicate logical task ID: {task_id}")
        expected_identities[task_id] = (
            episode_inputs,
            _episode_key(episode_inputs),
        )

    episodes_by_task_id: dict[str, dict[str, Any]] = {}
    pending_tasks: list[dict[str, Any]] = []
    for task in execution_tasks:
        task_id = str(task["task_id"])
        episode_inputs, episode_key = expected_identities[task_id]
        episode_path = episode_dir / _episode_filename(task_id)
        if args.resume and episode_path.exists():
            episode = read_json(episode_path)
            _validate_episode_identity(
                episode,
                task=task,
                episode_inputs=episode_inputs,
                episode_key=episode_key,
                episode_path=episode_path,
            )
            episodes_by_task_id[task_id] = episode
            continue
        pending_tasks.append(task)

    for fs_id, filesystem_tasks in group_tasks_by_filesystem(pending_tasks):
        fs_number = fs_id.removeprefix("fs")
        data_path = (
            Path(args.intercode_root)
            / "data"
            / "bash"
            / "nl2bash"
            / f"nl2bash_fs_{fs_number}.json"
        )
        environment_key = stable_id(
            "INTERCODE_BASH_FILESYSTEM_ENVIRONMENT_V1",
            checkpoint_identity,
            runner_identity,
            official_source_identity["identity"],
            fs_id,
            image_ids[fs_id],
        )
        next_task_index = 0
        while next_task_index < len(filesystem_tasks):
            with ReusableOfficialBashEnv(
                intercode_root=args.intercode_root,
                image_name=image_ids[fs_id],
                data_path=data_path,
                environment_key=environment_key,
            ) as environment:
                while next_task_index < len(filesystem_tasks):
                    task = filesystem_tasks[next_task_index]
                    task_id = str(task["task_id"])
                    episode_inputs, episode_key = expected_identities[task_id]
                    episode_path = episode_dir / _episode_filename(task_id)
                    episode = run_try_again_episode(
                        environment=environment,
                        model=model,
                        tokenizer=tokenizer,
                        task=task,
                        max_turns=args.max_turns,
                        max_new_tokens=args.max_new_tokens,
                        context_limit=args.context_limit,
                        system_prompt=system_prompt,
                    )
                    episode["run_key"] = episode_key
                    episode["run_inputs"] = episode_inputs
                    _validate_episode_identity(
                        episode,
                        task=task,
                        episode_inputs=episode_inputs,
                        episode_key=episode_key,
                        episode_path=episode_path,
                    )
                    write_json(episode_path, episode)
                    episodes_by_task_id[task_id] = episode
                    next_task_index += 1
                    if (
                        episode["termination_reason"]
                        == "official_status_parse_error"
                    ):
                        # A command that made the official status output
                        # unparsable may also have damaged files outside Git.
                        # Recreate the pair before evaluating the next task.
                        break

    episodes = [
        episodes_by_task_id[str(task["task_id"])]
        for task in execution_tasks
    ]

    if shard_specification is None and evaluation_complete:
        episodes = load_complete_episode_set(
            episode_dir,
            logical_tasks,
            expected_identities,
        )

    rewards = [float(episode["max_reward"]) for episode in episodes]
    per_filesystem = {}
    for fs_id in sorted({episode["fs_id"] for episode in episodes}):
        members = [episode for episode in episodes if episode["fs_id"] == fs_id]
        member_rewards = [float(episode["max_reward"]) for episode in members]
        member_released_rewards = [
            float(episode["max_released_reward"])
            for episode in members
        ]
        successes = sum(bool(episode["success"]) for episode in members)
        released_successes = sum(
            reward == 1.0 for reward in member_released_rewards
        )
        per_filesystem[fs_id] = {
            "tasks": len(members),
            "successes": successes,
            "success_rate": successes / len(members),
            "mean_continuous_reward": statistics.fmean(member_rewards),
            "released_successes": released_successes,
            "released_success_rate": released_successes / len(members),
            "mean_released_continuous_reward": statistics.fmean(
                member_released_rewards
            ),
            "mean_turns_taken": statistics.fmean(
                int(episode["turns_taken"]) for episode in members
            ),
            **episode_diagnostics(members),
        }
    diagnostics = episode_diagnostics(episodes)
    result_paper_ready = bool(
        input_paper_ready
        and diagnostics["context_overflow_episodes"] == 0
    )
    summary = {
        "schema": FORMAL_SUMMARY_SCHEMA,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_hashes": checkpoint_hashes,
        "artifact_schema": artifact_schema,
        "method": checkpoint_metadata["method"],
        "tasks": len(episodes),
        "max_turns": args.max_turns,
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "success_rate": (
            sum(bool(episode["success"]) for episode in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "mean_continuous_reward": statistics.fmean(rewards) if rewards else 0.0,
        "released_successes": sum(
            float(episode["max_released_reward"]) == 1.0
            for episode in episodes
        ),
        "released_success_rate": (
            sum(
                float(
                    episode["max_released_reward"]
                )
                == 1.0
                for episode in episodes
            )
            / len(episodes)
            if episodes
            else 0.0
        ),
        "mean_released_continuous_reward": (
            statistics.fmean(
                float(
                    episode["max_released_reward"]
                )
                for episode in episodes
            )
            if episodes
            else 0.0
        ),
        "mean_turns_taken": (
            statistics.fmean(int(episode["turns_taken"]) for episode in episodes)
            if episodes
            else 0.0
        ),
        "per_filesystem": per_filesystem,
        **diagnostics,
        "context_limit": args.context_limit,
        "max_new_tokens": args.max_new_tokens,
        "observation_policy": OBSERVATION_POLICY,
        "container_startup_policy": CONTAINER_STARTUP_POLICY,
        "environment_lifecycle_policy": ENVIRONMENT_LIFECYCLE_POLICY,
        "git_status_policy": GIT_STATUS_POLICY,
        "scorer_policy": SCORER_POLICY,
        "status_parse_failure_policy": STATUS_PARSE_FAILURE_POLICY,
        "dtype": args.dtype,
        "prompt_sha256": prompt_sha256,
        "catalog_token_count": (
            int(checkpoint_metadata["catalog_token_count"])
            if catalog_text is not None
            else 0
        ),
        "runner_protocol": RUNNER_PROTOCOL,
        "runner_identity": runner_identity,
        "runner_code_hashes": runner_code_hashes,
        "checkpoint_formal_ready": checkpoint_formal_ready,
        "evaluation_complete": evaluation_complete,
        "input_paper_ready": input_paper_ready,
        "paper_ready": result_paper_ready,
        "formal_ready": result_paper_ready,
        "selected_task_counts": selected_counts,
        "expected_task_counts": EXPECTED_INTERCODE_COUNTS,
        "image_map": image_map,
        "image_ids": image_ids,
        "image_details": image_details,
        "official_source_identity": official_source_identity,
        "runtime_identity": runtime_identity,
        "docker_runtime_identity": docker_runtime,
    }
    if shard_specification is None:
        write_json(output_dir / f"summary_{args.max_turns}_turn.json", summary)
        return summary

    shard_count, shard_index = shard_specification
    shard_task_counts = {
        fs_id: sum(task["fs_id"] == fs_id for task in execution_tasks)
        for fs_id in EXPECTED_INTERCODE_COUNTS
    }
    shard_summary = {
        **summary,
        "schema": SHARD_SUMMARY_SCHEMA,
        "global_tasks": len(logical_tasks),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "shard_assignment": SHARD_ASSIGNMENT,
        "shard_task_counts": shard_task_counts,
        "shard_task_ids": [str(task["task_id"]) for task in execution_tasks],
        "shard_complete": len(episodes) == len(execution_tasks),
        "global_input_paper_ready": input_paper_ready,
        "global_result_paper_ready": None,
        # A shard is never itself a formal result, even though its episode
        # identities are computed from the complete 200-task logical run.
        "paper_ready": False,
        "formal_ready": False,
        "formal_summary_written": False,
    }
    shard_summary_path = output_dir / (
        f"shard_summary_{args.max_turns}_turn_"
        f"{shard_index:03d}_of_{shard_count:03d}.json"
    )
    write_json(shard_summary_path, shard_summary)
    return shard_summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--intercode-root",
        default=str(PACKAGE_ROOT.parent / "datasets" / "intercode"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-turns", type=int, choices=(1, 10), default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--image-map", help="JSON object mapping fs1..fs4 to image names")
    parser.add_argument("--fs-id", choices=("fs1", "fs2", "fs3", "fs4"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--task-id",
        action="append",
        help=(
            "Select an exact task ID; repeat for a deterministic diagnostic "
            "subset. Cannot be combined with --fs-id or --limit."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help=(
            "Enable formal execution sharding into this many zero-based shards. "
            "The logical selection remains all 200 tasks. Disabled by default."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help=(
            "Zero-based formal shard index. Each filesystem assigns a task by "
            "local_index %% shard_count."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-exploratory-checkpoint",
        action="store_true",
        help="Permit a non-primary checkpoint for debugging, never for paper results",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.context_limit <= args.max_new_tokens:
        parser.error("--context-limit must exceed --max-new-tokens")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.task_id and (args.fs_id or args.limit is not None):
        parser.error("--task-id cannot be combined with --fs-id or --limit")
    try:
        specification = shard_configuration(args.shard_count, args.shard_index)
    except ValueError as exc:
        parser.error(str(exc))
    if specification is not None and (
        args.task_id or args.fs_id or args.limit is not None
    ):
        parser.error(
            "--shard-count/--shard-index cannot be combined with "
            "--task-id, --fs-id, or --limit"
        )
    return args


def main(argv: Iterable[str] | None = None) -> None:
    summary = evaluate(parse_args(argv))
    print(summary)


if __name__ == "__main__":
    main()
