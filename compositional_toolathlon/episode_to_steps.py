from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .context import normalize_workspace_paths
from .manifest import canonical_json
from .mcp_adapter import observation_reports_error


ALLOWED_SPLITS = ("train", "validation", "synthetic_test")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: episode must be an object")
                records.append(record)
        return records
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("episode JSON must contain a list of objects")
    return payload


def record_content_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _message_kind(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "assistant":
        return "call"
    if role == "tool":
        return "observation"
    raise ValueError(f"unsupported episode message role: {role!r}")


def _validate_real_execution_evidence(episode: dict[str, Any]) -> None:
    teacher = episode["teacher"]
    real_execution = teacher.get("real_execution")
    if "real_execution" in teacher and not isinstance(real_execution, bool):
        raise ValueError("teacher.real_execution must be a boolean when present")
    if real_execution is not True:
        return

    fresh_environment_id = teacher.get("fresh_environment_id")
    if (
        not isinstance(fresh_environment_id, str)
        or not fresh_environment_id.strip()
    ):
        raise ValueError(
            "real-execution episode requires teacher.fresh_environment_id"
        )
    workspace_root = episode.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        raise ValueError("real-execution episode requires workspace_root")
    if not Path(workspace_root).is_absolute():
        raise ValueError("real-execution episode workspace_root must be absolute")

    for index, observation_message in enumerate(episode["messages"][1::2], start=1):
        message_index = 2 * index - 1
        success = observation_message.get("success")
        if not isinstance(success, bool):
            raise ValueError(
                f"real-execution observation {message_index} requires boolean success"
            )

        runtime_metadata = observation_message.get("runtime_metadata")
        if not isinstance(runtime_metadata, dict):
            raise ValueError(
                f"real-execution observation {message_index} requires runtime_metadata"
            )

        has_mcp_status = (
            "is_error" in runtime_metadata
            or "semantic_error" in runtime_metadata
        )
        if has_mcp_status:
            for field in ("is_error", "semantic_error"):
                if not isinstance(runtime_metadata.get(field), bool):
                    raise ValueError(
                        "real-execution MCP observation "
                        f"{message_index} requires boolean runtime_metadata.{field}"
                    )

        has_return_code = "return_code" in runtime_metadata
        if has_return_code:
            return_code = runtime_metadata["return_code"]
            if isinstance(return_code, bool) or not isinstance(return_code, int):
                raise ValueError(
                    "real-execution local observation "
                    f"{message_index} requires integer runtime_metadata.return_code"
                )

        if not has_mcp_status and not has_return_code:
            raise ValueError(
                "real-execution observation "
                f"{message_index} lacks MCP error flags or a return_code"
            )

        if success and has_mcp_status and (
            runtime_metadata["is_error"]
            or runtime_metadata["semantic_error"]
        ):
            raise ValueError(
                f"real-execution observation {message_index} marks an error as success"
            )
        if success and has_return_code and runtime_metadata["return_code"] != 0:
            raise ValueError(
                "real-execution observation "
                f"{message_index} marks a nonzero return_code as success"
            )
        if success and runtime_metadata.get("timed_out") is True:
            raise ValueError(
                f"real-execution observation {message_index} marks a timeout as success"
            )
        if success and observation_reports_error(
            observation_message.get("observation")
        ):
            raise ValueError(
                "real-execution observation "
                f"{message_index} has Error:/Failed to content marked as success"
            )

        observation_sha256 = observation_message.get("observation_sha256")
        if (
            not isinstance(observation_sha256, str)
            or len(observation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in observation_sha256)
        ):
            raise ValueError(
                "real-execution observation "
                f"{message_index} requires a lowercase observation_sha256"
            )
        expected_sha256 = hashlib.sha256(
            canonical_json(observation_message.get("observation")).encode("utf-8")
        ).hexdigest()
        if observation_sha256 != expected_sha256:
            raise ValueError(
                f"real-execution observation {message_index} observation_sha256 mismatch"
            )


def validate_episode(
    episode: dict[str, Any],
    *,
    require_clean: bool = True,
    include_rejected: bool = False,
) -> None:
    accepted = episode.get("accepted")
    if include_rejected:
        if not isinstance(accepted, bool):
            raise ValueError("training episode must record a boolean accepted status")
    elif accepted is not True:
        raise ValueError("training episode must have accepted=true")
    rejection_reasons = episode.get("rejection_reasons")
    if accepted is True and rejection_reasons not in (None, []):
        raise ValueError("accepted episode cannot contain rejection reasons")
    if accepted is False and not isinstance(rejection_reasons, list):
        raise ValueError("rejected episode must record rejection_reasons")
    required_strings = ("episode_id", "task_id", "template_id", "instruction", "split")
    for field in required_strings:
        if not isinstance(episode.get(field), str) or not episode[field]:
            raise ValueError(f"episode requires non-empty string {field!r}")
    if episode["split"] not in ALLOWED_SPLITS:
        raise ValueError(
            f"episode {episode['episode_id']} has unsupported split {episode['split']!r}"
        )
    if not isinstance(episode.get("tool_manifest_hash"), str):
        raise ValueError("episode requires a tool_manifest_hash")
    teacher = episode.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError("accepted episode requires teacher provenance")
    if teacher.get("termination_reason") != "claim_done":
        raise ValueError("accepted teacher episode must terminate with claim_done")
    if teacher.get("visible_assistant_characters") != 0:
        raise ValueError("accepted teacher episode must not retain visible prose")

    available = episode.get("available_tool_ids")
    if not isinstance(available, list) or not available:
        raise ValueError("episode requires a non-empty available_tool_ids list")
    if len(available) != len(set(available)):
        raise ValueError("available_tool_ids contains duplicates")

    evaluator = episode.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("training episode requires evaluator metadata")
    evaluator_passed = evaluator.get("passed")
    if include_rejected:
        if not isinstance(evaluator_passed, bool):
            raise ValueError("training episode evaluator.passed must be boolean")
    elif evaluator_passed is not True:
        raise ValueError("accepted episode requires evaluator.passed=true")

    messages = episode.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("episode requires at least one call/observation pair")
    if len(messages) % 2 != 0:
        raise ValueError("clean episode messages must contain complete call/observation pairs")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        expected = "call" if index % 2 == 0 else "observation"
        actual = _message_kind(message)
        if actual != expected:
            raise ValueError(
                f"message {index} must be {expected}, found role={message.get('role')!r}"
            )
        tool_id = message.get("tool_id")
        if not isinstance(tool_id, str) or tool_id not in available:
            raise ValueError(f"message {index} references unavailable tool {tool_id!r}")
        if actual == "call":
            if not isinstance(message.get("arguments"), dict):
                raise ValueError(f"call message {index} requires object arguments")
            if message.get("content"):
                raise ValueError("teacher prose must not be stored in assistant call messages")
        else:
            preceding_tool_id = messages[index - 1].get("tool_id")
            if tool_id != preceding_tool_id:
                raise ValueError(
                    f"observation {index} tool_id does not match preceding call"
                )
            if "observation" not in message:
                raise ValueError(f"tool message {index} requires observation")
            if require_clean and message.get("success") is not True:
                raise ValueError("main training data accepts only clean successful calls")
    _validate_real_execution_evidence(episode)


def flatten_episode(
    episode: dict[str, Any],
    *,
    workspace_root: str | None = None,
    require_clean: bool = True,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    validate_episode(
        episode,
        require_clean=require_clean,
        include_rejected=include_rejected,
    )
    episode_workspace = episode.get("workspace_root")
    if workspace_root is None and isinstance(episode_workspace, str):
        workspace_root = episode_workspace
    messages = normalize_workspace_paths(episode["messages"], workspace_root)
    instruction = normalize_workspace_paths(episode["instruction"], workspace_root)
    call_count = len(messages) // 2
    history = []
    steps = []

    for call_index in range(call_count):
        call = messages[2 * call_index]
        observation = messages[2 * call_index + 1]
        step = {
            "sample_id": f"{episode['episode_id']}_step_{call_index:03d}",
            "episode_id": episode["episode_id"],
            "task_id": episode["task_id"],
            "template_id": episode["template_id"],
            "asset_seed": episode.get("asset_seed"),
            "split": episode["split"],
            "step_index": call_index,
            "episode_step_count": call_count,
            "instruction": instruction,
            "available_tool_ids": list(episode["available_tool_ids"]),
            "history": list(history),
            "target_tool_id": call["tool_id"],
            "target_arguments": call["arguments"],
            "is_terminal": bool(call.get("is_terminal", False)),
            "tool_manifest_hash": episode.get("tool_manifest_hash"),
        }
        if include_rejected or not require_clean:
            step.update(
                {
                    "episode_accepted": bool(episode["accepted"]),
                    "episode_evaluator_passed": bool(
                        episode["evaluator"]["passed"]
                    ),
                    "episode_rejection_reasons": list(
                        episode.get("rejection_reasons") or []
                    ),
                    "target_call_success": bool(observation.get("success")),
                }
            )
        steps.append(step)
        history.append(
            {
                "tool_id": call["tool_id"],
                "arguments": call["arguments"],
                "observation": observation["observation"],
                "success": bool(observation.get("success")),
            }
        )
    return steps


def validate_group_splits(episodes: Iterable[dict[str, Any]]) -> None:
    template_splits: dict[str, set[str]] = defaultdict(set)
    episode_ids = set()
    real_environment_ids: dict[str, str] = {}
    real_workspace_roots: dict[str, str] = {}
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        template_splits[str(episode.get("template_id"))].add(str(episode.get("split")))
        teacher = episode.get("teacher")
        if isinstance(teacher, dict) and teacher.get("real_execution") is True:
            fresh_environment_id = teacher.get("fresh_environment_id")
            workspace_root = episode.get("workspace_root")
            if isinstance(fresh_environment_id, str):
                previous_episode = real_environment_ids.setdefault(
                    fresh_environment_id,
                    str(episode_id),
                )
                if previous_episode != str(episode_id):
                    raise ValueError(
                        "real-execution episodes reuse fresh_environment_id "
                        f"{fresh_environment_id!r}: "
                        f"{previous_episode!r}, {episode_id!r}"
                    )
            if isinstance(workspace_root, str):
                normalized_root = str(Path(workspace_root).resolve())
                previous_episode = real_workspace_roots.setdefault(
                    normalized_root,
                    str(episode_id),
                )
                if previous_episode != str(episode_id):
                    raise ValueError(
                        "real-execution episodes reuse workspace_root "
                        f"{normalized_root!r}: "
                        f"{previous_episode!r}, {episode_id!r}"
                    )
    leaked = {
        template_id: sorted(splits)
        for template_id, splits in template_splits.items()
        if len(splits) > 1
    }
    if leaked:
        raise ValueError(f"template IDs cross data splits: {leaked}")


def as_compositional_sample(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": step["sample_id"],
        "user_input": step["instruction"],
        "tools": [step["target_tool_id"]],
        "function_calls": [canonical_json(step["target_arguments"])],
        "available_tools": list(step["available_tool_ids"]),
        "toolathlon_step": step,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_steps(
    episodes: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    workspace_root: str | None = None,
    include_rejected: bool = False,
    include_failed_calls: bool = False,
) -> dict[str, Any]:
    validate_group_splits(episodes)
    manifest_hashes = {episode.get("tool_manifest_hash") for episode in episodes}
    if len(manifest_hashes) != 1:
        raise ValueError("all episodes must use one identical tool manifest hash")
    output_dir = Path(output_dir)
    split_steps: dict[str, list[dict[str, Any]]] = {
        split: [] for split in ALLOWED_SPLITS
    }
    for episode in episodes:
        steps = flatten_episode(
            episode,
            workspace_root=workspace_root,
            require_clean=not include_failed_calls,
            include_rejected=include_rejected,
        )
        split_steps[episode["split"]].extend(steps)

    train_targets = {
        step["target_tool_id"] for step in split_steps["train"]
    }
    held_out_targets = {
        step["target_tool_id"]
        for split in ("validation", "synthetic_test")
        for step in split_steps[split]
    }
    missing_train_coverage = sorted(held_out_targets - train_targets)
    if missing_train_coverage:
        raise ValueError(
            "held-out splits contain target tools unseen in train: "
            + ", ".join(missing_train_coverage)
        )

    tool_counts = Counter(
        step["target_tool_id"]
        for split in ALLOWED_SPLITS
        for step in split_steps[split]
    )
    metadata = {
        "schema_version": 1,
        "episode_count": len(episodes),
        "step_count": sum(len(steps) for steps in split_steps.values()),
        "split_episode_counts": {
            split: sum(episode["split"] == split for episode in episodes)
            for split in ALLOWED_SPLITS
        },
        "split_step_counts": {
            split: len(steps) for split, steps in split_steps.items()
        },
        "split_step_content_hashes": {
            split: record_content_hash(steps)
            for split, steps in split_steps.items()
        },
        "target_tool_counts": dict(sorted(tool_counts.items())),
        "training_episode_policy": {
            "include_rejected": include_rejected,
            "include_failed_calls": include_failed_calls,
        },
        "episode_acceptance_counts": dict(
            sorted(
                Counter(
                    "accepted" if episode["accepted"] else "rejected"
                    for episode in episodes
                ).items()
            )
        ),
        "evaluator_status_counts": dict(
            sorted(
                Counter(
                    "passed" if episode["evaluator"]["passed"] else "failed"
                    for episode in episodes
                ).items()
            )
        ),
    }

    for split, steps in split_steps.items():
        write_jsonl(output_dir / f"{split}.jsonl", steps)
        write_json(
            output_dir / f"{split}_compositional.json",
            [as_compositional_sample(step) for step in steps],
        )
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert verified Toolathlon episodes into observation-aware step samples"
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="include complete collector-rejected episodes and retain their status",
    )
    parser.add_argument(
        "--include-failed-calls",
        action="store_true",
        help="include steps whose real tool observation reports failure",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = read_records(args.episodes)
    metadata = prepare_steps(
        episodes,
        args.output_dir,
        workspace_root=args.workspace_root,
        include_rejected=args.include_rejected,
        include_failed_calls=args.include_failed_calls,
    )
    print(canonical_json(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
