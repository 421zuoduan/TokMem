from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .context import normalize_workspace_paths
from .manifest import canonical_json


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


def _message_kind(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "assistant":
        return "call"
    if role == "tool":
        return "observation"
    raise ValueError(f"unsupported episode message role: {role!r}")


def validate_episode(
    episode: dict[str, Any],
    *,
    require_clean: bool = True,
) -> None:
    if episode.get("accepted") is not True:
        raise ValueError("training episode must have accepted=true")
    rejection_reasons = episode.get("rejection_reasons")
    if rejection_reasons not in (None, []):
        raise ValueError("accepted episode cannot contain rejection reasons")
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
    if not isinstance(evaluator, dict) or evaluator.get("passed") is not True:
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


def flatten_episode(
    episode: dict[str, Any],
    *,
    workspace_root: str | None = None,
    require_clean: bool = True,
) -> list[dict[str, Any]]:
    validate_episode(episode, require_clean=require_clean)
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
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        template_splits[str(episode.get("template_id"))].add(str(episode.get("split")))
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
        steps = flatten_episode(episode, workspace_root=workspace_root)
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
        "split_episode_counts": dict(Counter(episode["split"] for episode in episodes)),
        "split_step_counts": {
            split: len(steps) for split, steps in split_steps.items()
        },
        "target_tool_counts": dict(sorted(tool_counts.items())),
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = read_records(args.episodes)
    metadata = prepare_steps(
        episodes,
        args.output_dir,
        workspace_root=args.workspace_root,
    )
    print(canonical_json(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
