from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .episode_to_steps import validate_episode
from .manifest import canonical_json


def read_jsonl(paths: list[str]) -> list[dict[str, Any]]:
    records = []
    for path_string in paths:
        path = Path(path_string)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                records.append(record)
    return records


def select_canonical_episodes(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate.get("task_id"))].append(candidate)

    selected = []
    rejected = []
    for task_id, group in sorted(grouped.items()):
        environment_ids = [
            candidate.get("teacher", {}).get("fresh_environment_id")
            for candidate in group
        ]
        if any(not isinstance(value, str) or not value for value in environment_ids):
            raise ValueError(f"task {task_id} has no fresh_environment_id")
        if len(environment_ids) != len(set(environment_ids)):
            raise ValueError(f"task {task_id} reused a teacher environment")

        clean = []
        for candidate in group:
            try:
                validate_episode(candidate, require_clean=True)
            except ValueError:
                rejected.append(candidate)
            else:
                clean.append(candidate)
        if not clean:
            continue
        clean.sort(
            key=lambda candidate: (
                len(candidate["messages"]) // 2,
                int(candidate["teacher"]["candidate_index"]),
                candidate["episode_id"],
            )
        )
        winner = dict(clean[0])
        winner["canonical_selection"] = {
            "policy": "clean_then_fewest_calls_then_candidate_index",
            "candidate_count": len(group),
            "clean_candidate_count": len(clean),
        }
        selected.append(winner)
        rejected.extend(clean[1:])
    return selected, rejected


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one clean canonical teacher trajectory per synthetic task"
    )
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--accepted-output", required=True)
    parser.add_argument("--rejected-output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected, rejected = select_canonical_episodes(read_jsonl(args.candidates))
    write_jsonl(args.accepted_output, selected)
    write_jsonl(args.rejected_output, rejected)
    print(f"canonical={len(selected)} rejected={len(rejected)}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
