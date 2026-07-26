from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .generate_tasks import write_jsonl
from .verify_tasks import read_jsonl


def select_shard(
    tasks: list[dict[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[tuple[int, dict[str, Any]]]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return [
        (index, task)
        for index, task in enumerate(tasks)
        if index % num_shards == shard_index
    ]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "task"


def require_fresh_artifacts(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to reuse episode artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def episode_port(start_port: int, shard_index: int, local_index: int) -> int:
    return start_port + shard_index * 100 + local_index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect one real teacher episode for each verified task shard"
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--start-port", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    selected = select_shard(
        tasks,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    generation_config = json.loads(
        Path(args.generation_config).read_text(encoding="utf-8")
    )
    teacher_sessions = [
        session["session_id"]
        for session in generation_config["teacher_sessions"]
    ]
    if not teacher_sessions:
        raise ValueError("generation config requires teacher_sessions")

    shard_name = f"shard_{args.shard_index:02d}"
    output_root = Path(args.output_root) / shard_name
    runtime_root = Path(args.runtime_root) / shard_name
    task_dir = runtime_root / "task_specs"
    candidate_dir = output_root / "candidates"
    task_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    runner = PACKAGE_DIR / "scripts" / "collect_llm_episode.sh"

    episodes = []
    summary = []
    missing_outputs = 0
    for local_index, (global_index, task) in enumerate(selected):
        task_id = task["task_id"]
        stem = f"{global_index:03d}_{safe_name(task_id)}"
        task_path = task_dir / f"{stem}.json"
        workspace = runtime_root / "workspaces" / stem
        gateway_runtime = runtime_root / "gateways" / stem
        output_path = candidate_dir / f"{stem}.jsonl"
        require_fresh_artifacts(
            [task_path, workspace, gateway_runtime, output_path]
        )
        task_path.write_text(
            json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        teacher_session = teacher_sessions[global_index % len(teacher_sessions)]
        completed = subprocess.run(
            [
                "bash",
                str(runner),
                str(task_path),
                args.manifest,
                args.generation_config,
                teacher_session,
                str(workspace),
                str(gateway_runtime),
                str(
                    episode_port(
                        args.start_port,
                        args.shard_index,
                        local_index,
                    )
                ),
                str(output_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        print(completed.stdout.strip(), flush=True)
        if completed.stderr.strip():
            print(completed.stderr.strip(), flush=True)

        record = {
            "task_id": task_id,
            "teacher_session_id": teacher_session,
            "returncode": completed.returncode,
            "output": str(output_path),
        }
        if output_path.is_file():
            episode = json.loads(output_path.read_text(encoding="utf-8"))
            episodes.append(episode)
            record["episode_id"] = episode["episode_id"]
            record["accepted"] = episode["accepted"]
            record["rejection_reasons"] = episode["rejection_reasons"]
        else:
            missing_outputs += 1
            record["accepted"] = False
            record["rejection_reasons"] = ["collector produced no episode file"]
        summary.append(record)

    write_jsonl(output_root / "all.jsonl", episodes)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    accepted = sum(bool(record["accepted"]) for record in summary)
    print(
        f"shard={args.shard_index}/{args.num_shards} "
        f"episodes={len(episodes)} accepted={accepted} "
        f"rejected={len(summary) - accepted}",
        flush=True,
    )
    return 1 if missing_outputs else 0


if __name__ == "__main__":
    raise SystemExit(main())
