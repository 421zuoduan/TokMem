#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any


REVISION = "2aed2468858f15818acafa178518390cc4b0f5cb"
RAW_BASE = f"https://raw.githubusercontent.com/hkust-nlp/Toolathlon/{REVISION}/"
PREFIXES = (
    "configs/",
    "deployment/",
    "global_preparation/",
    "scripts/",
    "utils/",
)
ROOT_FILES = {
    ".gitignore",
    "DECOUPLED_AGENT_LOOP.md",
    "Dockerfile",
    "README.md",
    "main.py",
    "package-lock.json",
    "package.json",
    "pyproject.toml",
    "uv.lock",
}
EXTRA_FILES = {"local_binary/github-mcp-version.txt"}
REQUIRED_FILES = {
    "configs/global_configs_example.py",
    "global_preparation/install_env_minimal.sh",
    "scripts/run_single_decoupled.sh",
    "scripts/decoupled/container_eval.py",
    "scripts/decoupled/container_preprocess.py",
    "scripts/decoupled/container_tool_gateway.py",
    "scripts/decoupled/host_agent_loop.py",
    "utils/openai_agents_monkey_patch/tool_name_aliases.py",
}


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def selected_blob(entry: dict[str, Any]) -> bool:
    path = entry.get("path")
    return (
        entry.get("type") == "blob"
        and isinstance(path, str)
        and (
            path.startswith(PREFIXES)
            or path in ROOT_FILES
            or path in EXTRA_FILES
        )
    )


def download_blob(entry: dict[str, Any], staging: Path) -> dict[str, Any]:
    path = entry["path"]
    expected_sha = entry["sha"]
    expected_size = int(entry.get("size", -1))
    url = RAW_BASE + urllib.parse.quote(path, safe="/")
    output = staging / path
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(2):
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "--http1.1",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "25",
                    "--retry",
                    "1",
                    "--retry-delay",
                    "1",
                    "--retry-all-errors",
                    "--output",
                    str(output),
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip()
                    or f"curl exited {completed.returncode}"
                )
            content = output.read_bytes()
            if expected_size >= 0 and len(content) != expected_size:
                raise ValueError(
                    f"size mismatch for {path}: {len(content)} != {expected_size}"
                )
            actual_sha = git_blob_sha1(content)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Git blob mismatch for {path}: {actual_sha} != {expected_sha}"
                )
            if entry.get("mode") == "100755":
                output.chmod(0o755)
            return {
                "path": path,
                "git_blob_sha1": actual_sha,
                "size": len(content),
            }
        except Exception as exc:
            output.unlink(missing_ok=True)
            last_error = exc
    raise RuntimeError(f"failed to download {path}: {last_error}")


def fetch_overlay(
    *,
    tree_path: Path,
    target: Path,
    task_root: Path,
    workers: int,
) -> dict[str, Any]:
    if target.exists():
        raise ValueError(f"refusing to overwrite existing target: {target}")
    if not task_root.is_dir():
        raise ValueError(f"local Toolathlon task snapshot is missing: {task_root}")
    tree_bytes = tree_path.read_bytes()
    tree_payload = json.loads(tree_bytes)
    if tree_payload.get("truncated") is not False:
        raise ValueError("GitHub tree response is truncated or malformed")
    entries = [entry for entry in tree_payload["tree"] if selected_blob(entry)]
    paths = {entry["path"] for entry in entries}
    missing = sorted(REQUIRED_FILES - paths)
    if missing:
        raise ValueError(f"fixed tree lacks required runner files: {missing}")
    if not entries:
        raise ValueError("fixed tree selected no source files")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".toolathlon-source-",
            dir=str(target.parent),
        )
    )
    try:
        records = []
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = [
            pool.submit(download_blob, entry, staging)
            for entry in entries
        ]
        try:
            for completed_index, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                records.append(future.result())
                if completed_index % 25 == 0 or completed_index == len(entries):
                    print(
                        f"verified {completed_index}/{len(entries)} source blobs",
                        flush=True,
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        task_link = staging / "tasks"
        task_link.symlink_to(
            os.path.relpath(task_root.resolve(), staging),
            target_is_directory=True,
        )
        manifest = {
            "source_kind": "github_raw_sparse",
            "source_url": (
                "https://api.github.com/repos/hkust-nlp/Toolathlon/git/trees/"
                f"{REVISION}?recursive=1"
            ),
            "revision": REVISION,
            "tree_response_sha256": hashlib.sha256(tree_bytes).hexdigest(),
            "downloaded_file_count": len(records),
            "downloaded_bytes": sum(record["size"] for record in records),
            "tasks_source": str(task_root.resolve()),
            "git_blob_verification": True,
        }
        (staging / ".toolathlon-source.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the code-only overlay for the pinned Toolathlon revision"
    )
    parser.add_argument("--tree-json", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be in [1, 32]")
    result = fetch_overlay(
        tree_path=Path(args.tree_json),
        target=Path(args.target).resolve(),
        task_root=Path(args.task_root).resolve(),
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
