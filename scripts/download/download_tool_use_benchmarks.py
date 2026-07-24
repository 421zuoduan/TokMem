#!/usr/bin/env python3
"""Download the public Toolathlon and MCP-Atlas benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "datasets"

TOOLATHLON_REPOSITORY = "hkust-nlp/Toolathlon"
TOOLATHLON_REVISION = "2aed2468858f15818acafa178518390cc4b0f5cb"
TOOLATHLON_PREFIX = "tasks/finalpool/"

MCP_ATLAS_REPOSITORY = "ScaleAI/MCP-Atlas"
MCP_ATLAS_REVISION = "b5bcde2236c0b8772020e13dea4e481241e78677"
MCP_ATLAS_FILENAME = "MCP-Atlas.parquet"
MCP_ATLAS_SIZE = 15_638_757
MCP_ATLAS_SHA256 = "2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0"


def request_json(url: str, retries: int) -> dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "tokmem-benchmark-downloader"})
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError("unreachable")


def git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(
    url: str,
    target: Path,
    expected_size: int,
    retries: int,
    expected_git_sha1: str | None = None,
    expected_sha256: str | None = None,
    segments: int = 1,
    segment_threshold: int = 8 * 1024 * 1024,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_size:
        if expected_git_sha1 and git_blob_sha1(target, expected_size) == expected_git_sha1:
            return
        if expected_sha256 and sha256(target) == expected_sha256:
            return
        if not expected_git_sha1 and not expected_sha256:
            return

    partial = target.with_name(f"{target.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if segments > 1 and expected_size - offset >= segment_threshold:
        segment_ranges = split_ranges(offset, expected_size - 1, segments)
        segment_paths = [
            target.with_name(f"{target.name}.part.{start}-{end}")
            for start, end in segment_ranges
        ]
        with ThreadPoolExecutor(max_workers=len(segment_ranges)) as executor:
            futures = [
                executor.submit(
                    download_range,
                    url,
                    segment_path,
                    start,
                    end,
                    retries,
                )
                for (start, end), segment_path in zip(segment_ranges, segment_paths)
            ]
            for future in as_completed(futures):
                future.result()

        with partial.open("ab") as output:
            for segment_path in segment_paths:
                with segment_path.open("rb") as segment:
                    shutil.copyfileobj(segment, output, length=1024 * 1024)
                segment_path.unlink()

        if partial.stat().st_size != expected_size:
            raise OSError(f"segmented size mismatch for {target}")
        if expected_git_sha1 and git_blob_sha1(partial, expected_size) != expected_git_sha1:
            partial.unlink()
            raise OSError(f"Git blob SHA-1 mismatch for {target}")
        if expected_sha256 and sha256(partial) != expected_sha256:
            partial.unlink()
            raise OSError(f"SHA-256 mismatch for {target}")
        partial.replace(target)
        return

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            partial.unlink()
            offset = 0

        headers = {"User-Agent": "tokmem-benchmark-downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=120) as response:
                append = offset > 0 and getattr(response, "status", 200) == 206
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)

            if partial.stat().st_size != expected_size:
                raise OSError(
                    f"size mismatch for {target}: "
                    f"{partial.stat().st_size} != {expected_size}"
                )
            if expected_git_sha1 and git_blob_sha1(partial, expected_size) != expected_git_sha1:
                raise OSError(f"Git blob SHA-1 mismatch for {target}")
            if expected_sha256 and sha256(partial) != expected_sha256:
                raise OSError(f"SHA-256 mismatch for {target}")

            partial.replace(target)
            return
        except HTTPError as error:
            if error.code == 416 and partial.exists() and partial.stat().st_size == expected_size:
                if expected_git_sha1 and git_blob_sha1(partial, expected_size) != expected_git_sha1:
                    partial.unlink()
                elif expected_sha256 and sha256(partial) != expected_sha256:
                    partial.unlink()
                else:
                    partial.replace(target)
                    return
            if attempt == retries:
                raise
        except Exception:
            if attempt == retries:
                raise
        time.sleep(min(2 * attempt, 10))


def split_ranges(start: int, end: int, segments: int) -> list[tuple[int, int]]:
    total_size = end - start + 1
    chunk_size = (total_size + segments - 1) // segments
    return [
        (chunk_start, min(chunk_start + chunk_size - 1, end))
        for chunk_start in range(start, end + 1, chunk_size)
    ]


def download_range(
    url: str,
    target: Path,
    range_start: int,
    range_end: int,
    retries: int,
) -> None:
    expected_size = range_end - range_start + 1
    if target.exists() and target.stat().st_size > expected_size:
        target.unlink()

    for attempt in range(1, retries + 1):
        offset = target.stat().st_size if target.exists() else 0
        if offset == expected_size:
            return
        request = Request(
            url,
            headers={
                "User-Agent": "tokmem-benchmark-downloader",
                "Range": f"bytes={range_start + offset}-{range_end}",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                if getattr(response, "status", 200) != 206:
                    raise OSError(f"server ignored Range request for {target}")
                with target.open("ab") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if target.stat().st_size == expected_size:
                return
            raise OSError(f"range size mismatch for {target}")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))


def load_toolathlon_tree(args: argparse.Namespace) -> dict[str, Any]:
    if args.toolathlon_tree_json:
        with args.toolathlon_tree_json.open(encoding="utf-8") as handle:
            tree = json.load(handle)
    else:
        url = (
            f"https://api.github.com/repos/{TOOLATHLON_REPOSITORY}/git/trees/"
            f"{args.toolathlon_revision}?recursive=1"
        )
        tree = request_json(url, args.retries)
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated Toolathlon file tree")
    return tree


def download_toolathlon(args: argparse.Namespace) -> None:
    tree = load_toolathlon_tree(args)
    entries = [
        entry
        for entry in tree["tree"]
        if entry.get("type") == "blob"
        and entry["path"].startswith(TOOLATHLON_PREFIX)
    ]
    if not entries:
        raise RuntimeError("No Toolathlon final-pool files found")

    destination = args.output_root / "toolathlon"
    total_size = sum(entry["size"] for entry in entries)
    completed_size = 0
    failures: list[tuple[str, Exception]] = []
    print(
        f"Toolathlon: {len(entries)} files, "
        f"{total_size / 1024 / 1024:.1f} MiB, revision {args.toolathlon_revision}"
    )

    def download_entry(entry: dict[str, Any]) -> dict[str, Any]:
        relative_path = Path(entry["path"])
        target = destination / relative_path
        url = (
            f"{args.github_raw_prefix.rstrip('/')}/{TOOLATHLON_REPOSITORY}/"
            f"{args.toolathlon_revision}/{quote(entry['path'], safe='/')}"
        )
        download_with_resume(
            url,
            target,
            entry["size"],
            args.retries,
            expected_git_sha1=entry["sha"],
            segments=args.segments_per_large_file,
            segment_threshold=args.segment_threshold_mib * 1024 * 1024,
        )
        if entry.get("mode") == "100755":
            target.chmod(target.stat().st_mode | 0o111)
        return entry

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_entry = {
            executor.submit(download_entry, entry): entry for entry in entries
        }
        completed_files = 0
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            completed_files += 1
            try:
                future.result()
                completed_size += entry["size"]
            except Exception as error:
                failures.append((entry["path"], error))
            if completed_files % 100 == 0 or completed_files == len(entries):
                print(
                    f"  {completed_files}/{len(entries)} files checked, "
                    f"{completed_size / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MiB"
                )

    if failures:
        for path, error in failures[:20]:
            print(f"FAILED: {path}: {error}", file=sys.stderr)
        raise RuntimeError(f"{len(failures)} Toolathlon files failed; rerun to resume")

    metadata = {
        "benchmark": "Toolathlon-Verified",
        "source": f"https://github.com/{TOOLATHLON_REPOSITORY}",
        "revision": args.toolathlon_revision,
        "task_directory": TOOLATHLON_PREFIX.rstrip("/"),
        "task_count": 108,
        "file_count": len(entries),
        "total_bytes": total_size,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(destination / "DOWNLOAD_INFO.json", metadata)


def download_mcp_atlas(args: argparse.Namespace) -> None:
    destination = args.output_root / "mcp-atlas"
    target = destination / MCP_ATLAS_FILENAME
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    endpoints = list(dict.fromkeys([endpoint, "https://huggingface.co", "https://hf-mirror.com"]))
    failures: list[Exception] = []

    for current_endpoint in endpoints:
        url = (
            f"{current_endpoint}/datasets/{MCP_ATLAS_REPOSITORY}/resolve/"
            f"{MCP_ATLAS_REVISION}/{MCP_ATLAS_FILENAME}"
        )
        try:
            download_with_resume(
                url,
                target,
                MCP_ATLAS_SIZE,
                args.retries,
                expected_sha256=MCP_ATLAS_SHA256,
            )
            break
        except Exception as error:
            failures.append(error)
    else:
        raise RuntimeError(f"MCP-Atlas download failed: {failures[-1]}")

    metadata = {
        "benchmark": "MCP-Atlas public subset",
        "source": f"https://huggingface.co/datasets/{MCP_ATLAS_REPOSITORY}",
        "revision": MCP_ATLAS_REVISION,
        "filename": MCP_ATLAS_FILENAME,
        "rows": 500,
        "total_bytes": MCP_ATLAS_SIZE,
        "sha256": MCP_ATLAS_SHA256,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(destination / "DOWNLOAD_INFO.json", metadata)
    print(f"MCP-Atlas: verified {target}")


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("all", "toolathlon", "mcp-atlas"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--segments-per-large-file", type=int, default=1)
    parser.add_argument("--segment-threshold-mib", type=int, default=8)
    parser.add_argument("--toolathlon-revision", default=TOOLATHLON_REVISION)
    parser.add_argument("--toolathlon-tree-json", type=Path)
    parser.add_argument(
        "--github-raw-prefix",
        default="https://raw.githubusercontent.com",
        help="Raw GitHub base URL; mirrors are safe when Git blob verification is enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.segments_per_large_file < 1:
        raise ValueError("--segments-per-large-file must be positive")
    if args.benchmark in ("all", "toolathlon"):
        download_toolathlon(args)
    if args.benchmark in ("all", "mcp-atlas"):
        download_mcp_atlas(args)


if __name__ == "__main__":
    main()
