"""Download the pinned 9,305-row NL2Bash release into this experiment folder."""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

from .data_sources import (
    CANONICAL_NL2BASH_REPOSITORY,
    CANONICAL_NL2BASH_REVISION,
    CANONICAL_SPLIT_FILES,
    DEFAULT_CANONICAL_SPLIT_DIR,
    load_canonical_nl2bash_splits,
)
from .io_utils import ensure_output_path, sha256_file, write_json


def _download_one(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    replace: bool,
    timeout_seconds: int,
) -> str:
    if destination.exists():
        actual_hash = sha256_file(destination)
        if actual_hash == expected_sha256:
            return "already_present"
        if not replace:
            raise ValueError(
                f"{destination} exists but has SHA-256 {actual_hash}; "
                "pass --replace only if it is safe to replace this artifact"
            )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".download",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "tapmem-intercode-bash-data/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with temporary_path.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        actual_hash = sha256_file(temporary_path)
        if actual_hash != expected_sha256:
            raise ValueError(
                f"Downloaded SHA-256 mismatch for {url}: "
                f"expected={expected_sha256}, actual={actual_hash}"
            )
        os.replace(temporary_path, destination)
        return "downloaded"
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def download_canonical_split(
    output_dir: str | Path,
    *,
    endpoint: str,
    replace: bool = False,
    timeout_seconds: int = 120,
) -> dict:
    output_dir = ensure_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = (
        endpoint.rstrip("/")
        + "/datasets/"
        + CANONICAL_NL2BASH_REPOSITORY
        + "/resolve/"
        + CANONICAL_NL2BASH_REVISION
    )
    statuses = {}
    for split, specification in CANONICAL_SPLIT_FILES.items():
        destination = output_dir / str(specification["filename"])
        url = base_url + "/" + str(specification["remote_path"])
        statuses[split] = _download_one(
            url,
            destination,
            str(specification["sha256"]),
            replace=replace,
            timeout_seconds=timeout_seconds,
        )

    _pairs, manifest = load_canonical_nl2bash_splits(output_dir)
    manifest["download_status"] = statuses
    manifest["endpoint"] = endpoint
    write_json(output_dir / "download_manifest.json", manifest)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CANONICAL_SPLIT_DIR),
    )
    parser.add_argument(
        "--endpoint",
        default="https://huggingface.co",
        help="Use https://hf-mirror.com when the primary endpoint is unreachable",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = download_canonical_split(
        args.output_dir,
        endpoint=args.endpoint,
        replace=args.replace,
        timeout_seconds=args.timeout_seconds,
    )
    print(report)


if __name__ == "__main__":
    main()
