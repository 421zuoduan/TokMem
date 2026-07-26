"""Build the four distinct InterCode-Bash filesystem images."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

from .io_utils import PACKAGE_ROOT, ensure_output_path, write_json
from .official_env import (
    BENCHMARK_LABEL,
    BENCHMARK_LABEL_VALUE,
    FILESYSTEM_LABEL,
)


IMMUTABLE_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")
LOCAL_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intercode-root",
        default=str(PACKAGE_ROOT.parent / "datasets" / "intercode"),
    )
    parser.add_argument(
        "--base-image",
        required=True,
        help="Immutable Docker reference, for example ubuntu@sha256:<64 hex digits>",
    )
    parser.add_argument(
        "--single-uid-rootless",
        action="store_true",
        help=(
            "Use a locally normalized immutable image ID and the build-only "
            "single-UID ownership compatibility path."
        ),
    )
    parser.add_argument("--log-dir", default=str(PACKAGE_ROOT / "artifacts" / "docker"))
    args = parser.parse_args(argv)
    immutable_registry_ref = IMMUTABLE_IMAGE_RE.fullmatch(args.base_image)
    immutable_local_id = LOCAL_IMAGE_ID_RE.fullmatch(args.base_image)
    if not immutable_registry_ref and not (
        args.single_uid_rootless and immutable_local_id
    ):
        parser.error(
            "--base-image must be pinned by an immutable sha256 digest; "
            "single-UID rootless mode additionally accepts a local "
            "sha256:<64 hex digits> image ID"
        )
    intercode_root = Path(args.intercode_root).resolve()
    dockerfile = PACKAGE_ROOT / "docker" / "nl2bash_fs.Dockerfile"
    log_dir = ensure_output_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "intercode_bash_images_v1",
        "base_image": args.base_image,
        "single_uid_rootless": args.single_uid_rootless,
        "images": {},
    }
    for fs_number in range(1, 5):
        image_name = f"intercode-nl2bash-fs{fs_number}"
        command = [
            "docker",
            "build",
            "--file",
            str(dockerfile),
            "--build-arg",
            f"FILE_SYSTEM_VERSION={fs_number}",
            "--build-arg",
            f"BASE_IMAGE={args.base_image}",
            "--label",
            f"{FILESYSTEM_LABEL}=fs{fs_number}",
            "--label",
            f"{BENCHMARK_LABEL}={BENCHMARK_LABEL_VALUE}",
            "--tag",
            image_name,
            str(intercode_root),
        ]
        if args.single_uid_rootless:
            command[2:2] = ["--network", "host"]
            command[2:2] = ["--force-rm"]
            command[2:2] = [
                "--build-arg",
                "APT_SANDBOX_USER=root",
            ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        (log_dir / f"{image_name}.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (log_dir / f"{image_name}.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to build {image_name}; inspect logs under {log_dir}"
            )
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_name, "--format", "{{json .}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        inspect_value = json.loads(inspect.stdout)
        labels = inspect_value.get("Config", {}).get("Labels", {}) or {}
        expected_fs_id = f"fs{fs_number}"
        if (
            labels.get(FILESYSTEM_LABEL) != expected_fs_id
            or labels.get(BENCHMARK_LABEL) != BENCHMARK_LABEL_VALUE
        ):
            raise RuntimeError(
                f"Built image {image_name} has incorrect filesystem labels"
            )
        manifest["images"][expected_fs_id] = {
            "name": image_name,
            "image_id": inspect_value["Id"],
            "labels": {
                FILESYSTEM_LABEL: labels[FILESYSTEM_LABEL],
                BENCHMARK_LABEL: labels[BENCHMARK_LABEL],
            },
            "inspect": inspect_value,
        }
    image_ids = [
        value["image_id"] for value in manifest["images"].values()
    ]
    if len(set(image_ids)) != 4:
        raise RuntimeError("The four filesystem builds did not produce four image IDs")
    write_json(log_dir / "image_manifest.json", manifest)


if __name__ == "__main__":
    main()
