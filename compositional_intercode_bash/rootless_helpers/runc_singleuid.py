#!/usr/bin/python3
"""runc shim for a rootless namespace with only UID/GID 0 mapped.

Docker's generated OCI spec mounts devpts with ``gid=5``.  That group is
unmapped in the deliberately narrow single-GID namespace and the kernel rejects
the mount.  Before the real runc ``create`` call, this shim changes only that
mount option to ``gid=0``.  All other runc calls and OCI fields pass through.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


REAL_RUNC = str(
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "rootless-docker"
    / "bin"
    / "runc"
)


def _bundle_path(arguments: list[str]) -> Path | None:
    for index, argument in enumerate(arguments):
        if argument == "--bundle" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if argument.startswith("--bundle="):
            return Path(argument.split("=", 1)[1])
    return None


def _rewrite_devpts_gid(bundle: Path) -> None:
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    devpts_mounts = 0
    rewritten_options = 0
    for mount in config.get("mounts", []):
        if mount.get("type") != "devpts":
            continue
        devpts_mounts += 1
        options = mount.get("options", [])
        rewritten = []
        for option in options:
            if option == "gid=5":
                rewritten.append("gid=0")
                rewritten_options += 1
            else:
                rewritten.append(option)
        if rewritten != options:
            mount["options"] = rewritten
    if devpts_mounts != 1 or rewritten_options != 1:
        raise RuntimeError(
            "single-UID runtime expected exactly one devpts gid=5 option; "
            f"found mounts={devpts_mounts}, options={rewritten_options}"
        )
    original_mode = config_path.stat().st_mode & 0o777
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.singleuid.",
        suffix=".tmp",
        dir=bundle,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, original_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    arguments = sys.argv[1:]
    bundle = _bundle_path(arguments)
    if bundle is not None and "create" in arguments:
        _rewrite_devpts_gid(bundle)
    os.execv(REAL_RUNC, [REAL_RUNC, *arguments])


if __name__ == "__main__":
    main()
