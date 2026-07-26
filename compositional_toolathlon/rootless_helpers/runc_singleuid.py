#!/usr/bin/env python3
"""Adapt devpts to a rootless namespace with only GID 0 mapped."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


DEFAULT_RUNC = (
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
    changed = False
    for mount in config.get("mounts", []):
        if mount.get("type") != "devpts":
            continue
        options = mount.get("options", [])
        rewritten = [
            "gid=0" if option == "gid=5" else option for option in options
        ]
        if rewritten != options:
            mount["options"] = rewritten
            changed = True
    if changed:
        temporary = config_path.with_suffix(".json.singleuid.tmp")
        temporary.write_text(
            json.dumps(config, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, config_path)


def main() -> None:
    arguments = sys.argv[1:]
    bundle = _bundle_path(arguments)
    if bundle is not None and "create" in arguments:
        _rewrite_devpts_gid(bundle)
    real_runc = Path(
        os.environ.get("TOOLATHLON_ROOTLESS_RUNC", str(DEFAULT_RUNC))
    ).resolve()
    if not real_runc.is_file():
        raise SystemExit(f"rootless runc binary is missing: {real_runc}")
    os.execv(str(real_runc), [str(real_runc), *arguments])


if __name__ == "__main__":
    main()
