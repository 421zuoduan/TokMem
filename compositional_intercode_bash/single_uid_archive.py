"""Rewrite a Docker archive for a single-UID rootless daemon.

The fallback daemon used by this experiment maps only the invoking host user to
UID/GID 0 inside its user namespace.  A regular Docker archive may contain
files owned by other IDs, which such a daemon cannot unpack.  This utility
creates a separate archive whose layer entries are all owned by root.  The
image configuration and its content-addressed filename are updated to match
the rewritten layer digests; the downloaded source archive is never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import BinaryIO


def _normalize_tar(stream: BinaryIO) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="r:*") as source:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member in source:
                member.uid = 0
                member.gid = 0
                member.uname = "root"
                member.gname = "root"
                payload = source.extractfile(member) if member.isfile() else None
                target.addfile(member, payload)
    return output.getvalue()


def rewrite_archive(input_path: Path, output_path: Path) -> dict[str, object]:
    with tarfile.open(input_path, mode="r:*") as source:
        members: dict[str, tuple[tarfile.TarInfo, bytes]] = {}
        for member in source:
            if not member.isfile():
                continue
            payload = source.extractfile(member)
            if payload is None:
                raise ValueError(f"could not read archive member: {member.name}")
            members[member.name] = (member, payload.read())

    if "manifest.json" not in members:
        raise ValueError("Docker archive does not contain manifest.json")
    manifest = json.loads(members["manifest.json"][1])
    rewritten_layers: dict[str, bytes] = {}
    rewritten_configs: dict[str, tuple[str, bytes]] = {}

    for image in manifest:
        diff_ids: list[str] = []
        for layer_name in image["Layers"]:
            if layer_name not in rewritten_layers:
                rewritten_layers[layer_name] = _normalize_tar(
                    io.BytesIO(members[layer_name][1])
                )
            digest = hashlib.sha256(rewritten_layers[layer_name]).hexdigest()
            diff_ids.append(f"sha256:{digest}")

        old_config_name = image["Config"]
        config = json.loads(members[old_config_name][1])
        config.setdefault("rootfs", {})["diff_ids"] = diff_ids
        config_bytes = json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        new_config_name = f"{hashlib.sha256(config_bytes).hexdigest()}.json"
        rewritten_configs[old_config_name] = (new_config_name, config_bytes)
        image["Config"] = new_config_name

    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, mode="w:gz", format=tarfile.PAX_FORMAT) as target:
        emitted: set[str] = set()
        for name, (source_info, original_bytes) in members.items():
            if name in rewritten_configs:
                new_name, payload = rewritten_configs[name]
            elif name in rewritten_layers:
                new_name, payload = name, rewritten_layers[name]
            elif name == "manifest.json":
                new_name, payload = name, manifest_bytes
            else:
                new_name, payload = name, original_bytes
            info = tarfile.TarInfo(new_name)
            info.size = len(payload)
            info.mode = source_info.mode
            info.mtime = source_info.mtime
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            target.addfile(info, io.BytesIO(payload))
            emitted.add(new_name)

        # Legacy Docker archives contain directory entries for layer paths.
        for layer_name in rewritten_layers:
            directory = layer_name.rsplit("/", 1)[0] + "/"
            if directory in emitted:
                continue
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            target.addfile(info)

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "layers": {
            name: f"sha256:{hashlib.sha256(payload).hexdigest()}"
            for name, payload in rewritten_layers.items()
        },
        "configs": {
            old_name: new_name
            for old_name, (new_name, _) in rewritten_configs.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = rewrite_archive(args.input, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
