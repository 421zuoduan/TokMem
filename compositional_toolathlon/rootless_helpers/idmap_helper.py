"""Create the narrow ID map used by the no-sudo Toolathlon runtime.

This helper intentionally maps only container ID 0 to the invoking user's
host ID.  It does not grant subordinate IDs and must not be presented as a
standard multi-UID rootless Docker installation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _read_process_owner(pid: int, kind: str) -> int:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    prefix = "Uid:" if kind == "uid" else "Gid:"
    for line in status.splitlines():
        if line.startswith(prefix):
            return int(line.split()[1])
    raise RuntimeError(f"{prefix} missing from /proc/{pid}/status")


def write_single_id_map(kind: str, arguments: list[str]) -> None:
    if kind not in {"uid", "gid"}:
        raise ValueError(f"unsupported ID-map kind: {kind}")
    if len(arguments) < 4 or (len(arguments) - 1) % 3:
        raise ValueError("expected PID followed by one or more ID mapping triples")

    pid = int(arguments[0])
    requested = [int(value) for value in arguments[1:]]
    host_id = os.getuid() if kind == "uid" else os.getgid()
    if requested[:3] != [0, host_id, 1]:
        raise PermissionError(
            f"refusing mapping {requested[:3]}; "
            f"only [0, {host_id}, 1] is allowed"
        )
    if _read_process_owner(pid, kind) != host_id:
        raise PermissionError(
            f"process {pid} is not owned by host {kind} {host_id}"
        )

    process_root = Path(f"/proc/{pid}")
    if kind == "gid":
        setgroups = process_root / "setgroups"
        if setgroups.exists():
            setgroups.write_text("deny", encoding="ascii")
    (process_root / f"{kind}_map").write_text(
        f"0 {host_id} 1\n",
        encoding="ascii",
    )


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: idmap_helper.py <uid|gid> PID mappings...")
    write_single_id_map(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
