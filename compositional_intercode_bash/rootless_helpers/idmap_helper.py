"""Minimal ID-map helper for the experiment's single-UID rootless runtime.

This deliberately does *not* grant subordinate IDs.  It only writes the
one-to-one mapping that an unprivileged user is already allowed to create:
container ID 0 maps to the invoking user's own host UID/GID.
"""

from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path


def _read_process_owner(pid: int, kind: str) -> int:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    prefix = "Uid:" if kind == "uid" else "Gid:"
    for line in status.splitlines():
        if line.startswith(prefix):
            return int(line.split()[1])
    raise RuntimeError(f"{prefix} missing from /proc/{pid}/status")


def _configured_subid_triple(kind: str) -> list[int] | None:
    username = pwd.getpwuid(os.getuid()).pw_name
    path = Path("/etc/subuid" if kind == "uid" else "/etc/subgid")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(":")
        if len(fields) == 3 and fields[0] in {username, str(os.getuid())}:
            start, count = int(fields[1]), int(fields[2])
            return [1, start, count]
    return None


def write_single_id_map(kind: str, argv: list[str]) -> None:
    if kind not in {"uid", "gid"}:
        raise ValueError(f"unsupported ID-map kind: {kind}")
    if len(argv) not in {4, 7}:
        raise ValueError(
            "expected PID 0 HOST_ID 1, optionally followed by the one "
            "subordinate-ID range RootlessKit discovered"
        )

    pid = int(argv[0])
    requested = [int(value) for value in argv[1:]]
    host_id = os.getuid() if kind == "uid" else os.getgid()
    if requested[:3] != [0, host_id, 1]:
        raise PermissionError(
            f"refusing leading mapping {requested[:3]}; "
            f"only [0, {host_id}, 1] is allowed"
        )
    if len(requested) == 6:
        configured = _configured_subid_triple(kind)
        if configured is None or requested[3:] != configured:
            raise PermissionError(
                f"refusing unexpected subordinate-ID request {requested[3:]}; "
                f"configured={configured}"
            )
    if _read_process_owner(pid, kind) != host_id:
        raise PermissionError(f"process {pid} is not owned by host {kind} {host_id}")

    proc_dir = Path(f"/proc/{pid}")
    if kind == "gid":
        setgroups = proc_dir / "setgroups"
        if setgroups.exists():
            setgroups.write_text("deny", encoding="ascii")
    (proc_dir / f"{kind}_map").write_text(
        f"0 {host_id} 1\n",
        encoding="ascii",
    )
    actual = [
        [int(value) for value in line.split()]
        for line in (proc_dir / f"{kind}_map").read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if actual != [[0, host_id, 1]]:
        raise RuntimeError(f"kernel installed an unexpected {kind} map: {actual}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: idmap_helper.py <uid|gid> PID mappings...")
    write_single_id_map(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
