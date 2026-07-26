"""Deterministic artifact I/O used by the InterCode-Bash experiment."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash relative names and contents of every regular file in a directory."""

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def stable_id(namespace: str, *parts: object) -> str:
    payload = "\0".join([namespace, *(str(part) for part in parts)])
    return sha256_text(payload)


def json_safe_log_value(value: Any) -> Any:
    """Convert environment diagnostics to JSON without exposing custom objects."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else repr(numeric)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe_log_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe_log_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            json_safe_log_value(item)
            for item in sorted(value, key=lambda item: repr(item))
        ]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError, RuntimeError):
            pass
        else:
            if scalar is not value:
                return json_safe_log_value(scalar)
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return json_safe_log_value(list_method())
        except (TypeError, ValueError, RuntimeError):
            pass
    return repr(value)


def ensure_output_path(path: str | Path) -> Path:
    """Reject writes outside this experiment directory."""

    candidate = Path(path).resolve()
    try:
        candidate.relative_to(PACKAGE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Experiment outputs must stay below {PACKAGE_ROOT}; got {candidate}"
        ) from exc
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _atomic_replace(path: Path, payload: bytes) -> None:
    path = ensure_output_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_replace(Path(path), payload + b"\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [canonical_json_bytes(record) for record in records]
    _atomic_replace(Path(path), b"\n".join(lines) + (b"\n" if lines else b""))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        return {
            "path": str(path.resolve()),
            "sha256": sha256_directory(path),
            "bytes": sum(candidate.stat().st_size for candidate in files),
            "files": len(files),
            "kind": "directory",
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "kind": "file",
    }
