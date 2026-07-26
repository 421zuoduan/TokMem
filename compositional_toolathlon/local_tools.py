from __future__ import annotations

import argparse
import asyncio
import json
import posixpath
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .config import load_experiment_config
from .context import materialize_workspace_paths, normalize_workspace_paths
from .manifest import (
    finalize_manifest,
    load_manifest,
    schema_hash,
    stable_tool_id,
    validate_tool_arguments,
)


PYTHON_EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "maxLength": 1600},
        "filename": {"type": "string", "maxLength": 128},
        "timeout": {"type": "number", "maximum": 120, "default": 30},
    },
    "required": ["code"],
    "additionalProperties": False,
}
MAX_CAPTURED_STREAM_BYTES = 200_000
PYTHON_JAIL_WORKSPACE_ROOT = "/workspace"


async def _read_bounded_stream(
    stream: asyncio.StreamReader,
    limit: int = MAX_CAPTURED_STREAM_BYTES,
) -> tuple[bytes, bool]:
    retained = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(retained), truncated


def python_execute_manifest_record() -> dict[str, Any]:
    raw_name = "local-python-execute"
    digest = schema_hash(raw_name, PYTHON_EXECUTE_SCHEMA)
    return {
        "server": "toolathlon-host",
        "tool_name": raw_name,
        "wire_name": None,
        "raw_name": raw_name,
        "model_name": "local_python_execute",
        "origin": "host_local",
        "group_key": "python_execute",
        "dispatch_kind": "local_python",
        "state_requirements": [
            "episode_workspace",
            "bubblewrap_workspace_only",
            "network_namespace_disabled",
        ],
        "description": (
            "Execute a short Python file in a workspace-only, network-disabled "
            "sandbox with a timeout of at most 120 seconds."
        ),
        "input_schema": PYTHON_EXECUTE_SCHEMA,
        "schema_hash": digest,
        "stable_id": stable_tool_id("toolathlon-host", raw_name, digest),
    }


def augment_default_decoupled_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if any(record.get("origin") == "host_local" for record in manifest["tools"]):
        raise ValueError("manifest is already augmented with host-local tools")
    records = [dict(record) for record in manifest["tools"]]
    for record in records:
        record.pop("memory_slot", None)
    records.append(python_execute_manifest_record())
    config = load_experiment_config()
    augmented = finalize_manifest(
        records,
        manifest["benchmark_revision"],
        tapmem_capacity=config.interface.tapmem_tool_capacity,
    )
    augmented["gateway_capture"] = manifest.get("gateway_capture")
    augmented["runtime_interface"] = {
        "name": "toolathlon_default_decoupled",
        "gateway_tools": "raw tools/list including local-claim_done",
        "host_local_groups": ["python_execute"],
        "filtered_groups": [
            "claim_done",
            "manage_context",
            "history",
            "handle_overlong_tool_outputs",
        ],
    }
    return augmented


class PythonExecuteTool:
    def __init__(
        self,
        workspace_root: str | Path,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.is_dir():
            raise ValueError(f"episode workspace does not exist: {self.workspace_root}")
        self.bwrap_binary = shutil.which("bwrap")
        if self.bwrap_binary is None:
            raise RuntimeError(
                "local-python-execute requires bubblewrap; unsafe host execution "
                "has no fallback"
            )
        probe_command = [
            self.bwrap_binary,
            "--die-with-parent",
            "--unshare-user",
            "--unshare-net",
            "--new-session",
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(system_path).exists():
                probe_command.extend(["--ro-bind", system_path, system_path])
        probe_command.append("/usr/bin/true")
        probe = subprocess.run(
            probe_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "bubblewrap cannot create the required user/network jail; "
                "unsafe host execution has no fallback: "
                + probe.stderr.strip()[:500]
            )
        self.python_prefix = Path(sys.prefix).resolve()
        self.python_base_prefix = Path(sys.base_prefix).resolve()
        executable = Path(sys.executable).resolve()
        try:
            relative_executable = executable.relative_to(self.python_prefix)
        except ValueError:
            try:
                relative_executable = executable.relative_to(
                    self.python_base_prefix
                )
            except ValueError as exc:
                raise RuntimeError(
                    "current Python executable is outside sys.prefix and "
                    "sys.base_prefix and cannot be mounted into the tool jail"
                ) from exc
            self.jailed_python = Path("/base_runtime") / relative_executable
            self.jailed_pythonpath = (
                f"/runtime/lib/python{sys.version_info.major}."
                f"{sys.version_info.minor}/site-packages"
            )
        else:
            self.jailed_python = Path("/runtime") / relative_executable
            self.jailed_pythonpath = None

    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"code", "filename", "timeout"}:
            raise ValueError("local-python-execute received additional properties")
        code = arguments.get("code")
        if not isinstance(code, str):
            raise ValueError("local-python-execute requires string code")
        timeout = arguments.get("timeout", 30)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValueError("local-python-execute timeout must be numeric")
        timeout = float(timeout)
        if not 0 < timeout <= 120:
            raise ValueError("local-python-execute timeout must be in (0, 120]")
        filename = arguments.get("filename") or f"{uuid.uuid4().hex}.py"
        if not isinstance(filename, str) or not filename:
            raise ValueError("local-python-execute filename must be a string")
        filename = Path(filename).name
        if not filename.endswith(".py"):
            filename += ".py"
        temporary_dir = self.workspace_root / ".python_tmp"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        if temporary_dir.is_symlink() or not temporary_dir.resolve().is_relative_to(
            self.workspace_root
        ):
            raise RuntimeError("workspace Python temporary directory escaped the jail")
        script_name = f"{uuid.uuid4().hex}_{filename}"
        script_path = temporary_dir / script_name
        with script_path.open("x", encoding="utf-8") as handle:
            handle.write(code)

        command = [
            self.bwrap_binary,
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--new-session",
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(system_path).exists():
                command.extend(["--ro-bind", system_path, system_path])
        command.extend(
            [
                "--ro-bind",
                str(self.python_prefix),
                "/runtime",
            ]
        )
        if self.python_base_prefix != self.python_prefix:
            command.extend(
                [
                    "--ro-bind",
                    str(self.python_base_prefix),
                    "/base_runtime",
                ]
            )
        command.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--bind",
                str(self.workspace_root),
                "/workspace",
                "--chdir",
                "/workspace",
                "--setenv",
                "HOME",
                "/workspace",
                "--setenv",
                "PATH",
                "/runtime/bin:/usr/bin:/bin",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
            ]
        )
        if self.jailed_pythonpath is not None:
            command.extend(
                [
                    "--setenv",
                    "PYTHONPATH",
                    self.jailed_pythonpath,
                ]
            )
        command.extend(
            [
                str(self.jailed_python),
                f"/workspace/.python_tmp/{script_name}",
            ]
        )

        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                },
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_task = asyncio.create_task(
                _read_bounded_stream(process.stdout)
            )
            stderr_task = asyncio.create_task(
                _read_bounded_stream(process.stderr)
            )
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=timeout,
                )
                return_code = int(process.returncode)
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return_code = -9
                timed_out = True
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
        finally:
            script_path.unlink(missing_ok=True)
            try:
                temporary_dir.rmdir()
            except OSError:
                pass
        elapsed = time.monotonic() - start
        observation = (
            f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}\n"
            f"STDERR:\n{stderr.decode('utf-8', errors='replace')}\n"
            f"RETURN_CODE: {return_code}\n"
            f"TIME_SECONDS: {elapsed:.3f}\n"
            f"TIMED_OUT: {str(timed_out).lower()}\n"
            f"STDOUT_TRUNCATED: {str(stdout_truncated).lower()}\n"
            f"STDERR_TRUNCATED: {str(stderr_truncated).lower()}"
        )
        return {
            "success": return_code == 0 and not timed_out,
            "observation": observation,
            "runtime_metadata": {
                "return_code": return_code,
                "timed_out": timed_out,
                "sandbox": "bubblewrap_workspace_only",
                "network_disabled": True,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
        }


class CompositeToolExecutor:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        mcp_executor: Any,
        workspace_root: str | Path,
        mcp_workspace_root: str | Path | None = None,
        enable_python_execute: bool,
        validate_mcp_arguments: bool = True,
    ) -> None:
        self.manifest = manifest
        self.mcp_executor = mcp_executor
        self.workspace_root = Path(workspace_root).resolve()
        raw_mcp_workspace_root = str(
            mcp_workspace_root
            if mcp_workspace_root is not None
            else self.workspace_root
        )
        if (
            not posixpath.isabs(raw_mcp_workspace_root)
            or posixpath.normpath(raw_mcp_workspace_root)
            != raw_mcp_workspace_root
        ):
            raise ValueError("MCP workspace root must be a normalized absolute path")
        self.mcp_workspace_root = raw_mcp_workspace_root
        self.validate_mcp_arguments = validate_mcp_arguments
        self.records = {
            record["stable_id"]: record for record in manifest["tools"]
        }
        self.python_execute = (
            PythonExecuteTool(
                self.workspace_root,
            )
            if enable_python_execute
            else None
        )

    async def call_tool(
        self,
        stable_tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            record = self.records[stable_tool_id]
        except KeyError as exc:
            raise ValueError(f"unknown stable tool ID: {stable_tool_id}") from exc
        dispatch = record["dispatch_kind"]
        if dispatch in {"mcp", "terminal"}:
            runtime_arguments = materialize_workspace_paths(
                arguments,
                self.mcp_workspace_root,
            )
            if self.validate_mcp_arguments:
                validate_tool_arguments(record, runtime_arguments)
            outcome = await self.mcp_executor.call_tool(
                stable_tool_id,
                runtime_arguments,
            )
            if isinstance(outcome, dict) and "observation" in outcome:
                outcome = dict(outcome)
                outcome["observation"] = normalize_workspace_paths(
                    outcome["observation"],
                    self.mcp_workspace_root,
                )
            return outcome
        if dispatch == "local_python":
            if self.python_execute is None:
                raise RuntimeError("local-python-execute is not enabled for this task")
            runtime_arguments = materialize_workspace_paths(
                arguments,
                PYTHON_JAIL_WORKSPACE_ROOT,
            )
            validate_tool_arguments(record, runtime_arguments)
            outcome = await self.python_execute.call(runtime_arguments)
            if isinstance(outcome, dict) and "observation" in outcome:
                outcome = dict(outcome)
                outcome["observation"] = normalize_workspace_paths(
                    outcome["observation"],
                    PYTHON_JAIL_WORKSPACE_ROOT,
                )
            return outcome
        raise ValueError(f"unsupported tool dispatch kind: {dispatch}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment a raw gateway manifest for official default decoupled mode"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    augment = subparsers.add_parser("augment-manifest")
    augment.add_argument("--input", required=True)
    augment.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command != "augment-manifest":
        raise ValueError(f"unsupported command: {args.command}")
    augmented = augment_default_decoupled_manifest(load_manifest(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Augmented manifest with default decoupled host tools; "
        f"tool_count={augmented['tool_count']} hash={augmented['manifest_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
