from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, ExperimentConfig, load_experiment_config


def run_probe(
    command: list[str],
    timeout: int = 10,
    *,
    require_stdout: bool = False,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    ok = completed.returncode == 0 and (bool(stdout) or not require_stdout)
    return {
        "ok": ok,
        "returncode": completed.returncode,
        "output": stdout[:1000],
        "error": stderr[:1000],
    }


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    probe = run_probe(["git", "-C", str(path), "rev-parse", "HEAD"])
    return probe.get("output") if probe.get("ok") else None


def benchmark_provenance(path: Path) -> dict[str, Any]:
    revision = git_revision(path)
    if revision:
        return {"source_kind": "git", "revision": revision}
    marker = path / ".toolathlon-source.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "source_kind": "invalid_marker",
                "revision": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "source_kind": payload.get("source_kind"),
            "revision": payload.get("revision"),
            "source_url": payload.get("source_url"),
            "archive_sha256": payload.get("archive_sha256"),
        }
    return {"source_kind": "unknown", "revision": None}


def runner_hook_status(
    benchmark_root: Path,
    expected_revision: str,
) -> dict[str, Any]:
    runner = benchmark_root / "scripts" / "run_single_decoupled.sh"
    marker = benchmark_root / "scripts" / ".tokmem-runtime-hook.json"
    if not runner.is_file() or not marker.is_file():
        return {
            "ok": False,
            "error": "patched runner or hook marker is missing",
        }
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": bool(
            metadata.get("revision") == expected_revision
            and metadata.get("patched_sha256") == runner_sha256
            and metadata.get("original_sha256")
        ),
        "revision": metadata.get("revision"),
        "original_sha256": metadata.get("original_sha256"),
        "patched_sha256_expected": metadata.get("patched_sha256"),
        "patched_sha256_actual": runner_sha256,
    }


def find_benchmark_root(config: ExperimentConfig) -> Path | None:
    override = os.environ.get("TOOLATHLON_ROOT")
    candidates = [Path(override)] if override else []
    candidates.extend(config.benchmark_candidates())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def audit_environment(config: ExperimentConfig) -> dict[str, Any]:
    benchmark_root = find_benchmark_root(config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_revision_expected": config.benchmark.revision,
        "benchmark_root": str(benchmark_root) if benchmark_root else None,
        "checks": {},
    }
    checks = report["checks"]

    checks["tokmem_conda_env"] = {
        "ok": os.environ.get("CONDA_DEFAULT_ENV") == "tokmem",
        "value": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    checks["gpt56_endpoint"] = {
        "ok": bool(os.environ.get("TOOLATHLON_LLM_BASE_URL")),
        "base_url_configured": bool(
            os.environ.get("TOOLATHLON_LLM_BASE_URL")
        ),
        "api_key_configured": bool(
            os.environ.get("TOOLATHLON_LLM_API_KEY")
        ),
    }
    checks["gpt56_endpoint"]["ok"] = bool(
        checks["gpt56_endpoint"]["base_url_configured"]
        and checks["gpt56_endpoint"]["api_key_configured"]
    )
    for executable in ("git", "docker", "node", "npm", "npx", "uv", "bwrap"):
        path = shutil.which(executable)
        checks[f"executable_{executable}"] = {"ok": path is not None, "path": path}
    bwrap_path = checks["executable_bwrap"]["path"]
    if bwrap_path:
        checks["python_execute_jail"] = run_probe(
            [
                bwrap_path,
                "--die-with-parent",
                "--unshare-user",
                "--unshare-net",
                "--new-session",
                "--ro-bind",
                "/usr",
                "/usr",
                "/usr/bin/true",
            ]
        )
    else:
        checks["python_execute_jail"] = {
            "ok": False,
            "error": "bubblewrap executable is unavailable",
        }
    checks["tokmem_mcp_client"] = run_probe(
        [
            sys.executable,
            "-c",
            (
                "import mcp; from mcp import ClientSession; "
                "from mcp.client.sse import sse_client; print('mcp-client-ok')"
            ),
        ],
        require_stdout=True,
    )

    runtime_python = REPO_ROOT / "compositional_toolathlon" / ".toolathlon-venv" / "bin" / "python"
    if runtime_python.is_file():
        checks["official_python_runtime"] = run_probe(
            [
                str(runtime_python),
                "-c",
                (
                    "import agents, jsonschema, mcp, openai, platform; "
                    "assert platform.python_version() == '3.12.11'; "
                    "print(platform.python_version())"
                ),
            ],
            timeout=30,
            require_stdout=True,
        )
    else:
        checks["official_python_runtime"] = {
            "ok": False,
            "error": f"runtime not found: {runtime_python}",
        }

    if benchmark_root is None:
        checks["benchmark_checkout"] = {"ok": False, "error": "checkout not found"}
        checks["local_office_10_tasks"] = {
            "ok": False,
            "error": "checkout not found",
        }
        checks["official_runner_files"] = {
            "ok": False,
            "error": "checkout not found",
        }
        checks["tokmem_runner_hook"] = {
            "ok": False,
            "error": "checkout not found",
        }
    else:
        provenance = benchmark_provenance(benchmark_root)
        revision = provenance.get("revision")
        checks["benchmark_checkout"] = {
            "ok": revision == config.benchmark.revision,
            "revision": revision,
            "provenance": provenance,
        }
        task_root = benchmark_root / config.benchmark.task_directory
        missing_tasks = [
            task_id for task_id in config.task_ids if not (task_root / task_id).is_dir()
        ]
        checks["local_office_10_tasks"] = {
            "ok": not missing_tasks,
            "missing": missing_tasks,
        }
        missing_runner = [
            relative
            for relative in config.required_runner_paths
            if not (benchmark_root / relative).exists()
        ]
        checks["official_runner_files"] = {
            "ok": not missing_runner,
            "missing": missing_runner,
        }
        checks["tokmem_runner_hook"] = runner_hook_status(
            benchmark_root,
            config.benchmark.revision,
        )

    docker_probe = run_probe(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        require_stdout=True,
    )
    checks["docker_daemon"] = docker_probe
    if docker_probe.get("ok"):
        checks["prepared_image"] = run_probe(
            [
                "docker",
                "image",
                "inspect",
                config.benchmark.prepared_image,
                "--format",
                "{{json .RepoDigests}}",
            ],
            require_stdout=True,
        )
    else:
        checks["prepared_image"] = {
            "ok": False,
            "error": "not checked because Docker daemon is unavailable",
        }

    try:
        import torch

        checks["cuda"] = {
            "ok": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except ImportError:
        checks["cuda"] = {"ok": False, "error": "torch is not installed"}

    required = (
        "benchmark_checkout",
        "local_office_10_tasks",
        "official_runner_files",
        "tokmem_runner_hook",
        "official_python_runtime",
        "executable_bwrap",
        "python_execute_jail",
        "docker_daemon",
        "prepared_image",
    )
    report["ready_for_real_mcp"] = all(checks[name].get("ok", False) for name in required)
    report["ready_for_training"] = (
        checks["tokmem_conda_env"]["ok"] and checks["cuda"]["ok"]
    )
    report["ready_for_gpt56_trajectory_collection"] = (
        report["ready_for_real_mcp"]
        and checks["gpt56_endpoint"]["ok"]
    )
    return report


def _audit_command(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    report = audit_environment(config)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.strict and not report["ready_for_real_mcp"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the Toolathlon experiment environment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "--config",
        default=str(REPO_ROOT / "compositional_toolathlon" / "configs" / "experiment.json"),
    )
    audit.add_argument("--output", default=None)
    audit.add_argument("--strict", action="store_true")
    audit.set_defaults(func=_audit_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
