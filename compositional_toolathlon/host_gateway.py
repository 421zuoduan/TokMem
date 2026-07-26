from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import PACKAGE_DIR, ExperimentConfig, load_experiment_config
from .environment import benchmark_provenance, find_benchmark_root


TERMINAL_ALLOWED_COMMANDS = (
    "ls,cat,pwd,echo,cd,mkdir,touch,rm,cp,mv,find,grep,head,tail,wc,"
    "sort,uniq,diff,tree,chmod,stat,file,which,whoami,date,hostname,df,"
    "du,ps,env,history,clear,sed,awk,cut,tr,basename,dirname,realpath,"
    "md5sum,sha256sum,tar,gzip,gunzip,zip,unzip,less,more,python,wget,"
    "curl,ping,netstat,ifconfig,nslookup,traceroute,helm,kubectl,git"
)
FILESYSTEM_SERVER_PACKAGE = (
    "@modelcontextprotocol/server-filesystem@2026.7.10"
)
PDF_TOOLS_SERVER_PACKAGE = "pdf-tools-mcp==0.1.4"
TERMINAL_SERVER_VERSION = "0.2.4"
EXCEL_SERVER_VERSION = "0.1.4"


@dataclass(frozen=True)
class HostGatewayRuntime:
    root: Path
    config_dir: Path
    bundle_file: Path
    workspace: Path
    benchmark_root: Path
    servers: tuple[str, ...]


def normalize_server_names(
    requested: Sequence[str] | None,
    *,
    allowed: Sequence[str],
) -> tuple[str, ...]:
    allowed_names = tuple(allowed)
    if requested is None:
        return allowed_names

    parsed: list[str] = []
    for value in requested:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one MCP server is required")

    unknown = sorted(set(parsed) - set(allowed_names))
    if unknown:
        raise ValueError(
            f"unsupported host MCP servers: {unknown}; allowed={list(allowed_names)}"
        )
    requested_set = set(parsed)
    return tuple(name for name in allowed_names if name in requested_set)


def build_server_config(
    server_name: str,
    *,
    workspace: str | Path,
    runtime_root: str | Path,
    package_dir: str | Path,
    benchmark_root: str | Path,
    npx_command: str,
    uvx_command: str,
) -> dict[str, Any]:
    workspace_path = Path(workspace).resolve()
    runtime_path = Path(runtime_root).resolve()
    package_path = Path(package_dir).resolve()
    benchmark_path = Path(benchmark_root).resolve()
    runtime_bin = package_path / ".toolathlon-venv" / "bin"

    common: dict[str, Any] = {
        "type": "stdio",
        "name": server_name,
        "cache_tools_list": True,
    }
    if server_name == "filesystem":
        common.update(
            {
                "params": {
                    "command": npx_command,
                    "args": [
                        "-y",
                        FILESYSTEM_SERVER_PACKAGE,
                        str(workspace_path),
                    ],
                    "cwd": str(workspace_path),
                },
                "client_session_timeout_seconds": 300,
            }
        )
        return common
    if server_name == "terminal":
        common.update(
            {
                "params": {
                    "command": str(runtime_bin / "cli-mcp-server"),
                    "args": [],
                    "env": {
                        "ALLOWED_DIR": str(workspace_path),
                        "ALLOWED_COMMANDS": TERMINAL_ALLOWED_COMMANDS,
                        "ALLOWED_FLAGS": "all",
                        "MAX_COMMAND_LENGTH": "2048",
                        "COMMAND_TIMEOUT": "60",
                        "ALLOW_SHELL_OPERATORS": "true",
                        "MAX_OUTPUT_LENGTH": "10240",
                        "MAX_STDOUT_LENGTH": "8192",
                        "MAX_STDERR_LENGTH": "2048",
                        "CLI_PROXY_ENABLED": "false",
                        "CLI_PROXY_URL": "",
                    },
                    "cwd": str(workspace_path),
                },
                "client_session_timeout_seconds": 60,
            }
        )
        return common
    if server_name == "excel":
        common.update(
            {
                "params": {
                    "command": str(runtime_bin / "excel-mcp-server"),
                    "args": ["stdio"],
                    "cwd": str(workspace_path),
                },
                "client_session_timeout_seconds": 100,
            }
        )
        return common
    if server_name == "pdf-tools":
        pdf_temp = runtime_path / "pdf-tools-temp"
        common.update(
            {
                "params": {
                    "command": uvx_command,
                    "args": [
                        "--from",
                        PDF_TOOLS_SERVER_PACKAGE,
                        "python",
                        str(
                            benchmark_path
                            / "utils"
                            / "local_servers"
                            / "pdf_tools_local_wrapper.py"
                        ),
                        "--workspace_path",
                        str(workspace_path),
                        "--tempfile_dir",
                        str(pdf_temp),
                    ],
                    "cwd": str(workspace_path),
                },
                "client_session_timeout_seconds": 120,
            }
        )
        return common
    raise ValueError(f"unsupported host MCP server: {server_name}")


def build_gateway_bundle(
    *,
    workspace: str | Path,
    config_dir: str | Path,
    servers: Sequence[str],
    benchmark_root: str | Path,
    benchmark_revision: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_kind": "tokmem_host_synthetic_gateway",
        "needed_mcp_servers": list(servers),
        "eval_config": {
            "mcp": {
                "server_config_path": str(Path(config_dir).resolve()),
            }
        },
        "container_paths": {
            # The vendored gateway uses this historical field name on both
            # container and host. Here it is always the real host workspace.
            "agent_workspace": str(Path(workspace).resolve()),
        },
        "local_token_key_session": {},
        "host_gateway": {
            "benchmark_root": str(Path(benchmark_root).resolve()),
            "benchmark_revision": benchmark_revision,
        },
    }


def _write_json_yaml(path: Path, payload: dict[str, Any]) -> None:
    # JSON is valid YAML. Keeping generated configs in this strict subset
    # avoids adding a second YAML serializer to the TokMem environment.
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_host_gateway_runtime(
    *,
    workspace: str | Path,
    runtime_root: str | Path,
    servers: Sequence[str],
    benchmark_root: str | Path,
    experiment_config: ExperimentConfig,
    package_dir: str | Path = PACKAGE_DIR,
    npx_command: str = "npx",
    uvx_command: str = "uvx",
) -> HostGatewayRuntime:
    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        raise ValueError(f"synthetic workspace does not exist: {workspace_path}")
    root = Path(runtime_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "mcp_servers"
    config_dir.mkdir(parents=True, exist_ok=True)
    for stale_config in config_dir.glob("*.yaml"):
        stale_config.unlink()

    normalized_servers = normalize_server_names(
        servers,
        allowed=experiment_config.mcp_servers,
    )
    for server_name in normalized_servers:
        payload = build_server_config(
            server_name,
            workspace=workspace_path,
            runtime_root=root,
            package_dir=package_dir,
            benchmark_root=benchmark_root,
            npx_command=npx_command,
            uvx_command=uvx_command,
        )
        _write_json_yaml(config_dir / f"{server_name}.yaml", payload)
    if "pdf-tools" in normalized_servers:
        (root / "pdf-tools-temp").mkdir(parents=True, exist_ok=True)

    bundle_file = root / "gateway_bundle.json"
    bundle = build_gateway_bundle(
        workspace=workspace_path,
        config_dir=config_dir,
        servers=normalized_servers,
        benchmark_root=benchmark_root,
        benchmark_revision=experiment_config.benchmark.revision,
    )
    _write_json_yaml(bundle_file, bundle)
    return HostGatewayRuntime(
        root=root,
        config_dir=config_dir,
        bundle_file=bundle_file,
        workspace=workspace_path,
        benchmark_root=Path(benchmark_root).resolve(),
        servers=normalized_servers,
    )


def _validate_launch_files(runtime: HostGatewayRuntime, package_dir: Path) -> None:
    required = [
        runtime.benchmark_root
        / "scripts"
        / "decoupled"
        / "container_tool_gateway.py",
        runtime.benchmark_root / "utils" / "mcp" / "tool_servers.py",
    ]
    runtime_bin = package_dir / ".toolathlon-venv" / "bin"
    if "terminal" in runtime.servers:
        required.append(runtime_bin / "cli-mcp-server")
    if "excel" in runtime.servers:
        required.append(runtime_bin / "excel-mcp-server")
    if "pdf-tools" in runtime.servers:
        required.append(
            runtime.benchmark_root
            / "utils"
            / "local_servers"
            / "pdf_tools_local_wrapper.py"
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"host gateway runtime files are missing: {missing}")
    expected_packages = {}
    if "terminal" in runtime.servers:
        expected_packages["cli-mcp-server"] = TERMINAL_SERVER_VERSION
    if "excel" in runtime.servers:
        expected_packages["excel-mcp-server"] = EXCEL_SERVER_VERSION
    mismatched = {}
    for package_name, expected_version in expected_packages.items():
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
        if actual_version != expected_version:
            mismatched[package_name] = {
                "expected": expected_version,
                "actual": actual_version,
            }
    if mismatched:
        raise RuntimeError(f"host gateway server version drift: {mismatched}")


@contextmanager
def _runtime_root(output_dir: str | Path | None) -> Iterator[Path]:
    if output_dir is not None:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        yield root
        return
    with tempfile.TemporaryDirectory(prefix="tokmem-toolathlon-gateway-") as temp:
        yield Path(temp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Toolathlon's aggregated MCP gateway directly on one synthetic "
            "host workspace, without Docker"
        )
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--servers",
        nargs="+",
        default=None,
        help=(
            "space- or comma-separated subset of filesystem, terminal, excel, "
            "pdf-tools; defaults to all servers in experiment.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="keep generated gateway_bundle.json and mcp_servers configs here",
    )
    parser.add_argument(
        "--uvx-command",
        default=None,
        help="uvx executable used for pdf-tools (defaults to PATH lookup)",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")

    experiment_config = load_experiment_config()
    servers = normalize_server_names(
        args.servers,
        allowed=experiment_config.mcp_servers,
    )
    benchmark_root = find_benchmark_root(experiment_config)
    if benchmark_root is None:
        raise RuntimeError(
            "Toolathlon source is unavailable; run scripts/bootstrap_benchmark.sh "
            "or set TOOLATHLON_ROOT"
        )
    actual_revision = benchmark_provenance(benchmark_root).get("revision")
    if actual_revision != experiment_config.benchmark.revision:
        raise RuntimeError(
            "Toolathlon source revision drift: "
            f"expected={experiment_config.benchmark.revision} "
            f"actual={actual_revision}"
        )
    npx_command = shutil.which("npx") or "npx"
    uvx_command = args.uvx_command or shutil.which("uvx")
    if "pdf-tools" in servers and not uvx_command:
        raise RuntimeError(
            "pdf-tools requires uvx; install uv or pass --uvx-command "
            "/absolute/path/to/uvx"
        )

    with _runtime_root(args.output_dir) as runtime_root:
        runtime = prepare_host_gateway_runtime(
            workspace=args.workspace,
            runtime_root=runtime_root,
            servers=servers,
            benchmark_root=benchmark_root,
            experiment_config=experiment_config,
            npx_command=npx_command,
            uvx_command=uvx_command or "uvx",
        )
        _validate_launch_files(runtime, PACKAGE_DIR)

        os.chdir(runtime.benchmark_root)
        sys.path.insert(0, str(runtime.benchmark_root))
        try:
            from aiohttp import web
            from scripts.decoupled.container_tool_gateway import (
                ContainerToolGateway,
            )
        except ImportError as exc:
            raise RuntimeError(
                "launch this module with compositional_toolathlon/.toolathlon-venv/"
                "bin/python; the official Toolathlon runtime is incomplete"
            ) from exc

        launch = {
            "workspace": str(runtime.workspace),
            "servers": list(runtime.servers),
            "gateway_url": f"http://127.0.0.1:{args.port}/sse",
            "health_url": f"http://127.0.0.1:{args.port}/health",
            "bundle_file": str(runtime.bundle_file),
            "config_dir": str(runtime.config_dir),
            "runtime_is_temporary": args.output_dir is None,
        }
        print(json.dumps(launch, ensure_ascii=False), flush=True)

        gateway = ContainerToolGateway(
            bundle_file=str(runtime.bundle_file),
            debug=args.debug,
        )
        # aiohttp installs SIGINT/SIGTERM handlers and runs app cleanup, which
        # calls MCPServerManager.ensure_all_disconnected for every child server.
        web.run_app(
            gateway.create_app(),
            host="127.0.0.1",
            port=args.port,
            handle_signals=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
