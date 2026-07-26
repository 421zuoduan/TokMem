from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from .config import load_experiment_config
from .manifest import (
    build_manifest,
    canonical_json,
    load_manifest,
    normalized_input_schema,
)


RAW_GATEWAY_NAMESPACE = "toolathlon-gateway"
SEMANTIC_ERROR_PREFIXES = (
    "error:",
    "error ",
    "failed:",
    "failed to ",
    "tool call failed",
    "access denied:",
)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(value, dict):
        return value
    raise TypeError(f"MCP value is not serializable: {type(value).__name__}")


def observation_reports_error(observation: Any) -> bool:
    if isinstance(observation, str):
        normalized = observation.strip().casefold()
        return normalized.startswith(SEMANTIC_ERROR_PREFIXES)
    if isinstance(observation, list):
        return any(observation_reports_error(item) for item in observation)
    if isinstance(observation, dict):
        text = observation.get("text")
        return isinstance(text, str) and observation_reports_error(text)
    return False


def assert_runtime_tool_set(
    manifest: dict[str, Any],
    runtime_tools: list[dict[str, Any]],
    *,
    expected_stable_ids: set[str] | None = None,
) -> None:
    expected = {
        record["wire_name"]: canonical_json(record["input_schema"])
        for record in manifest["tools"]
        if record.get("origin") == "gateway_mcp"
        and (
            expected_stable_ids is None
            or record["stable_id"] in expected_stable_ids
        )
    }
    actual = {}
    for tool in runtime_tools:
        wire_name = tool.get("name")
        if not isinstance(wire_name, str) or not wire_name:
            raise ValueError("runtime tools/list returned an invalid name")
        if wire_name in actual:
            raise ValueError(f"runtime tools/list duplicated {wire_name!r}")
        actual[wire_name] = canonical_json(normalized_input_schema(tool))
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        schema_changed = sorted(
            name
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        raise ValueError(
            "runtime MCP tools drifted from the frozen manifest: "
            f"missing={missing}, extra={extra}, schema_changed={schema_changed}"
        )


def resolve_runtime_gateway_tool_ids(
    manifest: dict[str, Any],
    runtime_tools: list[dict[str, Any]],
    *,
    require_schema_match: bool = True,
    ignore_unknown: bool = False,
) -> list[str]:
    record_by_wire = {
        record["wire_name"]: record
        for record in manifest["tools"]
        if record.get("origin") == "gateway_mcp"
    }
    resolved = []
    seen_wire_names: set[str] = set()
    for tool in runtime_tools:
        wire_name = tool.get("name")
        if not isinstance(wire_name, str) or not wire_name:
            raise ValueError("runtime tools/list returned an invalid name")
        if wire_name in seen_wire_names:
            raise ValueError(f"runtime tools/list duplicated {wire_name!r}")
        seen_wire_names.add(wire_name)
        record = record_by_wire.get(wire_name)
        if record is None:
            if ignore_unknown:
                continue
            raise ValueError(f"runtime exposed a tool absent from manifest: {wire_name!r}")
        if (
            require_schema_match
            and canonical_json(normalized_input_schema(tool))
            != canonical_json(record["input_schema"])
        ):
            raise ValueError(f"runtime schema drift for MCP wire tool {wire_name!r}")
        resolved.append(record["stable_id"])
    if len(resolved) != len(set(resolved)):
        raise ValueError("runtime tools/list maps to duplicate stable IDs")
    return sorted(resolved)


async def fetch_health(url: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    def _read() -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("gateway health response must be a JSON object")
        return payload

    return await asyncio.to_thread(_read)


class RawSseMcpClient:
    """Direct MCP client for Toolathlon's raw gateway.

    Tool names are opaque wire identifiers. This class never splits a name on
    hyphens and never adds the host scaffold's ``gw-`` prefix.
    """

    def __init__(
        self,
        url: str,
        *,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 300.0,
    ) -> None:
        self.url = url
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None

    async def __aenter__(self) -> "RawSseMcpClient":
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP client is missing; install "
                "compositional_toolathlon/requirements-tokmem-runtime.txt "
                "in the Python environment running this agent"
            ) from exc

        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                sse_client(
                    self.url,
                    timeout=self.connect_timeout_seconds,
                    sse_read_timeout=self.read_timeout_seconds,
                )
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, traceback)
        self._stack = None
        self._session = None

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        session = self._require_session()
        tools: list[dict[str, Any]] = []
        cursor = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(_model_dump(tool) for tool in result.tools)
            cursor = getattr(result, "nextCursor", None)
            if not cursor:
                break
        return tools

    async def call_tool(
        self,
        wire_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be an object")
        result = await self._require_session().call_tool(
            name=wire_name,
            arguments=arguments,
        )
        payload = _model_dump(result)
        content = payload.get("content", [])
        if isinstance(content, list) and len(content) == 1:
            observation: Any = content[0]
        else:
            observation = content
        semantic_error = observation_reports_error(observation)
        return {
            # The MCP protocol's isError field is authoritative.  Text such as
            # "Error:" may be legitimate file content, so semantic detection
            # is diagnostic only and must not change rollout control flow.
            "success": not bool(payload.get("isError", False)),
            "observation": observation,
            "runtime_metadata": {
                "is_error": bool(payload.get("isError", False)),
                "semantic_error": semantic_error,
                "meta": payload.get("_meta"),
            },
        }


class ManifestMcpExecutor:
    """Map immutable memory-token IDs to exact runtime MCP wire names."""

    def __init__(self, client: RawSseMcpClient, manifest: dict[str, Any]) -> None:
        self.client = client
        self.manifest = manifest
        self._wire_by_stable_id = {
            record["stable_id"]: record["wire_name"]
            for record in manifest["tools"]
            if record.get("origin") == "gateway_mcp"
        }

    async def verify_runtime(
        self,
        expected_stable_ids: set[str] | None = None,
    ) -> None:
        assert_runtime_tool_set(
            self.manifest,
            await self.client.list_tools(),
            expected_stable_ids=expected_stable_ids,
        )

    async def call_tool(
        self,
        stable_tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            wire_name = self._wire_by_stable_id[stable_tool_id]
        except KeyError as exc:
            raise ValueError(
                f"stable tool ID is absent from manifest: {stable_tool_id}"
            ) from exc
        return await self.client.call_tool(wire_name, arguments)


async def capture_gateway(
    *,
    gateway_url: str,
    health_url: str | None,
    benchmark_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with RawSseMcpClient(gateway_url) as client:
        tools = await client.list_tools()
    health = await fetch_health(health_url) if health_url else {}
    health_tool_count = health.get("tool_count")
    if isinstance(health_tool_count, int) and health_tool_count != len(tools):
        raise ValueError(
            f"gateway health reports {health_tool_count} tools, tools/list returned {len(tools)}"
        )
    raw_capture = {
        "schema_version": 1,
        "gateway_url": gateway_url,
        "health": health,
        "servers": {
            RAW_GATEWAY_NAMESPACE: {
                "tools": tools,
            }
        },
    }
    manifest = build_manifest(
        raw_capture,
        benchmark_revision,
        tapmem_capacity=load_experiment_config().interface.tapmem_tool_capacity,
    )
    manifest["gateway_capture"] = {
        "namespace": RAW_GATEWAY_NAMESPACE,
        "health": health,
        "wire_names_are_opaque": True,
    }
    return raw_capture, manifest


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the exact raw Toolathlon MCP gateway tool interface"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--gateway-url", default="http://127.0.0.1:8000/sse")
    capture.add_argument("--health-url", default="http://127.0.0.1:8000/health")
    capture.add_argument("--raw-output", required=True)
    capture.add_argument("--manifest-output", required=True)
    capture.add_argument("--benchmark-revision", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command != "capture":
        raise ValueError(f"unsupported command: {args.command}")
    config = load_experiment_config()
    raw_capture, manifest = asyncio.run(
        capture_gateway(
            gateway_url=args.gateway_url,
            health_url=args.health_url,
            benchmark_revision=args.benchmark_revision
            or config.benchmark.revision,
        )
    )
    _write_json(args.raw_output, raw_capture)
    _write_json(args.manifest_output, manifest)
    print(
        f"Captured {manifest['tool_count']} opaque gateway tools; "
        f"manifest_hash={manifest['manifest_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
