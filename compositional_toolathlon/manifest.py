from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import validators

from .config import load_experiment_config


MANIFEST_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_input_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema", tool.get("input_schema", {}))
    if schema is None:
        return {}
    if not isinstance(schema, dict):
        raise ValueError(f"tool input schema must be an object: {tool.get('name')!r}")
    return schema


def validate_tool_arguments(
    tool_record: dict[str, Any],
    arguments: dict[str, Any],
) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    schema = tool_record.get("input_schema")
    if not isinstance(schema, dict):
        raise ValueError("tool record has no object input_schema")
    validator_class = validators.validator_for(schema)
    validator_class.check_schema(schema)
    errors = sorted(
        validator_class(schema).iter_errors(arguments),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(
            f"arguments violate schema for {tool_record.get('stable_id')}: "
            f"{location}: {error.message}"
        )


def schema_hash(tool_name: str, input_schema: dict[str, Any]) -> str:
    material = canonical_json({"name": tool_name, "input_schema": input_schema})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def stable_tool_id(server_name: str, tool_name: str, digest: str) -> str:
    if "::" in server_name or "::" in tool_name:
        raise ValueError("server and tool names cannot contain the stable-ID separator '::'")
    return f"{server_name}::{tool_name}::{digest}"


def _iter_server_payloads(payload: Any) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    if isinstance(payload, dict) and "servers" in payload:
        payload = payload["servers"]

    if isinstance(payload, dict):
        for server_name, tools_payload in payload.items():
            if isinstance(tools_payload, dict):
                tools_payload = tools_payload.get("tools", tools_payload.get("result", tools_payload))
                if isinstance(tools_payload, dict):
                    tools_payload = tools_payload.get("tools", [])
            if not isinstance(tools_payload, list):
                raise ValueError(f"tools/list payload for {server_name!r} must be a list")
            yield str(server_name), tools_payload
        return

    if isinstance(payload, list):
        for server_payload in payload:
            if not isinstance(server_payload, dict):
                raise ValueError("each server payload must be an object")
            server_name = server_payload.get("server", server_payload.get("server_name"))
            tools = server_payload.get("tools")
            if not isinstance(server_name, str) or not isinstance(tools, list):
                raise ValueError("server payload requires string server/server_name and list tools")
            yield server_name, tools
        return

    raise ValueError("tools/list input must be an object or list")


def finalize_manifest(
    records: list[dict[str, Any]],
    benchmark_revision: str,
    *,
    tapmem_capacity: int,
) -> dict[str, Any]:
    records = sorted(records, key=lambda record: record["stable_id"])
    if not records:
        raise ValueError("tools/list produced an empty manifest")
    if len(records) > tapmem_capacity:
        raise ValueError(
            f"manifest has {len(records)} tools, exceeding TapMem capacity {tapmem_capacity}"
        )
    stable_ids = [record["stable_id"] for record in records]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("manifest contains duplicate stable tool IDs")
    wire_names = [
        record["wire_name"]
        for record in records
        if record.get("origin") == "gateway_mcp"
    ]
    if len(wire_names) != len(set(wire_names)):
        raise ValueError("manifest contains duplicate gateway wire names")
    model_names = [record["model_name"] for record in records]
    if len(model_names) != len(set(model_names)):
        duplicates = sorted(
            {
                name
                for name in model_names
                if model_names.count(name) > 1
            }
        )
        raise ValueError(
            "manifest tool names collide after provider normalization: "
            f"{duplicates}"
        )
    for slot, record in enumerate(records):
        record["memory_slot"] = slot
    manifest_material = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_revision": benchmark_revision,
        "tools": records,
    }
    manifest_hash = hashlib.sha256(
        canonical_json(manifest_material).encode("utf-8")
    ).hexdigest()
    return {
        **manifest_material,
        "tool_count": len(records),
        "manifest_hash": manifest_hash,
    }


def build_manifest(
    raw_payload: Any,
    benchmark_revision: str,
    *,
    tapmem_capacity: int = 247,
) -> dict[str, Any]:
    records = []
    seen_ids = set()
    seen_wire_names = set()
    for server_name, tools in _iter_server_payloads(raw_payload):
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError(f"tool record in {server_name!r} must be an object")
            wire_name = tool.get("wire_name", tool.get("name"))
            if not isinstance(wire_name, str) or not wire_name:
                raise ValueError(f"tool in {server_name!r} is missing a non-empty name")
            source_server = tool.get("source_server", server_name)
            tool_name = tool.get("backend_name", tool.get("name"))
            if not isinstance(source_server, str) or not source_server:
                raise ValueError("source_server must be a non-empty string")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("backend_name/name must be a non-empty string")
            if wire_name in seen_wire_names:
                raise ValueError(f"duplicate MCP wire tool name: {wire_name}")
            seen_wire_names.add(wire_name)
            input_schema = normalized_input_schema(tool)
            digest = schema_hash(tool_name, input_schema)
            stable_id = stable_tool_id(source_server, tool_name, digest)
            if stable_id in seen_ids:
                raise ValueError(f"duplicate stable tool ID: {stable_id}")
            seen_ids.add(stable_id)
            records.append(
                {
                    "server": source_server,
                    "tool_name": tool_name,
                    "wire_name": wire_name,
                    "raw_name": tool.get("raw_name", wire_name),
                    "model_name": tool.get("model_name", wire_name.replace("-", "_")),
                    "origin": tool.get("origin", "gateway_mcp"),
                    "group_key": tool.get("group_key"),
                    "dispatch_kind": tool.get(
                        "dispatch_kind",
                        "terminal" if wire_name == "local-claim_done" else "mcp",
                    ),
                    "state_requirements": tool.get("state_requirements", []),
                    "description": tool.get("description", ""),
                    "input_schema": input_schema,
                    "schema_hash": digest,
                    "stable_id": stable_id,
                }
            )

    return finalize_manifest(
        records,
        benchmark_revision,
        tapmem_capacity=tapmem_capacity,
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported tool manifest schema version")
    tools = manifest.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool manifest must contain a non-empty tools list")
    stable_ids = [record.get("stable_id") for record in tools]
    if any(not isinstance(stable_id, str) for stable_id in stable_ids):
        raise ValueError("every manifest tool requires a stable_id")
    if stable_ids != sorted(stable_ids):
        raise ValueError("manifest tools must be sorted by stable_id")
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("manifest contains duplicate stable IDs")
    gateway_wire_names = [
        record.get("wire_name")
        for record in tools
        if record.get("origin") == "gateway_mcp"
    ]
    if any(
        not isinstance(wire_name, str) or not wire_name
        for wire_name in gateway_wire_names
    ):
        raise ValueError("every gateway manifest tool requires a non-empty wire_name")
    if len(gateway_wire_names) != len(set(gateway_wire_names)):
        raise ValueError("manifest contains duplicate MCP wire names")
    model_names = [record.get("model_name") for record in tools]
    if any(not isinstance(name, str) or not name for name in model_names):
        raise ValueError("every manifest tool requires a non-empty model_name")
    if len(model_names) != len(set(model_names)):
        raise ValueError("manifest contains duplicate provider-normalized model names")
    for expected_slot, record in enumerate(tools):
        server_name = record.get("server")
        tool_name = record.get("tool_name")
        input_schema = record.get("input_schema")
        if not isinstance(server_name, str) or not isinstance(tool_name, str):
            raise ValueError("every manifest tool requires server and tool_name")
        if not isinstance(input_schema, dict):
            raise ValueError("every manifest tool requires an object input_schema")
        origin = record.get("origin")
        if origin not in {"gateway_mcp", "host_local"}:
            raise ValueError(f"unsupported manifest tool origin: {origin!r}")
        if origin == "host_local" and record.get("wire_name") is not None:
            raise ValueError("host-local tools must not claim an MCP wire_name")
        for field in ("raw_name", "model_name", "dispatch_kind"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"every manifest tool requires {field}")
        if not isinstance(record.get("state_requirements"), list):
            raise ValueError("state_requirements must be a list")
        expected_schema_hash = schema_hash(tool_name, input_schema)
        if record.get("schema_hash") != expected_schema_hash:
            raise ValueError(f"schema hash mismatch for {server_name}::{tool_name}")
        if record["stable_id"] != stable_tool_id(
            server_name,
            tool_name,
            expected_schema_hash,
        ):
            raise ValueError(f"stable ID mismatch for {server_name}::{tool_name}")
        if record.get("memory_slot") != expected_slot:
            raise ValueError("manifest memory slots must be contiguous and ordered")

    if manifest.get("tool_count") != len(tools):
        raise ValueError("manifest tool_count does not match tools")
    material = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_revision": manifest.get("benchmark_revision"),
        "tools": tools,
    }
    expected_manifest_hash = hashlib.sha256(
        canonical_json(material).encode("utf-8")
    ).hexdigest()
    if manifest.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("tool manifest hash mismatch")
    return manifest


def _build_command(args: argparse.Namespace) -> int:
    with Path(args.input).open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    config = load_experiment_config(args.config)
    revision = args.benchmark_revision or config.benchmark.revision
    manifest = build_manifest(
        raw_payload,
        revision,
        tapmem_capacity=config.interface.tapmem_tool_capacity,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Wrote {manifest['tool_count']} tools to {output_path} "
        f"(manifest_hash={manifest['manifest_hash']})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a stable Toolathlon tool manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="convert captured tools/list JSON")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--benchmark-revision", default=None)
    build.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "experiment.json"),
    )
    build.set_defaults(func=_build_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
