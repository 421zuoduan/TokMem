from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .local_tools import CompositeToolExecutor
from .manifest import canonical_json, load_manifest
from .mcp_adapter import (
    ManifestMcpExecutor,
    RawSseMcpClient,
    resolve_runtime_gateway_tool_ids,
)
from .model_runtime import TokMemClosedLoopPolicy, load_tool_model
from .rollout import run_closed_loop_rollout
from .synthetic_workspace import workspace_digest
from .teacher import terminal_tool_ids


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"official agent bundle requires object {key!r}")
    return value


def load_official_agent_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("official agent bundle must be a JSON object")
    if bundle.get("schema_version") != 2:
        raise ValueError("official agent bundle must use schema_version=2")
    task_str = bundle.get("task_str")
    if not isinstance(task_str, str) or not task_str.strip():
        raise ValueError("official agent bundle has no canonical task_str")
    host_paths = _require_mapping(bundle, "host_paths")
    for field in ("task_root", "agent_workspace", "log_file"):
        if not isinstance(host_paths.get(field), str) or not host_paths[field]:
            raise ValueError(f"official bundle host_paths requires {field!r}")
    needed_local_tools = bundle.get("needed_local_tools")
    if (
        not isinstance(needed_local_tools, list)
        or any(not isinstance(item, str) for item in needed_local_tools)
    ):
        raise ValueError("official bundle requires needed_local_tools list")
    max_steps = bundle.get("max_steps_under_single_turn_mode")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("official bundle requires a positive max step count")
    if not isinstance(bundle.get("resolved_task_config"), dict):
        raise ValueError("official bundle requires resolved_task_config object")

    output_root = Path(host_paths["task_root"]).resolve()
    log_file = Path(host_paths["log_file"]).resolve()
    workspace = Path(host_paths["agent_workspace"]).resolve()
    for label, resolved_path in (
        ("agent workspace", workspace),
        ("trajectory log", log_file),
    ):
        if not resolved_path.is_relative_to(output_root):
            raise ValueError(f"official {label} must stay inside the dump root")
    # The official shell may keep this bundle in a private trusted-stash directory
    # outside the public dump root. The runner supplies the path; output paths are
    # still constrained to the dump root above.
    bundle["_resolved_bundle_path"] = str(bundle_path)
    bundle["_resolved_output_root"] = str(output_root)
    bundle["_resolved_workspace"] = str(workspace)
    bundle["_resolved_log_file"] = str(log_file)
    return bundle


def _official_model_tool_name(record: dict[str, Any]) -> str:
    if record["origin"] == "gateway_mcp":
        return "gw_" + str(record["wire_name"]).replace("-", "_")
    return record["model_name"]


def official_provider_tools(
    manifest: dict[str, Any],
    available_tool_ids: list[str],
) -> list[dict[str, Any]]:
    records = {
        record["stable_id"]: record for record in manifest["tools"]
    }
    tools = []
    names = []
    for stable_id in available_tool_ids:
        try:
            record = records[stable_id]
        except KeyError as exc:
            raise ValueError(
                f"available tool is absent from manifest: {stable_id}"
            ) from exc
        name = _official_model_tool_name(record)
        names.append(name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": record.get("description", ""),
                    "parameters": record["input_schema"],
                },
            }
        )
    if len(names) != len(set(names)):
        raise ValueError(
            "official model tool aliases collide after gateway normalization"
        )
    return tools


def rollout_to_official_messages(
    *,
    instruction: str,
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    records = {
        record["stable_id"]: record for record in manifest["tools"]
    }
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": instruction}
    ]
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        action = event["action"]
        stable_tool_id = action["tool_id"]
        record = records[stable_tool_id]
        call_id = f"tokmem_call_{int(event['step_index']):04d}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": _official_model_tool_name(record),
                            "arguments": canonical_json(action["arguments"]),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": canonical_json(event["observation"]),
            }
        )
    return messages


def build_official_trajectory_envelope(
    *,
    bundle: dict[str, Any],
    rollout: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    fresh_environment_id: str,
    started_at: str,
    failure: dict[str, str] | None = None,
) -> dict[str, Any]:
    if rollout is None:
        termination_reason = "setup_failure"
        messages = [{"role": "user", "content": bundle["task_str"]}]
        tool_call_count = 0
        status = "failed"
        exposed_tools: list[dict[str, Any]] = []
        agent_llm_requests = 0
    else:
        if manifest is None:
            raise ValueError("a completed rollout requires its manifest")
        termination_reason = rollout["termination_reason"]
        messages = rollout_to_official_messages(
            instruction=bundle["task_str"],
            events=rollout["events"],
            manifest=manifest,
        )
        tool_call_count = int(rollout["tool_call_count"])
        exposed_tools = official_provider_tools(
            manifest,
            list(rollout["available_tool_ids"]),
        )
        agent_llm_requests = tool_call_count + int(
            rollout.get("decode_error_count", 0)
        )
        status = (
            "success"
            if termination_reason == "claim_done"
            else "max_turns_reached"
            if termination_reason == "max_tool_calls"
            else "failed"
        )
    return {
        "config": bundle.get("resolved_task_config", {}),
        "request_id": str(uuid.uuid4()),
        "initial_run_time": bundle.get("launch_time", started_at),
        "completion_time": datetime.now(timezone.utc).isoformat(),
        "tool_calls": {
            # Recorded only after inference for official replay/provenance. The
            # student context itself never receives these definitions.
            "tools": exposed_tools,
            "tool_choice": "required",
        },
        "status": status,
        "messages": messages,
        "key_stats": {
            "tool_calls": tool_call_count,
            "agent_llm_requests": agent_llm_requests,
        },
        "agent_cost": {},
        "user_cost": {},
        "resumed": False,
        "history_file": None,
        "session_id": fresh_environment_id,
        "tokmem_runtime": {
            "termination_reason": termination_reason,
            "tool_manifest_hash": (
                manifest["manifest_hash"] if manifest is not None else None
            ),
            "failure": failure,
        },
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tokmem-{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


async def run_official_agent(
    *,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: str | Path,
    run_dir: str | Path,
    gateway_url: str,
    device: str | None,
    dtype: str | None,
    max_new_tokens: int,
    fresh_environment_id: str,
) -> dict[str, Any]:
    workspace = Path(bundle["_resolved_workspace"])
    if not workspace.is_dir():
        raise ValueError(f"official runner workspace is missing: {workspace}")
    initial_state_hash = workspace_digest(workspace, allow_symlinks=True)
    loaded = load_tool_model(
        run_dir=run_dir,
        manifest_path=manifest_path,
        device=device,
        dtype=dtype,
    )
    policy = TokMemClosedLoopPolicy(loaded, max_new_tokens=max_new_tokens)
    enable_python_execute = "python_execute" in set(bundle["needed_local_tools"])

    async with RawSseMcpClient(gateway_url) as client:
        runtime_tools = await client.list_tools()
        available_tool_ids = resolve_runtime_gateway_tool_ids(
            manifest,
            runtime_tools,
        )
        if enable_python_execute:
            local_python_ids = [
                record["stable_id"]
                for record in manifest["tools"]
                if record["dispatch_kind"] == "local_python"
            ]
            if len(local_python_ids) != 1:
                raise ValueError(
                    "default decoupled python_execute requires exactly one manifest tool"
                )
            available_tool_ids.extend(local_python_ids)
            available_tool_ids.sort()
        gateway_available_ids = {
            tool_id
            for tool_id in available_tool_ids
            if tool_id in {
                record["stable_id"]
                for record in manifest["tools"]
                if record["origin"] == "gateway_mcp"
            }
        }
        mcp_executor = ManifestMcpExecutor(client, manifest)
        await mcp_executor.verify_runtime(
            expected_stable_ids=gateway_available_ids,
        )
        executor = CompositeToolExecutor(
            manifest=manifest,
            mcp_executor=mcp_executor,
            workspace_root=workspace,
            enable_python_execute=enable_python_execute,
        )
        result = await run_closed_loop_rollout(
            instruction=bundle["task_str"].strip(),
            available_tool_ids=available_tool_ids,
            terminal_tool_ids=terminal_tool_ids(manifest),
            policy=policy,
            executor=executor,
            evaluator=None,
            max_tool_calls=int(bundle["max_steps_under_single_turn_mode"]),
        )
    result.update(
        {
            "method": loaded.method,
            "tool_manifest_hash": manifest["manifest_hash"],
            "initial_state_hash": initial_state_hash,
            "final_state_hash": workspace_digest(
                workspace,
                allow_symlinks=True,
            ),
            "fresh_environment_id": fresh_environment_id,
            "official_evaluator_pending": True,
            "task_dir": bundle.get("task_dir"),
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TokMem-family Step-5 host loop for Toolathlon run_single_decoupled.sh"
        )
    )
    parser.add_argument("--bundle-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--fresh-environment-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.fresh_environment_id.strip():
        print("TokMem official host loop failed: fresh environment ID is empty")
        return 2
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    try:
        bundle = load_official_agent_bundle(args.bundle_file)
    except Exception as exc:
        print(
            f"TokMem official host loop failed before bundle resolution: "
            f"{type(exc).__name__}: {exc}"
        )
        return 2
    manifest: dict[str, Any] | None = None
    rollout: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    exit_code = 2
    try:
        manifest = load_manifest(args.manifest)
        rollout = asyncio.run(
            run_official_agent(
                bundle=bundle,
                manifest=manifest,
                manifest_path=args.manifest,
                run_dir=args.run_dir,
                gateway_url=args.gateway_url,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                fresh_environment_id=args.fresh_environment_id,
            )
        )
        exit_code = 0 if rollout["termination_reason"] == "claim_done" else 1
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    envelope = build_official_trajectory_envelope(
        bundle=bundle,
        rollout=rollout,
        manifest=manifest,
        fresh_environment_id=args.fresh_environment_id,
        started_at=started_at,
        failure=failure,
    )
    envelope["tokmem_runtime"]["host_elapsed_seconds"] = (
        time.monotonic() - started_clock
    )
    try:
        atomic_write_json(bundle["_resolved_log_file"], envelope)
    except Exception as exc:
        print(
            "TokMem official host loop could not write the required trajectory: "
            f"{type(exc).__name__}: {exc}"
        )
        return 2
    raw_rollout_path = Path(bundle["_resolved_output_root"]) / "tokmem_rollout.json"
    try:
        atomic_write_json(
            raw_rollout_path,
            {
                "rollout": rollout,
                "failure": failure,
                "official_trajectory_path": bundle["_resolved_log_file"],
            },
        )
    except Exception as exc:
        print(
            "Warning: required trajectory is safe, but auxiliary TokMem "
            f"diagnostics could not be written: {type(exc).__name__}: {exc}"
        )
    if failure is not None:
        print(
            f"TokMem official host loop failed: "
            f"{failure['error_type']}: {failure['error']}"
        )
    else:
        print(
            f"method={rollout['method']} calls={rollout['tool_call_count']} "
            f"termination={rollout['termination_reason']}; "
            "official evaluator remains owned by run_single_decoupled.sh"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
