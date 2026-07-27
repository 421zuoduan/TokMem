from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import canonical_json, normalized_input_schema
from .mcp_adapter import RawSseMcpClient
from .model_client import CodexCliClient
from .run_official_agent import (
    atomic_write_json,
    load_official_agent_bundle,
)
from .synthetic_workspace import workspace_digest


PROMPTS = {
    "action-only": """
Solve the task in the fresh Toolathlon environment using only the supplied tools.
Reason internally and return exactly one structured tool call per turn. Use the
real observation before choosing the next action. Do not emit prose, plans, or
markdown. Inspect state when needed, make the smallest justified change, and
call claim_done only after the requested final state is complete. Never inspect
evaluators, ground truth, or protected benchmark files.
""".strip(),
    "state-first": """
Solve the task in the fresh Toolathlon environment using only the supplied tools.
Before each action, internally determine the current state, the requested final
state, and the smallest unresolved difference. Return exactly one structured
tool call and no visible prose. Base later decisions on real observations,
verify material writes when useful, and call claim_done only when no requested
difference remains. Never inspect evaluators, ground truth, or protected files.
""".strip(),
    "contract-first": """
Solve the task in the fresh Toolathlon environment using only the supplied tools.
Internally check the selected tool's argument contract before every call, then
return exactly one structured tool call with a schema-valid argument object and
no visible prose. Use observations to revise the plan, avoid repeating a failed
call unchanged, and call claim_done only after the final environment state is
complete. Never inspect evaluators, ground truth, or protected benchmark files.
""".strip(),
}


def _provider_tools(
    runtime_tools: list[dict[str, Any]],
    *,
    include_claim_done: bool,
) -> tuple[dict[str, str | None], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    alias_to_wire: dict[str, str | None] = {}
    provider_tools: list[dict[str, Any]] = []
    runtime_by_wire: dict[str, dict[str, Any]] = {}
    for position, tool in enumerate(runtime_tools):
        wire_name = tool.get("name")
        if not isinstance(wire_name, str) or not wire_name:
            raise ValueError("runtime gateway exposed a tool without a valid name")
        if wire_name in runtime_by_wire:
            raise ValueError(f"runtime gateway duplicated tool {wire_name!r}")
        alias = f"tool_{position:04d}"
        runtime_by_wire[wire_name] = tool
        alias_to_wire[alias] = wire_name
        provider_tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": str(tool.get("description", "")),
                    "parameters": normalized_input_schema(tool),
                },
            }
        )
    if include_claim_done:
        alias = f"tool_{len(provider_tools):04d}"
        alias_to_wire[alias] = None
        provider_tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": (
                        "Declare the task complete. Call this only after every "
                        "requested environment change has been made."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )
    return alias_to_wire, provider_tools, runtime_by_wire


def _trajectory_tool_name(wire_name: str | None) -> str:
    if wire_name is None:
        return "claim_done"
    return "gw_" + wire_name.replace("-", "_")


def _official_messages(
    instruction: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    for event in events:
        call_id = f"gpt56_call_{int(event['step_index']):04d}"
        function_name = _trajectory_tool_name(event["wire_name"])
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": canonical_json(event["arguments"]),
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


async def run_teacher(
    *,
    bundle: dict[str, Any],
    gateway_url: str,
    model: str,
    prompt_variant: str,
    max_tool_calls: int,
    trace_path: str | Path,
) -> dict[str, Any]:
    workspace = Path(bundle["_resolved_workspace"])
    if not workspace.is_dir():
        raise ValueError(f"official runner workspace is missing: {workspace}")
    initial_state_hash = workspace_digest(workspace, allow_symlinks=True)
    include_claim_done = "claim_done" in set(bundle["needed_local_tools"])
    client = CodexCliClient(model=model, trace_path=trace_path)
    events: list[dict[str, Any]] = []
    model_history: list[dict[str, Any]] = []
    termination_reason = "max_tool_calls"
    runtime_tools: list[dict[str, Any]] = []
    try:
        async with RawSseMcpClient(gateway_url) as gateway:
            runtime_tools = await gateway.list_tools()
            alias_to_wire, provider_tools, _ = _provider_tools(
                runtime_tools,
                include_claim_done=include_claim_done,
            )
            for step_index in range(max_tool_calls):
                action = await client.next_tool_action(
                    model=model,
                    system_prompt=PROMPTS[prompt_variant],
                    instruction=bundle["task_str"].strip(),
                    history=model_history,
                    tools=provider_tools,
                    max_completion_tokens=768,
                )
                if action.tool_alias not in alias_to_wire:
                    raise ValueError(
                        f"GPT-5.6 returned unknown tool alias {action.tool_alias!r}"
                    )
                wire_name = alias_to_wire[action.tool_alias]
                if wire_name is None:
                    outcome = {
                        "success": True,
                        "observation": {
                            "type": "text",
                            "text": "Task completion claimed.",
                        },
                    }
                    termination_reason = "claim_done"
                else:
                    outcome = await gateway.call_tool(wire_name, action.arguments)
                event = {
                    "step_index": step_index,
                    "tool_alias": action.tool_alias,
                    "wire_name": wire_name,
                    "arguments": action.arguments,
                    "success": bool(outcome["success"]),
                    "observation": outcome["observation"],
                    "runtime_metadata": outcome.get("runtime_metadata", {}),
                }
                events.append(event)
                model_history.append(
                    {
                        "call_id": action.call_id,
                        "tool_alias": action.tool_alias,
                        "arguments": action.arguments,
                        "observation": outcome["observation"],
                    }
                )
                if wire_name is None:
                    break
    finally:
        thread_id = client.teacher_thread_id
        client.close()
    return {
        "method": "gpt5.6_teacher",
        "model": model,
        "prompt_variant": prompt_variant,
        "instruction": bundle["task_str"].strip(),
        "events": events,
        "termination_reason": termination_reason,
        "tool_call_count": len(events),
        "execution_error_count": sum(not event["success"] for event in events),
        "initial_state_hash": initial_state_hash,
        "final_state_hash": workspace_digest(workspace, allow_symlinks=True),
        "task_dir": bundle.get("task_dir"),
        "runtime_gateway_tools": runtime_tools,
        "teacher_thread_id": thread_id,
    }


def _official_envelope(
    *,
    bundle: dict[str, Any],
    rollout: dict[str, Any] | None,
    fresh_environment_id: str,
    started_at: str,
    failure: dict[str, str] | None,
) -> dict[str, Any]:
    if rollout is None:
        events: list[dict[str, Any]] = []
        status = "failed"
        termination_reason = "setup_failure"
        runtime_tools: list[dict[str, Any]] = []
    else:
        events = rollout["events"]
        status = (
            "success"
            if rollout["termination_reason"] == "claim_done"
            else "max_turns_reached"
        )
        termination_reason = rollout["termination_reason"]
        runtime_tools = rollout["runtime_gateway_tools"]
    exposed_tools = [
        {
            "type": "function",
            "function": {
                "name": _trajectory_tool_name(str(tool["name"])),
                "description": str(tool.get("description", "")),
                "parameters": normalized_input_schema(tool),
            },
        }
        for tool in runtime_tools
    ]
    if "claim_done" in set(bundle["needed_local_tools"]):
        exposed_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "claim_done",
                    "description": (
                        "Declare the task complete after every requested "
                        "environment change has been made."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )
    return {
        "config": bundle.get("resolved_task_config", {}),
        "request_id": str(uuid.uuid4()),
        "initial_run_time": bundle.get("launch_time", started_at),
        "completion_time": datetime.now(timezone.utc).isoformat(),
        "tool_calls": {
            "tools": exposed_tools,
            "tool_choice": "required",
        },
        "status": status,
        "messages": _official_messages(bundle["task_str"], events),
        "key_stats": {
            "tool_calls": len(events),
            "agent_llm_requests": len(events),
        },
        "agent_cost": {},
        "user_cost": {},
        "resumed": False,
        "history_file": None,
        "session_id": fresh_environment_id,
        "gpt5.6_teacher": {
            "termination_reason": termination_reason,
            "failure": failure,
            "model": rollout.get("model") if rollout else None,
            "prompt_variant": rollout.get("prompt_variant") if rollout else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GPT-5.6 Codex teacher against one official Toolathlon task"
    )
    parser.add_argument("--bundle-file", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--fresh-environment-id", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--prompt-variant", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--max-tool-calls", type=int, default=50)
    parser.add_argument("--trace-path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    bundle = load_official_agent_bundle(args.bundle_file)
    rollout: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        rollout = asyncio.run(
            run_teacher(
                bundle=bundle,
                gateway_url=args.gateway_url,
                model=args.model,
                prompt_variant=args.prompt_variant,
                max_tool_calls=min(
                    args.max_tool_calls,
                    int(bundle["max_steps_under_single_turn_mode"]),
                ),
                trace_path=args.trace_path,
            )
        )
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
    envelope = _official_envelope(
        bundle=bundle,
        rollout=rollout,
        fresh_environment_id=args.fresh_environment_id,
        started_at=started_at,
        failure=failure,
    )
    envelope["gpt5.6_teacher"]["host_elapsed_seconds"] = (
        time.monotonic() - started_clock
    )
    atomic_write_json(bundle["_resolved_log_file"], envelope)
    raw_path = Path(bundle["_resolved_output_root"]) / "gpt5.6_rollout.json"
    atomic_write_json(raw_path, {"rollout": rollout, "failure": failure})
    if failure is not None:
        print(
            f"GPT-5.6 official teacher failed: "
            f"{failure['error_type']}: {failure['error']}"
        )
        return 2
    print(
        f"method=gpt5.6_teacher calls={rollout['tool_call_count']} "
        f"errors={rollout['execution_error_count']} "
        f"termination={rollout['termination_reason']}"
    )
    return 0 if rollout["termination_reason"] == "claim_done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
