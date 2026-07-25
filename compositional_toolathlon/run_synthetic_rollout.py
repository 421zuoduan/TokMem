from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .collect_teacher_episode import DeclarativeWorkspaceEvaluator
from .generate_tasks import validate_task_candidate
from .local_tools import CompositeToolExecutor
from .manifest import load_manifest
from .mcp_adapter import ManifestMcpExecutor, RawSseMcpClient
from .model_runtime import TokMemClosedLoopPolicy, load_tool_model
from .rollout import run_closed_loop_rollout
from .synthetic_workspace import (
    apply_workspace_recipe,
    verify_task_assets,
    workspace_digest,
)
from .teacher import terminal_tool_ids


async def run_one(
    *,
    task_spec: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: str | Path,
    run_dir: str | Path,
    workspace_root: str | Path,
    gateway_url: str,
    device: str | None,
    dtype: str | None,
    max_tool_calls: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    validate_task_candidate(task_spec, manifest)
    if task_spec.get("verified") is not True:
        raise ValueError("synthetic rollout requires a verified task")
    verification = verify_task_assets(task_spec)
    if not verification["passed"]:
        raise ValueError("task assets no longer pass their verification gate")

    workspace = Path(workspace_root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    apply_workspace_recipe(
        workspace,
        task_spec["initial_workspace"],
        require_empty=True,
    )
    if workspace_digest(workspace) != verification["initial_workspace_hash"]:
        raise ValueError("rollout initial workspace hash drifted")

    loaded = load_tool_model(
        run_dir=run_dir,
        manifest_path=manifest_path,
        device=device,
        dtype=dtype,
    )
    if loaded.manifest["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("loaded checkpoint manifest drifted")
    policy = TokMemClosedLoopPolicy(
        loaded,
        max_new_tokens=max_new_tokens,
    )
    python_ids = {
        record["stable_id"]
        for record in manifest["tools"]
        if record["dispatch_kind"] == "local_python"
    }
    async with RawSseMcpClient(gateway_url) as client:
        mcp_executor = ManifestMcpExecutor(client, manifest)
        await mcp_executor.verify_runtime()
        executor = CompositeToolExecutor(
            manifest=manifest,
            mcp_executor=mcp_executor,
            workspace_root=workspace,
            enable_python_execute=bool(
                python_ids & set(task_spec["available_tools"])
            ),
        )
        result = await run_closed_loop_rollout(
            instruction=task_spec["instruction"],
            available_tool_ids=list(task_spec["available_tools"]),
            terminal_tool_ids=terminal_tool_ids(manifest),
            policy=policy,
            executor=executor,
            evaluator=DeclarativeWorkspaceEvaluator(
                workspace,
                task_spec["evaluator"],
            ),
            max_tool_calls=max_tool_calls,
        )
    result.update(
        {
            "task_id": task_spec["task_id"],
            "template_id": task_spec["template_id"],
            "split": task_spec["split"],
            "method": loaded.method,
            "tool_manifest_hash": manifest["manifest_hash"],
            "initial_state_hash": verification["initial_workspace_hash"],
            "final_state_hash": workspace_digest(workspace),
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen TokMem-family checkpoint in a real synthetic MCP workspace"
    )
    parser.add_argument("--task-spec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000/sse")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    task_spec = json.loads(Path(args.task_spec).read_text(encoding="utf-8"))
    result = asyncio.run(
        run_one(
            task_spec=task_spec,
            manifest=manifest,
            manifest_path=args.manifest,
            run_dir=args.run_dir,
            workspace_root=args.workspace_root,
            gateway_url=args.gateway_url,
            device=args.device,
            dtype=args.dtype,
            max_tool_calls=args.max_tool_calls,
            max_new_tokens=args.max_new_tokens,
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"task={result['task_id']} method={result['method']} "
        f"passed={result['evaluator']['passed']} calls={result['tool_call_count']}"
    )
    return 0 if result["evaluator"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
