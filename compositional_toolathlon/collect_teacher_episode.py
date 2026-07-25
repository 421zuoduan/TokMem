from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .generate_tasks import validate_task_candidate
from .local_tools import CompositeToolExecutor
from .manifest import load_manifest
from .mcp_adapter import ManifestMcpExecutor, RawSseMcpClient
from .model_client import OpenAICompatibleClient, read_prompt
from .synthetic_workspace import (
    apply_workspace_recipe,
    evaluate_workspace,
    verify_task_assets,
    workspace_digest,
)
from .teacher import TeacherSettings, collect_teacher_candidate


class DeclarativeWorkspaceEvaluator:
    def __init__(self, workspace: str | Path, evaluator: dict[str, Any]) -> None:
        self.workspace = Path(workspace)
        self.evaluator = evaluator

    async def evaluate(self) -> dict[str, Any]:
        return evaluate_workspace(self.workspace, self.evaluator)


async def collect_one(
    *,
    task_spec: dict[str, Any],
    manifest: dict[str, Any],
    generation_config: dict[str, Any],
    workspace_root: str | Path,
    gateway_url: str,
    candidate_index: int,
    fresh_environment_id: str,
    client: OpenAICompatibleClient,
) -> dict[str, Any]:
    validate_task_candidate(task_spec, manifest)
    if task_spec.get("verified") is not True:
        raise ValueError("teacher collection requires a verified task spec")
    if task_spec["generation_provenance"]["tool_manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("task and runtime manifest hashes differ")
    asset_verification = verify_task_assets(task_spec)
    if not asset_verification["passed"]:
        raise ValueError(f"task asset verification failed: {asset_verification['reasons']}")

    workspace = Path(workspace_root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    apply_workspace_recipe(
        workspace,
        task_spec["initial_workspace"],
        require_empty=True,
    )
    actual_initial_hash = workspace_digest(workspace)
    if actual_initial_hash != asset_verification["initial_workspace_hash"]:
        raise ValueError("materialized teacher workspace hash differs from verifier")

    teacher_config = generation_config["teacher"]
    settings = TeacherSettings(
        model=generation_config["models"]["teacher"],
        prompt_version=teacher_config["prompt_version"],
        system_prompt=read_prompt(PACKAGE_DIR / "prompts" / teacher_config["prompt"]),
        max_tool_calls=int(teacher_config["max_tool_calls"]),
        max_completion_tokens_per_action=int(
            teacher_config["max_completion_tokens_per_action"]
        ),
        max_visible_assistant_characters=int(
            teacher_config["max_visible_assistant_characters"]
        ),
        max_argument_characters_per_call=int(
            teacher_config["max_argument_characters_per_call"]
        ),
    )
    python_tool_ids = {
        record["stable_id"]
        for record in manifest["tools"]
        if record["dispatch_kind"] == "local_python"
    }
    enable_python = bool(python_tool_ids & set(task_spec["available_tools"]))

    async with RawSseMcpClient(gateway_url) as mcp_client:
        mcp_executor = ManifestMcpExecutor(mcp_client, manifest)
        await mcp_executor.verify_runtime()
        executor = CompositeToolExecutor(
            manifest=manifest,
            mcp_executor=mcp_executor,
            workspace_root=workspace,
            enable_python_execute=enable_python,
        )
        episode = await collect_teacher_candidate(
            client=client,
            executor=executor,
            evaluator=DeclarativeWorkspaceEvaluator(
                workspace,
                task_spec["evaluator"],
            ),
            task_spec=task_spec,
            manifest=manifest,
            settings=settings,
            candidate_index=candidate_index,
        )
    episode["workspace_template"] = task_spec["task_id"] + "/initial_workspace"
    episode["initial_state_hash"] = actual_initial_hash
    episode["final_state_hash"] = workspace_digest(workspace)
    episode["teacher"]["fresh_environment_id"] = fresh_environment_id
    episode["teacher"]["gateway_url_recorded_as"] = "loopback-sse"
    return episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one compact teacher episode in one fresh Toolathlon environment"
    )
    parser.add_argument("--task-spec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000/sse")
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--fresh-environment-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--generation-config",
        default=str(PACKAGE_DIR / "configs" / "generation.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task_spec = json.loads(Path(args.task_spec).read_text(encoding="utf-8"))
    generation_config = json.loads(
        Path(args.generation_config).read_text(encoding="utf-8")
    )
    episode = asyncio.run(
        collect_one(
            task_spec=task_spec,
            manifest=load_manifest(args.manifest),
            generation_config=generation_config,
            workspace_root=args.workspace_root,
            gateway_url=args.gateway_url,
            candidate_index=args.candidate_index,
            fresh_environment_id=args.fresh_environment_id,
            client=OpenAICompatibleClient.from_env(),
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"episode={episode['episode_id']} accepted={episode['accepted']} "
        f"calls={len(episode['messages']) // 2}"
    )
    return 0 if episode["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
