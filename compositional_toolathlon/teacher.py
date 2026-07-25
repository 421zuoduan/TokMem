from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .manifest import canonical_json
from .model_client import OpenAICompatibleClient


class ToolExecutor(Protocol):
    async def call_tool(
        self,
        stable_tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Return {"success": bool, "observation": JSON-serializable value}."""


class FinalStateEvaluator(Protocol):
    async def evaluate(self) -> dict[str, Any]:
        """Return at least {"passed": bool, "version": str}."""


@dataclass(frozen=True)
class TeacherSettings:
    model: str
    prompt_version: str
    system_prompt: str
    max_tool_calls: int = 30
    max_completion_tokens_per_action: int = 768
    max_visible_assistant_characters: int = 0
    max_argument_characters_per_call: int = 1600


def build_tool_aliases(
    manifest: dict[str, Any],
    available_tool_ids: list[str],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    records = {record["stable_id"]: record for record in manifest["tools"]}
    unknown = sorted(set(available_tool_ids) - set(records))
    if unknown:
        raise ValueError(f"available tools are absent from manifest: {unknown}")

    alias_to_stable: dict[str, str] = {}
    stable_to_alias: dict[str, str] = {}
    provider_tools = []
    for position, stable_id in enumerate(available_tool_ids):
        alias = f"tool_{position:04d}"
        record = records[stable_id]
        alias_to_stable[alias] = stable_id
        stable_to_alias[stable_id] = alias
        provider_tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": record.get("description", ""),
                    "parameters": record["input_schema"],
                },
            }
        )
    return alias_to_stable, stable_to_alias, provider_tools


def terminal_tool_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        record["stable_id"]
        for record in manifest["tools"]
        if record.get("dispatch_kind") == "terminal"
    }


async def collect_teacher_candidate(
    *,
    client: OpenAICompatibleClient,
    executor: ToolExecutor,
    evaluator: FinalStateEvaluator,
    task_spec: dict[str, Any],
    manifest: dict[str, Any],
    settings: TeacherSettings,
    candidate_index: int,
) -> dict[str, Any]:
    available_tool_ids = list(task_spec["available_tools"])
    alias_to_stable, _, provider_tools = build_tool_aliases(
        manifest,
        available_tool_ids,
    )
    terminal_ids = terminal_tool_ids(manifest)
    model_history: list[dict[str, Any]] = []
    episode_messages: list[dict[str, Any]] = []
    rejection_reasons = []
    visible_character_count = 0
    termination_reason = "max_tool_calls"

    for step_index in range(settings.max_tool_calls):
        action = await client.next_tool_action(
            model=settings.model,
            system_prompt=settings.system_prompt,
            instruction=task_spec["instruction"],
            history=model_history,
            tools=provider_tools,
            max_completion_tokens=settings.max_completion_tokens_per_action,
        )
        if action.tool_alias not in alias_to_stable:
            raise ValueError(f"teacher returned unknown tool alias: {action.tool_alias}")
        stable_id = alias_to_stable[action.tool_alias]
        argument_characters = len(canonical_json(action.arguments))
        if argument_characters > settings.max_argument_characters_per_call:
            raise ValueError(
                "teacher tool arguments are too long "
                f"({argument_characters} > "
                f"{settings.max_argument_characters_per_call} chars)"
            )
        visible_character_count += len(action.visible_text.strip())
        outcome = await executor.call_tool(stable_id, action.arguments)
        if not isinstance(outcome, dict) or not isinstance(outcome.get("success"), bool):
            raise ValueError("tool executor must return success bool and observation")
        if "observation" not in outcome:
            raise ValueError("tool executor result is missing observation")

        is_terminal = stable_id in terminal_ids
        episode_messages.append(
            {
                "role": "assistant",
                "tool_id": stable_id,
                "arguments": action.arguments,
                "is_terminal": is_terminal,
            }
        )
        episode_messages.append(
            {
                "role": "tool",
                "tool_id": stable_id,
                "observation": outcome["observation"],
                "success": outcome["success"],
            }
        )
        model_history.append(
            {
                "call_id": action.call_id,
                "tool_alias": action.tool_alias,
                "arguments": action.arguments,
                "observation": outcome["observation"],
            }
        )
        if not outcome["success"]:
            rejection_reasons.append(f"tool execution failed at step {step_index}")
        if is_terminal:
            termination_reason = "claim_done"
            break

    evaluation = await evaluator.evaluate()
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("passed"), bool):
        raise ValueError("evaluator must return a passed bool")
    if not evaluation["passed"]:
        rejection_reasons.append("deterministic evaluator failed")
    if termination_reason != "claim_done":
        rejection_reasons.append("teacher did not call claim_done")
    if visible_character_count > settings.max_visible_assistant_characters:
        rejection_reasons.append(
            "teacher emitted visible prose "
            f"({visible_character_count} > {settings.max_visible_assistant_characters} chars)"
        )

    episode_id = (
        f"{task_spec['task_id']}_candidate_{candidate_index:02d}_"
        f"{uuid.uuid4().hex[:10]}"
    )
    return {
        "episode_id": episode_id,
        "task_id": task_spec["task_id"],
        "template_id": task_spec["template_id"],
        "asset_seed": task_spec.get("asset_seed"),
        "split": task_spec["split"],
        "instruction": task_spec["instruction"],
        "available_tool_ids": available_tool_ids,
        "messages": episode_messages,
        "evaluator": evaluation,
        "teacher": {
            "model": settings.model,
            "prompt_version": settings.prompt_version,
            "candidate_index": candidate_index,
            "visible_assistant_characters": visible_character_count,
            "max_argument_characters_per_call": (
                settings.max_argument_characters_per_call
            ),
            "termination_reason": termination_reason,
        },
        "tool_manifest_hash": manifest["manifest_hash"],
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
    }
