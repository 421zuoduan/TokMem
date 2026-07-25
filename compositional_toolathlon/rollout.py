from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Action:
    tool_id: str
    arguments: dict[str, Any]
    stop_reason: str = "policy"


class ClosedLoopPolicy(Protocol):
    async def next_action(
        self,
        *,
        instruction: str,
        available_tool_ids: list[str],
        history: list[dict[str, Any]],
    ) -> Action:
        ...


class RuntimeExecutor(Protocol):
    async def call_tool(
        self,
        stable_tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class RuntimeEvaluator(Protocol):
    async def evaluate(self) -> dict[str, Any]:
        ...


async def run_closed_loop_rollout(
    *,
    instruction: str,
    available_tool_ids: list[str],
    terminal_tool_ids: set[str],
    policy: ClosedLoopPolicy,
    executor: RuntimeExecutor,
    evaluator: RuntimeEvaluator | None,
    max_tool_calls: int,
    max_consecutive_errors: int = 3,
) -> dict[str, Any]:
    if max_tool_calls <= 0:
        raise ValueError("max_tool_calls must be positive")
    if max_consecutive_errors <= 0:
        raise ValueError("max_consecutive_errors must be positive")
    if not available_tool_ids:
        raise ValueError("available_tool_ids cannot be empty")

    history: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    consecutive_errors = 0
    termination_reason = "max_tool_calls"

    for step_index in range(max_tool_calls):
        try:
            action = await policy.next_action(
                instruction=instruction,
                available_tool_ids=available_tool_ids,
                history=list(history),
            )
        except Exception as exc:
            consecutive_errors += 1
            events.append(
                {
                    "step_index": step_index,
                    "kind": "decode_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if consecutive_errors >= max_consecutive_errors:
                termination_reason = "consecutive_decode_errors"
                break
            continue

        if action.tool_id not in available_tool_ids:
            outcome = {
                "success": False,
                "observation": {
                    "error": "unavailable_tool",
                    "tool_id": action.tool_id,
                },
            }
        else:
            try:
                outcome = await executor.call_tool(action.tool_id, action.arguments)
            except Exception as exc:
                outcome = {
                    "success": False,
                    "observation": {
                        "error": "tool_execution_exception",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
        if not isinstance(outcome, dict) or not isinstance(outcome.get("success"), bool):
            raise ValueError("executor must return success bool and observation")
        if "observation" not in outcome:
            raise ValueError("executor outcome is missing observation")

        event = {
            "step_index": step_index,
            "kind": "tool_call",
            "action": asdict(action),
            "success": outcome["success"],
            "observation": outcome["observation"],
        }
        events.append(event)
        history.append(
            {
                "tool_id": action.tool_id,
                "arguments": action.arguments,
                "observation": outcome["observation"],
                "success": outcome["success"],
            }
        )
        consecutive_errors = 0 if outcome["success"] else consecutive_errors + 1

        if action.tool_id in terminal_tool_ids and outcome["success"]:
            termination_reason = "claim_done"
            break
        if consecutive_errors >= max_consecutive_errors:
            termination_reason = "consecutive_tool_errors"
            break

    if evaluator is None:
        evaluation = {
            "deferred_to_official_runner": True,
        }
    else:
        evaluation = await evaluator.evaluate()
        if not isinstance(evaluation, dict) or not isinstance(
            evaluation.get("passed"),
            bool,
        ):
            raise ValueError("evaluator must return a passed bool")
    return {
        "instruction": instruction,
        "available_tool_ids": available_tool_ids,
        "events": events,
        "termination_reason": termination_reason,
        "tool_call_count": sum(event["kind"] == "tool_call" for event in events),
        "decode_error_count": sum(event["kind"] == "decode_error" for event in events),
        "execution_error_count": sum(
            event["kind"] == "tool_call" and not event["success"] for event in events
        ),
        "evaluator": evaluation,
    }
