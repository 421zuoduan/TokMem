"""InterCode Try-Again protocol with exact control-token stripping."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch

from .io_utils import json_safe_log_value
from .official_env import OfficialStatusParseError
from .training_data import (
    SYSTEM_PROMPT,
    apply_chat_template,
    response_end_token_ids,
    strip_control_token_ids,
)

OBSERVATION_MAX_UTF8_BYTES = 2048
OBSERVATION_HEAD_UTF8_BYTES = 896
OBSERVATION_TAIL_UTF8_BYTES = 896
INTERCODE_BASH_EPISODE_SCHEMA = "intercode_bash_episode_v7"
OBSERVATION_FEEDBACK_POLICY = "utf8_head_tail_v1"


class BashEnvironment(Protocol):
    def reset(self, index: int) -> tuple[str, Mapping[str, Any]]: ...

    def step(self, action: str) -> tuple[str, float, bool, Mapping[str, Any]]: ...

    def submit(self) -> tuple[str, float, bool, Mapping[str, Any]]: ...

    def close(self) -> None: ...


@dataclass
class InteractionTurn:
    action: str
    observation: str
    reward: float
    valid_action: bool


@dataclass(frozen=True)
class BoundedTextRecord:
    raw_utf8_bytes: int
    raw_characters: int
    sha256: str
    truncated: bool
    feedback_utf8_bytes: int
    kept_head_utf8_bytes: int
    kept_tail_utf8_bytes: int
    feedback: str


class ContextOverflowError(ValueError):
    """The newest bounded action/observation pair cannot fit in context."""

    def __init__(
        self,
        *,
        required_tokens: int,
        context_limit: int,
        retained_turns: int,
    ) -> None:
        self.required_tokens = int(required_tokens)
        self.context_limit = int(context_limit)
        self.retained_turns = int(retained_turns)
        super().__init__(
            "System/instruction/latest-observation context requires "
            f"{self.required_tokens} tokens, exceeding the "
            f"{self.context_limit}-token limit"
        )


def bound_observation_text(value: str) -> BoundedTextRecord:
    """Create deterministic, method-independent feedback from shell output."""

    raw_bytes = value.encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    truncated = len(raw_bytes) > OBSERVATION_MAX_UTF8_BYTES
    if not truncated:
        feedback = value
        kept_head_utf8_bytes = len(raw_bytes)
        kept_tail_utf8_bytes = 0
    else:
        head = raw_bytes[:OBSERVATION_HEAD_UTF8_BYTES].decode(
            "utf-8",
            errors="ignore",
        )
        tail = raw_bytes[-OBSERVATION_TAIL_UTF8_BYTES:].decode(
            "utf-8",
            errors="ignore",
        )
        kept_head_utf8_bytes = len(head.encode("utf-8"))
        kept_tail_utf8_bytes = len(tail.encode("utf-8"))
        marker = (
            "\n...[OBSERVATION TRUNCATED "
            f"raw_utf8_bytes={len(raw_bytes)} "
            f"sha256={digest} "
            f"kept_head_utf8_bytes={kept_head_utf8_bytes} "
            f"kept_tail_utf8_bytes={kept_tail_utf8_bytes}]...\n"
        )
        if not marker.isascii():
            raise AssertionError("Observation truncation marker must be ASCII")
        feedback = head + marker + tail
        if len(feedback.encode("utf-8")) > OBSERVATION_MAX_UTF8_BYTES:
            raise AssertionError(
                "Bounded observation exceeds OBSERVATION_MAX_UTF8_BYTES"
            )
    return BoundedTextRecord(
        raw_utf8_bytes=len(raw_bytes),
        raw_characters=len(value),
        sha256=digest,
        truncated=truncated,
        feedback_utf8_bytes=len(feedback.encode("utf-8")),
        kept_head_utf8_bytes=kept_head_utf8_bytes,
        kept_tail_utf8_bytes=kept_tail_utf8_bytes,
        feedback=feedback,
    )


def bounded_json_log_value(value: Any) -> Any:
    """Make diagnostics JSON-safe and bound every overlong string recursively."""

    def replace_long_strings(item: Any) -> Any:
        if isinstance(item, str):
            record = bound_observation_text(item)
            return asdict(record) if record.truncated else item
        if isinstance(item, dict):
            return {
                key: replace_long_strings(member)
                for key, member in item.items()
            }
        if isinstance(item, list):
            return [replace_long_strings(member) for member in item]
        return item

    return replace_long_strings(json_safe_log_value(value))


def conversation_messages(
    instruction: str,
    history: Sequence[InteractionTurn],
    *,
    turn_number: int,
    max_turns: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Task:\n{instruction}\n\nReturn one Bash command.",
        },
    ]
    for history_index, turn in enumerate(history):
        messages.append({"role": "assistant", "content": turn.action})
        final_retained_turn = history_index == len(history) - 1
        next_attempt = (
            f"\nThis is attempt {turn_number} of {max_turns}. "
            if final_retained_turn
            else "\n"
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Shell observation:\n"
                    f"{turn.observation}\n"
                    f"Previous reward: {turn.reward:.6g}. "
                    f"{next_attempt}"
                    "If the task is not complete, return a corrected Bash command."
                ),
            }
        )
    if not history:
        messages[-1]["content"] += (
            f"\nThis is attempt {turn_number} of {max_turns}."
        )
    return messages


def fit_history_to_context(
    tokenizer,
    instruction: str,
    history: Sequence[InteractionTurn],
    *,
    turn_number: int,
    max_turns: int,
    context_limit: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[list[dict[str, str]], list[int], int]:
    """Drop only complete oldest action/observation pairs."""

    retained = list(history)
    dropped = 0
    while True:
        messages = conversation_messages(
            instruction,
            retained,
            turn_number=turn_number,
            max_turns=max_turns,
            system_prompt=system_prompt,
        )
        prompt_ids = apply_chat_template(tokenizer, messages)
        if len(prompt_ids) <= context_limit:
            return messages, prompt_ids, dropped
        if len(retained) <= 1:
            raise ContextOverflowError(
                required_tokens=len(prompt_ids),
                context_limit=context_limit,
                retained_turns=len(retained),
            )
        retained.pop(0)
        dropped += 1


def run_try_again_episode(
    *,
    model,
    tokenizer,
    environment: BashEnvironment,
    task: Mapping[str, Any],
    max_turns: int,
    max_new_tokens: int = 512,
    context_limit: int = 8192,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Run the official n-turn pattern: action -> submit after every attempt.

    InterCode's ``submit`` returns ``done=True`` on every attempt.  The official
    Try-Again protocol intentionally ignores that flag and continues in the
    same agent container until reward is exactly 1.0 or the turn budget ends.
    """

    query, reset_info = environment.reset(int(task["local_index"]))
    if query != task["query"]:
        raise ValueError(
            f"Environment query differs for {task['task_id']}: {query!r} != {task['query']!r}"
        )
    end_ids = response_end_token_ids(tokenizer)
    stop_sequences = [end_ids]
    if tokenizer.eos_token_id is not None:
        eos_sequence = [int(tokenizer.eos_token_id)]
        if eos_sequence not in stop_sequences:
            stop_sequences.append(eos_sequence)
    prompt_limit = int(context_limit) - int(max_new_tokens)
    if prompt_limit <= 0:
        raise ValueError(
            f"context_limit={context_limit} must exceed "
            f"max_new_tokens={max_new_tokens}"
        )

    history: list[InteractionTurn] = []
    turn_logs: list[dict[str, Any]] = []
    max_reward = 0.0
    max_released_reward = 0.0
    success = False
    termination_reason = "max_turns"
    context_overflow: dict[str, int] | None = None
    status_parse_error: dict[str, Any] | None = None
    observed_max_reward_before_parse_error: float | None = None
    observed_max_released_reward_before_parse_error: float | None = None
    for turn_index in range(max_turns):
        try:
            messages, prompt_ids, dropped_turns = fit_history_to_context(
                tokenizer,
                str(task["query"]),
                history,
                turn_number=turn_index + 1,
                max_turns=max_turns,
                context_limit=prompt_limit,
                system_prompt=system_prompt,
            )
        except ContextOverflowError as exc:
            if not history:
                raise
            termination_reason = "context_overflow"
            context_overflow = {
                "required_tokens": exc.required_tokens,
                "context_limit": exc.context_limit,
                "retained_turns": exc.retained_turns,
                "next_turn": turn_index + 1,
            }
            break
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        generation = model.generate_tokens(
            input_ids,
            attention_mask,
            response_end_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
        )
        registry = getattr(model, "registry", None)
        procedure_token_ids = (
            registry.procedure_token_ids if registry is not None else ()
        )
        eoc_token_id = (
            registry.eoc_token_id
            if registry is not None and bool(getattr(model, "use_eoc", False))
            else None
        )
        ordinary_ids, missing_terminator = strip_control_token_ids(
            generation["generated_ids"],
            procedure_token_ids=procedure_token_ids,
            eoc_token_id=eoc_token_id,
            response_end_sequences=stop_sequences,
        )
        action = tokenizer.decode(
            ordinary_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        observation, _zero, _action_done, action_info = environment.step(action)
        valid_action = bool(action_info.get("action_executed", False))
        try:
            (
                _submit_observation,
                reward,
                submit_done,
                reward_info,
            ) = environment.submit()
            reward = float(reward)
            released_reward = float(
                reward_info.get("released_reward", reward)
            )
            max_reward = max(max_reward, reward)
            max_released_reward = max(
                max_released_reward,
                released_reward,
            )
        except OfficialStatusParseError as exc:
            status_parse_error = dict(exc.details)
            reward = 0.0
            released_reward = 0.0
            submit_done = False
            reward_info = {
                "official_status_parse_error": status_parse_error,
            }
        observation_record = bound_observation_text(str(observation))
        history.append(
            InteractionTurn(
                action=action,
                observation=observation_record.feedback,
                reward=reward,
                valid_action=valid_action,
            )
        )
        turn_logs.append(
            {
                "turn": turn_index + 1,
                "prompt_token_count": len(prompt_ids),
                "prompt_token_limit": prompt_limit,
                "dropped_history_turns": dropped_turns,
                "generated_ids": generation["generated_ids"],
                "generated_text_with_controls": tokenizer.decode(
                    generation["generated_ids"],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "ordinary_action_ids": ordinary_ids,
                "action": action,
                "observation": observation_record.feedback,
                "observation_record": asdict(observation_record),
                "valid_action": valid_action,
                "reward": reward,
                "released_reward": released_reward,
                "submit_done": bool(submit_done),
                "missing_terminator": bool(missing_terminator),
                "memory_bank_constraint_trigger_count": int(
                    generation.get(
                        "memory_bank_constraint_trigger_count",
                        0,
                    )
                ),
                "memory_bank_constraint_changed_token_count": int(
                    generation.get(
                        "memory_bank_constraint_changed_token_count",
                        0,
                    )
                ),
                # This is saved for analysis only and is never included in history.
                "reward_info": bounded_json_log_value(reward_info),
            }
        )
        if status_parse_error is not None:
            observed_max_reward_before_parse_error = max_reward
            observed_max_released_reward_before_parse_error = (
                max_released_reward
            )
            success = False
            termination_reason = "official_status_parse_error"
            break
        if reward == 1.0:
            success = True
            termination_reason = "success"
            break

    episode = {
        "schema": INTERCODE_BASH_EPISODE_SCHEMA,
        "task_id": task["task_id"],
        "fs_id": task["fs_id"],
        "local_index": int(task["local_index"]),
        "query": task["query"],
        "max_turns": max_turns,
        "turns_taken": len(turn_logs),
        "max_reward": max_reward,
        "max_released_reward": max_released_reward,
        "released_success": max_released_reward == 1.0,
        "success": success,
        "termination_reason": termination_reason,
        "context_overflow": context_overflow,
        "reset_info": bounded_json_log_value(reset_info),
        "turns": turn_logs,
    }
    if status_parse_error is not None:
        episode["official_status_parse_error"] = status_parse_error
        episode["observed_max_reward_before_parse_error"] = (
            observed_max_reward_before_parse_error
        )
        episode["observed_max_released_reward_before_parse_error"] = (
            observed_max_released_reward_before_parse_error
        )
    return episode
