from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import canonical_json


@dataclass(frozen=True)
class ProposedAction:
    call_id: str
    tool_alias: str
    arguments: dict[str, Any]
    visible_text: str


class OpenAICompatibleClient:
    """Small provider wrapper; API keys are read from env and never serialized."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed; use the Toolathlon runtime environment"
            ) from exc
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        base_url = os.environ.get("TOOLATHLON_LLM_BASE_URL")
        api_key = os.environ.get("TOOLATHLON_LLM_API_KEY")
        if not base_url:
            raise RuntimeError("TOOLATHLON_LLM_BASE_URL is required")
        if api_key is None:
            raise RuntimeError("TOOLATHLON_LLM_API_KEY is required; use 'dummy' locally")
        return cls(base_url=base_url, api_key=api_key)

    async def request_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int,
    ) -> dict[str, Any]:
        def request() -> Any:
            return self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=max_completion_tokens,
            )

        response = await asyncio.to_thread(request)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("JSON generation returned no text content")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("model JSON response must be an object")
        return payload

    async def next_tool_action(
        self,
        *,
        model: str,
        system_prompt: str,
        instruction: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int,
    ) -> ProposedAction:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        for event in history:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": event["call_id"],
                            "type": "function",
                            "function": {
                                "name": event["tool_alias"],
                                "arguments": canonical_json(event["arguments"]),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event["call_id"],
                    "content": canonical_json(event["observation"]),
                }
            )

        def request() -> Any:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                max_completion_tokens=max_completion_tokens,
            )

        response = await asyncio.to_thread(request)
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])
        if len(tool_calls) != 1:
            raise ValueError(
                f"teacher must return exactly one tool call, received {len(tool_calls)}"
            )
        raw_arguments = tool_calls[0].function.arguments
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"teacher returned invalid tool arguments: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("teacher tool arguments must be a JSON object")
        visible_text = message.content if isinstance(message.content, str) else ""
        return ProposedAction(
            call_id=str(tool_calls[0].id),
            tool_alias=str(tool_calls[0].function.name),
            arguments=arguments,
            visible_text=visible_text,
        )


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()
