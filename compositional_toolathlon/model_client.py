from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .manifest import canonical_json


CODEX_JSON_OBJECT_SCHEMA = (
    PACKAGE_DIR / "configs" / "codex_json_object_schema.json"
)


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
    def from_env(cls) -> "OpenAICompatibleClient | CodexCliClient":
        provider = os.environ.get("TOOLATHLON_LLM_PROVIDER", "openai")
        if provider == "codex_cli":
            return CodexCliClient.from_env()
        if provider != "openai":
            raise RuntimeError(
                "TOOLATHLON_LLM_PROVIDER must be 'openai' or 'codex_cli'"
            )
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
        response_schema: str | Path | None = None,
    ) -> dict[str, Any]:
        if response_schema is None:
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            schema_path = Path(response_schema)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_path.stem,
                    "strict": True,
                    "schema": json.loads(schema_path.read_text(encoding="utf-8")),
                },
            }

        def request() -> Any:
            return self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
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


class CodexCliClient:
    """Use stateless Codex turns for JSON and one resumed thread per teacher episode."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        trace_path: str | Path | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model = model
        self.trace_path = Path(
            trace_path
            or PACKAGE_DIR
            / "data"
            / "generated"
            / "provenance"
            / "codex_cli_invocations.jsonl"
        )
        self.timeout_seconds = timeout_seconds
        self.invocations: list[dict[str, Any]] = []
        self._trace_lock = threading.Lock()
        self._codex_workdir = tempfile.TemporaryDirectory(
            prefix="toolathlon_codex_cli_"
        )
        self._teacher_thread_id: str | None = None
        self._teacher_context_sha256: str | None = None
        self._teacher_history_length = 0

    @classmethod
    def from_env(cls) -> "CodexCliClient":
        return cls(
            model=os.environ.get("TOOLATHLON_CODEX_MODEL", "gpt-5.6-sol"),
            trace_path=os.environ.get("TOOLATHLON_CODEX_TRACE_PATH"),
        )

    def close(self) -> None:
        self._codex_workdir.cleanup()

    @property
    def teacher_thread_id(self) -> str | None:
        return self._teacher_thread_id

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _initial_command(
        model: str,
        workdir: str,
        *,
        ephemeral: bool,
        output_schema: str | Path = CODEX_JSON_OBJECT_SCHEMA,
    ) -> list[str]:
        command = [
            "codex",
            "exec",
            "--json",
            "-m",
            model,
            "-s",
            "read-only",
            "-C",
            workdir,
            "--skip-git-repo-check",
            "--ignore-rules",
            "--output-schema",
            str(output_schema),
            "-",
        ]
        if ephemeral:
            command.insert(3, "--ephemeral")
        return command

    @staticmethod
    def _resume_command(
        model: str,
        thread_id: str,
        *,
        output_schema: str | Path = CODEX_JSON_OBJECT_SCHEMA,
    ) -> list[str]:
        return [
            "codex",
            "exec",
            "resume",
            "--json",
            "-m",
            model,
            "--skip-git-repo-check",
            "--ignore-rules",
            "--output-schema",
            str(output_schema),
            thread_id,
            "-",
        ]

    @staticmethod
    def _render_prompt(
        *,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int,
        wrap_payload: bool,
    ) -> str:
        if wrap_payload:
            response_instruction = (
                "Form the requested JSON object, serialize it completely, and put "
                "that serialized JSON string in the sole `payload` field of the "
                "outer response."
            )
        else:
            response_instruction = (
                "Return the requested JSON object directly, matching the supplied "
                "output schema."
            )
        return (
            "Follow these instructions without inspecting the local workspace or "
            "using tools.\n\n"
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{user_prompt}\n\n"
            f"{response_instruction} Do not add markdown or commentary. "
            f"Keep it concise (at most about {max_completion_tokens} tokens)."
        )

    @staticmethod
    def _parse_jsonl(
        stdout: str,
        *,
        unwrap_payload: bool = True,
    ) -> tuple[str, str, dict[str, Any]]:
        thread_id: str | None = None
        response_text: str | None = None
        for line in stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") in {
                "command_execution",
                "file_change",
                "mcp_tool_call",
                "tool_call",
                "web_search",
            }:
                raise ValueError(
                    "codex provider attempted command or tool execution: "
                    f"{item.get('type')}"
                )
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                response_text = item["text"]
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("codex exec JSONL did not contain a thread_id")
        if response_text is None:
            raise ValueError("codex exec JSONL did not contain a final agent message")
        try:
            envelope = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"codex final response was not pure JSON: {exc}") from exc
        if not isinstance(envelope, dict):
            raise ValueError("codex final response must be a JSON object")
        if unwrap_payload:
            if set(envelope) != {"payload"} or not isinstance(
                envelope["payload"],
                str,
            ):
                raise ValueError("codex final response must use the payload envelope")
            try:
                payload = json.loads(envelope["payload"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"codex payload was not valid JSON: {exc}") from exc
        else:
            payload = envelope
        if not isinstance(payload, dict):
            raise ValueError("codex payload must be a JSON object")
        return thread_id, response_text, payload

    @staticmethod
    def _error_from_jsonl(stdout: str) -> str:
        messages = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error" and isinstance(
                event.get("message"),
                str,
            ):
                messages.append(event["message"])
            error = event.get("error")
            if event.get("type") == "turn.failed" and isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    messages.append(message)
        return messages[-1] if messages else ""

    def _record_invocation(
        self,
        *,
        thread_id: str,
        request_kind: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
    ) -> None:
        with self._trace_lock:
            record = {
                "invocation_index": len(self.invocations),
                "thread_id": thread_id,
                "model": self.model,
                "request_kind": request_kind,
                "system_prompt_sha256": self._sha256(system_prompt),
                "user_prompt_sha256": self._sha256(user_prompt),
                "response_sha256": self._sha256(response_text),
            }
            self.invocations.append(record)
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(record) + "\n")

    def _invoke(
        self,
        *,
        command: list[str],
        request_kind: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int,
        unwrap_payload: bool,
    ) -> tuple[str, dict[str, Any]]:
        prompt = self._render_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_completion_tokens=max_completion_tokens,
            wrap_payload=unwrap_payload,
        )
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
            cwd=self._codex_workdir.name,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or self._error_from_jsonl(
                completed.stdout
            )
            raise RuntimeError(
                f"codex exec failed with exit code {completed.returncode}: "
                f"{detail}"
            )
        thread_id, response_text, payload = self._parse_jsonl(
            completed.stdout,
            unwrap_payload=unwrap_payload,
        )
        self._record_invocation(
            thread_id=thread_id,
            request_kind=request_kind,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )
        return thread_id, payload

    async def request_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int,
        response_schema: str | Path | None = None,
    ) -> dict[str, Any]:
        if model != self.model:
            raise ValueError(
                f"configured model {model!r} differs from Codex model {self.model!r}"
            )
        _, payload = await asyncio.to_thread(
            self._invoke,
            command=self._initial_command(
                self.model,
                self._codex_workdir.name,
                ephemeral=True,
                output_schema=response_schema or CODEX_JSON_OBJECT_SCHEMA,
            ),
            request_kind="request_json",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_completion_tokens=max_completion_tokens,
            unwrap_payload=response_schema is None,
        )
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
        if model != self.model:
            raise ValueError(
                f"configured model {model!r} differs from Codex model {self.model!r}"
            )
        context = canonical_json(
            {
                "instruction": instruction,
                "tools": tools,
            }
        )
        context_sha256 = self._sha256(system_prompt + "\n" + context)
        if not history:
            user_prompt = canonical_json(
                {
                    "instruction": instruction,
                    "history": [],
                    "tools": tools,
                    "output": {
                        "tool_alias": "one supplied tool name",
                        "arguments": {},
                    },
                }
            )
            thread_id, payload = await asyncio.to_thread(
                self._invoke,
                command=self._initial_command(
                    self.model,
                    self._codex_workdir.name,
                    ephemeral=False,
                ),
                request_kind="next_tool_action_start",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=max_completion_tokens,
                unwrap_payload=True,
            )
            self._teacher_thread_id = thread_id
            self._teacher_context_sha256 = context_sha256
            self._teacher_history_length = 0
        else:
            if self._teacher_thread_id is None:
                raise ValueError("Codex teacher history has no active thread")
            if self._teacher_context_sha256 != context_sha256:
                raise ValueError("Codex teacher instruction or tool menu changed mid-episode")
            if len(history) != self._teacher_history_length + 1:
                raise ValueError("Codex teacher requires exactly one new observation per turn")
            user_prompt = canonical_json(
                {
                    "observation": history[-1]["observation"],
                    "output": {
                        "tool_alias": "one supplied tool name",
                        "arguments": {},
                    },
                }
            )
            thread_id, payload = await asyncio.to_thread(
                self._invoke,
                command=self._resume_command(
                    self.model,
                    self._teacher_thread_id,
                ),
                request_kind="next_tool_action_resume",
                system_prompt="",
                user_prompt=user_prompt,
                max_completion_tokens=max_completion_tokens,
                unwrap_payload=True,
            )
            if thread_id != self._teacher_thread_id:
                raise ValueError("codex exec resume returned a different thread_id")
            self._teacher_history_length = len(history)

        tool_alias = payload.get("tool_alias")
        arguments = payload.get("arguments")
        if not isinstance(tool_alias, str) or not tool_alias:
            raise ValueError("Codex teacher JSON requires a non-empty tool_alias")
        if not isinstance(arguments, dict):
            raise ValueError("Codex teacher JSON requires object arguments")
        return ProposedAction(
            call_id=f"codex_{uuid.uuid4().hex}",
            tool_alias=tool_alias,
            arguments=arguments,
            visible_text="",
        )


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()
