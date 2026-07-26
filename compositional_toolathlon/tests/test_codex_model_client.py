from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compositional_toolathlon.model_client import (
    CODEX_JSON_OBJECT_SCHEMA,
    CodexCliClient,
    OpenAICompatibleClient,
)


def _codex_result(thread_id: str, payload: dict) -> subprocess.CompletedProcess:
    response = json.dumps(
        {"payload": json.dumps(payload, separators=(",", ":"))},
        separators=(",", ":"),
    )
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": thread_id}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": response},
                }
            ),
        ]
    )
    return subprocess.CompletedProcess(
        args=["codex"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


async def _direct_to_thread(function, /, *args, **kwargs):
    return function(*args, **kwargs)


class ProviderSelectionTests(unittest.TestCase):
    def test_openai_remains_the_default_provider(self) -> None:
        environment = {
            "TOOLATHLON_LLM_BASE_URL": "http://example.invalid/v1",
            "TOOLATHLON_LLM_API_KEY": "test-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(OpenAICompatibleClient, "__init__", return_value=None) as init,
        ):
            client = OpenAICompatibleClient.from_env()
        self.assertIsInstance(client, OpenAICompatibleClient)
        init.assert_called_once_with(
            base_url="http://example.invalid/v1",
            api_key="test-key",
        )

    def test_codex_provider_is_selected_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "trace.jsonl"
            environment = {
                "TOOLATHLON_LLM_PROVIDER": "codex_cli",
                "TOOLATHLON_CODEX_MODEL": "gpt-5.6-sol",
                "TOOLATHLON_CODEX_TRACE_PATH": str(trace_path),
            }
            with patch.dict(os.environ, environment, clear=True):
                client = OpenAICompatibleClient.from_env()
        self.assertIsInstance(client, CodexCliClient)
        self.assertEqual(client.model, "gpt-5.6-sol")
        client.close()


class CodexCliClientTests(unittest.TestCase):
    def test_request_json_records_hashed_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "trace.jsonl"
            client = CodexCliClient(trace_path=trace_path)
            self.addCleanup(client.close)
            with (
                patch(
                    "compositional_toolathlon.model_client.subprocess.run",
                    return_value=_codex_result(
                        "thread-task",
                        {"task_id": "task-1"},
                    ),
                ) as run,
                patch(
                    "compositional_toolathlon.model_client.asyncio.to_thread",
                    new=_direct_to_thread,
                ),
            ):
                result = asyncio.run(
                    client.request_json(
                        model="gpt-5.6-sol",
                        system_prompt="generate a task",
                        user_prompt='{"seed":7}',
                        max_completion_tokens=200,
                    )
                )

            self.assertEqual(result, {"task_id": "task-1"})
            self.assertEqual(
                run.call_args.args[0],
                [
                    "codex",
                    "exec",
                    "--json",
                    "--ephemeral",
                    "-m",
                    "gpt-5.6-sol",
                    "-s",
                    "read-only",
                    "-C",
                    client._codex_workdir.name,
                    "--skip-git-repo-check",
                    "--ignore-rules",
                    "--output-schema",
                    str(CODEX_JSON_OBJECT_SCHEMA),
                    "-",
                ],
            )
            self.assertEqual(
                run.call_args.kwargs["cwd"],
                client._codex_workdir.name,
            )
            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["invocation_index"], 0)
            self.assertEqual(records[0]["thread_id"], "thread-task")
            self.assertEqual(records[0]["model"], "gpt-5.6-sol")
            self.assertEqual(records[0]["request_kind"], "request_json")
            self.assertEqual(
                records[0]["system_prompt_sha256"],
                hashlib.sha256(b"generate a task").hexdigest(),
            )
            self.assertEqual(
                records[0]["user_prompt_sha256"],
                hashlib.sha256(b'{"seed":7}').hexdigest(),
            )

    def test_teacher_resumes_thread_with_only_new_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "trace.jsonl"
            client = CodexCliClient(trace_path=trace_path)
            self.addCleanup(client.close)
            results = [
                _codex_result(
                    "thread-teacher",
                    {"tool_alias": "tool_0000", "arguments": {"path": "."}},
                ),
                _codex_result(
                    "thread-teacher",
                    {"tool_alias": "tool_0001", "arguments": {}},
                ),
            ]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_0000",
                        "description": "UNIQUE_LIST_DESCRIPTION",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "tool_0001",
                        "description": "finish",
                        "parameters": {"type": "object"},
                    },
                },
            ]
            with (
                patch(
                    "compositional_toolathlon.model_client.subprocess.run",
                    side_effect=results,
                ) as run,
                patch(
                    "compositional_toolathlon.model_client.asyncio.to_thread",
                    new=_direct_to_thread,
                ),
            ):
                first = asyncio.run(
                    client.next_tool_action(
                        model="gpt-5.6-sol",
                        system_prompt="return one action",
                        instruction="UNIQUE_TASK_INSTRUCTION",
                        history=[],
                        tools=tools,
                        max_completion_tokens=100,
                    )
                )
                history = [
                    {
                        "call_id": first.call_id,
                        "tool_alias": first.tool_alias,
                        "arguments": first.arguments,
                        "observation": {"entries": ["report.txt"]},
                    }
                ]
                second = asyncio.run(
                    client.next_tool_action(
                        model="gpt-5.6-sol",
                        system_prompt="return one action",
                        instruction="UNIQUE_TASK_INSTRUCTION",
                        history=history,
                        tools=tools,
                        max_completion_tokens=100,
                    )
                )

            self.assertEqual(first.tool_alias, "tool_0000")
            self.assertEqual(second.tool_alias, "tool_0001")
            self.assertEqual(second.visible_text, "")
            first_command = run.call_args_list[0].args[0]
            self.assertNotIn("--ephemeral", first_command)
            self.assertIn(client._codex_workdir.name, first_command)
            self.assertEqual(
                run.call_args_list[1].args[0],
                [
                    "codex",
                    "exec",
                    "resume",
                    "--json",
                    "-m",
                    "gpt-5.6-sol",
                    "--skip-git-repo-check",
                    "--ignore-rules",
                    "--output-schema",
                    str(CODEX_JSON_OBJECT_SCHEMA),
                    "thread-teacher",
                    "-",
                ],
            )
            resume_prompt = run.call_args_list[1].kwargs["input"]
            self.assertIn("report.txt", resume_prompt)
            self.assertNotIn("UNIQUE_TASK_INSTRUCTION", resume_prompt)
            self.assertNotIn("UNIQUE_LIST_DESCRIPTION", resume_prompt)
            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["request_kind"] for record in records],
                ["next_tool_action_start", "next_tool_action_resume"],
            )
            self.assertEqual(
                [record["thread_id"] for record in records],
                ["thread-teacher", "thread-teacher"],
            )
            self.assertEqual(
                [record["invocation_index"] for record in records],
                [0, 1],
            )

    def test_tool_execution_event_is_rejected(self) -> None:
        stdout = "\n".join(
            [
                json.dumps(
                    {"type": "thread.started", "thread_id": "thread-tool"}
                ),
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": "command_execution", "command": "ls"},
                    }
                ),
            ]
        )
        with self.assertRaisesRegex(ValueError, "command or tool execution"):
            CodexCliClient._parse_jsonl(stdout)


if __name__ == "__main__":
    unittest.main()
