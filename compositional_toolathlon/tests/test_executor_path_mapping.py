from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from compositional_toolathlon.local_tools import CompositeToolExecutor


class _CapturingMcpExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, stable_tool_id: str, arguments: dict):
        self.calls.append((stable_tool_id, arguments))
        return {
            "success": True,
            "observation": {
                "path": arguments.get("path"),
            },
        }


class ExecutorPathMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_arguments_use_container_workspace(self):
        manifest = {
            "tools": [
                {
                    "stable_id": "filesystem-read",
                    "dispatch_kind": "mcp",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        mcp = _CapturingMcpExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            executor = CompositeToolExecutor(
                manifest=manifest,
                mcp_executor=mcp,
                workspace_root=temporary,
                mcp_workspace_root="/workspace/dumps/task/agent_workspace",
                enable_python_execute=False,
            )
            outcome = await executor.call_tool(
                "filesystem-read",
                {"path": "<WORKSPACE>/input.txt"},
            )
        self.assertEqual(
            mcp.calls,
            [
                (
                    "filesystem-read",
                    {
                        "path": (
                            "/workspace/dumps/task/agent_workspace/input.txt"
                        )
                    },
                )
            ],
        )
        self.assertEqual(
            outcome["observation"],
            {"path": "<WORKSPACE>/input.txt"},
        )

    async def test_local_python_arguments_use_jail_workspace(self):
        manifest = {
            "tools": [
                {
                    "stable_id": "local-python",
                    "dispatch_kind": "local_python",
                    "input_schema": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        fake_python = AsyncMock()
        fake_python.call.return_value = {
            "success": True,
            "observation": "wrote /workspace/output.txt",
        }
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "compositional_toolathlon.local_tools.PythonExecuteTool",
                return_value=fake_python,
            ):
                executor = CompositeToolExecutor(
                    manifest=manifest,
                    mcp_executor=_CapturingMcpExecutor(),
                    workspace_root=temporary,
                    mcp_workspace_root="/workspace/dumps/task/agent_workspace",
                    enable_python_execute=True,
                )
            outcome = await executor.call_tool(
                "local-python",
                {"code": "open('<WORKSPACE>/output.txt', 'w').close()"},
            )
        fake_python.call.assert_awaited_once_with(
            {"code": "open('/workspace/output.txt', 'w').close()"}
        )
        self.assertEqual(
            outcome["observation"],
            "wrote <WORKSPACE>/output.txt",
        )

    async def test_official_runtime_can_defer_argument_validation_to_mcp_server(self):
        manifest = {
            "tools": [
                {
                    "stable_id": "filesystem-read",
                    "dispatch_kind": "mcp",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        mcp = _CapturingMcpExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            executor = CompositeToolExecutor(
                manifest=manifest,
                mcp_executor=mcp,
                workspace_root=temporary,
                mcp_workspace_root="/workspace/dumps/task/agent_workspace",
                enable_python_execute=False,
                validate_mcp_arguments=False,
            )
            await executor.call_tool(
                "filesystem-read",
                {"runtime_version_specific": True},
            )
        self.assertEqual(
            mcp.calls,
            [
                (
                    "filesystem-read",
                    {"runtime_version_specific": True},
                )
            ],
        )

    def test_mcp_workspace_must_be_normalized_absolute(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "normalized absolute"):
                CompositeToolExecutor(
                    manifest={"tools": []},
                    mcp_executor=_CapturingMcpExecutor(),
                    workspace_root=Path(temporary),
                    mcp_workspace_root="../container-workspace",
                    enable_python_execute=False,
                )


if __name__ == "__main__":
    unittest.main()
