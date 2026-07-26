import json
import tempfile
import unittest
from pathlib import Path

from compositional_toolathlon.config import load_experiment_config
from compositional_toolathlon.host_gateway import (
    FILESYSTEM_SERVER_PACKAGE,
    PDF_TOOLS_SERVER_PACKAGE,
    build_server_config,
    normalize_server_names,
    prepare_host_gateway_runtime,
)


class HostGatewayConfigTest(unittest.TestCase):
    def test_server_configs_use_host_workspace_and_locked_runtime(self):
        workspace = Path("/tmp/synthetic-workspace")
        runtime_root = Path("/tmp/gateway-runtime")
        package_dir = Path("/repo/compositional_toolathlon")
        benchmark_root = package_dir / "vendor" / "toolathlon"
        common = {
            "workspace": workspace,
            "runtime_root": runtime_root,
            "package_dir": package_dir,
            "benchmark_root": benchmark_root,
            "npx_command": "/node/bin/npx",
            "uvx_command": "/tools/bin/uvx",
        }

        filesystem = build_server_config("filesystem", **common)
        self.assertEqual(filesystem["params"]["args"][-1], str(workspace))
        self.assertEqual(filesystem["params"]["command"], "/node/bin/npx")
        self.assertEqual(
            filesystem["params"]["args"][1],
            FILESYSTEM_SERVER_PACKAGE,
        )

        terminal = build_server_config("terminal", **common)
        self.assertEqual(terminal["params"]["env"]["ALLOWED_DIR"], str(workspace))
        self.assertEqual(
            terminal["params"]["command"],
            str(package_dir / ".toolathlon-venv/bin/cli-mcp-server"),
        )

        excel = build_server_config("excel", **common)
        self.assertEqual(excel["params"]["args"], ["stdio"])
        self.assertEqual(
            excel["params"]["command"],
            str(package_dir / ".toolathlon-venv/bin/excel-mcp-server"),
        )

        pdf = build_server_config("pdf-tools", **common)
        self.assertEqual(pdf["params"]["command"], "/tools/bin/uvx")
        self.assertEqual(
            pdf["params"]["args"][:3],
            ["--from", PDF_TOOLS_SERVER_PACKAGE, "python"],
        )
        self.assertIn(
            str(
                benchmark_root
                / "utils/local_servers/pdf_tools_local_wrapper.py"
            ),
            pdf["params"]["args"],
        )
        self.assertIn(
            str(runtime_root / "pdf-tools-temp"),
            pdf["params"]["args"],
        )
        self.assertNotIn(
            str(workspace / ".pdf_tools_tempfiles"),
            pdf["params"]["args"],
        )

    def test_prepare_writes_selected_configs_and_minimal_bundle(self):
        config = load_experiment_config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime_root = root / "runtime"
            stale_config_dir = runtime_root / "mcp_servers"
            stale_config_dir.mkdir(parents=True)
            (stale_config_dir / "terminal.yaml").write_text(
                "{}\n",
                encoding="utf-8",
            )
            benchmark_root = root / "toolathlon"
            package_dir = root / "track"

            runtime = prepare_host_gateway_runtime(
                workspace=workspace,
                runtime_root=runtime_root,
                servers=("filesystem", "excel"),
                benchmark_root=benchmark_root,
                experiment_config=config,
                package_dir=package_dir,
                npx_command="/node/bin/npx",
                uvx_command="/tools/bin/uvx",
            )

            self.assertEqual(runtime.servers, ("filesystem", "excel"))
            self.assertEqual(
                sorted(path.name for path in runtime.config_dir.glob("*.yaml")),
                ["excel.yaml", "filesystem.yaml"],
            )
            bundle = json.loads(runtime.bundle_file.read_text(encoding="utf-8"))
            self.assertEqual(
                bundle["needed_mcp_servers"],
                ["filesystem", "excel"],
            )
            self.assertEqual(
                bundle["container_paths"]["agent_workspace"],
                str(workspace.resolve()),
            )
            self.assertEqual(
                bundle["eval_config"]["mcp"]["server_config_path"],
                str(runtime.config_dir.resolve()),
            )

    def test_server_selection_is_config_ordered_and_rejects_unknown(self):
        allowed = ("filesystem", "terminal", "excel", "pdf-tools")
        self.assertEqual(
            normalize_server_names(
                ("pdf-tools,filesystem", "excel"),
                allowed=allowed,
            ),
            ("filesystem", "excel", "pdf-tools"),
        )
        with self.assertRaisesRegex(ValueError, "unsupported host MCP servers"):
            normalize_server_names(("browser",), allowed=allowed)


if __name__ == "__main__":
    unittest.main()
