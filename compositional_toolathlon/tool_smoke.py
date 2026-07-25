from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import canonical_json, load_manifest
from .mcp_adapter import ManifestMcpExecutor, RawSseMcpClient
from .local_tools import CompositeToolExecutor


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("smoke cases must be a JSON list")
    for case in payload:
        if not isinstance(case, dict):
            raise ValueError("each smoke case must be an object")
        if not isinstance(case.get("tool_id"), str):
            raise ValueError("each smoke case requires tool_id")
        if not isinstance(case.get("arguments"), dict):
            raise ValueError("each smoke case requires object arguments")
    return payload


async def run_smoke_cases(
    *,
    gateway_url: str,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    workspace_root: str | Path,
) -> dict[str, Any]:
    known = {record["stable_id"] for record in manifest["tools"]}
    unknown = sorted({case["tool_id"] for case in cases} - known)
    if unknown:
        raise ValueError(f"smoke cases reference unknown tools: {unknown}")
    results = []
    async with RawSseMcpClient(gateway_url) as client:
        executor = ManifestMcpExecutor(client, manifest)
        await executor.verify_runtime()
        composite = CompositeToolExecutor(
            manifest=manifest,
            mcp_executor=executor,
            workspace_root=workspace_root,
            enable_python_execute=any(
                next(
                    record
                    for record in manifest["tools"]
                    if record["stable_id"] == case["tool_id"]
                )["dispatch_kind"]
                == "local_python"
                for case in cases
            ),
        )
        for case_index, case in enumerate(cases):
            try:
                outcome = await composite.call_tool(
                    case["tool_id"],
                    case["arguments"],
                )
                success = bool(outcome["success"])
                serialized = canonical_json(outcome["observation"])
                results.append(
                    {
                        "case_index": case_index,
                        "tool_id": case["tool_id"],
                        "success": success,
                        "observation_sha256": hashlib.sha256(
                            serialized.encode("utf-8")
                        ).hexdigest(),
                        "observation_preview": serialized[:500],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "case_index": case_index,
                        "tool_id": case["tool_id"],
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    usable = sorted(
        {
            result["tool_id"]
            for result in results
            if result["success"]
        }
    )
    return {
        "schema_version": 1,
        "tool_manifest_hash": manifest["manifest_hash"],
        "gateway_url": gateway_url,
        "disposable_workspace_required": True,
        "usable_tool_ids": usable,
        "case_results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute explicit MCP smoke cases in a disposable Toolathlon workspace"
    )
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000/sse")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--require-all-cases", action="store_true")
    parser.add_argument(
        "--confirm-disposable-workspace",
        action="store_true",
        help="required acknowledgement because smoke calls may mutate files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.confirm_disposable_workspace:
        raise RuntimeError(
            "refusing to execute tools without --confirm-disposable-workspace"
        )
    report = asyncio.run(
        run_smoke_cases(
            gateway_url=args.gateway_url,
            manifest=load_manifest(args.manifest),
            cases=load_cases(args.cases),
            workspace_root=args.workspace_root,
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = [result for result in report["case_results"] if not result["success"]]
    print(
        f"usable_tools={len(report['usable_tool_ids'])} "
        f"cases={len(report['case_results'])} failed={len(failed)}"
    )
    return 1 if args.require_all_cases and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
