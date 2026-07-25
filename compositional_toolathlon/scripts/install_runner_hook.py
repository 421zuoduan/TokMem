#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import uuid
from pathlib import Path


REVISION = "2aed2468858f15818acafa178518390cc4b0f5cb"
ORIGINAL_SHA256 = "4caa92af9889c4788d353131ee8e2e4c66a3cb960325b252019f2ece6da5c44a"


def replace_once(text: str, before: str, after: str, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise ValueError(
            f"runner hook expected one {label} block, found {count}; "
            "refusing source drift"
        )
    return text.replace(before, after, 1)


def patched_runner(original: str) -> str:
    text = replace_once(
        original,
        "#   - toolathlon_default\n#   - claude_agent_sdk\n",
        (
            "#   - toolathlon_default\n"
            "#   - claude_agent_sdk\n"
            "#   - tokmem_runtime\n"
        ),
        "framework header",
    )
    text = replace_once(
        text,
        """        claude_agent_sdk|claude_sdk|claude)
            echo "claude_agent_sdk"
            ;;
        *)
""",
        """        claude_agent_sdk|claude_sdk|claude)
            echo "claude_agent_sdk"
            ;;
        tokmem_runtime|tokmem)
            echo "tokmem_runtime"
            ;;
        *)
""",
        "framework resolver",
    )
    text = replace_once(
        text,
        """        claude_agent_sdk)
            echo "claude_sdk"
            ;;
        *)
""",
        """        claude_agent_sdk)
            echo "claude_sdk"
            ;;
        tokmem_runtime)
            echo "tokmem_runtime"
            ;;
        *)
""",
        "host backend resolver",
    )
    text = replace_once(
        text,
        """case "$host_loop_backend" in
    claude|claude_sdk|claude_agent_sdk)
        USE_UNIFIED_MODEL_ENV=false
        ;;
esac
""",
        """case "$host_loop_backend" in
    claude|claude_sdk|claude_agent_sdk|tokmem_runtime)
        USE_UNIFIED_MODEL_ENV=false
        ;;
esac
""",
        "credential passthrough policy",
    )
    text = replace_once(
        text,
        """    claude|claude_sdk|claude_agent_sdk)
        HOST_LOOP_CMD=(
            uv run python -m scripts.decoupled.host_agent_loop_claude_sdk
            --bundle_file "$HOST_AGENT_BUNDLE_FILE"
            --gateway_url "http://127.0.0.1:${gateway_port}/sse"
            --gateway_server_name "gw"
            --model "$modelname"
            --tool_call_mode "$tool_call_mode"
            --debug
        )
        ;;
    *)
        echo "✗ Unsupported host loop backend: $host_loop_backend"
        echo "Supported values: openai, openai_agents, claude, claude_sdk, claude_agent_sdk"
""",
        """    claude|claude_sdk|claude_agent_sdk)
        HOST_LOOP_CMD=(
            uv run python -m scripts.decoupled.host_agent_loop_claude_sdk
            --bundle_file "$HOST_AGENT_BUNDLE_FILE"
            --gateway_url "http://127.0.0.1:${gateway_port}/sse"
            --gateway_server_name "gw"
            --model "$modelname"
            --tool_call_mode "$tool_call_mode"
            --debug
        )
        ;;
    tokmem_runtime)
        if [ -z "${TOKMEM_PROJECT_ROOT:-}" ]; then
            echo "✗ TOKMEM_PROJECT_ROOT is required for tokmem_runtime" >&2
            exit 1
        fi
        TOKMEM_EPISODE_ID_VALUE="${TOKMEM_EPISODE_ID:-${CONTAINER_NAME}-$$}"
        HOST_LOOP_CMD=(
            bash
            "$TOKMEM_PROJECT_ROOT/compositional_toolathlon/scripts/run_official_host_from_bundle.sh"
            "$HOST_AGENT_BUNDLE_FILE"
            "http://127.0.0.1:${gateway_port}/sse"
            "$TOKMEM_EPISODE_ID_VALUE"
        )
        ;;
    *)
        echo "✗ Unsupported host loop backend: $host_loop_backend"
        echo "Supported values: openai, openai_agents, claude, claude_sdk, claude_agent_sdk, tokmem_runtime"
""",
        "Step 5 backend",
    )
    return text


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + f".tokmem-{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)


def install(runner: Path, patch_output: Path) -> dict[str, str]:
    original_bytes = runner.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    marker = runner.parent / ".tokmem-runtime-hook.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if (
            metadata.get("revision") == REVISION
            and metadata.get("patched_sha256") == original_sha
        ):
            return metadata
        raise ValueError("runner hook marker exists but runner content drifted")
    if original_sha != ORIGINAL_SHA256:
        raise ValueError(
            f"official runner SHA-256 drifted: {original_sha}; "
            f"expected {ORIGINAL_SHA256}"
        )

    original = original_bytes.decode("utf-8")
    patched = patched_runner(original)
    patched_sha = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"Toolathlon-{REVISION}/scripts/run_single_decoupled.sh",
            tofile="Toolathlon-tokmem/scripts/run_single_decoupled.sh",
        )
    )
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    patch_output.write_text(diff, encoding="utf-8")
    atomic_write(runner, patched)
    metadata = {
        "revision": REVISION,
        "original_sha256": original_sha,
        "patched_sha256": patched_sha,
        "patch_file": str(patch_output.resolve()),
    }
    marker.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Install the fail-closed TokMem backend into the pinned runner"
    )
    parser.add_argument("--runner", required=True)
    parser.add_argument(
        "--patch-output",
        default=str(package / "artifacts" / "toolathlon_runner_hook.patch"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = install(
        Path(args.runner).resolve(),
        Path(args.patch_output).resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
