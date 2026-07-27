#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 AGENT_BUNDLE_JSON GATEWAY_SSE_URL FRESH_ENVIRONMENT_ID" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${TOKMEM_PROJECT_ROOT:-$(cd "$TRACK_DIR/.." && pwd)}"
AGENT_PYTHON="${TOKMEM_AGENT_PYTHON:-}"
HOST_MODE="${TOKMEM_OFFICIAL_HOST_MODE:-tokmem}"
RUN_DIR="${TOKMEM_RUN_DIR:-}"
MANIFEST="${TOKMEM_TOOL_MANIFEST:-}"
BUNDLE_FILE="$1"
GATEWAY_URL="$2"
FRESH_ENVIRONMENT_ID="$3"

if [[ -z "$AGENT_PYTHON" || ! -x "$AGENT_PYTHON" ]]; then
    echo "TOKMEM_AGENT_PYTHON is not executable: $AGENT_PYTHON" >&2
    exit 2
fi
if [[ ! -f "$BUNDLE_FILE" ]]; then
    echo "Official trusted agent bundle is missing: $BUNDLE_FILE" >&2
    exit 2
fi
if [[ -z "$FRESH_ENVIRONMENT_ID" ]]; then
    echo "Fresh environment ID must be non-empty" >&2
    exit 2
fi
if [[ "$GATEWAY_URL" != http://127.0.0.1:*/sse && "$GATEWAY_URL" != http://localhost:*/sse ]]; then
    echo "Refusing a non-loopback or malformed gateway URL: $GATEWAY_URL" >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$HOST_MODE" == "codex_teacher" ]]; then
    PROMPT_VARIANT="${TOOLATHLON_CODEX_PROMPT_VARIANT:-action-only}"
    MODEL="${TOOLATHLON_CODEX_MODEL:-gpt-5.6-sol}"
    TRACE_PATH="${TOOLATHLON_CODEX_TRACE_PATH:-$TRACK_DIR/data/official_reference/provenance/codex_cli_invocations.jsonl}"
    MAX_TOOL_CALLS="${TOOLATHLON_CODEX_MAX_TOOL_CALLS:-50}"
    exec "$AGENT_PYTHON" \
        -m compositional_toolathlon.run_official_codex_teacher \
        --bundle-file "$BUNDLE_FILE" \
        --gateway-url "$GATEWAY_URL" \
        --fresh-environment-id "$FRESH_ENVIRONMENT_ID" \
        --model "$MODEL" \
        --prompt-variant "$PROMPT_VARIANT" \
        --max-tool-calls "$MAX_TOOL_CALLS" \
        --trace-path "$TRACE_PATH"
fi
if [[ "$HOST_MODE" != "tokmem" ]]; then
    echo "Unsupported TOKMEM_OFFICIAL_HOST_MODE: $HOST_MODE" >&2
    exit 2
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "TOKMEM_RUN_DIR must name an existing trained run directory" >&2
    exit 2
fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
    echo "TOKMEM_TOOL_MANIFEST must name the frozen manifest JSON" >&2
    exit 2
fi
exec "$AGENT_PYTHON" \
    -m compositional_toolathlon.run_official_agent \
    --bundle-file "$BUNDLE_FILE" \
    --manifest "$MANIFEST" \
    --run-dir "$RUN_DIR" \
    --gateway-url "$GATEWAY_URL" \
    --fresh-environment-id "$FRESH_ENVIRONMENT_ID"
