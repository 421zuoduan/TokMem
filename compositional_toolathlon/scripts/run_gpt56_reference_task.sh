#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "Usage: $0 TASK_NAME OUTPUT_ROOT PROMPT_VARIANT [GATEWAY_PORT]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRACK_DIR/.." && pwd)"
TASK_NAME="$1"
OUTPUT_ROOT="$2"
PROMPT_VARIANT="$3"
GATEWAY_PORT="${4:-}"
IMAGE="${TOOLATHLON_REFERENCE_IMAGE:-127.0.0.1:5055/lockon0927/toolathlon-task-image:1016beta-singleuid}"
DEFAULT_RUNTIME_PARENT="${XDG_RUNTIME_DIR:-/tmp}"
RUNTIME_ROOT="${TOOLATHLON_ROOTLESS_RUNTIME_ROOT:-$DEFAULT_RUNTIME_PARENT/compositional-toolathlon-rootless}"
ROOTLESSKIT_API_SOCKET="$RUNTIME_ROOT/state/api.sock"

case "$TASK_NAME" in
    arrange-workspace|courses-ta-hws|detect-revised-terms|excel-data-transformation|excel-market-research|imagenet|paper-checker|privacy-desensitization|reimbursement-form-filler|university-course-selection)
        echo "Refusing protected ten-task evaluation item: $TASK_NAME" >&2
        exit 2
        ;;
esac
case "$PROMPT_VARIANT" in
    action-only|state-first|contract-first) ;;
    *)
        echo "Unknown prompt variant: $PROMPT_VARIANT" >&2
        exit 2
        ;;
esac

export TOKMEM_PROJECT_ROOT="$PROJECT_ROOT"
export TOKMEM_AGENT_PYTHON="${TOOLATHLON_CODEX_AGENT_PYTHON:-/home/shilong/anaconda3/envs/tokmem/bin/python}"
export TOKMEM_OFFICIAL_HOST_MODE="codex_teacher"
export TOOLATHLON_CODEX_MODEL="${TOOLATHLON_CODEX_MODEL:-gpt-5.6-sol}"
export TOOLATHLON_CODEX_PROMPT_VARIANT="$PROMPT_VARIANT"
export TOOLATHLON_CODEX_TRACE_PATH="${TOOLATHLON_CODEX_TRACE_PATH:-$TRACK_DIR/data/official_reference/provenance/codex_cli_invocations.jsonl}"
export PATH="$SCRIPT_DIR/rootless-docker-cli:/home/shilong/.local/bin:/home/shilong/anaconda3/envs/tokmem/bin:$PATH"

if [[ -z "$GATEWAY_PORT" ]]; then
    GATEWAY_PORT="$("$TOKMEM_AGENT_PYTHON" -c \
        'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi
PORT_RESPONSE="$(
    curl --silent --show-error --fail-with-body \
        --unix-socket "$ROOTLESSKIT_API_SOCKET" \
        -H "Content-Type: application/json" \
        -d "{\"proto\":\"tcp\",\"parentIP\":\"127.0.0.1\",\"parentPort\":$GATEWAY_PORT,\"childIP\":\"127.0.0.1\",\"childPort\":$GATEWAY_PORT}" \
        http://localhost/v1/ports
)"
PORT_MAPPING_ID="$(
    printf '%s' "$PORT_RESPONSE" |
        "$TOKMEM_AGENT_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["id"])'
)"
cleanup_port_mapping() {
    curl --silent --show-error \
        --unix-socket "$ROOTLESSKIT_API_SOCKET" \
        -X DELETE \
        "http://localhost/v1/ports/$PORT_MAPPING_ID" >/dev/null || true
}
trap cleanup_port_mapping EXIT

runner_args=(
    "finalpool/$TASK_NAME"
    quickstart
    "$OUTPUT_ROOT"
    gpt-5.6-sol
    unified
    50
    scripts/formal_run_v0.json
    "$IMAGE"
    tokmem_runtime
    "$GATEWAY_PORT"
)

bash "$SCRIPT_DIR/run_pinned_official_task.sh" "${runner_args[@]}"
