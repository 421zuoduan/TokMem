#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  collect_llm_episode.sh TASK_JSON MANIFEST GENERATION_CONFIG TEACHER_SESSION \
    WORKSPACE RUNTIME_DIR PORT OUTPUT

Collect one teacher episode with one host gateway, workspace, and model client.
The workspace must be empty. The script does not retry collection or clean paths.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "$#" -ne 8 ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRACK_DIR/.." && pwd)"
PYTHON="$TRACK_DIR/.toolathlon-venv/bin/python"
UVX_COMMAND="${TOOLATHLON_UVX_COMMAND:-/home/shilong/anaconda3/envs/tokmem/bin/uvx}"

TASK_JSON="$1"
MANIFEST="$2"
GENERATION_CONFIG="$3"
TEACHER_SESSION="$4"
WORKSPACE="$5"
RUNTIME_DIR="$6"
PORT="$7"
OUTPUT="$8"
GATEWAY_URL="http://127.0.0.1:${PORT}/sse"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
FRESH_ENVIRONMENT_ID="$(basename "$RUNTIME_DIR")"

mkdir -p "$WORKSPACE" "$RUNTIME_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

setsid "$PYTHON" -m compositional_toolathlon.host_gateway \
    --workspace "$WORKSPACE" \
    --port "$PORT" \
    --uvx-command "$UVX_COMMAND" \
    --output-dir "$RUNTIME_DIR" \
    >"$RUNTIME_DIR/host_gateway.log" 2>&1 &
gateway_pid=$!

stop_gateway() {
    kill -KILL -- "-$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
}
trap stop_gateway EXIT

gateway_ready=false
for _ in {1..300}; do
    if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null 2>&1; then
        gateway_ready=true
        break
    fi
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
        echo "host gateway exited; see $RUNTIME_DIR/host_gateway.log" >&2
        exit 1
    fi
    sleep 0.2
done
if [[ "$gateway_ready" != true ]]; then
    echo "host gateway did not become healthy; see $RUNTIME_DIR/host_gateway.log" >&2
    exit 1
fi

"$PYTHON" -m compositional_toolathlon.collect_teacher_episode \
    --task-spec "$TASK_JSON" \
    --manifest "$MANIFEST" \
    --workspace-root "$WORKSPACE" \
    --gateway-url "$GATEWAY_URL" \
    --candidate-index 0 \
    --fresh-environment-id "$FRESH_ENVIRONMENT_ID" \
    --generation-config "$GENERATION_CONFIG" \
    --teacher-session-id "$TEACHER_SESSION" \
    --output "$OUTPUT"
