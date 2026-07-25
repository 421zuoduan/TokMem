#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
    echo "Usage: $0 TASK_DIR RUNMODE DUMP_PATH MODEL_NAME [official runner arguments ...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRACK_DIR/.." && pwd)"
BENCHMARK_DIR="${TOOLATHLON_ROOT:-$TRACK_DIR/vendor/toolathlon}"
OFFICIAL_ENV="${TOOLATHLON_PYTHON_ENV:-$TRACK_DIR/.toolathlon-venv}"
RUNNER="$BENCHMARK_DIR/scripts/run_single_decoupled.sh"
HOOK_MARKER="$BENCHMARK_DIR/scripts/.tokmem-runtime-hook.json"

if [[ ! -x "$OFFICIAL_ENV/bin/python" ]]; then
    echo "Pinned Toolathlon Python environment is missing: $OFFICIAL_ENV" >&2
    exit 2
fi
if [[ ! -f "$RUNNER" || ! -f "$HOOK_MARKER" ]]; then
    echo "Pinned runner or TokMem hook marker is missing under: $BENCHMARK_DIR" >&2
    exit 2
fi

export TOKMEM_PROJECT_ROOT="${TOKMEM_PROJECT_ROOT:-$PROJECT_ROOT}"
export UV_PROJECT_ENVIRONMENT="$OFFICIAL_ENV"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TRACK_DIR/.uv-cache}"
export UV_FROZEN=1
export UV_NO_SYNC=1
exec bash "$RUNNER" "$@"
