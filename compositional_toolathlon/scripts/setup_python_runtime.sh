#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$TRACK_DIR/.." && pwd)"
EXPECTED_REVISION="2aed2468858f15818acafa178518390cc4b0f5cb"
PYTHON_VERSION="3.12.11"
PYTHON_INSTALL_DIR="$TRACK_DIR/.python"
VENV_DIR="$TRACK_DIR/.toolathlon-venv"
UV_CACHE_DIR="${UV_CACHE_DIR:-$TRACK_DIR/.uv-cache}"

if [[ -n "${TOOLATHLON_ROOT:-}" ]]; then
    BENCHMARK_DIR="$TOOLATHLON_ROOT"
elif [[ -d "$TRACK_DIR/vendor/toolathlon/.git" || \
        -f "$TRACK_DIR/vendor/toolathlon/.toolathlon-source.json" ]]; then
    BENCHMARK_DIR="$TRACK_DIR/vendor/toolathlon"
else
    BENCHMARK_DIR="$REPO_DIR/datasets/toolathlon"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it in the tokmem conda environment first." >&2
    exit 2
fi
if [[ ! -f "$BENCHMARK_DIR/pyproject.toml" || ! -f "$BENCHMARK_DIR/uv.lock" ]]; then
    echo "Toolathlon pyproject.toml or uv.lock is missing: $BENCHMARK_DIR" >&2
    exit 2
fi
if [[ -d "$BENCHMARK_DIR/.git" ]]; then
    ACTUAL_REVISION="$(git -C "$BENCHMARK_DIR" rev-parse HEAD)"
elif [[ -f "$BENCHMARK_DIR/.toolathlon-source.json" ]]; then
    ACTUAL_REVISION="$(
        python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' \
            "$BENCHMARK_DIR/.toolathlon-source.json"
    )"
else
    echo "Toolathlon source provenance is missing: $BENCHMARK_DIR" >&2
    exit 2
fi
if [[ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]]; then
    echo "Unexpected Toolathlon revision: $ACTUAL_REVISION" >&2
    exit 2
fi

uv python install "$PYTHON_VERSION" \
    --install-dir "$PYTHON_INSTALL_DIR" \
    --no-bin \
    --cache-dir "$UV_CACHE_DIR"
PYTHON_BIN="$PYTHON_INSTALL_DIR/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "uv did not create the expected Python executable: $PYTHON_BIN" >&2
    exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    uv venv "$VENV_DIR" --python "$PYTHON_BIN"
fi
RUNTIME_VERSION="$("$VENV_DIR/bin/python" -c 'import platform; print(platform.python_version())')"
if [[ "$RUNTIME_VERSION" != "$PYTHON_VERSION" ]]; then
    echo "Existing runtime has Python $RUNTIME_VERSION, expected $PYTHON_VERSION" >&2
    exit 2
fi

UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync \
    --project "$BENCHMARK_DIR" \
    --frozen \
    --no-install-project \
    --no-binary-package xlsxwriter \
    --python "$PYTHON_BIN" \
    --cache-dir "$UV_CACHE_DIR"

uv pip install \
    --python "$VENV_DIR/bin/python" \
    --cache-dir "$UV_CACHE_DIR" \
    "jsonschema==4.25.1"

"$VENV_DIR/bin/python" -c \
    'import agents, jsonschema, mcp, openai; print("Toolathlon Python runtime imports: OK")'
