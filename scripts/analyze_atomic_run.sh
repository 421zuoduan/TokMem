#!/usr/bin/env bash

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$REPO_ROOT/atomic/analyze_baseline_failures.py" --run-dir "$1" "${@:2}"
