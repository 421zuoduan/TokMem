#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python "$ROOT_DIR/compositional/utils/analyze_generation_eoc_boundaries.py" "$@"
