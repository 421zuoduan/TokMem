#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

exec python "$ROOT_DIR/compositional/utils/run_memory_bank_constraint_eval.py" "$@"
