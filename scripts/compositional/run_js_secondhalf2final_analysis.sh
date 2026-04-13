#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

cd "$ROOT_DIR"
python compositional/utils/js_secondhalf2final_explore.py "$@"
