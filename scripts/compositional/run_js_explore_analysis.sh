#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/ruochen/tokmem"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

cd "$REPO_ROOT"
python compositional/utils/js_explore.py "$@"
