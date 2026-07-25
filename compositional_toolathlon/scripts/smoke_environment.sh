#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
cd "$REPO_ROOT"
export UV_CACHE_DIR="$REPO_ROOT/compositional_toolathlon/.uv-cache"

python -m compositional_toolathlon.environment audit \
    --output compositional_toolathlon/artifacts/environment_audit.json
python -m unittest discover -s compositional_toolathlon/tests -v
