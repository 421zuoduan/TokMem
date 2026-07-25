#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$TRACK_DIR/.." && pwd)"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
cd "$REPO_DIR"

python -m pip install \
    --index-url https://pypi.org/simple \
    "uv==0.11.32"
python -m pip install \
    --index-url https://pypi.org/simple \
    -r compositional_toolathlon/requirements-tokmem-runtime.txt

bash compositional_toolathlon/scripts/bootstrap_benchmark.sh
if [[ -n "${TOOLATHLON_ROOT:-}" ]]; then
    BENCHMARK_DIR="$TOOLATHLON_ROOT"
else
    BENCHMARK_DIR="$TRACK_DIR/vendor/toolathlon"
fi

python -m compositional_toolathlon.scripts.install_runner_hook \
    --runner "$BENCHMARK_DIR/scripts/run_single_decoupled.sh"
bash compositional_toolathlon/scripts/setup_python_runtime.sh

bash compositional_toolathlon/scripts/pull_official_image.sh
python -m compositional_toolathlon.environment audit --strict
