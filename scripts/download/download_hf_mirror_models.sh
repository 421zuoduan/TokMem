#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f /home/shilong/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/shilong/anaconda3/etc/profile.d/conda.sh
elif [ -f /data/ruochen/anaconda/etc/profile.d/conda.sh ]; then
    source /data/ruochen/anaconda/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1; then
    conda activate tokmem
fi

export HF_ENDPOINT="https://hf-mirror.com"

cd "$ROOT_DIR"
python -u utils/download_hf_mirror_models.py "$@"
