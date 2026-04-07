#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=4,5,6
export TOKMEM_COMPOSITIONAL_MODEL="$ROOT_DIR/models/Llama-3.2-1B-Instruct"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

echo "Running compositional TokMem with current repository defaults."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Model: $TOKMEM_COMPOSITIONAL_MODEL"
echo "Entrypoint: $ROOT_DIR/compositional/run_n_rounds_main.sh"

cd "$ROOT_DIR/compositional"
bash run_n_rounds_main.sh
