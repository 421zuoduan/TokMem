#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="compositional_icl_llama_1b_51_100_${RUN_ID}"
RUN_DIR="$ROOT_DIR/compositional/runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES=0

cd "$ROOT_DIR/compositional"

python xlam_datasets.py \
    --top_k "51-100" \
    --max_samples_per_tool 50 \
    --train_size 5000 \
    --test_size 500 \
    --train_max_function_calls 4 \
    --test_max_function_calls 4 \
    --train_multi_tool_ratios "0.5,0.5" \
    --test_multi_tool_ratios "0.5,0.5" \
    --output_dir "$ROOT_DIR/compositional/data"

python -u icl_baseline.py \
    --test_data "$ROOT_DIR/compositional/data/test/function_calling_test_tools51-100_4calls.json" \
    --tool_descriptions "$ROOT_DIR/compositional/data/tool_descriptions_tools51-100.json" \
    --model_name "$ROOT_DIR/models/Llama-3.2-1B-Instruct" \
    --batch_size 16 \
    --use_rag \
    --retrieval_k 5 \
    --run_root_dir "$ROOT_DIR/compositional/runs" \
    --run_name "$RUN_NAME" \
    --run_tag "llama_1b"
