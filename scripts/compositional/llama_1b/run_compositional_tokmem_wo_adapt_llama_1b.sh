#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="tokmem_wo_adapt_llama_1b_50tools_${RUN_ID}"
RUN_DIR="$ROOT_DIR/compositional/runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

while true; do
    {
        echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
    } >> "$RUN_DIR/gpu_monitor.log"
    sleep 10
done &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

python -u main_sequential.py \
    --training_rounds "51-100:1" \
    --epochs 3 \
    --batch_size 4 \
    --train_max_function_calls 4 \
    --test_max_function_calls 4 \
    --model_name "$ROOT_DIR/models/Llama-3.2-1B-Instruct" \
    --eval_after_each_round \
    --save_checkpoints \
    --data_dir "$ROOT_DIR/compositional/data" \
    --lr 5e-3 \
    --eval_batch_size 16 \
    --max_length 1024 \
    --seed 42 \
    --tensorboard \
    --renorm_active_tools \
    --run_root_dir "$ROOT_DIR/compositional/runs" \
    --run_name "$RUN_NAME" \
    --run_tag "llama_1b"
