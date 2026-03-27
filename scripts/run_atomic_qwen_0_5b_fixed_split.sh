#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/atomic"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/atomic/runs/atomic_qwen2.5_0.5b_fixed_split_700tasks_${RUN_ID}"

mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/$(basename "$0")"

export CUDA_VISIBLE_DEVICES=4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

while true; do
    {
        echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i 4,5,6
    } >> "$RUN_DIR/gpu_monitor.log"
    sleep 10
done &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

python -u main_in_domain_fixed_split.py \
    --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
    --num_tasks 700 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --device_map balanced \
    --split_cache_path "$ROOT_DIR/atomic/cached_splits/qwen2.5_0.5b_random700_from763_train500_val10_test50/tokmem_atomic_fixed_split_maxlen1024.pt" \
    --run_root_dir "$ROOT_DIR/atomic/runs" \
    --run_name "$(basename "$RUN_DIR")" \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --lr 0.005 \
    --val_batch_size 16 \
    --test_batch_size 256 \
    --validate_every_n_steps 1000 \
    --seed 42
