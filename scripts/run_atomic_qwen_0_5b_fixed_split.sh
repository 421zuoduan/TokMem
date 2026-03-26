#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/atomic"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
GPU_LOG="$ROOT_DIR/atomic/logs/gpu_monitor_${RUN_ID}.log"

mkdir -p "$ROOT_DIR/atomic/logs"

export CUDA_VISIBLE_DEVICES=4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

monitor_gpu() {
    while true; do
        {
            echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
            nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i 4,5,6
        } >> "$GPU_LOG"
        sleep 10
    done
}

monitor_gpu &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

python -u main_in_domain_fixed_split.py \
    --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
    --num_tasks 1000 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --device_map balanced \
    --split_cache_path "$ROOT_DIR/atomic/cached_splits/tokmem_atomic_fixed_split.pt" \
    --ignore_model_name_in_split_cache \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --val_batch_size 16 \
    --test_batch_size 64 \
    --validate_every_n_steps 1000 \
    --seed 42
