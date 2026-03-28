#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=700
SPLIT_NAME="all-models-task700-500-10-50-seed42"
SPLIT_DIR="$ROOT_DIR/atomic/cached_splits/$SPLIT_NAME"
SPLIT_CACHE="$SPLIT_DIR/tokmem_atomic_fixed_split_maxlen1024.pt"

if [ ! -f "$SPLIT_CACHE" ]; then
    bash "$ROOT_DIR/scripts/sample_atomic_all_models_fixed_split.sh" "$NUM_TASKS"
fi

cd "$ROOT_DIR/atomic"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="atomic_qwen2.5_0.5b_$SPLIT_NAME_$RUN_ID"
RUN_DIR="$ROOT_DIR/atomic/runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

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
    --num_tasks "$NUM_TASKS" \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --device_map balanced \
    --split_cache_path "$SPLIT_CACHE" \
    --run_root_dir "$ROOT_DIR/atomic/runs" \
    --run_name "$RUN_NAME" \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --lr 5e-3 \
    --generation_routing full_vocab_generation \
    --val_batch_size 16 \
    --test_batch_size 256 \
    --validate_every_n_steps 1000 \
    --seed 42
