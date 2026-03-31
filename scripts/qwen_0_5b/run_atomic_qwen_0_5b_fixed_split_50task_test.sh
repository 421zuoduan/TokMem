#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=50
SPLIT_NAME="task$NUM_TASKS-500-10-50-seed42"
SPLIT_DIR="$ROOT_DIR/atomic/cached_splits/$SPLIT_NAME"
SPLIT_CACHE="$SPLIT_DIR/tokmem_atomic_fixed_split_maxlen1024.pt"

if [ ! -f "$SPLIT_CACHE" ]; then
    bash "$ROOT_DIR/scripts/all_models/sample_atomic_all_models_fixed_split.sh" "$NUM_TASKS"
fi

cd "$ROOT_DIR/atomic"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="atomic_qwen2.5_0.5b_${NUM_TASKS}tasks_$RUN_ID"
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
    --shuffle_train \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --lr 5e-4 \
    --generation_routing full_vocab_generation \
    --use_task_loss False \
    --task_loss_weight 0.0 \
    --val_batch_size 16 \
    --test_batch_size 400 \
    --validate_every_n_steps 500 \
    --seed 42
