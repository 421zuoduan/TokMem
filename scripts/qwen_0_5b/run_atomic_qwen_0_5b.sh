#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=1000
RUN_TAG="runtime-split-task1000-500-10-50-seed42"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="atomic_qwen2.5_0.5b_$RUN_TAG_$RUN_ID"
RUN_DIR="$ROOT_DIR/atomic/runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

cd "$ROOT_DIR/atomic"

python main_in_domain.py \
    --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
    --num_tasks "$NUM_TASKS" \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --run_root_dir "$ROOT_DIR/atomic/runs" \
    --run_name "$RUN_NAME" \
    --num_epochs 1 \
    --batch_size 8 \
    --shuffle_train \
    --gradient_accumulation_steps 1 \
    --max_length 1280 \
    --generation_routing full_vocab_generation \
    --use_task_loss False \
    --task_loss_weight 0.0 \
    --val_batch_size 16 \
    --test_batch_size 64 \
    --validate_every_n_steps 1000 \
    --seed 42
