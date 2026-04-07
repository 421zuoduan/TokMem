#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POOL_DIR="$ROOT_DIR/atomic/cached_splits/qwen2.5-0.5b-pool-500-10-50-seed42"
SPLIT_DIR="$ROOT_DIR/atomic/cached_splits/qwen2.5-0.5b-task700-500-10-50-seed42"

if [ ! -f "$POOL_DIR/task_pool_manifest.json" ]; then
    bash "$ROOT_DIR/scripts/qwen_0_5b/build_atomic_qwen_0_5b_task_pool.sh"
fi

cd "$ROOT_DIR"

python -u "$ROOT_DIR/atomic/utils/sample_tasks_from_task_pool.py" \
    --pool_dir "$POOL_DIR" \
    --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --num_tasks 700 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --max_length 1024 \
    --seed 42 \
    --output_dir "$SPLIT_DIR"
