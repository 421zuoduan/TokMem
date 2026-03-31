#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POOL_DIR="$ROOT_DIR/atomic/cached_splits/llama-3.2-3b-pool-500-10-50-seed42"
SPLIT_DIR="$ROOT_DIR/atomic/cached_splits/llama-3.2-3b-task100-500-10-50-seed42"

if [ ! -f "$POOL_DIR/task_pool_manifest.json" ]; then
    bash "$ROOT_DIR/scripts/llama_3b/build_atomic_llama_3b_task_pool.sh"
fi

cd "$ROOT_DIR"

python -u "$ROOT_DIR/atomic/utils/sample_tasks_from_task_pool.py" \
    --pool_dir "$POOL_DIR" \
    --model_name "$ROOT_DIR/models/Llama-3.2-3B-Instruct" \
    --num_tasks 100 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --max_length 1280 \
    --seed 42 \
    --output_dir "$SPLIT_DIR"
