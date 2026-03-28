#!/usr/bin/env bash

NUM_TASKS="${1:-700}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOL_DIR="$ROOT_DIR/atomic/cached_splits/all-models-pool-500-10-50-seed42"
SPLIT_DIR="$ROOT_DIR/atomic/cached_splits/all-models-task${NUM_TASKS}-500-10-50-seed42"

if [ ! -f "$POOL_DIR/task_pool_manifest.json" ]; then
    bash "$ROOT_DIR/scripts/build_atomic_all_models_task_pool.sh"
fi

cd "$ROOT_DIR"

python -u "$ROOT_DIR/atomic/utils/sample_tasks_from_task_pool.py" \
    --pool_dir "$POOL_DIR" \
    --model_name "all-models" \
    --num_tasks "$NUM_TASKS" \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --max_length 1024 \
    --seed 42 \
    --output_dir "$SPLIT_DIR"
