#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python -u "$ROOT_DIR/atomic/utils/filter_tasks_for_all_models.py" \
    --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
    --train_size 500 \
    --eval_size 10 \
    --test_size 50 \
    --num_tasks 1000 \
    --max_length 1024 \
    --seed 42 \
    --output_dir "$ROOT_DIR/atomic/cached_splits/paper_model_common_pool"
