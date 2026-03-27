#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python -u "$ROOT_DIR/atomic/utils/filter_tasks_for_all_models.py" \
    --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
    --tokenizer "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --tokenizer "$ROOT_DIR/models/Llama-3.2-3B-Instruct" \
    --train_size 500 \
    --eval_size 10 \
    --test_size 50 \
    --num_tasks 700 \
    --max_length 1024 \
    --seed 42 \
    --output_dir "$ROOT_DIR/atomic/cached_splits/qwen2.5_0.5b_random700_from763_train500_val10_test50_seed42"
