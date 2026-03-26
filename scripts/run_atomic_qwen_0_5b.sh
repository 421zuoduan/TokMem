#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/atomic"

python main_in_domain.py \
    --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
    --num_tasks 1000 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --val_batch_size 16 \
    --test_batch_size 64 \
    --validate_every_n_steps 1000 \
    --seed 42
