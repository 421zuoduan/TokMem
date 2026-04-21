#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=0

python main_in_domain.py \
    --num_tasks 50 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "${REPO_ROOT}/models/Qwen2.5-0.5B-Instruct" \
    --split_cache_path "${SCRIPT_DIR}/cached_splits/task50-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt" \
    --use_logit_bias \
    --logit_bias_loss_weight 1.0 \
    --logit_bias_network linear \
    --logit_bias_scale 1.0 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --max_instruction_tokens 1024 \
    --val_batch_size 32 \
    --test_batch_size 32 \
    --validate_every_n_steps 1000
