#!/usr/bin/env bash
set -euo pipefail

cd /data/shilong/tokmem
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

MODEL_NAME="${MODEL_NAME:-models/Llama-3.2-1B-Instruct}"
SAMPLE_TYPES="${SAMPLE_TYPES:-node_chain}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LR="${LR:-5e-3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
LOGIT_BIAS_SCALE="${LOGIT_BIAS_SCALE:-1.0}"
LOGIT_BIAS_LOSS_WEIGHT="${LOGIT_BIAS_LOSS_WEIGHT:-0.1}"
SEED="${SEED:-42}"

python compositional_taskbench/main_taskbench.py \
  --method tapmem \
  --model_name "$MODEL_NAME" \
  --sample_types "$SAMPLE_TYPES" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --lr "$LR" \
  --max_length "$MAX_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --logit_bias_scale "$LOGIT_BIAS_SCALE" \
  --logit_bias_loss_weight "$LOGIT_BIAS_LOSS_WEIGHT" \
  --detach \
  --use_logit_train_add \
  --seed "$SEED" \
  --run_tag "${SAMPLE_TYPES}_tapmem"
