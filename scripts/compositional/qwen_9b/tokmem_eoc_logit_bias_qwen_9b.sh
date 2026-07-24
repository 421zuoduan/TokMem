#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="tokmem_eoc_logit_bias_qwen_9b_50tools_${RUN_ID}"
RUN_DIR="$ROOT_DIR/compositional/runs/$RUN_NAME"

MODEL_PATH="${TOKMEM_MODEL_PATH:-$ROOT_DIR/models/Qwen3.5-9B}"
DATA_DIR="${TOKMEM_DATA_DIR:-$ROOT_DIR/compositional/data}"
GPU="${TOKMEM_GPU:-0}"
BATCH_SIZE="${TOKMEM_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${TOKMEM_EVAL_BATCH_SIZE:-4}"
MAX_LENGTH="${TOKMEM_MAX_LENGTH:-512}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Qwen3.5 model config not found: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$ROOT_DIR/compositional"

python xlam_datasets.py \
    --top_k "51-100" \
    --max_samples_per_tool 50 \
    --train_size 5000 \
    --test_size 500 \
    --train_max_function_calls 4 \
    --test_max_function_calls 4 \
    --train_multi_tool_ratios "0.5,0.5" \
    --test_multi_tool_ratios "0.5,0.5" \
    --output_dir "$DATA_DIR"

python -u main_sequential.py \
    --training_rounds "51-100:1" \
    --epochs 3 \
    --batch_size "$BATCH_SIZE" \
    --train_max_function_calls 4 \
    --test_max_function_calls 4 \
    --model_name "$MODEL_PATH" \
    --eval_after_each_round \
    --save_checkpoints \
    --data_dir "$DATA_DIR" \
    --lr 5e-3 \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --seed 42 \
    --tensorboard \
    --use_eoc \
    --use_logit_bias \
    --detach \
    --use_logit_train_add \
    --run_root_dir "$ROOT_DIR/compositional/runs" \
    --run_name "$RUN_NAME" \
    --run_tag "qwen_9b_eoc_logit_bias"
