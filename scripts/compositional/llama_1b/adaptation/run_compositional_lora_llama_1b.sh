#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="compositional_lora_llama_1b_100tools_${RUN_ID}"
RUN_DIR="$ROOT_DIR/compositional/runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES=0

cd "$ROOT_DIR/compositional"

python xlam_datasets.py \
    --top_k "1-50" \
    --max_samples_per_tool 50 \
    --train_size 5000 \
    --test_size 500 \
    --train_max_function_calls 4 \
    --test_max_function_calls 4 \
    --train_multi_tool_ratios "0.5,0.5" \
    --test_multi_tool_ratios "0.5,0.5" \
    --output_dir "$ROOT_DIR/compositional/data"

python xlam_datasets.py \
    --top_k "51-100" \
    --max_samples_per_tool 50 \
    --train_size 5000 \
    --test_size 500 \
    --train_max_function_calls 4 \
    --test_max_function_calls 4 \
    --train_multi_tool_ratios "0.5,0.5" \
    --test_multi_tool_ratios "0.5,0.5" \
    --output_dir "$ROOT_DIR/compositional/data"

python -u lora_sequential.py \
    --training_rounds "1-50:1,51-100:1" \
    --batch_size 2 \
    --model_name "$ROOT_DIR/models/Llama-3.2-1B-Instruct" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj" \
    --eval_after_each_round \
    --save_checkpoints \
    --data_dir "$ROOT_DIR/compositional/data" \
    --lr 5e-5 \
    --eval_batch_size 16 \
    --seed 42 \
    --train_max_function_calls_per_round "4,4" \
    --test_max_function_calls_per_round "4,4" \
    --reinit_lora_after_each_round \
    --run_root_dir "$ROOT_DIR/compositional/runs" \
    --run_name "$RUN_NAME" \
    --run_tag "llama_1b"
