#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=700
SPLIT_NAME="task${NUM_TASKS}-500-10-50-seed42"
SPLIT_CACHE="${REPO_ROOT}/atomic/cached_splits/${SPLIT_NAME}/tokmem_atomic_fixed_split_maxlen1024.pt"

if [ ! -f "${SPLIT_CACHE}" ]; then
    echo "Missing split cache: ${SPLIT_CACHE}" >&2
    exit 1
fi

cd "${REPO_ROOT}/atomic"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="atomic_qwen2.5_0.5b_${NUM_TASKS}tasks_tokmem_baseline_${RUN_ID}"
RUN_DIR="${REPO_ROOT}/atomic/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
cp "${SCRIPT_PATH}" "${RUN_DIR}/$(basename "${SCRIPT_PATH}")"

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MONITOR_GPU_IDS="${CUDA_VISIBLE_DEVICES}"

while true; do
    {
        echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i "${MONITOR_GPU_IDS}"
    } >> "${RUN_DIR}/gpu_monitor.log"
    sleep 10
done &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

accelerate launch \
    --num_processes 3 \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    main_in_domain.py \
    --tasks_dir "${REPO_ROOT}/datasets/natural-instructions-2.8/tasks" \
    --num_tasks "${NUM_TASKS}" \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "${REPO_ROOT}/models/Qwen2.5-0.5B-Instruct" \
    --split_cache_path "${SPLIT_CACHE}" \
    --use_fsdp \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --max_instruction_tokens 1024 \
    --lr 5e-4 \
    --val_batch_size 8 \
    --test_batch_size 64 \
    --validate_every_n_steps 200 \
    --num_workers 4 \
    --pin_memory \
    --seed 42 \
    2>&1 | tee "${RUN_DIR}/stdout.log"

EXIT_CODE=${PIPESTATUS[0]}
printf '%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"
exit "${EXIT_CODE}"
