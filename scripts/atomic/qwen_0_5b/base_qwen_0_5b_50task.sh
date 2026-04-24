#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=50
SPLIT_NAME="task${NUM_TASKS}-500-10-50-seed42"
SPLIT_CACHE="${REPO_ROOT}/atomic/cached_splits/${SPLIT_NAME}/tokmem_atomic_fixed_split_maxlen1024.pt"

if [ ! -f "${SPLIT_CACHE}" ]; then
    echo "Missing split cache: ${SPLIT_CACHE}" >&2
    exit 1
fi

cd "${REPO_ROOT}/atomic"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="base_qwen_0_5b_${NUM_TASKS}task_${RUN_ID}"
RUN_DIR="${REPO_ROOT}/atomic/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
cp "${SCRIPT_PATH}" "${RUN_DIR}/$(basename "${SCRIPT_PATH}")"

export CUDA_VISIBLE_DEVICES=4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

while true; do
    {
        echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i 4,5,6
    } >> "${RUN_DIR}/gpu_monitor.log"
    sleep 10
done &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

python -u main_base_model.py \
    --tasks_dir "${REPO_ROOT}/datasets/natural-instructions-2.8/tasks" \
    --num_tasks "${NUM_TASKS}" \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "${REPO_ROOT}/models/Qwen2.5-0.5B-Instruct" \
    --device_map balanced \
    --split_cache_path "${SPLIT_CACHE}" \
    --max_length 1024 \
    --max_instruction_tokens 1024 \
    --max_new_tokens 256 \
    --test_batch_size 400 \
    --seed 42 \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/stdout.log"

EXIT_CODE=${PIPESTATUS[0]}
printf '%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"
exit "${EXIT_CODE}"
