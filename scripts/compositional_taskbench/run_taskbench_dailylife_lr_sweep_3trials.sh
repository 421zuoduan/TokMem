#!/usr/bin/env bash
set -euo pipefail

cd /data/shilong/tokmem
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

MODEL_NAME="${MODEL_NAME:-models/Llama-3.2-1B-Instruct}"
SAMPLE_TYPES="${SAMPLE_TYPES:-node_chain}"
SOURCE_PATH="${SOURCE_PATH:-datasets/taskbench/data_dailylifeapis/data.json}"
TOOL_DESC_PATH="${TOOL_DESC_PATH:-datasets/taskbench/data_dailylifeapis/tool_desc.json}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
TRAIN_SIZE="${TRAIN_SIZE:-}"
TEST_SIZE="${TEST_SIZE:-}"
LR_VALUES="${LR_VALUES:-8e-3,1e-3,5e-4,1e-4,5e-5,1e-5}"
METHODS="${METHODS:-tokmem,tapmem}"
TRIALS="${TRIALS:-3}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
LOGIT_BIAS_SCALE="${LOGIT_BIAS_SCALE:-1.0}"
LOGIT_BIAS_LOSS_WEIGHT="${LOGIT_BIAS_LOSS_WEIGHT:-0.1}"
SUITE_NAME="${SUITE_NAME:-taskbench_dailylife_${SAMPLE_TYPES}_lr_sweep_seed${SEED}_3trials_$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-/data/shilong/tokmem/compositional_taskbench/runs/${SUITE_NAME}}"
DATA_DIR="${DATA_DIR:-$SUITE_DIR/data}"

mkdir -p "$SUITE_DIR/logs" "$SUITE_DIR/runs" "$DATA_DIR"
TASKS_FILE="$SUITE_DIR/tasks.tsv"
QUEUE_FILE="$SUITE_DIR/queue.tsv"
MANIFEST="$SUITE_DIR/manifest.tsv"
STATUS_DIR="$SUITE_DIR/status"
LOCK_FILE="$SUITE_DIR/queue.lock"
SUITE_LOG="$SUITE_DIR/suite.log"
mkdir -p "$STATUS_DIR"

PREPARE_DATA_ARGS=(
  --source_path "$SOURCE_PATH"
  --sample_types "$SAMPLE_TYPES"
  --output_dir "$DATA_DIR"
  --train_ratio "$TRAIN_RATIO"
  --seed "$SEED"
)
MAIN_DATA_ARGS=(
  --source_path "$SOURCE_PATH"
  --tool_desc_path "$TOOL_DESC_PATH"
  --sample_types "$SAMPLE_TYPES"
  --train_ratio "$TRAIN_RATIO"
  --seed "$SEED"
  --data_dir "$DATA_DIR"
)
if [ -n "$TRAIN_SIZE" ]; then
  PREPARE_DATA_ARGS+=(--train_size "$TRAIN_SIZE")
  MAIN_DATA_ARGS+=(--train_size "$TRAIN_SIZE")
fi
if [ -n "$TEST_SIZE" ]; then
  PREPARE_DATA_ARGS+=(--test_size "$TEST_SIZE")
  MAIN_DATA_ARGS+=(--test_size "$TEST_SIZE")
fi

echo "[$(date -Is)] Preparing suite data under $DATA_DIR" | tee -a "$SUITE_LOG"
python compositional_taskbench/prepare_data.py "${PREPARE_DATA_ARGS[@]}" >> "$SUITE_LOG" 2>&1

normalize_lr_label() {
  echo "$1" | sed 's/+//g; s/-/m/g; s/\./p/g'
}

IFS=',' read -r -a LR_ARRAY <<< "$LR_VALUES"
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"

printf "task_id\tmethod\tlr\ttrial\n" > "$TASKS_FILE"
task_id=0
for lr in "${LR_ARRAY[@]}"; do
  lr="$(echo "$lr" | xargs)"
  for method in "${METHOD_ARRAY[@]}"; do
    method="$(echo "$method" | xargs)"
    for trial in $(seq 1 "$TRIALS"); do
      task_id=$((task_id + 1))
      printf "%s\t%s\t%s\t%s\n" "$task_id" "$method" "$lr" "$trial" >> "$TASKS_FILE"
    done
  done
done
tail -n +2 "$TASKS_FILE" > "$QUEUE_FILE"
printf "task_id\tmethod\tlr\ttrial\tgpu\tstatus\texit_code\trun_name\trun_dir\tlog_file\n" > "$MANIFEST"

claim_task() {
  local claimed_file="$1"
  flock "$LOCK_FILE" bash -c '
    queue_file="$1"
    claimed_file="$2"
    if [ ! -s "$queue_file" ]; then
      : > "$claimed_file"
      exit 0
    fi
    head -n 1 "$queue_file" > "$claimed_file"
    tail -n +2 "$queue_file" > "${queue_file}.tmp"
    mv "${queue_file}.tmp" "$queue_file"
  ' _ "$QUEUE_FILE" "$claimed_file"
}

append_manifest() {
  flock "$LOCK_FILE" bash -c '
    manifest="$1"
    shift
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$manifest"
  ' _ "$MANIFEST" "$@"
}

run_task() {
  local task_id="$1"
  local method="$2"
  local lr="$3"
  local trial="$4"
  local gpu="$5"
  local lr_label
  local run_name
  local run_dir
  local log_file
  local status_file
  local status
  local exit_code

  lr_label="$(normalize_lr_label "$lr")"
  run_name="taskbench_${method}_${SAMPLE_TYPES}_lr${lr_label}_seed${SEED}_trial${trial}"
  run_dir="$SUITE_DIR/runs/$run_name"
  log_file="$SUITE_DIR/logs/${run_name}.log"
  status_file="$STATUS_DIR/${task_id}.status"

  echo "[$(date -Is)] START task=${task_id} gpu=${gpu} method=${method} lr=${lr} trial=${trial}" | tee -a "$SUITE_LOG"

  set +e
  if [ "$method" = "tokmem" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python compositional_taskbench/main_taskbench.py \
      --method tokmem \
      --model_name "$MODEL_NAME" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --lr "$lr" \
      --max_length "$MAX_LENGTH" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      "${MAIN_DATA_ARGS[@]}" \
      --run_name "$run_name" \
      --run_root_dir "$SUITE_DIR/runs" \
      > "$log_file" 2>&1
  elif [ "$method" = "tapmem" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python compositional_taskbench/main_taskbench.py \
      --method tapmem \
      --model_name "$MODEL_NAME" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --lr "$lr" \
      --max_length "$MAX_LENGTH" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --logit_bias_scale "$LOGIT_BIAS_SCALE" \
      --logit_bias_loss_weight "$LOGIT_BIAS_LOSS_WEIGHT" \
      --detach \
      --use_logit_train_add \
      "${MAIN_DATA_ARGS[@]}" \
      --run_name "$run_name" \
      --run_root_dir "$SUITE_DIR/runs" \
      > "$log_file" 2>&1
  else
    echo "Unsupported method: $method" > "$log_file"
    false
  fi
  exit_code=$?
  set -e

  if [ "$exit_code" -eq 0 ]; then
    status="success"
  else
    status="failed"
  fi
  append_manifest "$task_id" "$method" "$lr" "$trial" "$gpu" "$status" "$exit_code" "$run_name" "$run_dir" "$log_file"
  printf "%s\n" "$status" > "$status_file"
  echo "[$(date -Is)] END task=${task_id} gpu=${gpu} status=${status} exit_code=${exit_code}" | tee -a "$SUITE_LOG"
}

worker() {
  local gpu="$1"
  local claimed_file="$SUITE_DIR/claimed_gpu${gpu}.tsv"
  while true; do
    claim_task "$claimed_file"
    if [ ! -s "$claimed_file" ]; then
      break
    fi
    IFS=$'\t' read -r task_id method lr trial < "$claimed_file"
    run_task "$task_id" "$method" "$lr" "$trial" "$gpu"
  done
}

echo "[$(date -Is)] Suite directory: $SUITE_DIR" | tee -a "$SUITE_LOG"
echo "[$(date -Is)] Tasks: $task_id, GPUs: $GPUS, LR_VALUES: $LR_VALUES, METHODS: $METHODS" | tee -a "$SUITE_LOG"

worker_pids=()
for gpu in "${GPU_ARRAY[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  worker "$gpu" &
  worker_pids+=("$!")
done

overall_exit=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    overall_exit=1
  fi
done

python scripts/compositional_taskbench/summarize_taskbench_lr_sweep.py "$SUITE_DIR" | tee -a "$SUITE_LOG"

if grep -q $'\tfailed\t' "$MANIFEST"; then
  overall_exit=1
fi
exit "$overall_exit"
