#!/usr/bin/env bash
set -euo pipefail

cd /data/shilong/tokmem
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

MODEL_NAME="${MODEL_NAME:-models/Llama-3.1-8B-Instruct}"
SAMPLE_TYPES="${SAMPLE_TYPES:-node_chain}"
SOURCE_PATH="${SOURCE_PATH:-datasets/taskbench/data_dailylifeapis/data.json}"
TOOL_DESC_PATH="${TOOL_DESC_PATH:-datasets/taskbench/data_dailylifeapis/tool_desc.json}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
TRAIN_SIZE="${TRAIN_SIZE:-}"
TEST_SIZE="${TEST_SIZE:-}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRIAL_SEEDS="${TRIAL_SEEDS:-42,43,44}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
TOKMEM_LR_VALUES="${TOKMEM_LR_VALUES:-2e-3,3e-3,5e-3,8e-3,1e-2,1.5e-2,2e-2,3e-2}"
TAPMEM_LR_VALUES="${TAPMEM_LR_VALUES:-2e-3,3e-3,5e-3,8e-3,1e-2,1.5e-2,2e-2,3e-2}"
BATCH_VALUES="${BATCH_VALUES:-4,8}"
EVAL_BATCH_SIZE_FOR_4="${EVAL_BATCH_SIZE_FOR_4:-16}"
EVAL_BATCH_SIZE_FOR_8="${EVAL_BATCH_SIZE_FOR_8:-32}"
BASE_LOGIT_BIAS_SCALE="${BASE_LOGIT_BIAS_SCALE:-1.0}"
BASE_LOGIT_BIAS_LOSS_WEIGHT="${BASE_LOGIT_BIAS_LOSS_WEIGHT:-0.1}"
TAPMEM_FOCUS_LR_VALUES="${TAPMEM_FOCUS_LR_VALUES:-2e-3,5e-3,8e-3,1e-2}"
TAPMEM_FOCUS_BATCH_SIZE="${TAPMEM_FOCUS_BATCH_SIZE:-8}"
TAPMEM_SCALE_VALUES="${TAPMEM_SCALE_VALUES:-0.5,0.8,1.2,1.5}"
TAPMEM_LOSS_WEIGHT_VALUES="${TAPMEM_LOSS_WEIGHT_VALUES:-0.05,0.2,0.3}"
FOCUS_TRIAL_SEEDS="${FOCUS_TRIAL_SEEDS:-42,43}"
SUITE_NAME="${SUITE_NAME:-taskbench_dailylife_${SAMPLE_TYPES}_8b_wide_sweep_split${SPLIT_SEED}_$(date +%Y%m%d_%H%M%S)}"
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
  --seed "$SPLIT_SEED"
)
if [ -n "$TRAIN_SIZE" ]; then
  PREPARE_DATA_ARGS+=(--train_size "$TRAIN_SIZE")
fi
if [ -n "$TEST_SIZE" ]; then
  PREPARE_DATA_ARGS+=(--test_size "$TEST_SIZE")
fi

echo "[$(date -Is)] Preparing fixed split under $DATA_DIR" | tee -a "$SUITE_LOG"
python compositional_taskbench/prepare_data.py "${PREPARE_DATA_ARGS[@]}" >> "$SUITE_LOG" 2>&1
TRAIN_PATH="$DATA_DIR/training/taskbench_dailylife_${SAMPLE_TYPES}_train.json"
TEST_PATH="$DATA_DIR/test/taskbench_dailylife_${SAMPLE_TYPES}_test.json"

normalize_label() {
  echo "$1" | sed 's/+//g; s/-/m/g; s/\./p/g'
}

eval_batch_for_batch() {
  case "$1" in
    4) echo "$EVAL_BATCH_SIZE_FOR_4" ;;
    8) echo "$EVAL_BATCH_SIZE_FOR_8" ;;
    *) echo "$EVAL_BATCH_SIZE_FOR_8" ;;
  esac
}

add_task() {
  local method="$1"
  local lr="$2"
  local batch_size="$3"
  local eval_batch_size="$4"
  local scale="$5"
  local loss_weight="$6"
  local trial="$7"
  local seed="$8"
  task_id=$((task_id + 1))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task_id" "$method" "$lr" "$batch_size" "$eval_batch_size" "$scale" "$loss_weight" "$trial" "$seed" \
    >> "$TASKS_FILE"
}

IFS=',' read -r -a TOKMEM_LR_ARRAY <<< "$TOKMEM_LR_VALUES"
IFS=',' read -r -a TAPMEM_LR_ARRAY <<< "$TAPMEM_LR_VALUES"
IFS=',' read -r -a BATCH_ARRAY <<< "$BATCH_VALUES"
IFS=',' read -r -a TRIAL_SEED_ARRAY <<< "$TRIAL_SEEDS"
IFS=',' read -r -a FOCUS_LR_ARRAY <<< "$TAPMEM_FOCUS_LR_VALUES"
IFS=',' read -r -a SCALE_ARRAY <<< "$TAPMEM_SCALE_VALUES"
IFS=',' read -r -a LOSS_WEIGHT_ARRAY <<< "$TAPMEM_LOSS_WEIGHT_VALUES"
IFS=',' read -r -a FOCUS_SEED_ARRAY <<< "$FOCUS_TRIAL_SEEDS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"

printf "task_id\tmethod\tlr\tbatch_size\teval_batch_size\tlogit_bias_scale\tlogit_bias_loss_weight\ttrial\tseed\n" > "$TASKS_FILE"
task_id=0

for batch_size in "${BATCH_ARRAY[@]}"; do
  batch_size="$(echo "$batch_size" | xargs)"
  eval_batch_size="$(eval_batch_for_batch "$batch_size")"
  for lr in "${TOKMEM_LR_ARRAY[@]}"; do
    lr="$(echo "$lr" | xargs)"
    trial=0
    for seed in "${TRIAL_SEED_ARRAY[@]}"; do
      seed="$(echo "$seed" | xargs)"
      trial=$((trial + 1))
      add_task tokmem "$lr" "$batch_size" "$eval_batch_size" NA NA "$trial" "$seed"
    done
  done

  for lr in "${TAPMEM_LR_ARRAY[@]}"; do
    lr="$(echo "$lr" | xargs)"
    trial=0
    for seed in "${TRIAL_SEED_ARRAY[@]}"; do
      seed="$(echo "$seed" | xargs)"
      trial=$((trial + 1))
      add_task tapmem "$lr" "$batch_size" "$eval_batch_size" "$BASE_LOGIT_BIAS_SCALE" "$BASE_LOGIT_BIAS_LOSS_WEIGHT" "$trial" "$seed"
    done
  done
done

for lr in "${FOCUS_LR_ARRAY[@]}"; do
  lr="$(echo "$lr" | xargs)"
  eval_batch_size="$(eval_batch_for_batch "$TAPMEM_FOCUS_BATCH_SIZE")"
  for scale in "${SCALE_ARRAY[@]}"; do
    scale="$(echo "$scale" | xargs)"
    trial=0
    for seed in "${FOCUS_SEED_ARRAY[@]}"; do
      seed="$(echo "$seed" | xargs)"
      trial=$((trial + 1))
      add_task tapmem "$lr" "$TAPMEM_FOCUS_BATCH_SIZE" "$eval_batch_size" "$scale" "$BASE_LOGIT_BIAS_LOSS_WEIGHT" "$trial" "$seed"
    done
  done
  for loss_weight in "${LOSS_WEIGHT_ARRAY[@]}"; do
    loss_weight="$(echo "$loss_weight" | xargs)"
    trial=0
    for seed in "${FOCUS_SEED_ARRAY[@]}"; do
      seed="$(echo "$seed" | xargs)"
      trial=$((trial + 1))
      add_task tapmem "$lr" "$TAPMEM_FOCUS_BATCH_SIZE" "$eval_batch_size" "$BASE_LOGIT_BIAS_SCALE" "$loss_weight" "$trial" "$seed"
    done
  done
done

tail -n +2 "$TASKS_FILE" > "$QUEUE_FILE"
printf "task_id\tmethod\tlr\tbatch_size\teval_batch_size\tlogit_bias_scale\tlogit_bias_loss_weight\ttrial\tseed\tgpu\tstatus\texit_code\trun_name\trun_dir\tlog_file\n" > "$MANIFEST"

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
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$manifest"
  ' _ "$MANIFEST" "$@"
}

run_task() {
  local task_id="$1"
  local method="$2"
  local lr="$3"
  local batch_size="$4"
  local eval_batch_size="$5"
  local scale="$6"
  local loss_weight="$7"
  local trial="$8"
  local seed="$9"
  local gpu="${10}"
  local lr_label scale_label loss_label
  local run_name run_dir log_file status_file status exit_code

  lr_label="$(normalize_label "$lr")"
  scale_label="$(normalize_label "$scale")"
  loss_label="$(normalize_label "$loss_weight")"
  run_name="taskbench_${method}_${SAMPLE_TYPES}_8b_lr${lr_label}_bs${batch_size}_scale${scale_label}_loss${loss_label}_seed${seed}_trial${trial}"
  run_dir="$SUITE_DIR/runs/$run_name"
  log_file="$SUITE_DIR/logs/${run_name}.log"
  status_file="$STATUS_DIR/${task_id}.status"

  echo "[$(date -Is)] START task=${task_id}/${task_id_total} gpu=${gpu} method=${method} lr=${lr} batch=${batch_size} scale=${scale} loss=${loss_weight} seed=${seed}" | tee -a "$SUITE_LOG"

  set +e
  if [ "$method" = "tokmem" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python compositional_taskbench/main_taskbench.py \
      --method tokmem \
      --model_name "$MODEL_NAME" \
      --epochs "$EPOCHS" \
      --batch_size "$batch_size" \
      --eval_batch_size "$eval_batch_size" \
      --lr "$lr" \
      --max_length "$MAX_LENGTH" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --source_path "$SOURCE_PATH" \
      --tool_desc_path "$TOOL_DESC_PATH" \
      --sample_types "$SAMPLE_TYPES" \
      --train_path "$TRAIN_PATH" \
      --test_path "$TEST_PATH" \
      --seed "$seed" \
      --run_name "$run_name" \
      --run_root_dir "$SUITE_DIR/runs" \
      > "$log_file" 2>&1
  elif [ "$method" = "tapmem" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python compositional_taskbench/main_taskbench.py \
      --method tapmem \
      --model_name "$MODEL_NAME" \
      --epochs "$EPOCHS" \
      --batch_size "$batch_size" \
      --eval_batch_size "$eval_batch_size" \
      --lr "$lr" \
      --max_length "$MAX_LENGTH" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --logit_bias_scale "$scale" \
      --logit_bias_loss_weight "$loss_weight" \
      --detach \
      --use_logit_train_add \
      --source_path "$SOURCE_PATH" \
      --tool_desc_path "$TOOL_DESC_PATH" \
      --sample_types "$SAMPLE_TYPES" \
      --train_path "$TRAIN_PATH" \
      --test_path "$TEST_PATH" \
      --seed "$seed" \
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
  append_manifest "$task_id" "$method" "$lr" "$batch_size" "$eval_batch_size" "$scale" "$loss_weight" "$trial" "$seed" "$gpu" "$status" "$exit_code" "$run_name" "$run_dir" "$log_file"
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
    IFS=$'\t' read -r task_id method lr batch_size eval_batch_size scale loss_weight trial seed < "$claimed_file"
    run_task "$task_id" "$method" "$lr" "$batch_size" "$eval_batch_size" "$scale" "$loss_weight" "$trial" "$seed" "$gpu"
  done
}

task_id_total="$task_id"
echo "[$(date -Is)] Suite directory: $SUITE_DIR" | tee -a "$SUITE_LOG"
echo "[$(date -Is)] Tasks: $task_id_total, GPUs: $GPUS, split seed: $SPLIT_SEED, trial seeds: $TRIAL_SEEDS" | tee -a "$SUITE_LOG"
echo "[$(date -Is)] Data: train=$TRAIN_PATH test=$TEST_PATH" | tee -a "$SUITE_LOG"

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

python scripts/compositional_taskbench/summarize_taskbench_8b_wide_sweep.py "$SUITE_DIR" | tee -a "$SUITE_LOG"

if grep -q $'\tfailed\t' "$MANIFEST"; then
  overall_exit=1
fi
exit "$overall_exit"
