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
TRIALS="${TRIALS:-3}"
SUITE_NAME="${SUITE_NAME:-taskbench_dailylife_${SAMPLE_TYPES}_seed${SEED}_3trials_$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-/data/shilong/tokmem/compositional_taskbench/runs/${SUITE_NAME}}"
DATA_DIR="${DATA_DIR:-$SUITE_DIR/data}"

mkdir -p "$SUITE_DIR" "$DATA_DIR"
MANIFEST="$SUITE_DIR/manifest.tsv"
SUMMARY_LOG="$SUITE_DIR/suite.log"
printf "method\ttrial\tstatus\texit_code\trun_name\trun_dir\n" > "$MANIFEST"

echo "[$(date -Is)] Preparing suite data under $DATA_DIR" | tee -a "$SUMMARY_LOG"
python compositional_taskbench/prepare_data.py \
  --sample_types "$SAMPLE_TYPES" \
  --output_dir "$DATA_DIR" \
  --seed "$SEED" \
  >> "$SUMMARY_LOG" 2>&1

run_one() {
  local method="$1"
  local trial="$2"
  local run_name="taskbench_${method}_${SAMPLE_TYPES}_seed${SEED}_trial${trial}"
  local run_dir="$SUITE_DIR/runs/$run_name"
  local status="success"
  local exit_code=0

  echo "[$(date -Is)] START method=${method} trial=${trial} run=${run_name}" | tee -a "$SUMMARY_LOG"

  set +e
  if [ "$method" = "tokmem" ]; then
    python compositional_taskbench/main_taskbench.py \
      --method tokmem \
      --model_name "$MODEL_NAME" \
      --sample_types "$SAMPLE_TYPES" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --lr "$LR" \
      --max_length "$MAX_LENGTH" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --seed "$SEED" \
      --data_dir "$DATA_DIR" \
      --run_name "$run_name" \
      --run_root_dir "$SUITE_DIR/runs" \
      > "$SUITE_DIR/${run_name}.log" 2>&1
  else
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
      --data_dir "$DATA_DIR" \
      --run_name "$run_name" \
      --run_root_dir "$SUITE_DIR/runs" \
      > "$SUITE_DIR/${run_name}.log" 2>&1
  fi
  exit_code=$?
  set -e

  if [ "$exit_code" -ne 0 ]; then
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$method" "$trial" "$status" "$exit_code" "$run_name" "$run_dir" >> "$MANIFEST"
  echo "[$(date -Is)] END method=${method} trial=${trial} status=${status} exit_code=${exit_code}" | tee -a "$SUMMARY_LOG"
  return "$exit_code"
}

overall_exit=0
for method in tokmem tapmem; do
  for trial in $(seq 1 "$TRIALS"); do
    if ! run_one "$method" "$trial"; then
      overall_exit=1
    fi
  done
done

python scripts/compositional_taskbench/summarize_taskbench_trials.py "$SUITE_DIR" | tee -a "$SUMMARY_LOG"
exit "$overall_exit"
