#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=50
SPLIT_NAME="task50-500-10-50-seed42"
SPLIT_DIR="$ROOT_DIR/atomic/cached_splits/$SPLIT_NAME"
SPLIT_CACHE="$SPLIT_DIR/tokmem_atomic_fixed_split_maxlen1024.pt"
WEIGHTS=(0.5 0.2 0.1 0.05 0.03 0.01 0.005 0.001)
TAU="0.3"

if [ ! -f "$SPLIT_CACHE" ]; then
    bash "$ROOT_DIR/scripts/all_models/sample_atomic_all_models_fixed_split.sh" "$NUM_TASKS"
fi

SWEEP_ID="$(date -u +%Y%m%d_%H%M%S)"
SWEEP_NAME="atomic_qwen2.5_0.5b_${NUM_TASKS}tasks_sep_loss_tau03_sweep_$SWEEP_ID"
SWEEP_DIR="$ROOT_DIR/atomic/runs/$SWEEP_NAME"
MANIFEST_PATH="$SWEEP_DIR/run_manifest.tsv"
SUMMARY_PATH="$SWEEP_DIR/summary.md"

mkdir -p "$SWEEP_DIR"
cp "$SCRIPT_PATH" "$SWEEP_DIR/$(basename "$SCRIPT_PATH")"

printf "weight\trun_name\trun_dir\texit_code\n" > "$MANIFEST_PATH"

export CUDA_VISIBLE_DEVICES=4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

cd "$ROOT_DIR/atomic"

for WEIGHT in "${WEIGHTS[@]}"; do
    WEIGHT_TAG="${WEIGHT/./p}"
    RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
    RUN_NAME="atomic_qwen2.5_0.5b_${NUM_TASKS}tasks_sep_loss_tau03_w${WEIGHT_TAG}_$RUN_ID"
    RUN_DIR="$ROOT_DIR/atomic/runs/$RUN_NAME"

    mkdir -p "$RUN_DIR"
    cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

    echo "============================================================"
    echo "Starting run: $RUN_NAME"
    echo "sep_loss_weight = $WEIGHT"
    echo "sep_loss_tau = $TAU"
    echo "============================================================"

    while true; do
        {
            echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
            nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i 4,5,6
        } >> "$RUN_DIR/gpu_monitor.log"
        sleep 10
    done &
    MONITOR_PID=$!

    python -u main_in_domain_fixed_split.py \
        --tasks_dir "$ROOT_DIR/datasets/natural-instructions-2.8/tasks" \
        --num_tasks "$NUM_TASKS" \
        --train_size 500 \
        --val_size 10 \
        --test_size 50 \
        --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
        --device_map balanced \
        --split_cache_path "$SPLIT_CACHE" \
        --run_root_dir "$ROOT_DIR/atomic/runs" \
        --run_name "$RUN_NAME" \
        --num_epochs 1 \
        --batch_size 8 \
        --shuffle_train \
        --gradient_accumulation_steps 1 \
        --max_length 1024 \
        --lr 5e-4 \
        --generation_routing full_vocab_generation \
        --use_task_loss False \
        --task_loss_weight 0.0 \
        --use_mean_loss False \
        --mean_loss_weight 0.0 \
        --use_angular_margin_loss True \
        --angular_margin_loss_weight 0.01 \
        --use_hard_negative_loss True \
        --hard_negative_loss_weight 0.01 \
        --hard_negative_margin 0.2 \
        --use_sep_loss True \
        --sep_loss_weight "$WEIGHT" \
        --sep_loss_tau "$TAU" \
        --val_batch_size 16 \
        --test_batch_size 400 \
        --validate_every_n_steps 500 \
        --seed 42
    EXIT_CODE=$?

    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true

    printf "%s\t%s\t%s\t%s\n" "$WEIGHT" "$RUN_NAME" "$RUN_DIR" "$EXIT_CODE" >> "$MANIFEST_PATH"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Run failed: $RUN_NAME (exit code $EXIT_CODE)"
    else
        echo "Run finished: $RUN_NAME"
    fi
done

python - "$MANIFEST_PATH" "$SUMMARY_PATH" <<'PY'
import csv
import json
import math
import os
import sys

manifest_path = sys.argv[1]
summary_path = sys.argv[2]

rows = []
with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

summaries = []
for row in rows:
    weight = row["weight"]
    run_name = row["run_name"]
    run_dir = row["run_dir"]
    exit_code = int(row["exit_code"])
    entry = {
        "weight": weight,
        "run_name": run_name,
        "run_dir": run_dir,
        "exit_code": exit_code,
        "status": "failed" if exit_code != 0 else "completed",
        "best_val_loss": None,
        "iq_task_acc": None,
        "iq_rouge_l": None,
        "q_task_acc": None,
        "q_rouge_l": None,
    }

    train_results_path = os.path.join(run_dir, "train_results.json")
    eval_results_path = os.path.join(run_dir, "evaluation_results.json")

    if exit_code == 0 and os.path.exists(train_results_path) and os.path.exists(eval_results_path):
        with open(train_results_path, "r", encoding="utf-8") as f:
            train_results = json.load(f)
        with open(eval_results_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

        iq = eval_results["instruction_and_query"]
        q = eval_results["query_only"]
        entry["best_val_loss"] = float(train_results["best_val_loss"])
        entry["iq_task_acc"] = float(iq["task_accuracy"])
        entry["iq_rouge_l"] = float(iq["ni_rouge_l"])
        entry["q_task_acc"] = float(q["task_accuracy"])
        entry["q_rouge_l"] = float(q["ni_rouge_l"])
    else:
        entry["status"] = "failed"

    summaries.append(entry)

completed = [x for x in summaries if x["status"] == "completed" and x["iq_task_acc"] is not None]

def rank_key(item):
    return (
        item["iq_task_acc"],
        item["iq_rouge_l"],
        -item["best_val_loss"],
        item["q_task_acc"],
        item["q_rouge_l"],
    )

best = max(completed, key=rank_key) if completed else None

lines = []
lines.append("# 50-task sep_loss tau=0.3 sweep summary")
lines.append("")
lines.append("排序规则：先看 `instruction_and_query` 的 `Task Acc`，再看 `instruction_and_query` 的 `ROUGE-L`，然后看更低的 `best val loss`，最后才看 `query_only`。")
lines.append("")
lines.append("| weight | status | Best val loss | I+Q Task Acc | I+Q ROUGE-L | Query-only Task Acc | Query-only ROUGE-L | run_name |")
lines.append("|---|---|---:|---:|---:|---:|---:|---|")

for item in summaries:
    def fmt(value, digits=4):
        if value is None:
            return "NA"
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "NA"
        return f"{value:.{digits}f}"

    lines.append(
        "| {weight} | {status} | {best_val_loss} | {iq_task_acc} | {iq_rouge_l} | {q_task_acc} | {q_rouge_l} | {run_name} |".format(
            weight=item["weight"],
            status=item["status"],
            best_val_loss=fmt(item["best_val_loss"]),
            iq_task_acc=fmt(item["iq_task_acc"]),
            iq_rouge_l=fmt(item["iq_rouge_l"]),
            q_task_acc=fmt(item["q_task_acc"]),
            q_rouge_l=fmt(item["q_rouge_l"]),
            run_name=item["run_name"],
        )
    )

lines.append("")
if best is None:
    lines.append("没有成功完成的 run，无法给出 best weight。")
else:
    lines.append("## Best Weight")
    lines.append("")
    lines.append(f"- best weight: `{best['weight']}`")
    lines.append(f"- run: `{best['run_name']}`")
    lines.append(f"- `Best val loss = {best['best_val_loss']:.4f}`")
    lines.append(f"- `I+Q Task Acc = {best['iq_task_acc']:.4f}`")
    lines.append(f"- `I+Q ROUGE-L = {best['iq_rouge_l']:.4f}`")
    lines.append(f"- `Query-only Task Acc = {best['q_task_acc']:.4f}`")
    lines.append(f"- `Query-only ROUGE-L = {best['q_rouge_l']:.4f}`")

text = "\n".join(lines) + "\n"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(text)

print(text, end="")
PY

echo "Sweep manifest: $MANIFEST_PATH"
echo "Sweep summary: $SUMMARY_PATH"
