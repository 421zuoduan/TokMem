#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=700
SPLIT_NAME="task${NUM_TASKS}-500-10-50-seed42"
SPLIT_CACHE="${REPO_ROOT}/atomic/cached_splits/${SPLIT_NAME}/tokmem_atomic_fixed_split_maxlen1024.pt"
RESULTS_MD="${REPO_ROOT}/results/atomic_mean_results.md"
TMP_METRICS="$(mktemp)"
CUDA_DEVICE="0"
MONITOR_PID=""

trap 'rm -f "${TMP_METRICS}"; if [ -n "${MONITOR_PID}" ]; then kill "${MONITOR_PID}" 2>/dev/null || true; fi' EXIT

if [ ! -f "${SPLIT_CACHE}" ]; then
    echo "Missing split cache: ${SPLIT_CACHE}" >&2
    exit 1
fi

cd "${REPO_ROOT}/atomic"

printf 'method\trun_name\trun_dir\ttask_accuracy\trouge_l\n' > "${TMP_METRICS}"

for METHOD in tokmem_baseline logit_bias; do
    for REPEAT in 1 2 3; do
        RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
        RUN_NAME="atomic_qwen2.5_0.5b_${NUM_TASKS}tasks_${METHOD}_${RUN_ID}_repeat${REPEAT}"
        RUN_DIR="${REPO_ROOT}/atomic/runs/${RUN_NAME}"

        mkdir -p "${RUN_DIR}"
        cp "${SCRIPT_PATH}" "${RUN_DIR}/$(basename "${SCRIPT_PATH}")"

        export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        while true; do
            {
                echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
                nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i "${CUDA_DEVICE}"
            } >> "${RUN_DIR}/gpu_monitor.log"
            sleep 10
        done &
        MONITOR_PID=$!

        CMD=(
            accelerate launch
            --num_processes 1
            --num_machines 1
            --mixed_precision bf16
            --dynamo_backend no
            main_in_domain.py
            --tasks_dir "${REPO_ROOT}/datasets/natural-instructions-2.8/tasks"
            --num_tasks "${NUM_TASKS}"
            --train_size 500
            --val_size 10
            --test_size 50
            --model_name "${REPO_ROOT}/models/Qwen2.5-0.5B-Instruct"
            --split_cache_path "${SPLIT_CACHE}"
            --run_dir "${RUN_DIR}"
            --num_epochs 1
            --batch_size 4
            --gradient_accumulation_steps 1
            --max_length 1024
            --max_instruction_tokens 1024
            --lr 5e-4
            --val_batch_size 16
            --test_batch_size 64
            --validate_every_n_steps 1000
            --num_workers 4
            --pin_memory
            --seed 42
        )

        if [ "${METHOD}" = "logit_bias" ]; then
            CMD+=(
                --use_logit_bias
                --logit_bias_loss_weight 0.1
                --logit_bias_network linear
                --logit_bias_scale 1.0
            )
        fi

        echo "Running ${METHOD} repeat ${REPEAT}: ${RUN_NAME}"
        set +e
        "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/stdout.log"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        kill "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""

        printf '%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"
        if [ "${EXIT_CODE}" -ne 0 ]; then
            echo "Run failed: ${RUN_NAME}" >&2
            exit "${EXIT_CODE}"
        fi

        METRIC_LINE="$(python - "${RUN_DIR}" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
evaluation_path = run_dir / "evaluation_results.json"
if evaluation_path.exists():
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    print(f"{payload['task_accuracy']}\t{payload['ni_rouge_l']}")
    raise SystemExit(0)

stdout_path = run_dir / "stdout.log"
text = stdout_path.read_text(encoding="utf-8", errors="replace")
task_matches = re.findall(r"Task Prediction Accuracy:\s*([0-9.]+)", text)
score_matches = re.findall(r"Average Response Score:\s*([0-9.]+)", text)
if not task_matches or not score_matches:
    raise SystemExit(f"Missing metrics in {stdout_path}")

task_accuracy = float(task_matches[-1])
rouge_l = float(score_matches[-1]) * 100.0
print(f"{task_accuracy}\t{rouge_l}")
PY
)"

        TASK_ACCURACY="$(printf '%s\n' "${METRIC_LINE}" | cut -f1)"
        ROUGE_L="$(printf '%s\n' "${METRIC_LINE}" | cut -f2)"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${METHOD}" \
            "${RUN_NAME}" \
            "${RUN_DIR}" \
            "${TASK_ACCURACY}" \
            "${ROUGE_L}" >> "${TMP_METRICS}"
    done
done

python - "${TMP_METRICS}" "${RESULTS_MD}" <<'PY'
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime

metrics_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])

rows = []
with metrics_path.open(encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        row["task_accuracy"] = float(row["task_accuracy"])
        row["rouge_l"] = float(row["rouge_l"])
        rows.append(row)

by_method = defaultdict(list)
for row in rows:
    by_method[row["method"]].append(row)

results_path.parent.mkdir(parents=True, exist_ok=True)
if not results_path.exists():
    results_path.write_text(
        "# Atomic Mean Results\n\n"
        "记录 `atomic` 里 baseline 与 `logit_bias` 的多次重跑均值。\n\n",
        encoding="utf-8",
    )

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    f"## {timestamp} `700-task / Qwen2.5-0.5B / seed=42 / 3 repeats`",
    "",
    "共同底座：",
    "",
    "- split cache：`task700-500-10-50-seed42`",
    "- `train/val/test per task = 500/10/50`",
    "- `batch_size = 4`",
    "- `gradient_accumulation_steps = 1`",
    "- `max_length = 1024`",
    "- `lr = 5e-4`",
    "- `val_batch_size = 16`",
    "- `test_batch_size = 64`",
    "- `validate_every_n_steps = 1000`",
    "- `accelerate launch --num_processes 1 --mixed_precision bf16`",
    "- `seed = 42`",
    "",
    "| Run | 类型 | 关键改动 | 重复次数 | I+Q Task Acc | I+Q ROUGE-L | 简评 |",
    "|---|---|---|---:|---:|---:|---|",
]

for method in ("tokmem_baseline", "logit_bias"):
    method_rows = by_method[method]
    run_names = "<br>".join(f"`{row['run_name']}`" for row in method_rows)
    task_acc_mean = statistics.mean(row["task_accuracy"] for row in method_rows) * 100.0
    rouge_l_mean = statistics.mean(row["rouge_l"] for row in method_rows)
    if method == "tokmem_baseline":
        method_label = "baseline"
        change_label = "无 `use_logit_bias`"
    else:
        method_label = "`logit_bias`"
        change_label = "`use_logit_bias=True`, `loss_weight=0.1`, `network=linear`, `scale=1.0`"
    lines.append(
        f"| {run_names} | {method_label} | {change_label} | 3 | {task_acc_mean:.2f}% | {rouge_l_mean:.4f}% | `seed=42` 固定重跑三次均值。 |"
    )

lines.extend(
    [
        "",
        "### 单次结果",
        "",
        "| Method | Run | I+Q Task Acc | I+Q ROUGE-L | Run Dir |",
        "|---|---|---:|---:|---|",
    ]
)

for method in ("tokmem_baseline", "logit_bias"):
    label = "baseline" if method == "tokmem_baseline" else "logit_bias"
    for row in by_method[method]:
        lines.append(
            f"| {label} | `{row['run_name']}` | {row['task_accuracy'] * 100.0:.2f}% | {row['rouge_l']:.4f}% | `{row['run_dir']}` |"
        )

with results_path.open("a", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\n\n")
PY

echo "Mean results appended to ${RESULTS_MD}"
