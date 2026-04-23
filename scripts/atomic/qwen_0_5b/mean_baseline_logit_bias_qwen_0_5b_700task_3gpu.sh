#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
NUM_TASKS=700
SPLIT_NAME="task${NUM_TASKS}-500-10-50-seed42"
SPLIT_CACHE="${REPO_ROOT}/atomic/cached_splits/${SPLIT_NAME}/tokmem_atomic_fixed_split_maxlen1024.pt"
RESULTS_PREFIX="atomic_mean_results_3gpu_345_bs8"
RESULTS_MD="${REPO_ROOT}/results/${RESULTS_PREFIX}.md"
LAUNCHER_LOG="${REPO_ROOT}/results/${RESULTS_PREFIX}_$(date -u +%Y%m%d_%H%M%S).log"
TMP_ROOT="$(mktemp -d)"
TRAIN_GPU_IDS="3,4,5"
MONITOR_PIDS=()

cleanup() {
    for pid in "${MONITOR_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    rm -rf "${TMP_ROOT}"
}

trap cleanup EXIT

mkdir -p "${REPO_ROOT}/results"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

if [ ! -f "${SPLIT_CACHE}" ]; then
    echo "Missing split cache: ${SPLIT_CACHE}" >&2
    exit 1
fi

echo "Launcher log: ${LAUNCHER_LOG}"
echo "Mean results file: ${RESULTS_MD}"
echo "Training GPUs: ${TRAIN_GPU_IDS}"

cd "${REPO_ROOT}/atomic"

run_single_job() {
    local method="$1"
    local repeat="$2"
    local metrics_file="$3"
    local run_id
    local run_name
    local run_dir
    local monitor_pid=""
    local exit_code
    local metric_line
    local task_accuracy
    local rouge_l
    local cmd=()

    run_id="$(date -u +%Y%m%d_%H%M%S)"
    run_name="atomic_qwen2.5_0.5b_${NUM_TASKS}tasks_${method}_gpus345_bs8_${run_id}_repeat${repeat}"
    run_dir="${REPO_ROOT}/atomic/runs/${run_name}"

    mkdir -p "${run_dir}"
    cp "${SCRIPT_PATH}" "${run_dir}/$(basename "${SCRIPT_PATH}")"

    export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    while true; do
        {
            echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
            nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i "${TRAIN_GPU_IDS}"
        } >> "${run_dir}/gpu_monitor.log"
        sleep 10
    done &
    monitor_pid=$!
    MONITOR_PIDS+=("${monitor_pid}")

    cmd=(
        accelerate launch
        --num_processes 3
        --num_machines 1
        --multi_gpu
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
        --run_dir "${run_dir}"
        --use_fsdp
        --num_epochs 1
        --batch_size 4
        --gradient_accumulation_steps 1
        --max_length 1024
        --max_instruction_tokens 1024
        --lr 5e-4
        --val_batch_size 8
        --test_batch_size 64
        --validate_every_n_steps 1000
        --num_workers 4
        --pin_memory
        --seed 42
    )

    if [ "${method}" = "logit_bias" ]; then
        cmd+=(
            --use_logit_bias
            --logit_bias_loss_weight 0.1
            --logit_bias_network linear
            --logit_bias_scale 1.0
        )
    fi

    echo "Running ${method} repeat ${repeat} on GPUs ${TRAIN_GPU_IDS}: ${run_name}"
    set +e
    "${cmd[@]}" 2>&1 | tee "${run_dir}/stdout.log"
    exit_code=${PIPESTATUS[0]}
    set -e

    kill "${monitor_pid}" 2>/dev/null || true

    printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"
    if [ "${exit_code}" -ne 0 ]; then
        echo "Run failed: ${run_name}" >&2
        return "${exit_code}"
    fi

    metric_line="$(python - "${run_dir}" <<'PY'
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

    task_accuracy="$(printf '%s\n' "${metric_line}" | cut -f1)"
    rouge_l="$(printf '%s\n' "${metric_line}" | cut -f2)"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${method}" \
        "${run_name}" \
        "${run_dir}" \
        "${task_accuracy}" \
        "${rouge_l}" > "${metrics_file}"
}

for method in tokmem_baseline logit_bias; do
    for repeat in 1 2 3; do
        run_single_job "${method}" "${repeat}" "${TMP_ROOT}/${method}_repeat${repeat}.tsv"
    done
done

python - "${TMP_ROOT}" "${RESULTS_MD}" <<'PY'
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime

metrics_root = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])

rows = []
for metrics_path in sorted(metrics_root.glob("*.tsv")):
    with metrics_path.open(encoding="utf-8") as f:
        row = next(csv.reader(f, delimiter="\t"))
        rows.append(
            {
                "method": row[0],
                "run_name": row[1],
                "run_dir": row[2],
                "task_accuracy": float(row[3]),
                "rouge_l": float(row[4]),
            }
        )

by_method = defaultdict(list)
for row in rows:
    by_method[row["method"]].append(row)

results_path.parent.mkdir(parents=True, exist_ok=True)
if not results_path.exists():
    results_path.write_text(
        "# Atomic Mean Results\n\n"
        "记录 `atomic` 里在 `GPU 3,4,5 / batch_size=4 / 三卡联合训练` 设置下的 baseline 与 `logit_bias` 多次重跑均值。\n\n",
        encoding="utf-8",
    )

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    f"## {timestamp} `700-task / Qwen2.5-0.5B / seed=42 / GPUs 3,4,5 / batch_size=4 / 3 repeats`",
    "",
    "共同底座：",
    "",
    "- split cache：`task700-500-10-50-seed42`",
    "- `train/val/test per task = 500/10/50`",
    "- `batch_size = 4`",
    "- `gradient_accumulation_steps = 1`",
    "- `max_length = 1024`",
    "- `lr = 5e-4`",
    "- `val_batch_size = 8`",
    "- `test_batch_size = 64`",
    "- `validate_every_n_steps = 200`",
    "- `accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16`",
    "- `use_fsdp = True`",
    "- `seed = 42`",
    "- `repeat1/2/3` 都固定使用 `GPU 3,4,5` 做三卡联合训练",
    "",
    "| Run | 类型 | 关键改动 | 重复次数 | I+Q Task Acc | I+Q ROUGE-L | 简评 |",
    "|---|---|---|---:|---:|---:|---|",
]

for method in ("tokmem_baseline", "logit_bias"):
    method_rows = sorted(by_method[method], key=lambda row: row["run_name"])
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
        f"| {run_names} | {method_label} | {change_label} | 3 | {task_acc_mean:.2f}% | {rouge_l_mean:.4f}% | `GPU 3,4,5` 三卡联合训练重跑三次均值。 |"
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
    for row in sorted(by_method[method], key=lambda item: item["run_name"]):
        lines.append(
            f"| {label} | `{row['run_name']}` | {row['task_accuracy'] * 100.0:.2f}% | {row['rouge_l']:.4f}% | `{row['run_dir']}` |"
        )

with results_path.open("a", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\n\n")
PY

echo "Mean results appended to ${RESULTS_MD}"
