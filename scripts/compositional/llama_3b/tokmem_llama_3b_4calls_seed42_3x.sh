#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_REL="${SCRIPT_PATH#$ROOT_DIR/}"
RESULTS_ROOT="${TOKMEM_RESULTS_ROOT:-$ROOT_DIR/results/compositional}"
SUITE_ID="${TOKMEM_SUITE_ID:-$(date -u +%Y%m%d_%H%M%S)}"
METHOD_NAME="tokmem_llama_3b"
RUN_NAME_PREFIX="tokmem_llama_3b_4calls_seed42_3x"
RUN_TAG="llama3b_tokmem_4calls"
SUITE_NAME="${METHOD_NAME}_4calls_seed42_3x_${SUITE_ID}"
SUITE_DIR="$RESULTS_ROOT/$SUITE_NAME"
MANIFEST_FILE="$SUITE_DIR/manifest.tsv"
SUMMARY_FILE="$SUITE_DIR/summary.md"
ARTIFACTS_FILE="$SUITE_DIR/results.json"
DATA_DIR="${TOKMEM_DATA_DIR:-$SUITE_DIR/data}"
HF_CACHE_DIR="${TOKMEM_HF_CACHE_DIR:-$SUITE_DIR/hf-cache}"
METHOD_FLAGS=()
METHOD_FLAGS_TEXT="none"

mkdir -p "$SUITE_DIR" "$HF_CACHE_DIR"
cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"
if [[ -n "${TOKMEM_CHILD_SUITE_DIR_FILE:-}" ]]; then
    printf "%s\n" "$SUITE_DIR" > "$TOKMEM_CHILD_SUITE_DIR_FILE"
fi

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TOKENIZERS_PARALLELISM=false

printf "trial\tseed\trun_name\trun_dir\tevaluation_results\ttraining_summary\tstdout_log\n" > "$MANIFEST_FILE"

for trial in 1 2 3; do
    run_name="${RUN_NAME_PREFIX}_${SUITE_ID}_trial${trial}"
    run_dir="$ROOT_DIR/compositional/runs/$run_name"
    stdout_log="$SUITE_DIR/trial_${trial}.stdout.log"

    mkdir -p "$run_dir"
    cp "$SCRIPT_PATH" "$run_dir/$(basename "$SCRIPT_PATH")"

    echo "[trial $trial/3] running $run_name on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    (
        cd "$ROOT_DIR/compositional"

        if [[ "${SKIP_XLAM_DATASET:-0}" != "1" ]]; then
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
        fi

        python -u main_sequential.py \
            --training_rounds "51-100:1" \
            --epochs 3 \
            --batch_size 16 \
            --train_max_function_calls 4 \
            --test_max_function_calls 4 \
            --model_name "$ROOT_DIR/models/Llama-3.2-3B-Instruct" \
            --eval_after_each_round \
            --save_checkpoints \
            --data_dir "$DATA_DIR" \
            --lr 5e-3 \
            --eval_batch_size 192 \
            --max_length 1024 \
            --seed 42 \
            --tensorboard \
            "${METHOD_FLAGS[@]}" \
            --run_root_dir "$ROOT_DIR/compositional/runs" \
            --run_name "$run_name" \
            --run_tag "$RUN_TAG"
    ) 2>&1 | tee "$stdout_log"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$trial" \
        "42" \
        "$run_name" \
        "$run_dir" \
        "$run_dir/evaluation_results.json" \
        "$run_dir/training_summary.json" \
        "$stdout_log" \
        >> "$MANIFEST_FILE"
done

python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$ARTIFACTS_FILE" "$SUITE_NAME" "$METHOD_NAME" "$SCRIPT_REL" "$METHOD_FLAGS_TEXT" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
artifacts_path = Path(sys.argv[3])
suite_name = sys.argv[4]
method_name = sys.argv[5]
script_rel = sys.argv[6]
method_flags = sys.argv[7]

metric_fields = (
    ("tool_accuracy", "Routing Acc"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("tool_exact_match_acc", "Tool Exact Match Acc"),
    ("exact_accuracy", "Exact Match Acc"),
    ("parse_error_rate", "Parse Error Rate"),
)

loss_fields = (
    ("avg_total_loss", "Avg Total Loss"),
    ("avg_ar_loss", "Avg AR Loss"),
    ("avg_logit_bias_loss", "Avg Logit Bias Loss"),
)


def mean_or_none(values):
    return statistics.mean(values) if values else None


def stdev_or_none(values):
    return statistics.stdev(values) if len(values) >= 2 else None


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def load_last_eval_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("rounds"):
        return payload["rounds"][-1].get("eval_results") or {}
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    return payload


def load_last_training_round(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["rounds"][-1]
    if isinstance(payload, list):
        return payload[-1]
    raise TypeError(f"Unsupported training summary type: {type(payload)!r}")


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
metric_values = {key: [] for key, _ in metric_fields}
loss_values = {key: [] for key, _ in loss_fields}
trial_summaries = []

for row in rows:
    eval_path = Path(row["evaluation_results"])
    train_path = Path(row["training_summary"])
    trial_entry = {
        "trial": int(row["trial"]),
        "seed": int(row["seed"]),
        "run_name": row["run_name"],
        "run_dir": row["run_dir"],
        "evaluation_results": row["evaluation_results"],
        "training_summary": row["training_summary"],
        "stdout_log": row["stdout_log"],
        "metrics": {},
        "losses": {},
    }

    if eval_path.exists():
        eval_metrics = load_last_eval_metrics(eval_path)
        for key, _ in metric_fields:
            value = eval_metrics.get(key)
            if value is not None:
                metric_values[key].append(value)
                trial_entry["metrics"][key] = value

    if train_path.exists():
        train_round = load_last_training_round(train_path)
        for key, _ in loss_fields:
            value = train_round.get(key)
            if value is not None:
                loss_values[key].append(value)
                trial_entry["losses"][key] = value

    trial_summaries.append(trial_entry)

metric_stats = {
    key: {
        "mean": mean_or_none(values),
        "stdev": stdev_or_none(values),
        "count": len(values),
    }
    for key, values in metric_values.items()
}
loss_stats = {
    key: {
        "mean": mean_or_none(values),
        "stdev": stdev_or_none(values),
        "count": len(values),
    }
    for key, values in loss_values.items()
}

artifacts = {
    "suite_name": suite_name,
    "method": method_name,
    "script": script_rel,
    "method_flags": method_flags,
    "model": "Llama-3.2-3B-Instruct",
    "scope": "tools 51-100 / 4 calls",
    "training_rounds": "51-100:1",
    "epochs": 3,
    "batch_size": 16,
    "eval_batch_size": 192,
    "max_length": 1024,
    "lr": 5e-3,
    "trials_requested": 3,
    "seed": 42,
    "metric_stats": metric_stats,
    "loss_stats": loss_stats,
    "trials": trial_summaries,
}
artifacts_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

summary_lines = [
    f"# {suite_name}",
    "",
    f"- method: `{method_name}`",
    "- model: `Llama-3.2-3B-Instruct`",
    "- scope: `tools 51-100 / 4 calls`",
    "- trials: `3`",
    "- seed: `42` for every trial",
    "- paper-suite aligned settings: `training_rounds=51-100:1`, `epochs=3`, `batch_size=16`, `eval_batch_size=192`, `max_length=1024`, `lr=5e-3`",
    f"- script: `{script_rel}`",
    f"- method flags: `{method_flags}`",
    "",
    "## Mean Metrics",
    "",
    "| Metric | Mean | Std | Count |",
    "| --- | ---: | ---: | ---: |",
]
for key, label in metric_fields:
    stats = metric_stats[key]
    summary_lines.append(
        f"| {label} | {fmt(stats['mean'])} | {fmt(stats['stdev'])} | {stats['count']} |"
    )

summary_lines.extend(
    [
        "",
        "## Mean Losses",
        "",
        "| Loss | Mean | Std | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
)
for key, label in loss_fields:
    stats = loss_stats[key]
    summary_lines.append(
        f"| {label} | {fmt(stats['mean'])} | {fmt(stats['stdev'])} | {stats['count']} |"
    )

summary_lines.extend(
    [
        "",
        "## Trials",
        "",
        "| Trial | Run | Routing Acc | Tool F1 | Arguments F1 | Exact Match Acc |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
)
for trial in trial_summaries:
    metrics = trial["metrics"]
    summary_lines.append(
        "| "
        + " | ".join(
            [
                str(trial["trial"]),
                f"`{trial['run_name']}`",
                fmt(metrics.get("tool_accuracy")),
                fmt(metrics.get("avg_tool_f1_score")),
                fmt(metrics.get("avg_f1_score")),
                fmt(metrics.get("exact_accuracy")),
            ]
        )
        + " |"
    )

summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
PY

echo "Summary written to $SUMMARY_FILE"
echo "JSON results written to $ARTIFACTS_FILE"
