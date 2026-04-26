#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_REL="${SCRIPT_PATH#$ROOT_DIR/}"

RESULTS_ROOT="${TOKMEM_RESULTS_ROOT:-$ROOT_DIR/results/compositional}"
SUITE_ID="${TOKMEM_SUITE_ID:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_NAME="qwen_0_5b_logit_bias_scale_ablation_4calls_seed42_3x_${SUITE_ID}"
SUITE_DIR="$RESULTS_ROOT/$SUITE_NAME"
RUNS_ROOT="$SUITE_DIR/runs"
DATA_DIR="${TOKMEM_DATA_DIR:-$SUITE_DIR/data}"
HF_CACHE_DIR="${TOKMEM_HF_CACHE_DIR:-$SUITE_DIR/hf-cache}"
SUMMARY_FILE="$SUITE_DIR/summary.md"
RESULTS_JSON="$SUITE_DIR/results.json"
MANIFEST_FILE="$SUITE_DIR/manifest.tsv"
DATASET_LOG="$SUITE_DIR/dataset.log"
SCHEDULER_LOG="$SUITE_DIR/scheduler.log"

SCALES=(0.1 0.5 2)
GPUS=(5 6 7)
TRIALS=(1 2 3)
SEED=42

mkdir -p "$SUITE_DIR" "$RUNS_ROOT" "$DATA_DIR" "$HF_CACHE_DIR"
cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TOKENIZERS_PARALLELISM=false

printf "scale\tgpu\ttrial\tseed\trun_name\trun_dir\tevaluation_results\ttraining_summary\tstdout_log\n" > "$MANIFEST_FILE"
touch "$SCHEDULER_LOG"

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$SCHEDULER_LOG"
}

scale_slug() {
    local value="$1"
    value="${value//./p}"
    echo "$value"
}

generate_dataset() {
    local train_file="$DATA_DIR/training/function_calling_train_tools51-100_4calls.json"
    local test_file="$DATA_DIR/test/function_calling_test_tools51-100_4calls.json"
    local tool_file="$DATA_DIR/tool_descriptions_tools51-100.json"

    if [[ -f "$train_file" && -f "$test_file" && -f "$tool_file" ]]; then
        log "Dataset already exists under $DATA_DIR"
        return
    fi

    log "Generating compositional dataset under $DATA_DIR"
    (
        cd "$ROOT_DIR/compositional"
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
    ) 2>&1 | tee "$DATASET_LOG"
}

run_scale() {
    local scale="$1"
    local gpu="$2"
    local slug
    slug="$(scale_slug "$scale")"
    local scale_dir="$SUITE_DIR/scale_${slug}"
    mkdir -p "$scale_dir"

    export CUDA_VISIBLE_DEVICES="$gpu"

    for trial in "${TRIALS[@]}"; do
        local run_name="tokmem_eoc_logit_bias_scale${slug}_qwen_0_5b_4calls_seed42_3x_${SUITE_ID}_trial${trial}"
        local run_dir="$RUNS_ROOT/$run_name"
        local stdout_log="$scale_dir/trial_${trial}.stdout.log"

        mkdir -p "$run_dir"
        cp "$SCRIPT_PATH" "$run_dir/$(basename "$SCRIPT_PATH")"

        log "scale=$scale trial=$trial/3 running on gpu=$gpu run=$run_name"
        (
            cd "$ROOT_DIR/compositional"
            python -u main_sequential.py \
                --training_rounds "51-100:1" \
                --epochs 3 \
                --batch_size 16 \
                --train_max_function_calls 4 \
                --test_max_function_calls 4 \
                --model_name "$ROOT_DIR/models/Qwen2.5-0.5B-Instruct" \
                --eval_after_each_round \
                --save_checkpoints \
                --data_dir "$DATA_DIR" \
                --lr 5e-3 \
                --eval_batch_size 64 \
                --max_length 512 \
                --seed "$SEED" \
                --tensorboard \
                --use_eoc \
                --use_logit_bias \
                --logit_bias_scale "$scale" \
                --run_root_dir "$RUNS_ROOT" \
                --run_name "$run_name" \
                --run_tag "qwen0_5b_tokmem_eoc_logit_bias_scale${slug}_4calls"
        ) 2>&1 | tee "$stdout_log"

        (
            flock -x 9
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$scale" \
                "$gpu" \
                "$trial" \
                "$SEED" \
                "$run_name" \
                "$run_dir" \
                "$run_dir/evaluation_results.json" \
                "$run_dir/training_summary.json" \
                "$stdout_log" \
                >> "$MANIFEST_FILE"
        ) 9>>"$MANIFEST_FILE.lock"
    done
}

write_summary() {
    python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$RESULTS_JSON" "$SUITE_NAME" "$SCRIPT_REL" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
results_path = Path(sys.argv[3])
suite_name = sys.argv[4]
script_rel = sys.argv[5]

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


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    return payload if isinstance(payload, dict) else {}


def load_last_training_round(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["rounds"][-1]
    if isinstance(payload, list):
        return payload[-1]
    return {}


def stats_for(values):
    return {
        "mean": mean_or_none(values),
        "stdev": stdev_or_none(values),
        "count": len(values),
    }


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
groups = defaultdict(list)
for row in rows:
    groups[row["scale"]].append(row)

scale_summaries = {}
for scale, scale_rows in sorted(groups.items(), key=lambda item: float(item[0])):
    metric_values = {key: [] for key, _ in metric_fields}
    loss_values = {key: [] for key, _ in loss_fields}
    trials = []

    for row in sorted(scale_rows, key=lambda item: int(item["trial"])):
        trial = {
            "scale": as_float(row["scale"]),
            "gpu": row["gpu"],
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
        eval_path = Path(row["evaluation_results"])
        train_path = Path(row["training_summary"])

        if eval_path.exists():
            metrics = load_last_eval_metrics(eval_path)
            for key, _ in metric_fields:
                value = as_float(metrics.get(key))
                if value is not None:
                    metric_values[key].append(value)
                    trial["metrics"][key] = value

        if train_path.exists():
            train_round = load_last_training_round(train_path)
            for key, _ in loss_fields:
                value = as_float(train_round.get(key))
                if value is not None:
                    loss_values[key].append(value)
                    trial["losses"][key] = value

        trials.append(trial)

    scale_summaries[scale] = {
        "metric_stats": {key: stats_for(values) for key, values in metric_values.items()},
        "loss_stats": {key: stats_for(values) for key, values in loss_values.items()},
        "trials": trials,
    }

payload = {
    "suite_name": suite_name,
    "script": script_rel,
    "model": "Qwen2.5-0.5B-Instruct",
    "scope": "tools 51-100 / 4 calls",
    "method": "tokmem_eoc_logit_bias",
    "training_rounds": "51-100:1",
    "epochs": 3,
    "batch_size": 16,
    "eval_batch_size": 64,
    "max_length": 512,
    "lr": 5e-3,
    "seed": 42,
    "trials_per_scale": 3,
    "logit_bias_loss_weight": 0.1,
    "logit_bias_network": "linear",
    "scales": scale_summaries,
}
results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    f"# {suite_name}",
    "",
    "- model: `Qwen2.5-0.5B-Instruct`",
    "- method: `tokmem_eoc_logit_bias`",
    "- scope: `tools 51-100 / 4 calls`",
    "- trials per scale: `3`",
    "- seed: `42` for every trial",
    "- scales: `0.1`, `0.5`, `2`",
    "- gpu assignment: `0.1 -> 5`, `0.5 -> 6`, `2 -> 7`",
    "- paper-suite aligned settings: `training_rounds=51-100:1`, `epochs=3`, `train_size=5000`, `test_size=500`, `max_function_calls=4`, `max_length=512`, `lr=5e-3`",
    "- qwen batch settings: `batch_size=16`, `eval_batch_size=64`",
    "- logit-bias defaults: `logit_bias_network=linear`, `logit_bias_loss_weight=0.1`",
    f"- script: `{script_rel}`",
    "",
    "## Mean Metrics",
    "",
    "| Scale | Routing Acc | Routing Std | Rouge-L | Rouge-L Std | Tool F1 | Exact Match | Count |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for scale, summary in sorted(scale_summaries.items(), key=lambda item: float(item[0])):
    metric_stats = summary["metric_stats"]
    lines.append(
        "| "
        + " | ".join(
            [
                scale,
                fmt(metric_stats["tool_accuracy"]["mean"]),
                fmt(metric_stats["tool_accuracy"]["stdev"]),
                fmt(metric_stats["avg_f1_score"]["mean"]),
                fmt(metric_stats["avg_f1_score"]["stdev"]),
                fmt(metric_stats["avg_tool_f1_score"]["mean"]),
                fmt(metric_stats["exact_accuracy"]["mean"]),
                str(metric_stats["tool_accuracy"]["count"]),
            ]
        )
        + " |"
    )

lines.extend(
    [
        "",
        "## Trial Details",
        "",
        "| Scale | GPU | Trial | Run | Routing Acc | Rouge-L | Tool F1 | Exact Match |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
)
for scale, summary in sorted(scale_summaries.items(), key=lambda item: float(item[0])):
    for trial in summary["trials"]:
        metrics = trial["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    scale,
                    trial["gpu"],
                    str(trial["trial"]),
                    f"`{trial['run_name']}`",
                    fmt(metrics.get("tool_accuracy")),
                    fmt(metrics.get("avg_f1_score")),
                    fmt(metrics.get("avg_tool_f1_score")),
                    fmt(metrics.get("exact_accuracy")),
                ]
            )
            + " |"
        )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

generate_dataset

declare -a PIDS=()
for index in "${!SCALES[@]}"; do
    run_scale "${SCALES[$index]}" "${GPUS[$index]}" &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

write_summary

log "Summary written to $SUMMARY_FILE"
log "JSON results written to $RESULTS_JSON"
exit "$status"
