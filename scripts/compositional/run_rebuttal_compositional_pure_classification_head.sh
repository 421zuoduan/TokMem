#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_REL="${SCRIPT_PATH#$ROOT_DIR/}"

SUITE_TIMESTAMP="${TOKMEM_SUITE_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
MODEL_KEY="${TOKMEM_MODEL_KEY:-llama1b}"
case "$MODEL_KEY" in
    llama1b)
        MODEL_LABEL="Llama-3.2-1B-Instruct"
        MODEL_PATH="$ROOT_DIR/models/Llama-3.2-1B-Instruct"
        TRAIN_BATCH_SIZE=24
        EVAL_BATCH_SIZE=256
        ;;
    llama8b)
        MODEL_LABEL="Llama-3.1-8B-Instruct"
        MODEL_PATH="$ROOT_DIR/models/Llama-3.1-8B-Instruct"
        TRAIN_BATCH_SIZE=8
        EVAL_BATCH_SIZE=64
        ;;
    *)
        echo "Unsupported TOKMEM_MODEL_KEY: $MODEL_KEY" >&2
        echo "Supported values: llama1b, llama8b" >&2
        exit 2
        ;;
esac

TRIAL_SPEC="${TOKMEM_TRIALS:-1,2,3}"
IFS=',' read -r -a TRIALS <<< "$TRIAL_SPEC"
if [[ "${#TRIALS[@]}" -eq 0 ]]; then
    echo "TOKMEM_TRIALS must name at least one trial" >&2
    exit 2
fi

declare -A SEEN_TRIALS=()
for trial in "${TRIALS[@]}"; do
    if [[ ! "$trial" =~ ^[1-9][0-9]*$ ]]; then
        echo "Trial identifiers must be positive integers, got: $trial" >&2
        exit 2
    fi
    if [[ -n "${SEEN_TRIALS[$trial]:-}" ]]; then
        echo "Trial identifiers must be unique, duplicate: $trial" >&2
        exit 2
    fi
    SEEN_TRIALS[$trial]=1
done

TRIAL_COUNT="${#TRIALS[@]}"
SUITE_NAME="rebuttal_compositional_pure_classification_head_${MODEL_KEY}_4calls_seed42_${TRIAL_COUNT}trials_${SUITE_TIMESTAMP}"
SUITE_DIR="${TOKMEM_RESULTS_ROOT:-$ROOT_DIR/results/compositional}/$SUITE_NAME"
RUNS_ROOT="$SUITE_DIR/runs"
DATA_DIR="${TOKMEM_DATA_DIR:-$SUITE_DIR/data}"
HF_CACHE_DIR="${TOKMEM_HF_CACHE_DIR:-$SUITE_DIR/hf-cache}"
MANIFEST_FILE="$SUITE_DIR/manifest.tsv"
SUMMARY_FILE="$SUITE_DIR/summary.md"
RESULTS_JSON="$SUITE_DIR/results.json"
DATASET_LOG="$SUITE_DIR/dataset.log"
SCHEDULER_LOG="$SUITE_DIR/scheduler.log"
TRIAL_STATUS_DIR="$SUITE_DIR/trial_status"

SEED=42
GPU_SPEC="${TOKMEM_GPUS:-${TOKMEM_GPU:-${CUDA_VISIBLE_DEVICES:-0,1,2}}}"
IFS=',' read -r -a REQUESTED_GPUS <<< "$GPU_SPEC"

if [[ "${#REQUESTED_GPUS[@]}" -ne "${#TRIALS[@]}" ]]; then
    echo "Expected exactly $TRIAL_COUNT comma-separated GPUs, got: $GPU_SPEC" >&2
    echo "Set TOKMEM_GPUS, for example: TOKMEM_GPUS=0,1,2" >&2
    exit 2
fi

declare -a TRIAL_GPUS=()
declare -A SEEN_GPUS=()
for requested_gpu in "${REQUESTED_GPUS[@]}"; do
    gpu="${requested_gpu//[[:space:]]/}"
    if [[ -z "$gpu" ]]; then
        echo "GPU identifiers must not be empty: $GPU_SPEC" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
        echo "Each trial must use a different GPU; duplicate GPU: $gpu" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu]=1
    TRIAL_GPUS+=("$gpu")
done

printf -v VISIBLE_GPUS '%s,' "${TRIAL_GPUS[@]}"
VISIBLE_GPUS="${VISIBLE_GPUS%,}"

mkdir -p "$SUITE_DIR" "$RUNS_ROOT" "$DATA_DIR" "$HF_CACHE_DIR" "$TRIAL_STATUS_DIR"
cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TOKENIZERS_PARALLELISM=false

printf "trial\tseed\tgpu\tstatus\texit_code\trun_name\trun_dir\tevaluation_results\ttraining_summary\tstdout_log\n" > "$MANIFEST_FILE"
touch "$SCHEDULER_LOG"

log() {
    local timestamp
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$timestamp] $*" | tee -a "$SCHEDULER_LOG"
}

generate_dataset() {
    local train_file="$DATA_DIR/training/function_calling_train_tools51-100_4calls.json"
    local test_file="$DATA_DIR/test/function_calling_test_tools51-100_4calls.json"
    local description_file="$DATA_DIR/tool_descriptions_tools51-100.json"

    if [[ -f "$train_file" && -f "$test_file" && -f "$description_file" ]]; then
        log "Reusing compositional data under $DATA_DIR"
        return
    fi

    log "Generating the shared tools 51-100 / 4-call dataset under $DATA_DIR"
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

write_summary() {
    python - \
        "$MANIFEST_FILE" \
        "$SUMMARY_FILE" \
        "$RESULTS_JSON" \
        "$SUITE_NAME" \
        "$SCRIPT_REL" \
        "$VISIBLE_GPUS" \
        "$MODEL_LABEL" \
        "$TRAIN_BATCH_SIZE" \
        "$EVAL_BATCH_SIZE" \
        "$MODEL_KEY" \
        "$TRIAL_COUNT" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path


manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
results_path = Path(sys.argv[3])
suite_name = sys.argv[4]
script_rel = sys.argv[5]
visible_gpus = sys.argv[6]
model_label = sys.argv[7]
train_batch_size = int(sys.argv[8])
eval_batch_size = int(sys.argv[9])
model_key = sys.argv[10]
trials_requested = int(sys.argv[11])

metric_fields = (
    ("tool_accuracy", "Tool Accuracy"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("tool_exact_match_acc", "Tool Exact Match Accuracy"),
    ("exact_accuracy", "Exact Match Accuracy"),
    ("parse_error_rate", "Parse Error Rate"),
)
loss_fields = (
    ("avg_total_loss", "Average Total Loss"),
    ("avg_ar_loss", "Average AR Loss"),
    ("avg_logit_bias_loss", "Average Routing Loss"),
)


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats_for(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) >= 2 else None,
        "count": len(values),
    }


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
    if isinstance(payload, dict) and payload.get("rounds"):
        return payload["rounds"][-1]
    if isinstance(payload, list) and payload:
        return payload[-1]
    return {}


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
metric_values = {key: [] for key, _ in metric_fields}
loss_values = {key: [] for key, _ in loss_fields}
trials = []

for row in sorted(rows, key=lambda item: int(item["trial"])):
    successful = row.get("status") == "success"
    trial = {
        "trial": int(row["trial"]),
        "seed": int(row["seed"]),
        "gpu": row["gpu"],
        "status": row.get("status") or "unknown",
        "exit_code": int(row["exit_code"]) if row.get("exit_code") else None,
        "run_name": row["run_name"],
        "run_dir": row["run_dir"],
        "evaluation_results": row["evaluation_results"],
        "training_summary": row["training_summary"],
        "stdout_log": row["stdout_log"],
        "metrics": {},
        "losses": {},
    }

    eval_path = Path(row["evaluation_results"])
    if successful and eval_path.exists():
        metrics = load_last_eval_metrics(eval_path)
        for key, _ in metric_fields:
            value = as_float(metrics.get(key))
            if value is not None:
                metric_values[key].append(value)
                trial["metrics"][key] = value

    training_path = Path(row["training_summary"])
    if successful and training_path.exists():
        training_round = load_last_training_round(training_path)
        for key, _ in loss_fields:
            value = as_float(training_round.get(key))
            if value is not None:
                loss_values[key].append(value)
                trial["losses"][key] = value

    trials.append(trial)

successful_trials = sum(trial["status"] == "success" for trial in trials)
metric_stats = {key: stats_for(values) for key, values in metric_values.items()}
loss_stats = {key: stats_for(values) for key, values in loss_values.items()}

payload = {
    "suite_name": suite_name,
    "script": script_rel,
    "model": model_label,
    "model_key": model_key,
    "method": "tapmem_pure_classification_routing_head",
    "scope": "tools 51-100 / 4 calls",
    "training_rounds": "51-100:1",
    "epochs": 3,
    "batch_size": train_batch_size,
    "eval_batch_size": eval_batch_size,
    "max_length": 512,
    "max_new_tokens": 512,
    "lr": 5e-3,
    "seed": 42,
    "trials_requested": trials_requested,
    "trials_successful": successful_trials,
    "complete": successful_trials == trials_requested,
    "gpu_ids": visible_gpus.split(","),
    "use_eoc": True,
    "use_logit_bias": True,
    "use_logit_train_add": True,
    "detach": True,
    "detach_head_from_ar_loss": True,
    "logit_bias_loss_weight": 0.1,
    "logit_bias_network": "linear",
    "logit_bias_scale": 1.0,
    "metric_stats": metric_stats,
    "loss_stats": loss_stats,
    "trials": trials,
}
results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    f"# {suite_name}",
    "",
    f"- model: `{model_label}`",
    "- method: `TapMem with a pure-classification routing head`",
    "- scope: `tools 51-100 / 4 calls`",
    f"- trials: `{trials_requested}`; seed: `42` for every trial",
    f"- completed trials: `{successful_trials}/{trials_requested}`",
    f"- TapMem-aligned settings: `epochs=3`, `batch_size={train_batch_size}`, `eval_batch_size={eval_batch_size}`, `max_length=512`, `max_new_tokens=512`, `lr=5e-3`",
    "- routing settings: `use_eoc=true`, `use_logit_bias=true`, `use_logit_train_add=true`, `detach=true`, `detach_head_from_ar_loss=true`, `routing_loss_weight=0.1`, `routing_head=linear`, `logit_bias_scale=1.0`",
    f"- parallel trial GPUs: `{visible_gpus}`",
    f"- script: `{script_rel}`",
    "",
]
if successful_trials != trials_requested:
    lines.extend(
        [
            "> Warning: fewer than the requested trials succeeded; means below use successful trials only.",
            "",
        ]
    )

lines.extend(
    [
        "## Mean Metrics",
        "",
        "| Metric | Mean | Std | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
)
for key, label in metric_fields:
    stats = metric_stats[key]
    lines.append(f"| {label} | {fmt(stats['mean'])} | {fmt(stats['stdev'])} | {stats['count']} |")

lines.extend(
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
    lines.append(f"| {label} | {fmt(stats['mean'])} | {fmt(stats['stdev'])} | {stats['count']} |")

lines.extend(
    [
        "",
        "## Trial Details",
        "",
        "| Trial | Seed | GPU | Status | Run | Tool F1 | Arguments F1 | Exact Match Accuracy |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
)
for trial in trials:
    metrics = trial["metrics"]
    lines.append(
        "| "
        + " | ".join(
            [
                str(trial["trial"]),
                str(trial["seed"]),
                trial["gpu"],
                trial["status"],
                f"`{trial['run_name']}`",
                fmt(metrics.get("avg_tool_f1_score")),
                fmt(metrics.get("avg_f1_score")),
                fmt(metrics.get("exact_accuracy")),
            ]
        )
        + " |"
    )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

generate_dataset

run_trial() {
    local trial="$1"
    local gpu="$2"
    local run_name run_dir stdout_log status_file trial_exit_code trial_status

    run_name="tapmem_pure_classification_head_${MODEL_KEY}_4calls_seed42_${SUITE_TIMESTAMP}_trial${trial}"
    run_dir="$RUNS_ROOT/$run_name"
    stdout_log="$SUITE_DIR/trial_${trial}.stdout.log"
    status_file="$TRIAL_STATUS_DIR/trial_${trial}.tsv"

    mkdir -p "$run_dir"
    cp "$SCRIPT_PATH" "$run_dir/$(basename "$SCRIPT_PATH")"

    log "trial=$trial/$TRIAL_COUNT seed=$SEED running on CUDA_VISIBLE_DEVICES=$gpu run=$run_name"
    set +e
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        cd "$ROOT_DIR/compositional"
        python -u main_sequential.py \
            --training_rounds "51-100:1" \
            --epochs 3 \
            --batch_size "$TRAIN_BATCH_SIZE" \
            --train_max_function_calls 4 \
            --test_max_function_calls 4 \
            --model_name "$MODEL_PATH" \
            --eval_after_each_round \
            --save_checkpoints \
            --data_dir "$DATA_DIR" \
            --lr 5e-3 \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            --max_length 512 \
            --max_new_tokens 512 \
            --dtype bfloat16 \
            --seed "$SEED" \
            --tensorboard \
            --use_eoc \
            --use_logit_bias \
            --use_logit_train_add \
            --detach \
            --detach_head_from_ar_loss \
            --logit_bias_loss_weight 0.1 \
            --logit_bias_network linear \
            --logit_bias_scale 1.0 \
            --run_root_dir "$RUNS_ROOT" \
            --run_name "$run_name" \
            --run_tag "${MODEL_KEY}_pure_classification_routing_head_4calls"
    ) 2>&1 | tee "$stdout_log"
    trial_exit_code="${PIPESTATUS[0]}"
    set -e

    trial_status="failed"
    if [[ "$trial_exit_code" -eq 0 ]]; then
        trial_status="success"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$trial" \
        "$SEED" \
        "$gpu" \
        "$trial_status" \
        "$trial_exit_code" \
        "$run_name" \
        "$run_dir" \
        "$run_dir/evaluation_results.json" \
        "$run_dir/training_summary.json" \
        "$stdout_log" \
        > "$status_file"

    log "trial=$trial/$TRIAL_COUNT gpu=$gpu status=$trial_status exit_code=$trial_exit_code run=$run_name"
    return "$trial_exit_code"
}

declare -a TRIAL_PIDS=()
for trial_index in "${!TRIALS[@]}"; do
    trial="${TRIALS[$trial_index]}"
    gpu="${TRIAL_GPUS[$trial_index]}"
    : > "$TRIAL_STATUS_DIR/trial_${trial}.tsv"
    run_trial "$trial" "$gpu" &
    TRIAL_PIDS[$trial_index]=$!
done

overall_status=0
for trial_index in "${!TRIALS[@]}"; do
    if ! wait "${TRIAL_PIDS[$trial_index]}"; then
        overall_status=1
    fi
done

for trial in "${TRIALS[@]}"; do
    status_file="$TRIAL_STATUS_DIR/trial_${trial}.tsv"
    if [[ ! -s "$status_file" ]]; then
        log "trial=$trial did not write a non-empty status file: $status_file"
        overall_status=1
        continue
    fi
    cat "$status_file" >> "$MANIFEST_FILE"
done

write_summary

log "Manifest written to $MANIFEST_FILE"
log "Summary written to $SUMMARY_FILE"
log "JSON results written to $RESULTS_JSON"
exit "$overall_status"
