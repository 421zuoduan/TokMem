#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_REL="${SCRIPT_PATH#$ROOT_DIR/}"

SUITE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="paper_compositional_logit_bias_loss_weight_ablation_llama1b_4calls_${SUITE_TIMESTAMP}"
GPU_IDS_CSV="0,1,2,3,4,5,6,7"
POLL_SECONDS=5
GPU_MEMORY_LIMIT_MIB=2048
GPU_IDLE_REQUIRED_SECONDS=240
GPU_LOCK_DIR="/tmp/tokmem_gpu_locks"

usage() {
    cat <<EOF
Usage: bash scripts/compositional/llama_1b/rerun_paper_compositional_logit_bias_loss_weight_ablation.sh [--gpus 0,1,2,3,4,5,6,7] [--suite-name NAME] [--poll-seconds N] [--gpu-memory-limit-mib N] [--gpu-idle-required-seconds N]

Runs the compositional logit-bias loss weight ablation for:
- model: llama1b
- method: tokmem_eoc_logit_bias
- tools: 51-100
- call scope: 4 calls
- loss weights: 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1
- 3 trials per loss weight, seed fixed to 42 for every trial
- GPU list is used as worker slots; extra loss weights are queued round-robin.
- each GPU worker takes a per-GPU flock lock, then starts after memory.used stays <= 2048 MiB for 240 consecutive seconds.

Artifacts:
- runs:    results/compositional/<suite-name>/runs/
- summary: results/compositional/<suite-name>/summary.md
- results: results/compositional/<suite-name>/results.json
- manifest: results/compositional/<suite-name>/manifest.tsv
- gpu availability: results/compositional/<suite-name>/gpu_availability.log
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_IDS_CSV="$2"
            shift 2
            ;;
        --suite-name)
            SUITE_NAME="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --gpu-memory-limit-mib)
            GPU_MEMORY_LIMIT_MIB="$2"
            shift 2
            ;;
        --gpu-idle-required-seconds)
            GPU_IDLE_REQUIRED_SECONDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SUITE_DIR="$ROOT_DIR/results/compositional/$SUITE_NAME"
RUNS_ROOT="$SUITE_DIR/runs"
DATA_DIR="$SUITE_DIR/data"
HF_CACHE_DIR="$SUITE_DIR/hf-cache"
SUMMARY_FILE="$SUITE_DIR/summary.md"
RESULTS_JSON="$SUITE_DIR/results.json"
MANIFEST_FILE="$SUITE_DIR/manifest.tsv"
DATASET_LOG="$SUITE_DIR/dataset.log"
SCHEDULER_LOG="$SUITE_DIR/scheduler.log"
GPU_AVAILABILITY_LOG="$SUITE_DIR/gpu_availability.log"

LOSS_WEIGHTS=(0.01 0.05 0.1 0.15 0.2 0.3 0.5 1)
TRIALS=(1 2 3)
SEED=42

IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
    echo "Need at least one GPU id in --gpus" >&2
    exit 1
fi

mkdir -p "$SUITE_DIR" "$RUNS_ROOT" "$DATA_DIR" "$HF_CACHE_DIR"
cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TOKENIZERS_PARALLELISM=false

printf "loss_weight\tgpu\ttrial\tseed\tstatus\texit_code\trun_name\trun_dir\tevaluation_results\ttraining_summary\tstdout_log\n" > "$MANIFEST_FILE"
touch "$SCHEDULER_LOG" "$GPU_AVAILABILITY_LOG"

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$SCHEDULER_LOG"
}

loss_weight_slug() {
    local value="$1"
    value="${value//./p}"
    echo "$value"
}

gpu_lock_file() {
    local gpu="$1"
    local lock_name="${gpu//[^A-Za-z0-9_.-]/_}"
    echo "$GPU_LOCK_DIR/gpu_${lock_name}.lock"
}

gpu_memory_used_mib() {
    local gpu="$1"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null \
        | awk 'NR == 1 {gsub(/^[ \t]+|[ \t]+$/, "", $0); print $0}'
}

wait_for_gpu_ready() {
    local gpu="$1"
    local idle_since=""
    local memory_used now_epoch now_iso idle_seconds

    while true; do
        now_epoch="$(date +%s)"
        now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

        if memory_used="$(gpu_memory_used_mib "$gpu")" && [[ "$memory_used" =~ ^[0-9]+$ ]]; then
            if (( memory_used <= GPU_MEMORY_LIMIT_MIB )); then
                if [[ -z "$idle_since" ]]; then
                    idle_since="$now_epoch"
                    log "GPU availability window started gpu=$gpu memory_mib=$memory_used limit_mib=$GPU_MEMORY_LIMIT_MIB required_seconds=$GPU_IDLE_REQUIRED_SECONDS"
                fi
                idle_seconds=$((now_epoch - idle_since))
                printf '[%s]\tgpu=%s\tstate=idle_window\tmemory_mib=%s\tidle_seconds=%s\n' \
                    "$now_iso" "$gpu" "$memory_used" "$idle_seconds" >> "$GPU_AVAILABILITY_LOG"
                if (( idle_seconds >= GPU_IDLE_REQUIRED_SECONDS )); then
                    log "GPU ready gpu=$gpu memory_mib=$memory_used idle_seconds=$idle_seconds"
                    return 0
                fi
            else
                if [[ -n "$idle_since" ]]; then
                    log "GPU availability window reset gpu=$gpu memory_mib=$memory_used limit_mib=$GPU_MEMORY_LIMIT_MIB"
                fi
                idle_since=""
                printf '[%s]\tgpu=%s\tstate=busy\tmemory_mib=%s\tidle_seconds=0\n' \
                    "$now_iso" "$gpu" "$memory_used" >> "$GPU_AVAILABILITY_LOG"
            fi
        else
            if [[ -n "$idle_since" ]]; then
                log "GPU availability window reset gpu=$gpu reason=nvidia-smi-query-failed"
            fi
            idle_since=""
            printf '[%s]\tgpu=%s\tstate=query_failed\tmemory_mib=unknown\tidle_seconds=0\n' \
                "$now_iso" "$gpu" >> "$GPU_AVAILABILITY_LOG"
        fi

        sleep "$POLL_SECONDS"
    done
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

run_loss_weight() {
    local loss_weight="$1"
    local gpu="$2"
    local slug
    slug="$(loss_weight_slug "$loss_weight")"
    local loss_weight_dir="$SUITE_DIR/loss_weight_${slug}"
    local loss_weight_status=0
    mkdir -p "$loss_weight_dir"

    export CUDA_VISIBLE_DEVICES="$gpu"

    for trial in "${TRIALS[@]}"; do
        local run_name="tokmem_eoc_logit_bias_lossw${slug}_llama_1b_4calls_seed42_3x_${SUITE_TIMESTAMP}_trial${trial}"
        local run_dir="$RUNS_ROOT/$run_name"
        local stdout_log="$loss_weight_dir/trial_${trial}.stdout.log"

        mkdir -p "$run_dir"
        cp "$SCRIPT_PATH" "$run_dir/$(basename "$SCRIPT_PATH")"

        log "loss_weight=$loss_weight trial=$trial/3 running on gpu=$gpu run=$run_name"
        set +e
        (
            cd "$ROOT_DIR/compositional"
            python -u main_sequential.py \
                --training_rounds "51-100:1" \
                --epochs 3 \
                --batch_size 24 \
                --train_max_function_calls 4 \
                --test_max_function_calls 4 \
                --model_name "$ROOT_DIR/models/Llama-3.2-1B-Instruct" \
                --eval_after_each_round \
                --save_checkpoints \
                --data_dir "$DATA_DIR" \
                --lr 5e-3 \
                --eval_batch_size 256 \
                --max_length 512 \
                --seed "$SEED" \
                --tensorboard \
                --use_eoc \
                --use_logit_bias \
                --detach \
                --use_logit_train_add \
                --logit_bias_loss_weight "$loss_weight" \
                --logit_bias_network linear \
                --logit_bias_scale 1.0 \
                --run_root_dir "$RUNS_ROOT" \
                --run_name "$run_name" \
                --run_tag "llama1b_tokmem_eoc_logit_bias_lossw${slug}_4calls"
        ) 2>&1 | tee "$stdout_log"
        local_exit_code=${PIPESTATUS[0]}
        set -e

        local_status="failed"
        if [[ "$local_exit_code" -eq 0 ]]; then
            local_status="success"
        else
            loss_weight_status=1
        fi

        (
            flock -x 9
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$loss_weight" \
                "$gpu" \
                "$trial" \
                "$SEED" \
                "$local_status" \
                "$local_exit_code" \
                "$run_name" \
                "$run_dir" \
                "$run_dir/evaluation_results.json" \
                "$run_dir/training_summary.json" \
                "$stdout_log" \
                >> "$MANIFEST_FILE"
        ) 9>>"$MANIFEST_FILE.lock"
    done

    return "$loss_weight_status"
}

write_summary() {
    python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$RESULTS_JSON" "$SUITE_NAME" "$SCRIPT_REL" "$GPU_IDS_CSV" "$POLL_SECONDS" "$GPU_MEMORY_LIMIT_MIB" "$GPU_IDLE_REQUIRED_SECONDS" <<'PY'
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
gpu_ids_csv = sys.argv[6]
poll_seconds = int(sys.argv[7])
gpu_memory_limit_mib = int(sys.argv[8])
gpu_idle_required_seconds = int(sys.argv[9])

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
    if isinstance(payload, dict):
        return payload["rounds"][-1]
    if isinstance(payload, list):
        return payload[-1]
    return {}


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
groups = defaultdict(list)
for row in rows:
    groups[row["loss_weight"]].append(row)

loss_weight_summaries = {}
for loss_weight, loss_weight_rows in sorted(groups.items(), key=lambda item: float(item[0])):
    metric_values = {key: [] for key, _ in metric_fields}
    loss_values = {key: [] for key, _ in loss_fields}
    trials = []

    for row in sorted(loss_weight_rows, key=lambda item: int(item["trial"])):
        trial = {
            "loss_weight": as_float(row["loss_weight"]),
            "gpu": row["gpu"],
            "trial": int(row["trial"]),
            "seed": int(row["seed"]),
            "status": row.get("status") or "unknown",
            "exit_code": int(row["exit_code"]) if row.get("exit_code") not in (None, "") else None,
            "run_name": row["run_name"],
            "run_dir": row["run_dir"],
            "evaluation_results": row["evaluation_results"],
            "training_summary": row["training_summary"],
            "stdout_log": row["stdout_log"],
            "metrics": {},
            "losses": {},
        }

        row_success = (row.get("status") == "success")

        eval_path = Path(row["evaluation_results"])
        if row_success and eval_path.exists():
            metrics = load_last_eval_metrics(eval_path)
            for key, _ in metric_fields:
                value = as_float(metrics.get(key))
                if value is not None:
                    metric_values[key].append(value)
                    trial["metrics"][key] = value

        train_path = Path(row["training_summary"])
        if row_success and train_path.exists():
            train_round = load_last_training_round(train_path)
            for key, _ in loss_fields:
                value = as_float(train_round.get(key))
                if value is not None:
                    loss_values[key].append(value)
                    trial["losses"][key] = value

        trials.append(trial)

    loss_weight_summaries[loss_weight] = {
        "metric_stats": {key: stats_for(values) for key, values in metric_values.items()},
        "loss_stats": {key: stats_for(values) for key, values in loss_values.items()},
        "trials": trials,
    }

payload = {
    "suite_name": suite_name,
    "script": script_rel,
    "model": "Llama-3.2-1B-Instruct",
    "scope": "tools 51-100 / 4 calls",
    "method": "tokmem_eoc_logit_bias",
    "training_rounds": "51-100:1",
    "epochs": 3,
    "batch_size": 24,
    "eval_batch_size": 256,
    "max_length": 512,
    "lr": 5e-3,
    "seed": 42,
    "trials_per_loss_weight": 3,
    "detach": True,
    "use_logit_train_add": True,
    "logit_bias_scale": 1.0,
    "logit_bias_network": "linear",
    "gpu_ids": gpu_ids_csv.split(","),
    "scheduler": {
        "poll_seconds": poll_seconds,
        "gpu_memory_limit_mib": gpu_memory_limit_mib,
        "gpu_idle_required_seconds": gpu_idle_required_seconds,
        "gpu_lock_dir": "/tmp/tokmem_gpu_locks",
        "gpu_availability_log": "gpu_availability.log",
    },
    "loss_weights": loss_weight_summaries,
}
results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

loss_weight_labels = ", ".join(f"`{loss_weight}`" for loss_weight in sorted(loss_weight_summaries, key=float))
gpu_assignments = ", ".join(
    f"`{loss_weight} -> {loss_weight_summaries[loss_weight]['trials'][0]['gpu']}`"
    for loss_weight in sorted(loss_weight_summaries, key=float)
    if loss_weight_summaries[loss_weight]["trials"]
)

lines = [
    f"# {suite_name}",
    "",
    "- model: `Llama-3.2-1B-Instruct`",
    "- method: `tokmem_eoc_logit_bias`",
    "- scope: `tools 51-100 / 4 calls`",
    "- trials per loss weight: `3`",
    "- seed: `42` for every trial",
    f"- loss weights: {loss_weight_labels}",
    f"- gpu assignment: {gpu_assignments}",
    "- paper-suite aligned settings: `training_rounds=51-100:1`, `epochs=3`, `train_size=5000`, `test_size=500`, `max_function_calls=4`, `max_length=512`, `lr=5e-3`",
    "- Llama-1B TokMem batch settings: `batch_size=24`, `eval_batch_size=256`",
    "- logit-bias settings: `detach=true`, `use_logit_train_add=true`, `logit_bias_scale=1.0`, `logit_bias_network=linear`",
    f"- GPU scheduling: per-GPU `flock` locks under `/tmp/tokmem_gpu_locks`, launch after `memory.used <= {gpu_memory_limit_mib} MiB` for `{gpu_idle_required_seconds}` seconds, poll interval `{poll_seconds}` seconds",
    f"- script: `{script_rel}`",
    "",
    "## Mean Metrics",
    "",
        "| Loss Weight | Routing Acc | Routing Std | Arguments F1 | Arguments F1 Std | Tool F1 | Exact Match | Count |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for loss_weight, summary in sorted(loss_weight_summaries.items(), key=lambda item: float(item[0])):
    metric_stats = summary["metric_stats"]
    lines.append(
        "| "
        + " | ".join(
            [
                loss_weight,
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
        "| Loss Weight | GPU | Trial | Status | Exit Code | Run | Routing Acc | Arguments F1 | Tool F1 | Exact Match |",
        "| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
)
for loss_weight, summary in sorted(loss_weight_summaries.items(), key=lambda item: float(item[0])):
    for trial in summary["trials"]:
        metrics = trial["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    loss_weight,
                    trial["gpu"],
                    str(trial["trial"]),
                    trial["status"],
                    "" if trial["exit_code"] is None else str(trial["exit_code"]),
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

run_gpu_queue() {
    local gpu_index="$1"
    local gpu_count="$2"
    local gpu="${GPUS[$gpu_index]}"
    local lock_file
    lock_file="$(gpu_lock_file "$gpu")"
    mkdir -p "$GPU_LOCK_DIR"

    log "Waiting for GPU lock gpu=$gpu lock_file=$lock_file"
    (
        flock -x 200
        log "GPU lock acquired gpu=$gpu lock_file=$lock_file"
        wait_for_gpu_ready "$gpu"

        worker_status=0
        for loss_weight_index in "${!LOSS_WEIGHTS[@]}"; do
            if (( loss_weight_index % gpu_count == gpu_index )); then
                if ! run_loss_weight "${LOSS_WEIGHTS[$loss_weight_index]}" "$gpu"; then
                    worker_status=1
                fi
            fi
        done

        exit "$worker_status"
    ) 200>"$lock_file"
}

declare -a PIDS=()
GPU_COUNT="${#GPUS[@]}"
for gpu_index in "${!GPUS[@]}"; do
    run_gpu_queue "$gpu_index" "$GPU_COUNT" &
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
