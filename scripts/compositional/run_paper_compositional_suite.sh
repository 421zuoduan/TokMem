#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

SUITE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="paper_compositional_51_100_4calls_10calls_${SUITE_TIMESTAMP}"
GPU_IDS_CSV="4,5,6,7"
POLL_SECONDS=5
GPU_MEMORY_LIMIT_MIB=2048
GPU_IDLE_REQUIRED_SECONDS=300
RERUN_FAILED=0
EXPLICIT_SUITE_NAME=0

usage() {
    cat <<EOF
Usage: bash scripts/compositional/run_paper_compositional_suite.sh [--gpus 0,1,2,3] [--suite-name NAME] [--poll-seconds N] [--rerun-failed]

Runs the compositional paper suite for:
- tools 51-100
- 4 calls main comparison
- 10 calls TokMem-family stress test with separate result tables
- models: llama1b, llama3b, llama8b
- methods: icl, rag, lora, tokmem, tokmem_eoc, tokmem_eoc_logit_bias, adap_tokmem, adap_tokmem_eoc, adap_tokmem_eoc_logit_bias
- 5 trials per model/method, seed fixed to 42 for every trial

Artifacts:
- runs:   results/compositional/<suite-name>/runs/
- summary: results/compositional/<suite-name>/summary.md
- status: results/compositional/<suite-name>/task_status.json

Notes:
- GPU scheduling only considers the IDs passed by --gpus.
- Suite-owned tasks keep their GPU reserved for the full process lifetime.
- GPUs that are already free, or just finished a suite-owned task, are eligible immediately.
- A GPU that was externally busy is eligible after memory.used stays <= 2048 MiB for 300 consecutive seconds.
- --rerun-failed requires --suite-name <existing-suite>
- rerunning an existing --suite-name reconciles the manifest with the current method set
- --rerun-failed reuses the existing manifest and only retries unfinished tasks already recorded there
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
            EXPLICIT_SUITE_NAME=1
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --rerun-failed)
            RERUN_FAILED=1
            shift
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
MANIFEST_FILE="$SUITE_DIR/task_manifest.tsv"
STATUS_FILE="$SUITE_DIR/task_status.json"
SUMMARY_FILE="$SUITE_DIR/summary.md"
SUMMARY_JSON_FILE="$SUITE_DIR/summary.json"
SCHEDULER_LOG="$SUITE_DIR/scheduler.log"
GPU_AVAILABILITY_LOG="$SUITE_DIR/gpu_availability.log"
SUITE_CONFIG_FILE="$SUITE_DIR/suite_config.json"
DATASET_LOG="$SUITE_DIR/dataset.log"
RERUNS_DIR="$SUITE_DIR/reruns"
RERUN_ID=""
RERUN_DIR=""
RERUN_SCRIPT_SNAPSHOT=""
RERUN_CONFIG_FILE=""
RERUN_MANIFEST_SNAPSHOT=""

if [[ "$RERUN_FAILED" -eq 1 && "$EXPLICIT_SUITE_NAME" -ne 1 ]]; then
    echo "--rerun-failed requires --suite-name <existing-suite>" >&2
    exit 1
fi

if [[ "$RERUN_FAILED" -eq 1 && ! -d "$SUITE_DIR" ]]; then
    echo "Suite directory does not exist for rerun: $SUITE_DIR" >&2
    exit 1
fi

if [[ "$RERUN_FAILED" -eq 1 ]]; then
    RERUN_ID="rerun_$(date -u +%Y%m%d_%H%M%S)"
    RERUN_DIR="$RERUNS_DIR/$RERUN_ID"
    RERUN_SCRIPT_SNAPSHOT="$RERUN_DIR/$(basename "$SCRIPT_PATH")"
    RERUN_CONFIG_FILE="$RERUN_DIR/suite_config.json"
    RERUN_MANIFEST_SNAPSHOT="$RERUN_DIR/task_manifest.tsv"
fi

TOP_K="51-100"
TRAINING_TOOLS="51-100"
TOKMEM_TRAINING_ROUNDS="${TRAINING_TOOLS}:1"
LORA_TRAINING_ROUNDS="${TRAINING_TOOLS}:3"
MAX_SAMPLES_PER_TOOL=50
CALL_SCOPES=(4calls 10calls)
TRIAL_COUNT=5
TRIAL_SEEDS=(42 42 42 42 42)

TOKMEM_EPOCHS=3
TOKMEM_LR="5e-3"
LORA_LR="5e-5"

MODEL_KEYS=(llama1b llama3b llama8b)
METHODS=(
    icl
    rag
    lora
    tokmem
    tokmem_eoc
    tokmem_eoc_logit_bias
    adap_tokmem
    adap_tokmem_eoc
    adap_tokmem_eoc_logit_bias
)
TRAINING_METHODS=(
    lora
    tokmem
    tokmem_eoc
    tokmem_eoc_logit_bias
    adap_tokmem
    adap_tokmem_eoc
    adap_tokmem_eoc_logit_bias
)

declare -A METHODS_BY_SCOPE=(
    [4calls]="${METHODS[*]}"
    [10calls]="tokmem tokmem_eoc_logit_bias adap_tokmem adap_tokmem_eoc_logit_bias"
)

declare -A TRAIN_SIZE_BY_SCOPE=(
    [4calls]=5000
    [10calls]=8000
)

declare -A TEST_SIZE_BY_SCOPE=(
    [4calls]=500
    [10calls]=800
)

declare -A TRAIN_MAX_CALLS_BY_SCOPE=(
    [4calls]=4
    [10calls]=10
)

declare -A TEST_MAX_CALLS_BY_SCOPE=(
    [4calls]=4
    [10calls]=10
)

declare -A TRAIN_MULTI_TOOL_RATIOS_BY_SCOPE=(
    [4calls]="0.5,0.5"
    [10calls]="0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125"
)

declare -A TEST_MULTI_TOOL_RATIOS_BY_SCOPE=(
    [4calls]="0.5,0.5"
    [10calls]="0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125"
)

declare -A MAX_LENGTH_BY_SCOPE=(
    [4calls]=512
    [10calls]=1024
)

declare -A MODEL_PATHS=(
    [llama1b]="$ROOT_DIR/models/Llama-3.2-1B-Instruct"
    [llama3b]="$ROOT_DIR/models/Llama-3.2-3B-Instruct"
    [llama8b]="$ROOT_DIR/models/Llama-3.1-8B-Instruct"
)

declare -A METHOD_EXPERIMENT_TYPES=(
    [icl]="icl"
    [rag]="rag"
    [lora]="lora"
    [tokmem]="tokmem"
    [tokmem_eoc]="tokmem_eoc"
    [tokmem_eoc_logit_bias]="tokmem_eoc_logit_bias"
    [adap_tokmem]="adap_tokmem"
    [adap_tokmem_eoc]="adap_tokmem_eoc"
    [adap_tokmem_eoc_logit_bias]="adap_tokmem_eoc_logit_bias"
)

declare -A TOKMEM_BATCH_SIZES=(
    [llama1b]=24
    [llama3b]=16
    [llama8b]=8
)

declare -A TOKMEM_EVAL_BATCH_SIZES=(
    [llama1b]=256
    [llama3b]=192
    [llama8b]=64
)

declare -A TOKMEM_10CALL_BATCH_SIZES=(
    [llama1b]=4
    [llama3b]=2
    [llama8b]=1
)

declare -A TOKMEM_10CALL_EVAL_BATCH_SIZES=(
    [llama1b]=16
    [llama3b]=8
    [llama8b]=4
)

declare -A LORA_BATCH_SIZES=(
    [llama1b]=16
    [llama3b]=8
    [llama8b]=4
)

declare -A LORA_EVAL_BATCH_SIZES=(
    [llama1b]=128
    [llama3b]=96
    [llama8b]=32
)

declare -A ADAP_BATCH_SIZES=(
    [llama1b]="16,24"
    [llama3b]="8,16"
    [llama8b]="4,8"
)

declare -A ADAP_10CALL_BATCH_SIZES=(
    [llama1b]="16,4"
    [llama3b]="8,2"
    [llama8b]="4,1"
)

declare -A ICL_BATCH_SIZES=(
    [llama1b]=64
    [llama3b]=32
    [llama8b]=24
)

declare -A RAG_BATCH_SIZES=(
    [llama1b]=256
    [llama3b]=192
    [llama8b]=128
)

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"

mkdir -p "$SUITE_DIR" "$RUNS_ROOT" "$DATA_DIR" "$HF_CACHE_DIR"
if [[ "$RERUN_FAILED" -eq 1 ]]; then
    mkdir -p "$RERUN_DIR"
    cp "$SCRIPT_PATH" "$RERUN_SCRIPT_SNAPSHOT"
else
    cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"
fi

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TOKENIZERS_PARALLELISM=false

touch "$SCHEDULER_LOG" "$GPU_AVAILABILITY_LOG"

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$SCHEDULER_LOG"
}

write_suite_config() {
    local output_path="$1"
    local metadata_scope="$2"
    local rerun_id="$3"
    python - "$output_path" "$SUITE_NAME" "$GPU_IDS_CSV" "$POLL_SECONDS" "$GPU_MEMORY_LIMIT_MIB" \
        "$GPU_IDLE_REQUIRED_SECONDS" "$RERUN_FAILED" "$metadata_scope" "$rerun_id" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "suite_name": sys.argv[2],
    "gpu_ids": sys.argv[3].split(","),
    "poll_seconds": int(sys.argv[4]),
    "gpu_memory_limit_mib": int(sys.argv[5]),
    "gpu_idle_required_seconds": int(sys.argv[6]),
    "rerun_failed": bool(int(sys.argv[7])),
    "metadata_scope": sys.argv[8],
    "rerun_id": sys.argv[9] or None,
    "dataset": {
        "top_k": "51-100",
        "evaluation_top_k": "51-100",
        "evaluation_scopes": {
            "4calls": {
                "description": "tools 51-100 / 4 calls",
                "train_size": 5000,
                "test_size": 500,
                "train_max_calls": 4,
                "test_max_calls": 4,
                "max_length": 512,
                "train_multi_tool_ratios": "0.5,0.5",
                "test_multi_tool_ratios": "0.5,0.5",
                "methods": [
                    "icl",
                    "rag",
                    "lora",
                    "tokmem",
                    "tokmem_eoc",
                    "tokmem_eoc_logit_bias",
                    "adap_tokmem",
                    "adap_tokmem_eoc",
                    "adap_tokmem_eoc_logit_bias",
                ],
            },
            "10calls": {
                "description": "tools 51-100 / 10 calls",
                "train_size": 8000,
                "test_size": 800,
                "train_max_calls": 10,
                "test_max_calls": 10,
                "max_length": 1024,
                "train_multi_tool_ratios": "0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125",
                "test_multi_tool_ratios": "0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125",
                "methods": [
                    "tokmem",
                    "tokmem_eoc_logit_bias",
                    "adap_tokmem",
                    "adap_tokmem_eoc_logit_bias",
                ],
            },
        },
        "method_training_rounds": {
            "icl": None,
            "rag": None,
            "lora": "51-100:3",
            "tokmem": "51-100:3",
            "tokmem_eoc": "51-100:3",
            "tokmem_eoc_logit_bias": "51-100:3",
            "adap_tokmem": "1-50:1,51-100:3",
            "adap_tokmem_eoc": "1-50:1,51-100:3",
            "adap_tokmem_eoc_logit_bias": "1-50:1,51-100:3",
        },
        "adaptation_scopes": {
            "4calls": {
                "enabled_methods": [
                    "adap_tokmem",
                    "adap_tokmem_eoc",
                    "adap_tokmem_eoc_logit_bias",
                ],
                "pre_adaptation_top_k": "1-50",
                "training_rounds": "1-50:1,51-100:3",
                "train_max_calls_per_round": "4,4",
                "test_max_calls_per_round": "4,4",
            },
            "10calls": {
                "enabled_methods": [
                    "adap_tokmem",
                    "adap_tokmem_eoc_logit_bias",
                ],
                "pre_adaptation_top_k": "1-50",
                "training_rounds": "1-50:1,51-100:3",
                "train_max_calls_per_round": "4,10",
                "test_max_calls_per_round": "4,10",
            },
        },
    },
    "trial_count": 5,
    "trial_seeds": [42, 42, 42, 42, 42],
    "batch_settings": {
        "tokmem_family": {
            "llama1b": {"train_batch_size": 24, "eval_batch_size": 256},
            "llama3b": {"train_batch_size": 16, "eval_batch_size": 192},
            "llama8b": {"train_batch_size": 8, "eval_batch_size": 64}
        },
        "tokmem_family_10calls": {
            "llama1b": {"train_batch_size": 4, "eval_batch_size": 16},
            "llama3b": {"train_batch_size": 2, "eval_batch_size": 8},
            "llama8b": {"train_batch_size": 1, "eval_batch_size": 4}
        },
        "lora": {
            "llama1b": {"train_batch_size": 16, "eval_batch_size": 128},
            "llama3b": {"train_batch_size": 8, "eval_batch_size": 96},
            "llama8b": {"train_batch_size": 4, "eval_batch_size": 32}
        },
        "adaptation_tokmem_family": {
            "llama1b": {"batch_size_per_round": "16,24", "eval_batch_size": 256},
            "llama3b": {"batch_size_per_round": "8,16", "eval_batch_size": 192},
            "llama8b": {"batch_size_per_round": "4,8", "eval_batch_size": 64}
        },
        "adaptation_tokmem_family_10calls": {
            "llama1b": {"batch_size_per_round": "16,4", "eval_batch_size": 16},
            "llama3b": {"batch_size_per_round": "8,2", "eval_batch_size": 8},
            "llama8b": {"batch_size_per_round": "4,1", "eval_batch_size": 4}
        },
        "icl": {
            "llama1b": {"batch_size": 64},
            "llama3b": {"batch_size": 32},
            "llama8b": {"batch_size": 24}
        },
        "rag": {
            "llama1b": {"batch_size": 256},
            "llama3b": {"batch_size": 192},
            "llama8b": {"batch_size": 128}
        },
    },
    "models": {
        "llama1b": "models/Llama-3.2-1B-Instruct",
        "llama3b": "models/Llama-3.2-3B-Instruct",
        "llama8b": "models/Llama-3.1-8B-Instruct",
    },
    "methods": [
        "icl",
        "rag",
        "lora",
        "tokmem",
        "tokmem_eoc",
        "tokmem_eoc_logit_bias",
        "adap_tokmem",
        "adap_tokmem_eoc",
        "adap_tokmem_eoc_logit_bias",
    ],
}
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

generate_dataset() {
    local pretrain_train_file="$DATA_DIR/training/function_calling_train_tools1-50_4calls.json"
    local pretrain_test_file="$DATA_DIR/test/function_calling_test_tools1-50_4calls.json"
    local pretrain_tool_file="$DATA_DIR/tool_descriptions_tools1-50.json"
    local train_file_4calls="$DATA_DIR/training/function_calling_train_tools51-100_4calls.json"
    local test_file_4calls="$DATA_DIR/test/function_calling_test_tools51-100_4calls.json"
    local train_file_10calls="$DATA_DIR/training/function_calling_train_tools51-100_10calls.json"
    local test_file_10calls="$DATA_DIR/test/function_calling_test_tools51-100_10calls.json"
    local tool_file="$DATA_DIR/tool_descriptions_tools51-100.json"
    if [[ -f "$pretrain_train_file" && -f "$pretrain_test_file" && -f "$pretrain_tool_file" \
        && -f "$train_file_4calls" && -f "$test_file_4calls" \
        && -f "$train_file_10calls" && -f "$test_file_10calls" && -f "$tool_file" ]]; then
        log "Dataset already exists under $DATA_DIR"
        return
    fi

    log "Generating compositional dataset under $DATA_DIR"
    (
        cd "$ROOT_DIR/compositional"
        python xlam_datasets.py \
            --top_k "1-50" \
            --max_samples_per_tool "$MAX_SAMPLES_PER_TOOL" \
            --train_size "${TRAIN_SIZE_BY_SCOPE[4calls]}" \
            --test_size "${TEST_SIZE_BY_SCOPE[4calls]}" \
            --train_max_function_calls "${TRAIN_MAX_CALLS_BY_SCOPE[4calls]}" \
            --test_max_function_calls "${TEST_MAX_CALLS_BY_SCOPE[4calls]}" \
            --train_multi_tool_ratios "${TRAIN_MULTI_TOOL_RATIOS_BY_SCOPE[4calls]}" \
            --test_multi_tool_ratios "${TEST_MULTI_TOOL_RATIOS_BY_SCOPE[4calls]}" \
            --output_dir "$DATA_DIR"
        for call_scope in "${CALL_SCOPES[@]}"; do
            python xlam_datasets.py \
                --top_k "$TOP_K" \
                --max_samples_per_tool "$MAX_SAMPLES_PER_TOOL" \
                --train_size "${TRAIN_SIZE_BY_SCOPE[$call_scope]}" \
                --test_size "${TEST_SIZE_BY_SCOPE[$call_scope]}" \
                --train_max_function_calls "${TRAIN_MAX_CALLS_BY_SCOPE[$call_scope]}" \
                --test_max_function_calls "${TEST_MAX_CALLS_BY_SCOPE[$call_scope]}" \
                --train_multi_tool_ratios "${TRAIN_MULTI_TOOL_RATIOS_BY_SCOPE[$call_scope]}" \
                --test_multi_tool_ratios "${TEST_MULTI_TOOL_RATIOS_BY_SCOPE[$call_scope]}" \
                --output_dir "$DATA_DIR"
        done
    ) 2>&1 | tee "$DATASET_LOG"
}

build_task_command() {
    local model_key="$1"
    local method="$2"
    local task_name="$3"
    local seed="$4"
    local call_scope="$5"
    local model_path="${MODEL_PATHS[$model_key]}"
    local train_max_calls="${TRAIN_MAX_CALLS_BY_SCOPE[$call_scope]}"
    local test_max_calls="${TEST_MAX_CALLS_BY_SCOPE[$call_scope]}"
    local max_length="${MAX_LENGTH_BY_SCOPE[$call_scope]}"
    local test_data_file="$DATA_DIR/test/function_calling_test_tools51-100_${test_max_calls}calls.json"
    local tokmem_batch_size="${TOKMEM_BATCH_SIZES[$model_key]}"
    local tokmem_eval_batch_size="${TOKMEM_EVAL_BATCH_SIZES[$model_key]}"
    local lora_batch_size="${LORA_BATCH_SIZES[$model_key]}"
    local lora_eval_batch_size="${LORA_EVAL_BATCH_SIZES[$model_key]}"
    local adap_batch_size="${ADAP_BATCH_SIZES[$model_key]}"
    local adap_train_max_calls="4,4"
    local adap_test_max_calls="4,4"
    local icl_batch_size="${ICL_BATCH_SIZES[$model_key]}"
    local rag_batch_size="${RAG_BATCH_SIZES[$model_key]}"

    if [[ "$call_scope" == "10calls" ]]; then
        tokmem_batch_size="${TOKMEM_10CALL_BATCH_SIZES[$model_key]}"
        tokmem_eval_batch_size="${TOKMEM_10CALL_EVAL_BATCH_SIZES[$model_key]}"
        adap_batch_size="${ADAP_10CALL_BATCH_SIZES[$model_key]}"
        adap_train_max_calls="4,10"
        adap_test_max_calls="4,10"
    fi

    case "$method" in
        icl)
            TASK_CMD=(
                python -u icl_baseline.py
                --test_data "$test_data_file"
                --tool_descriptions "$DATA_DIR/tool_descriptions_tools51-100.json"
                --model_name "$model_path"
                --batch_size "$icl_batch_size"
                --seed "$seed"
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_name"
                --run_tag "${model_key}_${method}"
            )
            ;;
        rag)
            TASK_CMD=(
                python -u icl_baseline.py
                --test_data "$test_data_file"
                --tool_descriptions "$DATA_DIR/tool_descriptions_tools51-100.json"
                --model_name "$model_path"
                --retriever_model_name "$ROOT_DIR/models/all-MiniLM-L6-v2"
                --batch_size "$rag_batch_size"
                --seed "$seed"
                --use_rag
                --retrieval_k 5
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_name"
                --run_tag "${model_key}_${method}"
            )
            ;;
        lora)
            TASK_CMD=(
                python -u lora_sequential.py
                --training_rounds "$LORA_TRAINING_ROUNDS"
                --batch_size "$lora_batch_size"
                --train_max_function_calls "$train_max_calls"
                --test_max_function_calls "$test_max_calls"
                --model_name "$model_path"
                --lora_r 8
                --lora_alpha 32
                --lora_dropout 0.1
                --lora_target_modules "q_proj,v_proj"
                --eval_after_each_round
                --save_checkpoints
                --data_dir "$DATA_DIR"
                --lr "$LORA_LR"
                --eval_batch_size "$lora_eval_batch_size"
                --max_length "$max_length"
                --seed "$seed"
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_name"
                --run_tag "${model_key}_${method}"
            )
            ;;
        adap_tokmem|adap_tokmem_eoc|adap_tokmem_eoc_logit_bias)
            TASK_CMD=(
                python -u main_sequential.py
                --training_rounds "1-50:1,51-100:3"
                --batch_size_per_round "$adap_batch_size"
                --train_max_function_calls_per_round "$adap_train_max_calls"
                --test_max_function_calls_per_round "$adap_test_max_calls"
                --model_name "$model_path"
                --use_lora
                --lora_r 8
                --lora_alpha 32
                --lora_dropout 0.1
                --lora_target_modules "q_proj,v_proj"
                --freeze_lora_after_first
                --eval_after_each_round
                --save_checkpoints
                --data_dir "$DATA_DIR"
                --lr "$TOKMEM_LR"
                --lora_lr "$LORA_LR"
                --eval_batch_size "$tokmem_eval_batch_size"
                --max_length "$max_length"
                --seed "$seed"
                --tensorboard
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_name"
                --run_tag "${model_key}_${method}"
            )
            if [[ "$method" == "adap_tokmem_eoc" || "$method" == "adap_tokmem_eoc_logit_bias" ]]; then
                TASK_CMD+=(--use_eoc)
            fi
            if [[ "$method" == "adap_tokmem_eoc_logit_bias" ]]; then
                TASK_CMD+=(--use_logit_bias)
            fi
            ;;
        tokmem|tokmem_eoc|tokmem_eoc_logit_bias)
            TASK_CMD=(
                python -u main_sequential.py
                --training_rounds "$TOKMEM_TRAINING_ROUNDS"
                --epochs "$TOKMEM_EPOCHS"
                --batch_size "$tokmem_batch_size"
                --train_max_function_calls "$train_max_calls"
                --test_max_function_calls "$test_max_calls"
                --model_name "$model_path"
                --eval_after_each_round
                --save_checkpoints
                --data_dir "$DATA_DIR"
                --lr "$TOKMEM_LR"
                --eval_batch_size "$tokmem_eval_batch_size"
                --max_length "$max_length"
                --seed "$seed"
                --tensorboard
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_name"
                --run_tag "${model_key}_${method}_${call_scope}"
            )
            if [[ "$method" == "tokmem_eoc" || "$method" == "tokmem_eoc_logit_bias" ]]; then
                TASK_CMD+=(--use_eoc)
            fi
            if [[ "$method" == "tokmem_eoc_logit_bias" ]]; then
                TASK_CMD+=(--use_logit_bias)
            fi
            ;;
        *)
            echo "Unknown method: $method" >&2
            exit 1
            ;;
    esac
}

command_string() {
    printf '%q ' "${TASK_CMD[@]}"
}

is_training_method() {
    local method="$1"
    case "$method" in
        lora|tokmem|tokmem_eoc|tokmem_eoc_logit_bias|adap_tokmem|adap_tokmem_eoc|adap_tokmem_eoc_logit_bias)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

task_success_marker() {
    echo "$1/SUCCESS"
}

task_failed_marker() {
    echo "$1/FAILED"
}

task_status_file() {
    echo "$1/task_status.json"
}

task_should_run() {
    local task_dir="$1"
    if [[ -f "$(task_success_marker "$task_dir")" ]]; then
        return 1
    fi
    if [[ "$RERUN_FAILED" -eq 0 && -f "$(task_failed_marker "$task_dir")" ]]; then
        return 1
    fi
    return 0
}

declare -a TASK_MODEL=()
declare -a TASK_CALL_SCOPE=()
declare -a TASK_METHOD=()
declare -a TASK_TRIAL=()
declare -a TASK_SEED=()
declare -a TASK_NAME=()
declare -a TASK_DIR=()
declare -a TASK_IS_TRAINING=()
declare -a TASK_PENDING_INDEXES=()

reset_task_inventory() {
    TASK_MODEL=()
    TASK_CALL_SCOPE=()
    TASK_METHOD=()
    TASK_TRIAL=()
    TASK_SEED=()
    TASK_NAME=()
    TASK_DIR=()
    TASK_IS_TRAINING=()
    TASK_PENDING_INDEXES=()
}

append_task_record() {
    local task_id="$1"
    local call_scope="$2"
    local model_key="$3"
    local method="$4"
    local trial="$5"
    local seed="$6"
    local training_flag="$7"
    local task_name="$8"
    local task_dir="$9"
    local task_index="${#TASK_NAME[@]}"

    TASK_MODEL+=("$model_key")
    TASK_CALL_SCOPE+=("$call_scope")
    TASK_METHOD+=("$method")
    TASK_TRIAL+=("$trial")
    TASK_SEED+=("$seed")
    TASK_NAME+=("$task_name")
    TASK_DIR+=("$task_dir")
    TASK_IS_TRAINING+=("$training_flag")

    if task_should_run "$task_dir"; then
        TASK_PENDING_INDEXES+=("$task_index")
    fi
}

task_name_for_scope() {
    local call_scope="$1"
    local model_key="$2"
    local method="$3"
    local trial="$4"
    local seed="$5"

    if [[ "$call_scope" == "4calls" ]]; then
        echo "${model_key}_${method}_trial${trial}_seed${seed}"
    else
        echo "${model_key}_${method}_${call_scope}_trial${trial}_seed${seed}"
    fi
}

migrate_manifest_call_scope_column() {
    local manifest_path="$1"
    set +e
    python - "$manifest_path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open("r", encoding="utf-8"), delimiter="\t"))
if not rows:
    return_code = 0
elif "call_scope" in rows[0]:
    return_code = 0
else:
    fieldnames = [
        "task_id",
        "call_scope",
        "model",
        "method",
        "trial",
        "seed",
        "is_training",
        "task_name",
        "task_dir",
    ]
    migrated_rows = []
    for row in rows:
        migrated_rows.append(
            {
                "task_id": row["task_id"],
                "call_scope": "4calls",
                "model": row["model"],
                "method": row["method"],
                "trial": row["trial"],
                "seed": row["seed"],
                "is_training": row["is_training"],
                "task_name": row["task_name"],
                "task_dir": row["task_dir"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(migrated_rows)
    return_code = 1
sys.exit(return_code)
PY
    local status=$?
    set -e
    if [[ "$status" -eq 1 ]]; then
        log "Migrated existing task manifest with call_scope=4calls: $manifest_path"
    elif [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
}

build_and_write_task_manifest() {
    local manifest_path="$1"
    printf "task_id\tcall_scope\tmodel\tmethod\ttrial\tseed\tis_training\ttask_name\ttask_dir\n" > "$manifest_path"

    local call_scope trial model_key method seed task_name task_dir training_flag task_id
    task_id=0
    for call_scope in "${CALL_SCOPES[@]}"; do
        for trial in $(seq 1 "$TRIAL_COUNT"); do
            seed="${TRIAL_SEEDS[$((trial - 1))]}"
            for model_key in "${MODEL_KEYS[@]}"; do
                for method in ${METHODS_BY_SCOPE[$call_scope]}; do
                    task_name="$(task_name_for_scope "$call_scope" "$model_key" "$method" "$trial" "$seed")"
                    task_dir="$RUNS_ROOT/$task_name"
                    training_flag=0
                    if is_training_method "$method"; then
                        training_flag=1
                    fi

                    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        "$task_id" "$call_scope" "$model_key" "$method" "$trial" "$seed" "$training_flag" "$task_name" "$task_dir" \
                        >> "$manifest_path"

                    task_id=$((task_id + 1))
                done
            done
        done
    done
}

append_missing_tasks_to_manifest() {
    local manifest_path="$1"
    local task_id call_scope model_key method trial seed training_flag task_name task_dir
    local max_task_id=-1
    local next_task_id
    local appended_count=0
    declare -A existing_task_names=()

    while IFS=$'\t' read -r task_id call_scope model_key method trial seed training_flag task_name task_dir; do
        if [[ "$task_id" == "task_id" || -z "$task_id" ]]; then
            continue
        fi
        if [[ -n "${MODEL_PATHS[$call_scope]+x}" ]]; then
            task_dir="$task_name"
            task_name="$training_flag"
            training_flag="$seed"
            seed="$trial"
            trial="$method"
            method="$model_key"
            model_key="$call_scope"
            call_scope="4calls"
        fi
        existing_task_names["$task_name"]=1
        if (( task_id > max_task_id )); then
            max_task_id=$task_id
        fi
    done < "$manifest_path"

    next_task_id=$((max_task_id + 1))

    for call_scope in "${CALL_SCOPES[@]}"; do
        for trial in $(seq 1 "$TRIAL_COUNT"); do
            seed="${TRIAL_SEEDS[$((trial - 1))]}"
            for model_key in "${MODEL_KEYS[@]}"; do
                for method in ${METHODS_BY_SCOPE[$call_scope]}; do
                    task_name="$(task_name_for_scope "$call_scope" "$model_key" "$method" "$trial" "$seed")"
                    if [[ -n "${existing_task_names[$task_name]+x}" ]]; then
                        continue
                    fi

                    task_dir="$RUNS_ROOT/$task_name"
                    training_flag=0
                    if is_training_method "$method"; then
                        training_flag=1
                    fi

                    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        "$next_task_id" "$call_scope" "$model_key" "$method" "$trial" "$seed" "$training_flag" "$task_name" "$task_dir" \
                        >> "$manifest_path"
                    existing_task_names["$task_name"]=1
                    next_task_id=$((next_task_id + 1))
                    appended_count=$((appended_count + 1))
                done
            done
        done
    done

    log "Appended $appended_count missing task(s) into existing manifest: $manifest_path"
}

load_task_manifest() {
    local manifest_path="$1"
    local task_id call_scope model_key method trial seed training_flag task_name task_dir

    while IFS=$'\t' read -r task_id call_scope model_key method trial seed training_flag task_name task_dir; do
        if [[ "$task_id" == "task_id" ]]; then
            continue
        fi
        if [[ -n "${MODEL_PATHS[$call_scope]+x}" ]]; then
            task_dir="$task_name"
            task_name="$training_flag"
            training_flag="$seed"
            seed="$trial"
            trial="$method"
            method="$model_key"
            model_key="$call_scope"
            call_scope="4calls"
        fi
        append_task_record \
            "$task_id" \
            "$call_scope" \
            "$model_key" \
            "$method" \
            "$trial" \
            "$seed" \
            "$training_flag" \
            "$task_name" \
            "$task_dir"
    done < "$manifest_path"
}

prepare_task_manifest() {
    reset_task_inventory

    if [[ "$RERUN_FAILED" -eq 1 && -f "$MANIFEST_FILE" ]]; then
        log "Reusing existing task manifest for --rerun-failed: $MANIFEST_FILE"
        cp "$MANIFEST_FILE" "$RERUN_MANIFEST_SNAPSHOT"
        load_task_manifest "$MANIFEST_FILE"
        return
    fi

    if [[ -f "$MANIFEST_FILE" ]]; then
        log "Reconciling existing task manifest: $MANIFEST_FILE"
        migrate_manifest_call_scope_column "$MANIFEST_FILE"
        append_missing_tasks_to_manifest "$MANIFEST_FILE"
    else
        log "Writing task manifest: $MANIFEST_FILE"
        build_and_write_task_manifest "$MANIFEST_FILE"
    fi

    if [[ "$RERUN_FAILED" -eq 1 ]]; then
        cp "$MANIFEST_FILE" "$RERUN_MANIFEST_SNAPSHOT"
    fi

    load_task_manifest "$MANIFEST_FILE"
}

write_task_status_json() {
    local status_path="$1"
    local task_name="$2"
    local call_scope="$3"
    local model_key="$4"
    local method="$5"
    local trial="$6"
    local seed="$7"
    local gpu_id="$8"
    local started_at="$9"
    local finished_at="${10}"
    local duration_seconds="${11}"
    local exit_code="${12}"
    local status="${13}"
    local task_dir="${14}"
    local is_training="${15}"
    python - "$status_path" "$task_name" "$call_scope" "$model_key" "$method" "$trial" "$seed" "$gpu_id" \
        "$started_at" "$finished_at" "$duration_seconds" "$exit_code" "$status" "$task_dir" "$is_training" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
task_dir = Path(sys.argv[14])
payload = {
    "task_name": sys.argv[2],
    "call_scope": sys.argv[3],
    "model": sys.argv[4],
    "method": sys.argv[5],
    "trial": int(sys.argv[6]),
    "seed": int(sys.argv[7]),
    "gpu_id": sys.argv[8],
    "started_at": sys.argv[9],
    "finished_at": sys.argv[10],
    "duration_seconds": int(sys.argv[11]),
    "exit_code": int(sys.argv[12]),
    "status": sys.argv[13],
    "task_dir": str(task_dir),
    "is_training_method": bool(int(sys.argv[15])),
    "run_config": str(task_dir / "run_config.json"),
    "evaluation_results": str(task_dir / "evaluation_results.json"),
    "training_summary": str(task_dir / "training_summary.json"),
    "stdout_log": str(task_dir / "stdout.log"),
    "evaluation_log": str(task_dir / "evaluation.log"),
}
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

launch_task() {
    local task_index="$1"
    local gpu_id="$2"
    local model_key="${TASK_MODEL[$task_index]}"
    local call_scope="${TASK_CALL_SCOPE[$task_index]}"
    local method="${TASK_METHOD[$task_index]}"
    local trial="${TASK_TRIAL[$task_index]}"
    local seed="${TASK_SEED[$task_index]}"
    local task_name="${TASK_NAME[$task_index]}"
    local task_dir="${TASK_DIR[$task_index]}"
    local training_flag="${TASK_IS_TRAINING[$task_index]}"

    mkdir -p "$task_dir"
    build_task_command "$model_key" "$method" "$task_name" "$seed" "$call_scope"
    cat > "$task_dir/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
export CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu_id")
export HF_HOME=$(printf '%q' "$HF_CACHE_DIR")
export HF_DATASETS_CACHE=$(printf '%q' "$HF_CACHE_DIR/datasets")
export HUGGINGFACE_HUB_CACHE=$(printf '%q' "$HF_CACHE_DIR/hub")
export TOKENIZERS_PARALLELISM=false
cd $(printf '%q' "$ROOT_DIR/compositional")
exec $(command_string)
EOF
    chmod +x "$task_dir/command.sh"

    log "Launching task=$task_name gpu=$gpu_id scope=$call_scope method=$method model=$model_key trial=$trial"

    (
        set +e
        local_start_epoch="$(date +%s)"
        local_start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        rm -f "$(task_success_marker "$task_dir")" "$(task_failed_marker "$task_dir")"

        bash "$task_dir/command.sh" > "$task_dir/stdout.log" 2>&1
        local_exit_code=$?

        local_end_epoch="$(date +%s)"
        local_end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        local_duration=$((local_end_epoch - local_start_epoch))
        local_status="failed"
        if [[ "$local_exit_code" -eq 0 ]]; then
            local_status="success"
            touch "$(task_success_marker "$task_dir")"
            rm -f "$(task_failed_marker "$task_dir")"
        else
            touch "$(task_failed_marker "$task_dir")"
            rm -f "$(task_success_marker "$task_dir")"
        fi

        write_task_status_json \
            "$(task_status_file "$task_dir")" \
            "$task_name" \
            "$call_scope" \
            "$model_key" \
            "$method" \
            "$trial" \
            "$seed" \
            "$gpu_id" \
            "$local_start_iso" \
            "$local_end_iso" \
            "$local_duration" \
            "$local_exit_code" \
            "$local_status" \
            "$task_dir" \
            "$training_flag"

        exit "$local_exit_code"
    ) &

    unset GPU_IDLE_SINCE["$gpu_id"]
    unset GPU_EXTERNAL_COOLDOWN_REQUIRED["$gpu_id"]
    GPU_PID["$gpu_id"]=$!
    GPU_TASK_INDEX["$gpu_id"]=$task_index
}

harvest_finished_tasks() {
    local gpu_id pid task_index task_name exit_code harvested_any
    harvested_any=0
    for gpu_id in "${GPU_IDS[@]}"; do
        pid="${GPU_PID[$gpu_id]-}"
        if [[ -z "${pid}" ]]; then
            continue
        fi

        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        task_index="${GPU_TASK_INDEX[$gpu_id]}"
        task_name="${TASK_NAME[$task_index]}"

        set +e
        wait "$pid"
        exit_code=$?
        set -e

        if [[ "$exit_code" -eq 0 ]]; then
            log "Completed task=$task_name gpu=$gpu_id status=success"
        else
            log "Completed task=$task_name gpu=$gpu_id status=failed exit_code=$exit_code"
        fi

        unset GPU_PID["$gpu_id"]
        unset GPU_TASK_INDEX["$gpu_id"]
        harvested_any=1
    done

    if [[ "$harvested_any" -eq 1 ]]; then
        write_suite_status_json
    fi
}

all_gpus_idle() {
    local gpu_id
    for gpu_id in "${GPU_IDS[@]}"; do
        if [[ -n "${GPU_PID[$gpu_id]-}" ]]; then
            return 1
        fi
    done
    return 0
}

gpu_memory_used_mib() {
    local gpu_id="$1"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null \
        | awk 'NR == 1 {gsub(/^[ \t]+|[ \t]+$/, "", $0); print $0}'
}

record_gpu_availability() {
    local now_epoch="$1"
    local now_iso gpu_id memory_used idle_since idle_seconds state cooldown_required
    now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    for gpu_id in "${GPU_IDS[@]}"; do
        if [[ -n "${GPU_PID[$gpu_id]-}" ]]; then
            unset GPU_IDLE_SINCE["$gpu_id"]
            printf '[%s]\tgpu=%s\tstate=suite_busy\tmemory_mib=unknown\tidle_seconds=0\n' \
                "$now_iso" "$gpu_id" >> "$GPU_AVAILABILITY_LOG"
            continue
        fi

        if memory_used="$(gpu_memory_used_mib "$gpu_id")" && [[ "$memory_used" =~ ^[0-9]+$ ]]; then
            if (( memory_used <= GPU_MEMORY_LIMIT_MIB )); then
                cooldown_required="${GPU_EXTERNAL_COOLDOWN_REQUIRED[$gpu_id]-0}"
                if [[ "$cooldown_required" == "1" ]]; then
                    if [[ -z "${GPU_IDLE_SINCE[$gpu_id]-}" ]]; then
                        GPU_IDLE_SINCE["$gpu_id"]="$now_epoch"
                        log "GPU external availability window started gpu=$gpu_id memory_mib=$memory_used limit_mib=$GPU_MEMORY_LIMIT_MIB required_seconds=$GPU_IDLE_REQUIRED_SECONDS"
                    fi
                    idle_since="${GPU_IDLE_SINCE[$gpu_id]}"
                    idle_seconds=$((now_epoch - idle_since))
                    state="external_cooldown"
                    if (( idle_seconds >= GPU_IDLE_REQUIRED_SECONDS )); then
                        state="ready"
                    fi
                else
                    if [[ -z "${GPU_IDLE_SINCE[$gpu_id]-}" ]]; then
                        GPU_IDLE_SINCE["$gpu_id"]=$((now_epoch - GPU_IDLE_REQUIRED_SECONDS))
                        log "GPU available for immediate suite launch gpu=$gpu_id memory_mib=$memory_used limit_mib=$GPU_MEMORY_LIMIT_MIB"
                    fi
                    idle_since="${GPU_IDLE_SINCE[$gpu_id]}"
                    idle_seconds=$((now_epoch - idle_since))
                    state="ready"
                fi
            else
                if [[ -n "${GPU_IDLE_SINCE[$gpu_id]-}" ]]; then
                    log "GPU availability window reset by external occupancy gpu=$gpu_id memory_mib=$memory_used limit_mib=$GPU_MEMORY_LIMIT_MIB"
                fi
                GPU_EXTERNAL_COOLDOWN_REQUIRED["$gpu_id"]=1
                unset GPU_IDLE_SINCE["$gpu_id"]
                idle_seconds=0
                state="busy"
            fi
            printf '[%s]\tgpu=%s\tstate=%s\tmemory_mib=%s\tidle_seconds=%s\n' \
                "$now_iso" "$gpu_id" "$state" "$memory_used" "$idle_seconds" >> "$GPU_AVAILABILITY_LOG"
        else
            if [[ -n "${GPU_IDLE_SINCE[$gpu_id]-}" ]]; then
                log "GPU availability window reset gpu=$gpu_id reason=nvidia-smi-query-failed"
            fi
            unset GPU_IDLE_SINCE["$gpu_id"]
            printf '[%s]\tgpu=%s\tstate=query_failed\tmemory_mib=unknown\tidle_seconds=0\n' \
                "$now_iso" "$gpu_id" >> "$GPU_AVAILABILITY_LOG"
        fi
    done
}

gpu_ready_for_launch() {
    local gpu_id="$1"
    local now_epoch="$2"
    local idle_since="${GPU_IDLE_SINCE[$gpu_id]-}"
    local cooldown_required="${GPU_EXTERNAL_COOLDOWN_REQUIRED[$gpu_id]-0}"

    if [[ -n "${GPU_PID[$gpu_id]-}" || -z "$idle_since" ]]; then
        return 1
    fi

    if [[ "$cooldown_required" != "1" ]]; then
        return 0
    fi

    if (( now_epoch - idle_since >= GPU_IDLE_REQUIRED_SECONDS )); then
        return 0
    fi

    return 1
}

write_suite_status_json() {
    python - "$MANIFEST_FILE" "$STATUS_FILE" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])

rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
tasks = []
for row in rows:
    task_dir = Path(row["task_dir"])
    task_status_path = task_dir / "task_status.json"
    if task_status_path.exists():
        tasks.append(json.loads(task_status_path.read_text(encoding="utf-8")))
        continue

    fallback_status = "pending"
    if (task_dir / "SUCCESS").exists():
        fallback_status = "success"
    elif (task_dir / "FAILED").exists():
        fallback_status = "failed"

    tasks.append(
        {
            "task_name": row["task_name"],
            "call_scope": row.get("call_scope") or "4calls",
            "model": row["model"],
            "method": row["method"],
            "trial": int(row["trial"]),
            "seed": int(row["seed"]),
            "status": fallback_status,
            "task_dir": row["task_dir"],
            "is_training_method": bool(int(row["is_training"])),
            "evaluation_results": str(task_dir / "evaluation_results.json"),
            "training_summary": str(task_dir / "training_summary.json"),
            "stdout_log": str(task_dir / "stdout.log"),
        }
    )

status_path.write_text(json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

write_suite_summary() {
    python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$SUMMARY_JSON_FILE" "$STATUS_FILE" "$SUITE_NAME" "$TRIAL_COUNT" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
summary_json_path = Path(sys.argv[3])
status_path = Path(sys.argv[4])
suite_name = sys.argv[5]
trial_count = int(sys.argv[6])

metric_fields = (
    ("tool_accuracy", "Tool Acc"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("tool_exact_match_acc", "Tool Exact Match Acc"),
    ("exact_accuracy", "Exact Match Acc"),
    ("parse_error_rate", "Parse Error Rate"),
)

training_methods = {
    "lora",
    "tokmem",
    "tokmem_eoc",
    "tokmem_eoc_logit_bias",
    "adap_tokmem",
    "adap_tokmem_eoc",
    "adap_tokmem_eoc_logit_bias",
}

rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
task_status = json.loads(status_path.read_text(encoding="utf-8"))
status_by_task = {item["task_name"]: item for item in task_status["tasks"]}


def fmt(value):
    if value is None:
        return ""
    return f"{value:.3f}"


def fmt_seconds(seconds):
    if seconds is None:
        return ""
    hours = seconds / 3600
    return f"{hours:.2f}h"


def mean_or_none(values):
    return statistics.mean(values) if values else None


def load_eval_metrics(path: Path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("rounds"):
        last_round = payload["rounds"][-1]
        metrics = last_round.get("eval_results") or {}
    elif isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
    else:
        metrics = payload

    if not metrics:
        return None

    exact_accuracy = metrics.get("exact_accuracy", metrics.get("exact_match_accuracy"))
    tool_f1 = metrics.get("avg_tool_f1_score", metrics.get("tool_selection_f1"))
    arguments_f1 = metrics.get("avg_f1_score", metrics.get("average_f1_score"))
    return {
        "tool_accuracy": metrics.get("tool_accuracy"),
        "avg_tool_f1_score": tool_f1,
        "avg_f1_score": arguments_f1,
        "tool_exact_match_acc": metrics.get("tool_exact_match_acc"),
        "exact_accuracy": exact_accuracy,
        "parse_error_rate": metrics.get("parse_error_rate"),
    }


call_scope_labels = {
    "4calls": "tools 51-100 / 4 calls",
    "10calls": "tools 51-100 / 10 calls",
}
call_scope_order = ["4calls", "10calls"]

grouped = defaultdict(list)
for row in rows:
    call_scope = row.get("call_scope") or "4calls"
    grouped[(call_scope, row["model"], row["method"])].append(row)

summary_lines = [
    f"# Compositional Paper Suite Summary",
    "",
    f"- suite: `{suite_name}`",
    f"- trials per model/method: `{trial_count}`",
    "- evaluation scopes: `tools 51-100 / 4 calls`, `tools 51-100 / 10 calls`",
    "- max length by scope: `4calls=512`, `10calls=1024`",
    "- 10-call results are summarized in separate tables because they use a different synthesized dataset.",
    "- models: `llama1b`, `llama3b`, `llama8b`",
    "- methods: `icl`, `rag`, `lora`, `tokmem`, `tokmem_eoc`, `tokmem_eoc_logit_bias`, `adap_tokmem`, `adap_tokmem_eoc`, `adap_tokmem_eoc_logit_bias`",
    "- adaptation 4calls scope: `adap_tokmem* uses 1-50:1,51-100:3 with 4 calls in both rounds`",
    "- adaptation 10calls scope: `adap_tokmem` and `adap_tokmem_eoc_logit_bias` use `1-50:1,51-100:3` with `4,10` calls by round",
    "",
    "## Batch Settings",
    "",
    "- `tokmem*`: `llama1b train/eval=24/256`, `llama3b train/eval=16/192`, `llama8b train/eval=8/64`",
    "- `tokmem* 10calls`: `llama1b train/eval=4/16`, `llama3b train/eval=2/8`, `llama8b train/eval=1/4`",
    "- `adap_tokmem*`: `llama1b train=16,24 eval=256`, `llama3b train=8,16 eval=192`, `llama8b train=4,8 eval=64`, rounds=`1-50:1,51-100:3`",
    "- `adap_tokmem* 10calls`: `llama1b train=16,4 eval=16`, `llama3b train=8,2 eval=8`, `llama8b train=4,1 eval=4`, rounds=`1-50:1,51-100:3`",
    "- `lora`: `llama1b train/eval=16/128`, `llama3b train/eval=8/96`, `llama8b train/eval=4/32`",
    "- `icl`: `llama1b=64`, `llama3b=32`, `llama8b=24`",
    "- `rag`: `llama1b=256`, `llama3b=192`, `llama8b=128`",
]

failed_tasks = []
incomplete_groups = []
training_duration_rows = []
successful_training_trials = []
mean_results = []

for call_scope in call_scope_order:
    scope_keys = sorted(key for key in grouped if key[0] == call_scope)
    if not scope_keys:
        continue

    summary_lines.extend(
        [
            "",
            f"## Mean Results - {call_scope_labels.get(call_scope, call_scope)}",
            "",
            "| Model | Method | Success | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate | Avg Runtime |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, model, method in scope_keys:
        rows_for_group = sorted(grouped[(call_scope, model, method)], key=lambda item: int(item["trial"]))
        metrics_by_field = defaultdict(list)
        runtimes = []
        success_count = 0

        for row in rows_for_group:
            status = status_by_task.get(row["task_name"], {})
            if status.get("status") != "success":
                failed_tasks.append(
                    {
                        "task_name": row["task_name"],
                        "call_scope": call_scope,
                        "model": model,
                        "method": method,
                        "trial": int(row["trial"]),
                        "status": status.get("status", "pending"),
                        "exit_code": status.get("exit_code"),
                        "stdout_log": status.get("stdout_log", str(Path(row["task_dir"]) / "stdout.log")),
                    }
                )
                continue

            success_count += 1
            runtime = status.get("duration_seconds")
            if runtime is not None:
                runtimes.append(runtime)

            eval_metrics = load_eval_metrics(Path(status["evaluation_results"]))
            if eval_metrics is None:
                continue

            for key, _ in metric_fields:
                value = eval_metrics.get(key)
                if value is not None:
                    metrics_by_field[key].append(value)

        if success_count < trial_count:
            incomplete_groups.append(
                {
                    "call_scope": call_scope,
                    "model": model,
                    "method": method,
                    "success_count": success_count,
                    "trial_count": trial_count,
                }
            )

        metric_means = {}
        if success_count == trial_count:
            for key, _ in metric_fields:
                metric_means[key] = mean_or_none(metrics_by_field[key])
        else:
            for key, _ in metric_fields:
                metric_means[key] = None

        avg_runtime = mean_or_none(runtimes)
        mean_results.append(
            {
                "call_scope": call_scope,
                "model": model,
                "method": method,
                "success_count": success_count,
                "trial_count": trial_count,
                "metrics": metric_means,
                "avg_runtime_seconds": avg_runtime,
            }
        )
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    method,
                    f"{success_count}/{trial_count}",
                    fmt(metric_means["tool_accuracy"]),
                    fmt(metric_means["avg_tool_f1_score"]),
                    fmt(metric_means["avg_f1_score"]),
                    fmt(metric_means["tool_exact_match_acc"]),
                    fmt(metric_means["exact_accuracy"]),
                    fmt(metric_means["parse_error_rate"]),
                    fmt_seconds(avg_runtime),
                ]
            )
            + " |"
        )

        if method in training_methods and runtimes:
            training_duration_rows.append(
                {
                    "call_scope": call_scope,
                    "model": model,
                    "method": method,
                    "avg_runtime_seconds": avg_runtime,
                    "success_count": success_count,
                }
            )

            for row in rows_for_group:
                status = status_by_task.get(row["task_name"], {})
                runtime = status.get("duration_seconds")
                if status.get("status") == "success" and runtime is not None:
                    successful_training_trials.append(
                        {
                            "task_name": row["task_name"],
                            "call_scope": call_scope,
                            "model": model,
                            "method": method,
                            "trial": int(row["trial"]),
                            "duration_seconds": runtime,
                            "stdout_log": status.get("stdout_log", str(Path(row["task_dir"]) / "stdout.log")),
                        }
                    )

summary_lines.extend(["", "## Incomplete Or Failed Experiment Groups", ""])
if incomplete_groups:
    summary_lines.append("| Scope | Model | Method | Successful Trials |")
    summary_lines.append("| --- | --- | --- | ---: |")
    for item in incomplete_groups:
        summary_lines.append(
            f"| {call_scope_labels.get(item['call_scope'], item['call_scope'])} | {item['model']} | {item['method']} | {item['success_count']}/{item['trial_count']} |"
        )
else:
    summary_lines.append("All model/method groups finished all trials successfully.")

summary_lines.extend(["", "## Failed Or Unfinished Trials", ""])
if failed_tasks:
    summary_lines.append("| Task | Scope | Model | Method | Trial | Status | Exit Code | Stdout Log |")
    summary_lines.append("| --- | --- | --- | --- | ---: | --- | ---: | --- |")
    for item in failed_tasks:
        exit_code = "" if item["exit_code"] is None else str(item["exit_code"])
        summary_lines.append(
            f"| {item['task_name']} | {call_scope_labels.get(item['call_scope'], item['call_scope'])} | {item['model']} | {item['method']} | {item['trial']} | "
            f"{item['status']} | {exit_code} | {item['stdout_log']} |"
        )
else:
    summary_lines.append("All trials finished successfully.")

summary_lines.extend(["", "## Training Runtime Ranking", ""])
if training_duration_rows:
    for call_scope in call_scope_order:
        rows_for_scope = [item for item in training_duration_rows if item["call_scope"] == call_scope]
        if not rows_for_scope:
            continue
        rows_for_scope.sort(key=lambda item: item["avg_runtime_seconds"], reverse=True)
        summary_lines.extend(["", f"### {call_scope_labels.get(call_scope, call_scope)}", ""])
        summary_lines.append("| Rank | Model | Method | Avg Runtime | Successful Trials |")
        summary_lines.append("| ---: | --- | --- | ---: | ---: |")
        for rank, item in enumerate(rows_for_scope, start=1):
            summary_lines.append(
                f"| {rank} | {item['model']} | {item['method']} | {fmt_seconds(item['avg_runtime_seconds'])} | "
                f"{item['success_count']}/{trial_count} |"
            )
else:
    summary_lines.append("No successful training runtimes available.")

long_runtime_trials = []
grouped_training_trials = defaultdict(list)
for item in successful_training_trials:
    grouped_training_trials[(item["call_scope"], item["model"], item["method"])].append(item)

for key, items in grouped_training_trials.items():
    runtimes = [item["duration_seconds"] for item in items]
    median_runtime = statistics.median(runtimes)
    threshold = max(median_runtime * 1.5, median_runtime + 600)
    for item in items:
        if item["duration_seconds"] >= threshold:
            item = dict(item)
            item["group_median_seconds"] = median_runtime
            item["threshold_seconds"] = threshold
            long_runtime_trials.append(item)

summary_lines.extend(["", "## Longer Training Trials", ""])
if long_runtime_trials:
    long_runtime_trials.sort(key=lambda item: item["duration_seconds"], reverse=True)
    summary_lines.append("| Task | Scope | Model | Method | Trial | Runtime | Group Median | Stdout Log |")
    summary_lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for item in long_runtime_trials:
        summary_lines.append(
            f"| {item['task_name']} | {call_scope_labels.get(item['call_scope'], item['call_scope'])} | {item['model']} | {item['method']} | {item['trial']} | "
            f"{fmt_seconds(item['duration_seconds'])} | {fmt_seconds(item['group_median_seconds'])} | "
            f"{item['stdout_log']} |"
        )
else:
    summary_lines.append("No longer-running successful training trials were flagged.")

summary_lines.extend(
    [
        "",
        "## Notes",
        "",
        "- Mean metrics are only reported for groups with `5/5` successful trials.",
        "- Results are grouped by call scope so `4 calls` and `10 calls` datasets are not averaged or ranked together in the mean-result tables.",
        "- Incomplete groups are listed separately and should not be treated as final paper numbers.",
        "- Training runtime ranking is sorted within each call scope by average successful-trial runtime among training methods only.",
        "- Longer training trials are flagged when runtime is at least `max(1.5 x group median, group median + 600s)` within the same call-scope/model/method group.",
    ]
)

summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
summary_json_path.write_text(
    json.dumps(
        {
            "suite_name": suite_name,
            "trial_count": trial_count,
            "mean_results": mean_results,
            "failed_tasks": failed_tasks,
            "incomplete_groups": incomplete_groups,
            "training_runtime_ranking": training_duration_rows,
            "long_runtime_trials": long_runtime_trials,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

declare -A GPU_PID=()
declare -A GPU_TASK_INDEX=()
declare -A GPU_IDLE_SINCE=()
declare -A GPU_EXTERNAL_COOLDOWN_REQUIRED=()

if [[ "$RERUN_FAILED" -eq 1 ]]; then
    if [[ -f "$SUITE_CONFIG_FILE" ]]; then
        write_suite_config "$RERUN_CONFIG_FILE" "rerun" "$RERUN_ID"
    else
        write_suite_config "$SUITE_CONFIG_FILE" "suite" ""
        write_suite_config "$RERUN_CONFIG_FILE" "rerun" "$RERUN_ID"
    fi
else
    write_suite_config "$SUITE_CONFIG_FILE" "suite" ""
fi
generate_dataset
prepare_task_manifest
write_suite_status_json

log "Suite directory: $SUITE_DIR"
if [[ "$RERUN_FAILED" -eq 1 ]]; then
    log "Rerun metadata directory: $RERUN_DIR"
fi
log "Pending tasks: ${#TASK_PENDING_INDEXES[@]}"

pending_cursor=0
while true; do
    harvest_finished_tasks
    now_epoch="$(date +%s)"
    record_gpu_availability "$now_epoch"

    for gpu_id in "${GPU_IDS[@]}"; do
        if [[ -n "${GPU_PID[$gpu_id]-}" ]]; then
            continue
        fi
        if [[ "$pending_cursor" -ge "${#TASK_PENDING_INDEXES[@]}" ]]; then
            continue
        fi
        if ! gpu_ready_for_launch "$gpu_id" "$now_epoch"; then
            continue
        fi

        next_task_index="${TASK_PENDING_INDEXES[$pending_cursor]}"
        pending_cursor=$((pending_cursor + 1))
        launch_task "$next_task_index" "$gpu_id"
    done

    if [[ "$pending_cursor" -ge "${#TASK_PENDING_INDEXES[@]}" ]]; then
        if all_gpus_idle; then
            break
        fi
    fi

    sleep "$POLL_SECONDS"
done

harvest_finished_tasks
write_suite_status_json
write_suite_summary

if python - "$STATUS_FILE" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
payload = json.loads(status_path.read_text(encoding="utf-8"))
failed = [
    item["task_name"]
    for item in payload.get("tasks", [])
    if item.get("status") != "success"
]
if failed:
    print(f"{len(failed)} task(s) did not succeed.", file=sys.stderr)
    sys.exit(1)
PY
then
    log "Suite finished. Summary written to $SUMMARY_FILE"
else
    log "Suite finished with failures. Summary written to $SUMMARY_FILE"
    exit 1
fi
