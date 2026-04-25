#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

SUITE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="paper_atomic_700task_${SUITE_TIMESTAMP}"
GPU_IDS_CSV="0,1,2,3"
POLL_SECONDS=5
GPU_MEMORY_LIMIT_MIB=2048
GPU_IDLE_REQUIRED_SECONDS=300
RERUN_FAILED=0
EXPLICIT_SUITE_NAME=0

usage() {
    cat <<EOF
Usage: bash scripts/atomic/run_paper_atomic_suite.sh [--gpus 0,1,2,3] [--suite-name NAME] [--poll-seconds N] [--rerun-failed] [--num-tasks 700] [--split-cache PATH]

Runs the maintained atomic fixed-split suite for:
- default split: 700 tasks / train 500 / val 10 / test 50 / seed 42
- models: qwen0_5b, llama3b, llama8b
- methods: base, rag, lora, tokmem, tokmem_logit_bias
- 3 trials per model/method, seed fixed to 42 for every trial to match the cached split metadata

Artifacts:
- runs:    results/atomic/<suite-name>/runs/
- summary: results/atomic/<suite-name>/summary.md
- status:  results/atomic/<suite-name>/task_status.json

Notes:
- GPU scheduling only considers the IDs passed by --gpus.
- Suite-owned tasks keep their GPU reserved for the full process lifetime.
- GPUs that are already free, or just finished a suite-owned task, are eligible immediately.
- A GPU that was externally busy is eligible after memory.used stays <= 2048 MiB for 300 consecutive seconds.
- RAG tasks share one suite-level few-shot corpus cache under data/.
- --rerun-failed requires --suite-name <existing-suite>
- the default split cache must exist before launch
- the suite exits nonzero after writing summary/status when any task did not succeed
EOF
}

NUM_TASKS=700
TRAIN_SIZE=500
VAL_SIZE=10
TEST_SIZE=50
SEED=42
MAX_LENGTH=1024
MAX_INSTRUCTION_TOKENS=1024
MAX_NEW_TOKENS=256
TASKS_DIR="$ROOT_DIR/datasets/natural-instructions-2.8/tasks"
SPLIT_CACHE="$ROOT_DIR/atomic/cached_splits/task700-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt"

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
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --split-cache)
            SPLIT_CACHE="$2"
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

SUITE_DIR="$ROOT_DIR/results/atomic/$SUITE_NAME"
RUNS_ROOT="$SUITE_DIR/runs"
DATA_DIR="$SUITE_DIR/data"
MANIFEST_FILE="$SUITE_DIR/task_manifest.tsv"
STATUS_FILE="$SUITE_DIR/task_status.json"
SUMMARY_FILE="$SUITE_DIR/summary.md"
SUMMARY_JSON_FILE="$SUITE_DIR/summary.json"
SCHEDULER_LOG="$SUITE_DIR/scheduler.log"
GPU_AVAILABILITY_LOG="$SUITE_DIR/gpu_availability.log"
SUITE_CONFIG_FILE="$SUITE_DIR/suite_config.json"

if [[ "$RERUN_FAILED" -eq 1 && "$EXPLICIT_SUITE_NAME" -ne 1 ]]; then
    echo "--rerun-failed requires --suite-name <existing-suite>" >&2
    exit 1
fi

if [[ "$RERUN_FAILED" -eq 1 && ! -d "$SUITE_DIR" ]]; then
    echo "Suite directory does not exist for rerun: $SUITE_DIR" >&2
    exit 1
fi

RAG_RETRIEVAL_TOP_K=3
RAG_RETRIEVER_MODEL="$ROOT_DIR/models/all-MiniLM-L6-v2"
RAG_SPLIT_CACHE_STEM="$(basename "$SPLIT_CACHE")"
RAG_SPLIT_CACHE_STEM="${RAG_SPLIT_CACHE_STEM%.*}"
RAG_SPLIT_CACHE_STEM="${RAG_SPLIT_CACHE_STEM//[^A-Za-z0-9_.-]/_}"
RAG_RETRIEVER_STEM="$(basename "$RAG_RETRIEVER_MODEL")"
RAG_RETRIEVER_STEM="${RAG_RETRIEVER_STEM//[^A-Za-z0-9_.-]/_}"
RAG_CORPUS_CACHE_PATH="$DATA_DIR/rag_corpus_${RAG_SPLIT_CACHE_STEM}_${RAG_RETRIEVER_STEM}_top${RAG_RETRIEVAL_TOP_K}.pt"
TOKMEM_EPOCHS=1
TOKMEM_LR="5e-3"
TOKMEM_VALIDATE_EVERY_N_STEPS=1000
LOGIT_BIAS_LOSS_WEIGHT="0.1"
LOGIT_BIAS_NETWORK="linear"
LOGIT_BIAS_SCALE="1.0"
TRIAL_COUNT=3
TRIAL_SEEDS=(42 42 42)

MODEL_KEYS=(qwen0_5b llama3b llama8b)
METHODS=(base rag lora tokmem tokmem_logit_bias)
TRAINING_METHODS=(lora tokmem tokmem_logit_bias)

declare -A MODEL_PATHS=(
    [qwen0_5b]="$ROOT_DIR/models/Qwen2.5-0.5B-Instruct"
    [llama3b]="$ROOT_DIR/models/Llama-3.2-3B-Instruct"
    [llama8b]="$ROOT_DIR/models/Llama-3.1-8B-Instruct"
)

declare -A LORA_TRAIN_BATCH_SIZES=(
    [qwen0_5b]=8
    [llama3b]=2
    [llama8b]=2
)

declare -A LORA_EVAL_BATCH_SIZES=(
    [qwen0_5b]=32
    [llama3b]=16
    [llama8b]=8
)

declare -A LORA_GRADIENT_ACCUMULATION_STEPS=(
    [qwen0_5b]=1
    [llama3b]=2
    [llama8b]=2
)

declare -A TOKMEM_TRAIN_BATCH_SIZES=(
    [qwen0_5b]=16
    [llama3b]=8
    [llama8b]=4
)

declare -A TOKMEM_EVAL_BATCH_SIZES=(
    [qwen0_5b]=64
    [llama3b]=32
    [llama8b]=16
)

declare -A TOKMEM_GRADIENT_ACCUMULATION_STEPS=(
    [qwen0_5b]=1
    [llama3b]=1
    [llama8b]=2
)

declare -A BASE_TEST_BATCH_SIZES=(
    [qwen0_5b]=512
    [llama3b]=256
    [llama8b]=128
)

declare -A RAG_TEST_BATCH_SIZES=(
    [qwen0_5b]=256
    [llama3b]=128
    [llama8b]=64
)

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"

mkdir -p "$SUITE_DIR" "$RUNS_ROOT" "$DATA_DIR"
cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

export TOKENIZERS_PARALLELISM=false

touch "$SCHEDULER_LOG" "$GPU_AVAILABILITY_LOG"

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$SCHEDULER_LOG"
}

write_suite_config() {
    python - "$SUITE_CONFIG_FILE" "$SUITE_NAME" "$GPU_IDS_CSV" "$POLL_SECONDS" "$GPU_MEMORY_LIMIT_MIB" \
        "$GPU_IDLE_REQUIRED_SECONDS" "$RERUN_FAILED" "$TASKS_DIR" "$SPLIT_CACHE" \
        "$RAG_CORPUS_CACHE_PATH" "$RAG_RETRIEVER_MODEL" "$RAG_RETRIEVAL_TOP_K" \
        "$NUM_TASKS" "$TRAIN_SIZE" "$VAL_SIZE" "$TEST_SIZE" "$SEED" \
        "$MAX_LENGTH" "$MAX_INSTRUCTION_TOKENS" "$MAX_NEW_TOKENS" <<'PY'
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
    "tasks_dir": sys.argv[8],
    "split_cache_path": sys.argv[9],
    "rag": {
        "corpus_cache_path": sys.argv[10],
        "retriever_model": sys.argv[11],
        "retrieval_top_k": int(sys.argv[12]),
    },
    "dataset": {
        "num_tasks": int(sys.argv[13]),
        "train_size": int(sys.argv[14]),
        "val_size": int(sys.argv[15]),
        "test_size": int(sys.argv[16]),
        "seed": int(sys.argv[17]),
        "max_length": int(sys.argv[18]),
        "max_instruction_tokens": int(sys.argv[19]),
        "max_new_tokens": int(sys.argv[20]),
    },
    "trial_count": 3,
    "trial_seeds": [42, 42, 42],
        "batch_settings": {
            "qwen0_5b": {
            "lora": {"train_batch_size": 8, "eval_batch_size": 32, "gradient_accumulation_steps": 1},
            "tokmem_family": {"train_batch_size": 16, "eval_batch_size": 64, "gradient_accumulation_steps": 1},
            "base": {"test_batch_size": 512},
            "rag": {"test_batch_size": 256, "retrieval_top_k": 3},
        },
        "llama3b": {
            "lora": {"train_batch_size": 2, "eval_batch_size": 16, "gradient_accumulation_steps": 2},
            "tokmem_family": {"train_batch_size": 8, "eval_batch_size": 32, "gradient_accumulation_steps": 1},
            "base": {"test_batch_size": 256},
            "rag": {"test_batch_size": 128, "retrieval_top_k": 3},
        },
        "llama8b": {
            "lora": {"train_batch_size": 2, "eval_batch_size": 8, "gradient_accumulation_steps": 2},
            "tokmem_family": {"train_batch_size": 4, "eval_batch_size": 16, "gradient_accumulation_steps": 2},
            "base": {"test_batch_size": 128},
            "rag": {"test_batch_size": 64, "retrieval_top_k": 3},
        },
        "shared_training": {
            "epochs": 1,
            "lr": 5e-3,
            "validate_every_n_steps": 1000,
        },
    },
    "models": {
        "qwen0_5b": "models/Qwen2.5-0.5B-Instruct",
        "llama3b": "models/Llama-3.2-3B-Instruct",
        "llama8b": "models/Llama-3.1-8B-Instruct",
    },
    "methods": [
        "base",
        "rag",
        "lora",
        "tokmem",
        "tokmem_logit_bias",
    ],
}
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

validate_inputs() {
    if [[ ! -d "$TASKS_DIR" ]]; then
        echo "Tasks directory not found: $TASKS_DIR" >&2
        exit 1
    fi

    if [[ ! -f "$SPLIT_CACHE" ]]; then
        echo "Split cache not found: $SPLIT_CACHE" >&2
        exit 1
    fi

    local model_key
    for model_key in "${MODEL_KEYS[@]}"; do
        if [[ ! -d "${MODEL_PATHS[$model_key]}" ]]; then
            echo "Model path not found for $model_key: ${MODEL_PATHS[$model_key]}" >&2
            exit 1
        fi
    done

    if [[ ! -d "$RAG_RETRIEVER_MODEL" ]]; then
        echo "Retriever model path not found: $RAG_RETRIEVER_MODEL" >&2
        exit 1
    fi

    python - "$SPLIT_CACHE" "$NUM_TASKS" "$TRAIN_SIZE" "$VAL_SIZE" "$TEST_SIZE" "$SEED" "$MAX_LENGTH" <<'PY'
import sys
import torch

cache_path = sys.argv[1]
expected = {
    "num_tasks": int(sys.argv[2]),
    "train_size": int(sys.argv[3]),
    "val_size": int(sys.argv[4]),
    "test_size": int(sys.argv[5]),
    "seed": int(sys.argv[6]),
    "max_length": int(sys.argv[7]),
}
payload = torch.load(cache_path, map_location="cpu")
metadata = payload.get("metadata", {})
mismatches = []
for key, expected_value in expected.items():
    cached_value = metadata.get(key)
    if cached_value != expected_value:
        mismatches.append(f"{key}: expected={expected_value!r}, cached={cached_value!r}")

if mismatches:
    raise SystemExit("Split cache metadata mismatch:\n" + "\n".join(mismatches))
PY
}

build_task_command() {
    local model_key="$1"
    local method="$2"
    local task_name="$3"
    local seed="$4"
    local task_dir="$5"
    local model_path="${MODEL_PATHS[$model_key]}"
    local base_test_batch_size="${BASE_TEST_BATCH_SIZES[$model_key]}"
    local rag_test_batch_size="${RAG_TEST_BATCH_SIZES[$model_key]}"
    local lora_train_batch_size="${LORA_TRAIN_BATCH_SIZES[$model_key]}"
    local lora_eval_batch_size="${LORA_EVAL_BATCH_SIZES[$model_key]}"
    local lora_gradient_accumulation_steps="${LORA_GRADIENT_ACCUMULATION_STEPS[$model_key]}"
    local tokmem_train_batch_size="${TOKMEM_TRAIN_BATCH_SIZES[$model_key]}"
    local tokmem_eval_batch_size="${TOKMEM_EVAL_BATCH_SIZES[$model_key]}"
    local tokmem_gradient_accumulation_steps="${TOKMEM_GRADIENT_ACCUMULATION_STEPS[$model_key]}"

    case "$method" in
        base)
            TASK_CMD=(
                python -u "$ROOT_DIR/atomic/main_base_model.py"
                --tasks_dir "$TASKS_DIR"
                --num_tasks "$NUM_TASKS"
                --train_size "$TRAIN_SIZE"
                --val_size "$VAL_SIZE"
                --test_size "$TEST_SIZE"
                --model_name "$model_path"
                --device_map balanced
                --split_cache_path "$SPLIT_CACHE"
                --max_length "$MAX_LENGTH"
                --max_instruction_tokens "$MAX_INSTRUCTION_TOKENS"
                --max_new_tokens "$MAX_NEW_TOKENS"
                --test_batch_size "$base_test_batch_size"
                --seed "$seed"
                --run_dir "$task_dir"
            )
            ;;
        rag)
            TASK_CMD=(
                python -u "$ROOT_DIR/atomic/main_rag_baseline.py"
                --tasks_dir "$TASKS_DIR"
                --num_tasks "$NUM_TASKS"
                --train_size "$TRAIN_SIZE"
                --val_size "$VAL_SIZE"
                --test_size "$TEST_SIZE"
                --model_name "$model_path"
                --retriever_model "$RAG_RETRIEVER_MODEL"
                --device_map balanced
                --split_cache_path "$SPLIT_CACHE"
                --corpus_cache_path "$RAG_CORPUS_CACHE_PATH"
                --retrieval_top_k "$RAG_RETRIEVAL_TOP_K"
                --max_length "$MAX_LENGTH"
                --max_instruction_tokens "$MAX_INSTRUCTION_TOKENS"
                --max_new_tokens "$MAX_NEW_TOKENS"
                --test_batch_size "$rag_test_batch_size"
                --seed "$seed"
                --run_dir "$task_dir"
            )
            ;;
        lora)
            TASK_CMD=(
                python -u "$ROOT_DIR/atomic/main_lora_baseline.py"
                --tasks_dir "$TASKS_DIR"
                --num_tasks "$NUM_TASKS"
                --train_size "$TRAIN_SIZE"
                --val_size "$VAL_SIZE"
                --test_size "$TEST_SIZE"
                --model_name "$model_path"
                --device_map balanced
                --split_cache_path "$SPLIT_CACHE"
                --max_length "$MAX_LENGTH"
                --max_instruction_tokens "$MAX_INSTRUCTION_TOKENS"
                --batch_size "$lora_train_batch_size"
                --val_batch_size "$lora_eval_batch_size"
                --test_batch_size "$lora_eval_batch_size"
                --num_epochs "$TOKMEM_EPOCHS"
                --lr "$TOKMEM_LR"
                --gradient_accumulation_steps "$lora_gradient_accumulation_steps"
                --validate_every_n_steps "$TOKMEM_VALIDATE_EVERY_N_STEPS"
                --seed "$seed"
                --save_path "$task_dir/saved_models/lora_best"
                --run_dir "$task_dir"
            )
            ;;
        tokmem|tokmem_logit_bias)
            TASK_CMD=(
                python -u "$ROOT_DIR/atomic/main_in_domain.py"
                --tasks_dir "$TASKS_DIR"
                --num_tasks "$NUM_TASKS"
                --train_size "$TRAIN_SIZE"
                --val_size "$VAL_SIZE"
                --test_size "$TEST_SIZE"
                --model_name "$model_path"
                --device_map balanced
                --split_cache_path "$SPLIT_CACHE"
                --num_epochs "$TOKMEM_EPOCHS"
                --batch_size "$tokmem_train_batch_size"
                --gradient_accumulation_steps "$tokmem_gradient_accumulation_steps"
                --max_length "$MAX_LENGTH"
                --max_instruction_tokens "$MAX_INSTRUCTION_TOKENS"
                --lr "$TOKMEM_LR"
                --val_batch_size "$tokmem_eval_batch_size"
                --test_batch_size "$tokmem_eval_batch_size"
                --validate_every_n_steps "$TOKMEM_VALIDATE_EVERY_N_STEPS"
                --seed "$seed"
                --run_dir "$task_dir"
            )
            if [[ "$method" == "tokmem_logit_bias" ]]; then
                TASK_CMD+=(
                    --use_logit_bias
                    --logit_bias_loss_weight "$LOGIT_BIAS_LOSS_WEIGHT"
                    --logit_bias_network "$LOGIT_BIAS_NETWORK"
                    --logit_bias_scale "$LOGIT_BIAS_SCALE"
                )
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
    local training_method
    for training_method in "${TRAINING_METHODS[@]}"; do
        if [[ "$method" == "$training_method" ]]; then
            return 0
        fi
    done
    return 1
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
declare -a TASK_METHOD=()
declare -a TASK_TRIAL=()
declare -a TASK_SEED=()
declare -a TASK_NAME=()
declare -a TASK_DIR=()
declare -a TASK_IS_TRAINING=()
declare -a TASK_PENDING_INDEXES=()

build_task_manifest() {
    printf "task_id\tmodel\tmethod\ttrial\tseed\tis_training\ttask_name\ttask_dir\n" > "$MANIFEST_FILE"

    local trial model_key method seed task_name task_dir training_flag task_id
    task_id=0
    for trial in $(seq 1 "$TRIAL_COUNT"); do
        seed="${TRIAL_SEEDS[$((trial - 1))]}"
        for model_key in "${MODEL_KEYS[@]}"; do
            for method in "${METHODS[@]}"; do
                task_name="${model_key}_${method}_trial${trial}_seed${seed}"
                task_dir="$RUNS_ROOT/$task_name"
                training_flag=0
                if is_training_method "$method"; then
                    training_flag=1
                fi

                TASK_MODEL+=("$model_key")
                TASK_METHOD+=("$method")
                TASK_TRIAL+=("$trial")
                TASK_SEED+=("$seed")
                TASK_NAME+=("$task_name")
                TASK_DIR+=("$task_dir")
                TASK_IS_TRAINING+=("$training_flag")

                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$task_id" "$model_key" "$method" "$trial" "$seed" "$training_flag" "$task_name" "$task_dir" \
                    >> "$MANIFEST_FILE"

                if task_should_run "$task_dir"; then
                    TASK_PENDING_INDEXES+=("$task_id")
                fi

                task_id=$((task_id + 1))
            done
        done
    done
}

write_task_status_json() {
    local status_path="$1"
    local task_name="$2"
    local model_key="$3"
    local method="$4"
    local trial="$5"
    local seed="$6"
    local gpu_id="$7"
    local started_at="$8"
    local finished_at="$9"
    local duration_seconds="${10}"
    local exit_code="${11}"
    local status="${12}"
    local task_dir="${13}"
    local is_training="${14}"
    python - "$status_path" "$task_name" "$model_key" "$method" "$trial" "$seed" "$gpu_id" \
        "$started_at" "$finished_at" "$duration_seconds" "$exit_code" "$status" "$task_dir" "$is_training" <<'PY'
import json
import sys
from pathlib import Path

def latest_file(task_dir: Path, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(task_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return str(matches[0])

def latest_parent_dir(task_dir: Path, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(task_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return str(matches[0].parent)

path = Path(sys.argv[1])
task_dir = Path(sys.argv[13])
payload = {
    "task_name": sys.argv[2],
    "model": sys.argv[3],
    "method": sys.argv[4],
    "trial": int(sys.argv[5]),
    "seed": int(sys.argv[6]),
    "gpu_id": sys.argv[7],
    "started_at": sys.argv[8],
    "finished_at": sys.argv[9],
    "duration_seconds": int(sys.argv[10]),
    "exit_code": int(sys.argv[11]),
    "status": sys.argv[12],
    "task_dir": str(task_dir),
    "is_training_method": bool(int(sys.argv[14])),
    "run_config": str(task_dir / "run_config.json"),
    "run_summary": str(task_dir / "run_summary.json"),
    "evaluation_results": str(task_dir / "evaluation_results.json"),
    "stdout_log": str(task_dir / "stdout.log"),
    "gpu_monitor_log": str(task_dir / "gpu_monitor.log"),
    "training_log": latest_file(task_dir, ["training_*.log", "logs/training_*.log"]),
    "evaluation_log": latest_file(task_dir, ["evaluation_*.log", "logs/evaluation_*.log"]),
    "best_task_tokens": latest_file(task_dir, ["saved_models/task_tokens_*_best.pt"]),
    "best_lora_adapter": latest_parent_dir(task_dir, ["**/adapter_config.json"]),
}
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

launch_task() {
    local task_index="$1"
    local gpu_id="$2"
    local model_key="${TASK_MODEL[$task_index]}"
    local method="${TASK_METHOD[$task_index]}"
    local trial="${TASK_TRIAL[$task_index]}"
    local seed="${TASK_SEED[$task_index]}"
    local task_name="${TASK_NAME[$task_index]}"
    local task_dir="${TASK_DIR[$task_index]}"
    local training_flag="${TASK_IS_TRAINING[$task_index]}"

    mkdir -p "$task_dir"
    build_task_command "$model_key" "$method" "$task_name" "$seed" "$task_dir"
    cat > "$task_dir/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
export CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu_id")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
cd $(printf '%q' "$task_dir")

while true; do
    {
        echo "===== \$(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i $(printf '%q' "$gpu_id")
    } >> gpu_monitor.log
    sleep 10
done &
MONITOR_PID=\$!
trap 'kill "\$MONITOR_PID" 2>/dev/null || true' EXIT

exec $(command_string)
EOF
    chmod +x "$task_dir/command.sh"

    log "Launching task=$task_name gpu=$gpu_id method=$method model=$model_key trial=$trial"

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
            "model": row["model"],
            "method": row["method"],
            "trial": int(row["trial"]),
            "seed": int(row["seed"]),
            "status": fallback_status,
            "task_dir": row["task_dir"],
            "is_training_method": bool(int(row["is_training"])),
            "evaluation_results": str(task_dir / "evaluation_results.json"),
            "stdout_log": str(task_dir / "stdout.log"),
            "gpu_monitor_log": str(task_dir / "gpu_monitor.log"),
        }
    )

status_path.write_text(json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

write_suite_summary() {
    python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$SUMMARY_JSON_FILE" "$STATUS_FILE" "$SUITE_NAME" \
        "$TRIAL_COUNT" "$SPLIT_CACHE" "$NUM_TASKS" "$TRAIN_SIZE" "$VAL_SIZE" "$TEST_SIZE" <<'PY'
import csv
import json
import re
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
split_cache = sys.argv[7]
num_tasks = int(sys.argv[8])
train_size = int(sys.argv[9])
val_size = int(sys.argv[10])
test_size = int(sys.argv[11])

metric_fields = (
    ("task_accuracy", "Routing Acc"),
    ("rouge_l", "Rouge-L"),
    ("exact_match", "Exact Match"),
    ("retrieval_top1_accuracy", "Retrieval Top-1"),
    ("retrieval_topk_accuracy", "Retrieval Top-K"),
)

training_methods = {"lora", "tokmem", "tokmem_logit_bias"}

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


def latest_file(task_dir: Path, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(task_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def load_eval_metrics(task_dir: Path, method: str):
    evaluation_results_path = task_dir / "evaluation_results.json"
    if evaluation_results_path.exists():
        payload = json.loads(evaluation_results_path.read_text(encoding="utf-8"))
        exact_match = payload.get("exact_accuracy")
        rouge_l = payload.get("avg_response_score")
        if rouge_l is None and payload.get("ni_rouge_l") is not None:
            rouge_l = float(payload["ni_rouge_l"]) / 100.0
        task_accuracy = payload.get("task_accuracy") if method in {"tokmem", "tokmem_logit_bias"} else None
        retrieval_top1_accuracy = payload.get("retrieval_top1_accuracy") if method == "rag" else None
        retrieval_topk_accuracy = payload.get("retrieval_topk_accuracy") if method == "rag" else None
        return {
            "task_accuracy": task_accuracy,
            "rouge_l": rouge_l,
            "exact_match": exact_match,
            "retrieval_top1_accuracy": retrieval_top1_accuracy,
            "retrieval_topk_accuracy": retrieval_topk_accuracy,
        }

    eval_log_path = latest_file(task_dir, ["evaluation_*.log", "logs/evaluation_*.log"])
    stdout_path = task_dir / "stdout.log"
    candidates = []
    if eval_log_path is not None:
        candidates.append(eval_log_path.read_text(encoding="utf-8", errors="replace"))
    if stdout_path.exists():
        candidates.append(stdout_path.read_text(encoding="utf-8", errors="replace"))

    if method in {"tokmem", "tokmem_logit_bias"}:
        pattern = re.compile(
            r"EVALUATION COMPLETE - TaskAcc:(?P<task>[0-9.]+) ExactMatch:(?P<exact>[0-9.]+)% RougeL:(?P<rouge>[0-9.]+)%"
        )
        for text in candidates:
            match = pattern.search(text)
            if match:
                return {
                    "task_accuracy": float(match.group("task")),
                    "rouge_l": float(match.group("rouge")) / 100.0,
                    "exact_match": float(match.group("exact")) / 100.0,
                    "retrieval_top1_accuracy": None,
                    "retrieval_topk_accuracy": None,
                }

    return None


grouped = defaultdict(list)
for row in rows:
    grouped[(row["model"], row["method"])].append(row)

summary_lines = [
    "# Atomic Paper Suite Summary",
    "",
    f"- suite: `{suite_name}`",
    f"- trials per model/method: `{trial_count}`",
    f"- scope: `{num_tasks}-task fixed split / train {train_size} / val {val_size} / test {test_size}`",
    f"- split cache: `{split_cache}`",
    "- models: `qwen0_5b`, `llama3b`, `llama8b`",
    "- methods: `base`, `rag`, `lora`, `tokmem`, `tokmem_logit_bias`",
    "",
    "## Batch Settings",
    "",
    "| Model | LoRA train | LoRA grad acc | LoRA eval | TokMem train | TokMem grad acc | TokMem effective train | TokMem eval | base test | rag test |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    "| qwen0_5b | 8 | 1 | 32 | 16 | 1 | 16 | 64 | 512 | 256 |",
    "| llama3b | 2 | 2 | 16 | 8 | 1 | 8 | 32 | 256 | 128 |",
    "| llama8b | 2 | 2 | 8 | 4 | 2 | 8 | 16 | 128 | 64 |",
    "",
    "- current launcher methods: `base`, `rag`, `lora`, `tokmem`, `tokmem_logit_bias`",
    "- shared training settings: epochs `1`, lr `5e-3`, validate_every_n_steps `1000`",
    "- `rag` retrieval top-k: `3`",
    "",
    "## Mean Results",
    "",
    "| Model | Method | Success | Routing Acc | Rouge-L | Exact Match | Retrieval Top-1 | Retrieval Top-K | Avg Runtime |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

failed_tasks = []
incomplete_groups = []
training_duration_rows = []
successful_training_trials = []
mean_results = []

for model, method in sorted(grouped):
    rows_for_group = sorted(grouped[(model, method)], key=lambda item: int(item["trial"]))
    metrics_by_field = defaultdict(list)
    runtimes = []
    success_count = 0

    for row in rows_for_group:
        status = status_by_task.get(row["task_name"], {})
        if status.get("status") != "success":
            failed_tasks.append(
                {
                    "task_name": row["task_name"],
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

        eval_metrics = load_eval_metrics(Path(status["task_dir"]), method)
        if eval_metrics is None:
            continue

        for key, _ in metric_fields:
            value = eval_metrics.get(key)
            if value is not None:
                metrics_by_field[key].append(value)

    if success_count < trial_count:
        incomplete_groups.append(
            {
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
                fmt(metric_means["task_accuracy"]),
                fmt(metric_means["rouge_l"]),
                fmt(metric_means["exact_match"]),
                fmt(metric_means["retrieval_top1_accuracy"]),
                fmt(metric_means["retrieval_topk_accuracy"]),
                fmt_seconds(avg_runtime),
            ]
        )
        + " |"
    )

    if method in training_methods and runtimes:
        training_duration_rows.append(
            {
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
                        "model": model,
                        "method": method,
                        "trial": int(row["trial"]),
                        "duration_seconds": runtime,
                        "stdout_log": status.get("stdout_log", str(Path(row["task_dir"]) / "stdout.log")),
                    }
                )

summary_lines.extend(["", "## Incomplete Or Failed Experiment Groups", ""])
if incomplete_groups:
    summary_lines.append("| Model | Method | Successful Trials |")
    summary_lines.append("| --- | --- | ---: |")
    for item in incomplete_groups:
        summary_lines.append(
            f"| {item['model']} | {item['method']} | {item['success_count']}/{item['trial_count']} |"
        )
else:
    summary_lines.append("All model/method groups finished all trials successfully.")

summary_lines.extend(["", "## Failed Or Unfinished Trials", ""])
if failed_tasks:
    summary_lines.append("| Task | Model | Method | Trial | Status | Exit Code | Stdout Log |")
    summary_lines.append("| --- | --- | --- | ---: | --- | ---: | --- |")
    for item in failed_tasks:
        exit_code = "" if item["exit_code"] is None else str(item["exit_code"])
        summary_lines.append(
            f"| {item['task_name']} | {item['model']} | {item['method']} | {item['trial']} | "
            f"{item['status']} | {exit_code} | {item['stdout_log']} |"
        )
else:
    summary_lines.append("All trials finished successfully.")

summary_lines.extend(["", "## Training Runtime Ranking", ""])
if training_duration_rows:
    training_duration_rows.sort(key=lambda item: item["avg_runtime_seconds"], reverse=True)
    summary_lines.append("| Rank | Model | Method | Avg Runtime | Successful Trials |")
    summary_lines.append("| ---: | --- | --- | ---: | ---: |")
    for rank, item in enumerate(training_duration_rows, start=1):
        summary_lines.append(
            f"| {rank} | {item['model']} | {item['method']} | {fmt_seconds(item['avg_runtime_seconds'])} | "
            f"{item['success_count']}/{trial_count} |"
        )
else:
    summary_lines.append("No successful training runtimes available.")

long_runtime_trials = []
grouped_training_trials = defaultdict(list)
for item in successful_training_trials:
    grouped_training_trials[(item["model"], item["method"])].append(item)

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
    summary_lines.append("| Task | Model | Method | Trial | Runtime | Group Median | Stdout Log |")
    summary_lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for item in long_runtime_trials:
        summary_lines.append(
            f"| {item['task_name']} | {item['model']} | {item['method']} | {item['trial']} | "
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
        f"- Mean metrics are only reported for groups with `{trial_count}/{trial_count}` successful trials.",
        "- `Routing Acc` is populated for `tokmem` and `tokmem_logit_bias` only.",
        "- `Retrieval Top-1` and `Retrieval Top-K` are populated for `rag` only.",
        "- `lora` uses the shared training settings and keeps replay disabled in the launcher command.",
        "- `Rouge-L` and `Exact Match` are normalized to `0-1` decimals for every method.",
        "- Longer training trials are flagged when runtime is at least `max(1.5 x group median, group median + 600s)` within the same model/method group.",
    ]
)

summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
summary_json_path.write_text(
    json.dumps(
        {
            "suite_name": suite_name,
            "trial_count": trial_count,
            "split_cache": split_cache,
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

validate_inputs
write_suite_config
build_task_manifest
write_suite_status_json

log "Suite directory: $SUITE_DIR"
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
