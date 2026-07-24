#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

SUITE_NAME="qwen35_table1_rebuttal_$(date -u +%Y%m%d_%H%M%S)"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
MODEL_LOAD_STAGGER_SECONDS="${MODEL_LOAD_STAGGER_SECONDS:-20}"
DRY_RUN=0

usage() {
    cat <<EOF
Usage:
  bash $0 [--suite-name NAME] [--gpus 0,1,...] [--poll-seconds N] [--dry-run]

Runs the Qwen3.5 compositional Table 1 rebuttal study in this order:
  1. Qwen3.5-9B TapMem seed-42 learning-rate sweep.
  2. Qwen3.5-9B ICL, RAG, TokMem, TapMem, LoRA, and adaptation runs
     for seeds 40, 41, and 42.
  3. The same sweep and final experiment matrix for Qwen3.5-4B.
  4. Bash/jq/awk aggregation into metrics.tsv and summary.md.

The scheduler dynamically uses every GPU in --gpus whose memory.used is at
most ${GPU_MEMORY_LIMIT_MIB} MiB. It holds the repository-wide per-GPU flock
for the entire task. Existing successful tasks are reused when the command is
unchanged, so passing the same --suite-name resumes a stopped suite.

The script intentionally does not save full model checkpoints. A Qwen3.5-9B
checkpoint is about 18 GiB, and saving one for every sweep/seed would consume
hundreds of GiB without being needed for the requested result table.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite-name)
            SUITE_NAME="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS_CSV="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
    echo "--gpus must contain at least one GPU index." >&2
    exit 2
fi
for gpu in "${GPUS[@]}"; do
    if ! [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "Invalid GPU index: $gpu" >&2
        exit 2
    fi
done
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
    echo "--gpus must not contain duplicate indexes." >&2
    exit 2
fi
for value in "$POLL_SECONDS" "$GPU_MEMORY_LIMIT_MIB" "$MODEL_LOAD_STAGGER_SECONDS"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Polling, memory, and stagger settings must be nonnegative integers." >&2
        exit 2
    fi
done
if [[ "$POLL_SECONDS" -eq 0 ]]; then
    echo "--poll-seconds must be positive." >&2
    exit 2
fi

RESULTS_ROOT="$ROOT_DIR/results/compositional"
SUITE_DIR="$RESULTS_ROOT/$SUITE_NAME"
RUNS_ROOT="$SUITE_DIR/runs"
DATA_DIR="$ROOT_DIR/results/compositional/all_methods/data"
HF_CACHE_DIR="$SUITE_DIR/hf-cache"
MANIFEST_FILE="$SUITE_DIR/task_manifest.tsv"
STATUS_FILE="$SUITE_DIR/task_status.tsv"
METRICS_FILE="$SUITE_DIR/metrics.tsv"
SUMMARY_FILE="$SUITE_DIR/summary.md"
SCHEDULER_LOG="$SUITE_DIR/scheduler.log"
GPU_LOCK_DIR="/tmp/tokmem_gpu_locks"
JQ_BIN="/home/shilong/anaconda3/bin/jq"

TRAIN_1_50="$DATA_DIR/training/function_calling_train_tools1-50_4calls.json"
TEST_1_50="$DATA_DIR/test/function_calling_test_tools1-50_4calls.json"
TRAIN_51_100="$DATA_DIR/training/function_calling_train_tools51-100_4calls.json"
TEST_51_100="$DATA_DIR/test/function_calling_test_tools51-100_4calls.json"
TOOL_DESCRIPTIONS="$DATA_DIR/tool_descriptions_tools51-100.json"
RETRIEVER_MODEL="$ROOT_DIR/models/all-MiniLM-L6-v2"

MODEL_KEYS=(qwen9b qwen4b)
METHODS=(icl rag tokmem tapmem lora adap_tokmem adap_tapmem)
SEEDS=(42 40 41)
INITIAL_SWEEP_LRS=(1e-3 2e-3 3e-3 5e-3 7e-3 1e-2)

declare -A MODEL_PATHS=(
    [qwen9b]="$ROOT_DIR/models/Qwen3.5-9B"
    [qwen4b]="$ROOT_DIR/models/Qwen3.5-4B"
)
declare -A TOKMEM_BATCH_SIZES=(
    [qwen9b]=4
    [qwen4b]=4
)
declare -A TOKMEM_EVAL_BATCH_SIZES=(
    [qwen9b]=16
    [qwen4b]=32
)
declare -A LORA_BATCH_SIZES=(
    [qwen9b]=2
    [qwen4b]=4
)
declare -A LORA_EVAL_BATCH_SIZES=(
    [qwen9b]=16
    [qwen4b]=32
)
declare -A ADAPT_BATCH_SIZES=(
    [qwen9b]="2,4"
    [qwen4b]="4,4"
)
declare -A ICL_BATCH_SIZES=(
    [qwen9b]=4
    [qwen4b]=8
)
declare -A RAG_BATCH_SIZES=(
    [qwen9b]=16
    [qwen4b]=32
)
declare -A LORA_LRS=(
    [qwen9b]=8e-5
    [qwen4b]=5e-5
)
declare -A ADAP_TAPMEM_LORA_LRS=(
    [qwen9b]=8e-5
    [qwen4b]=8e-5
)

log() {
    local message
    message="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
    echo "$message"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        echo "$message" >> "$SCHEDULER_LOG"
    fi
}

sanitize_lr() {
    echo "$1" | sed -e 's/+/_plus_/g' -e 's/-/m/g' -e 's/\./p/g'
}

model_display_name() {
    case "$1" in
        qwen9b) echo "Qwen3.5-9B" ;;
        qwen4b) echo "Qwen3.5-4B" ;;
        *) echo "$1" ;;
    esac
}

method_display_name() {
    case "$1" in
        icl) echo "ICL" ;;
        rag) echo "RAG" ;;
        tokmem) echo "TokMem" ;;
        tapmem) echo "TapMem" ;;
        lora) echo "Fine-Tuning (LoRA)" ;;
        adap_tokmem) echo "TokMem + adaptation" ;;
        adap_tapmem) echo "TapMem + adaptation" ;;
        *) echo "$1" ;;
    esac
}

preflight() {
    local required_file
    for required_file in \
        "$TRAIN_1_50" \
        "$TEST_1_50" \
        "$TRAIN_51_100" \
        "$TEST_51_100" \
        "$TOOL_DESCRIPTIONS" \
        "$RETRIEVER_MODEL/config.json" \
        "${MODEL_PATHS[qwen9b]}/config.json" \
        "${MODEL_PATHS[qwen4b]}/config.json"; do
        if [[ ! -f "$required_file" ]]; then
            echo "Required input is missing: $required_file" >&2
            exit 2
        fi
    done
    for command_name in nvidia-smi flock sha256sum awk sed sort; do
        if ! command -v "$command_name" >/dev/null; then
            echo "Required command is missing: $command_name" >&2
            exit 2
        fi
    done
    if [[ ! -x "$JQ_BIN" ]]; then
        echo "Required jq executable is missing: $JQ_BIN" >&2
        exit 2
    fi
    if [[ "$("$JQ_BIN" 'length' "$TRAIN_51_100")" -ne 5000 ]]; then
        echo "Expected 5000 training examples in $TRAIN_51_100" >&2
        exit 2
    fi
    if [[ "$("$JQ_BIN" 'length' "$TEST_51_100")" -ne 500 ]]; then
        echo "Expected 500 test examples in $TEST_51_100" >&2
        exit 2
    fi
    local available_kib
    available_kib="$(df -Pk "$RESULTS_ROOT" | awk 'NR == 2 {print $4}')"
    if ! [[ "$available_kib" =~ ^[0-9]+$ ]] || \
        (( available_kib < 100 * 1024 * 1024 )); then
        echo "At least 100 GiB of free result-disk space is required." >&2
        exit 2
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
        local gpu
        for gpu in "${GPUS[@]}"; do
            if ! nvidia-smi -i "$gpu" --query-gpu=memory.used \
                --format=csv,noheader,nounits >/dev/null 2>&1; then
                echo "Cannot query GPU $gpu with nvidia-smi." >&2
                exit 2
            fi
        done
    fi
}

initialize_suite() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return
    fi
    mkdir -p "$RUNS_ROOT" "$HF_CACHE_DIR" "$GPU_LOCK_DIR"
    if [[ ! -f "$SUITE_DIR/$(basename "$SCRIPT_PATH")" ]]; then
        cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"
    elif ! cmp -s "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"; then
        echo "Refusing to resume because the launcher differs from its suite snapshot." >&2
        exit 2
    fi
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        printf 'phase\tmodel\tmethod\tseed\tmemory_lr\tlora_lr\ttask_dir\n' \
            > "$MANIFEST_FILE"
    fi
    if [[ ! -f "$STATUS_FILE" ]]; then
        printf 'task\tstatus\texit_code\tgpu\tstarted_at\tfinished_at\n' \
            > "$STATUS_FILE"
    fi
    if [[ ! -f "$SUITE_DIR/input_sha256.txt" ]]; then
        sha256sum \
            "$TRAIN_1_50" \
            "$TEST_1_50" \
            "$TRAIN_51_100" \
            "$TEST_51_100" \
            "$TOOL_DESCRIPTIONS" \
            "${MODEL_PATHS[qwen9b]}/config.json" \
            "${MODEL_PATHS[qwen9b]}/model.safetensors.index.json" \
            "${MODEL_PATHS[qwen9b]}/tokenizer_config.json" \
            "${MODEL_PATHS[qwen4b]}/config.json" \
            "${MODEL_PATHS[qwen4b]}/model.safetensors.index.json" \
            "${MODEL_PATHS[qwen4b]}/tokenizer_config.json" \
            "$SCRIPT_PATH" \
            "$ROOT_DIR"/compositional/*.py \
            > "$SUITE_DIR/input_sha256.txt"
        git -C "$ROOT_DIR" rev-parse HEAD > "$SUITE_DIR/source_commit.txt"
        git -C "$ROOT_DIR" status --short > "$SUITE_DIR/source_worktree_status.txt"
    elif ! sha256sum --check --status "$SUITE_DIR/input_sha256.txt"; then
        echo "Refusing to resume because a recorded source, model config, or data input changed." >&2
        exit 2
    elif [[ "$(git -C "$ROOT_DIR" rev-parse HEAD)" != "$(cat "$SUITE_DIR/source_commit.txt")" ]]; then
        echo "Refusing to resume because the Git commit changed." >&2
        exit 2
    fi
}

acquire_suite_lock() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return
    fi
    exec 199>"$SUITE_DIR/suite.lock"
    if ! flock -n 199; then
        echo "Another launcher or inherited worker still owns this suite: $SUITE_DIR" >&2
        exit 2
    fi
    echo "$$" > "$SUITE_DIR/launcher.pid"
    cleanup_suite_pid() {
        if [[ -f "$SUITE_DIR/launcher.pid" ]] && \
            [[ "$(cat "$SUITE_DIR/launcher.pid")" == "$$" ]]; then
            rm -f "$SUITE_DIR/launcher.pid"
        fi
    }
    trap cleanup_suite_pid EXIT
    trap 'log "Launcher received an interrupt; active workers retain the suite lock until they exit"; exit 130' INT TERM
}

declare -a TASK_IDS=()
declare -a TASK_PHASES=()
declare -a TASK_MODELS=()
declare -a TASK_METHODS=()
declare -a TASK_SEEDS=()
declare -a TASK_MEMORY_LRS=()
declare -a TASK_LORA_LRS=()

clear_tasks() {
    TASK_IDS=()
    TASK_PHASES=()
    TASK_MODELS=()
    TASK_METHODS=()
    TASK_SEEDS=()
    TASK_MEMORY_LRS=()
    TASK_LORA_LRS=()
}

register_manifest_row() {
    local row="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return
    fi
    if ! grep -Fqx "$row" "$MANIFEST_FILE"; then
        echo "$row" >> "$MANIFEST_FILE"
    fi
}

add_task() {
    local task_id="$1"
    local phase="$2"
    local model="$3"
    local method="$4"
    local seed="$5"
    local memory_lr="$6"
    local lora_lr="$7"
    local index="${#TASK_IDS[@]}"
    local task_dir="$RUNS_ROOT/$task_id"

    TASK_IDS[index]="$task_id"
    TASK_PHASES[index]="$phase"
    TASK_MODELS[index]="$model"
    TASK_METHODS[index]="$method"
    TASK_SEEDS[index]="$seed"
    TASK_MEMORY_LRS[index]="$memory_lr"
    TASK_LORA_LRS[index]="$lora_lr"
    register_manifest_row "$phase	$model	$method	$seed	$memory_lr	$lora_lr	$task_dir"
}

build_task_command() {
    local index="$1"
    local task_id="${TASK_IDS[$index]}"
    local model="${TASK_MODELS[$index]}"
    local method="${TASK_METHODS[$index]}"
    local seed="${TASK_SEEDS[$index]}"
    local memory_lr="${TASK_MEMORY_LRS[$index]}"
    local lora_lr="${TASK_LORA_LRS[$index]}"
    local model_path="${MODEL_PATHS[$model]}"
    local tokmem_batch="${TOKMEM_BATCH_SIZES[$model]}"
    local tokmem_eval_batch="${TOKMEM_EVAL_BATCH_SIZES[$model]}"
    local lora_batch="${LORA_BATCH_SIZES[$model]}"
    local lora_eval_batch="${LORA_EVAL_BATCH_SIZES[$model]}"
    local adapt_batch="${ADAPT_BATCH_SIZES[$model]}"
    local icl_batch="${ICL_BATCH_SIZES[$model]}"
    local rag_batch="${RAG_BATCH_SIZES[$model]}"

    case "$method" in
        icl)
            COMMAND=(
                python -u icl_baseline.py
                --test_data "$TEST_51_100"
                --tool_descriptions "$TOOL_DESCRIPTIONS"
                --model_name "$model_path"
                --batch_size "$icl_batch"
                --seed "$seed"
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_id"
                --run_tag "${model}_${method}_table1"
            )
            ;;
        rag)
            COMMAND=(
                python -u icl_baseline.py
                --test_data "$TEST_51_100"
                --tool_descriptions "$TOOL_DESCRIPTIONS"
                --model_name "$model_path"
                --retriever_model_name "$RETRIEVER_MODEL"
                --batch_size "$rag_batch"
                --seed "$seed"
                --use_rag
                --retrieval_k 5
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_id"
                --run_tag "${model}_${method}_table1"
            )
            ;;
        lora)
            COMMAND=(
                python -u lora_sequential.py
                --training_rounds "51-100:3"
                --batch_size "$lora_batch"
                --train_max_function_calls 4
                --test_max_function_calls 4
                --model_name "$model_path"
                --lora_r 8
                --lora_alpha 32
                --lora_dropout 0.1
                --lora_target_modules "q_proj,v_proj"
                --eval_after_each_round
                --data_dir "$DATA_DIR"
                --lr "$lora_lr"
                --eval_batch_size "$lora_eval_batch"
                --max_length 512
                --seed "$seed"
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_id"
                --run_tag "${model}_${method}_table1"
            )
            ;;
        tokmem|tapmem)
            COMMAND=(
                python -u main_sequential.py
                --training_rounds "51-100:1"
                --epochs 3
                --batch_size "$tokmem_batch"
                --train_max_function_calls 4
                --test_max_function_calls 4
                --model_name "$model_path"
                --eval_after_each_round
                --data_dir "$DATA_DIR"
                --lr "$memory_lr"
                --eval_batch_size "$tokmem_eval_batch"
                --max_length 512
                --max_new_tokens 512
                --seed "$seed"
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_id"
                --run_tag "${model}_${method}_table1"
            )
            if [[ "$method" == "tapmem" ]]; then
                COMMAND+=(
                    --use_eoc
                    --use_logit_bias
                    --use_logit_train_add
                    --detach
                    --logit_bias_loss_weight 0.1
                    --logit_bias_network linear
                    --logit_bias_scale 1.0
                )
            fi
            ;;
        adap_tokmem|adap_tapmem)
            COMMAND=(
                python -u main_sequential.py
                --training_rounds "1-50:1,51-100:3"
                --batch_size_per_round "$adapt_batch"
                --train_max_function_calls_per_round "4,4"
                --test_max_function_calls_per_round "4,4"
                --model_name "$model_path"
                --use_lora
                --lora_r 8
                --lora_alpha 32
                --lora_dropout 0.1
                --lora_target_modules "q_proj,v_proj"
                --freeze_lora_after_first
                --eval_after_each_round
                --data_dir "$DATA_DIR"
                --lr "$memory_lr"
                --lora_lr "$lora_lr"
                --eval_batch_size "$tokmem_eval_batch"
                --max_length 512
                --max_new_tokens 512
                --seed "$seed"
                --run_root_dir "$RUNS_ROOT"
                --run_name "$task_id"
                --run_tag "${model}_${method}_table1"
            )
            if [[ "$method" == "adap_tapmem" ]]; then
                COMMAND+=(
                    --use_eoc
                    --use_logit_bias
                    --use_logit_train_add
                    --detach
                    --logit_bias_loss_weight 0.1
                    --logit_bias_network linear
                    --logit_bias_scale 1.0
                )
            fi
            ;;
        *)
            echo "Unknown method: $method" >&2
            exit 2
            ;;
    esac
}

command_line_for_task() {
    local index="$1"
    build_task_command "$index"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
}

task_is_complete() {
    local index="$1"
    local task_dir="$RUNS_ROOT/${TASK_IDS[$index]}"
    if [[ ! -f "$task_dir/SUCCESS" || ! -f "$task_dir/evaluation_results.json" ]]; then
        return 1
    fi
    local expected_command
    expected_command="$(command_line_for_task "$index")"
    if [[ ! -f "$task_dir/command.txt" ]] || \
        [[ "$(cat "$task_dir/command.txt")" != "$expected_command" ]]; then
        echo "Refusing to reuse ${TASK_IDS[$index]} because its command changed." >&2
        return 2
    fi
    return 0
}

gpu_memory_used_mib() {
    nvidia-smi -i "$1" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NR == 1 {gsub(/^[ \t]+|[ \t]+$/, ""); print}'
}

gpu_is_free() {
    local memory_used
    memory_used="$(gpu_memory_used_mib "$1")" || return 1
    [[ "$memory_used" =~ ^[0-9]+$ ]] || return 1
    (( memory_used <= GPU_MEMORY_LIMIT_MIB ))
}

gpu_lock_file() {
    echo "$GPU_LOCK_DIR/gpu_$1.lock"
}

record_status() {
    local task_id="$1"
    local status="$2"
    local exit_code="$3"
    local gpu="$4"
    local started_at="$5"
    local finished_at="$6"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$task_id" "$status" "$exit_code" "$gpu" "$started_at" "$finished_at" \
            >> "$STATUS_FILE"
    fi
}

run_task() {
    local index="$1"
    local gpu="$2"
    local task_id="${TASK_IDS[$index]}"
    local task_dir="$RUNS_ROOT/$task_id"
    local started_at finished_at exit_code

    mkdir -p "$task_dir"
    local attempt_dir=""
    local existing_file
    for existing_file in \
        command.txt command.sh stdout.log evaluation.log evaluation_results.json \
        run_config.json training_summary.json exit_code.txt SUCCESS FAILED; do
        if [[ -e "$task_dir/$existing_file" ]]; then
            if [[ -z "$attempt_dir" ]]; then
                attempt_dir="$task_dir/previous_attempts/$(date -u +%Y%m%d_%H%M%S)"
                mkdir -p "$attempt_dir"
            fi
            mv "$task_dir/$existing_file" "$attempt_dir/$existing_file"
        fi
    done
    build_task_command "$index"
    {
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    } > "$task_dir/command.txt"
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'source /home/shilong/anaconda3/etc/profile.d/conda.sh\n'
        printf 'conda activate tokmem\n'
        printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$gpu"
        printf 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n'
        printf 'export HF_HOME=%q\n' "$HF_CACHE_DIR"
        printf 'export HF_DATASETS_CACHE=%q\n' "$HF_CACHE_DIR/datasets"
        printf 'export HUGGINGFACE_HUB_CACHE=%q\n' "$HF_CACHE_DIR/hub"
        printf 'export TOKENIZERS_PARALLELISM=false\n'
        printf 'cd %q\n' "$ROOT_DIR/compositional"
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    } > "$task_dir/command.sh"
    chmod +x "$task_dir/command.sh"
    rm -f "$task_dir/SUCCESS" "$task_dir/FAILED" "$task_dir/exit_code.txt"

    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    log "Starting $task_id on GPU $gpu"
    if (
        cd "$ROOT_DIR/compositional"
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "${COMMAND[@]}"
    ) > "$task_dir/stdout.log" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "$exit_code" > "$task_dir/exit_code.txt"

    if [[ "$exit_code" -eq 0 && -f "$task_dir/evaluation_results.json" ]]; then
        touch "$task_dir/SUCCESS"
        record_status "$task_id" success 0 "$gpu" "$started_at" "$finished_at"
        log "Completed $task_id on GPU $gpu"
        return 0
    fi

    if [[ "$exit_code" -eq 0 ]]; then
        exit_code=3
        echo "$exit_code" > "$task_dir/exit_code.txt"
    fi
    touch "$task_dir/FAILED"
    record_status "$task_id" failed "$exit_code" "$gpu" "$started_at" "$finished_at"
    log "Failed $task_id on GPU $gpu with exit code $exit_code"
    return "${exit_code:-1}"
}

run_task_locked() {
    local index="$1"
    local gpu="$2"
    local lock_file
    lock_file="$(gpu_lock_file "$gpu")"
    (
        flock -x 200
        while ! gpu_is_free "$gpu"; do
            sleep "$POLL_SECONDS"
        done
        run_task "$index" "$gpu"
    ) 200>"$lock_file"
}

run_task_queue() {
    local stage_label="$1"
    local -a pending_indexes=()
    local index
    for index in "${!TASK_IDS[@]}"; do
        if task_is_complete "$index"; then
            log "Reusing completed task ${TASK_IDS[$index]}"
        else
            local status=$?
            if [[ "$status" -eq 2 ]]; then
                exit 2
            fi
            pending_indexes+=("$index")
        fi
    done

    if [[ "$DRY_RUN" -eq 1 ]]; then
        for index in "${pending_indexes[@]}"; do
            printf '[dry-run] %s: ' "${TASK_IDS[$index]}"
            command_line_for_task "$index"
        done
        return
    fi
    if [[ "${#pending_indexes[@]}" -eq 0 ]]; then
        log "$stage_label already complete"
        return
    fi

    log "$stage_label queue: ${#pending_indexes[@]} pending task(s)"
    declare -A gpu_pids=()
    declare -A gpu_tasks=()
    local next_task=0
    local task_failed=0

    while [[ "$next_task" -lt "${#pending_indexes[@]}" || "${#gpu_pids[@]}" -gt 0 ]]; do
        local made_progress=0
        local gpu pid

        for gpu in "${GPUS[@]}"; do
            pid="${gpu_pids[$gpu]:-}"
            if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
                if ! wait "$pid"; then
                    task_failed=1
                fi
                unset 'gpu_pids[$gpu]'
                unset 'gpu_tasks[$gpu]'
                made_progress=1
            fi
        done

        for gpu in "${GPUS[@]}"; do
            if [[ "$next_task" -ge "${#pending_indexes[@]}" ]]; then
                break
            fi
            if [[ -n "${gpu_pids[$gpu]:-}" ]]; then
                continue
            fi
            if ! gpu_is_free "$gpu"; then
                continue
            fi

            index="${pending_indexes[$next_task]}"
            log "Dispatching ${TASK_IDS[$index]} to GPU $gpu"
            run_task_locked "$index" "$gpu" &
            gpu_pids["$gpu"]=$!
            gpu_tasks["$gpu"]="${TASK_IDS[$index]}"
            next_task=$((next_task + 1))
            made_progress=1
            if [[ "$MODEL_LOAD_STAGGER_SECONDS" -gt 0 ]]; then
                sleep "$MODEL_LOAD_STAGGER_SECONDS"
            fi
        done

        if [[ "$made_progress" -eq 0 ]]; then
            log "Waiting for GPU/task completion: pending=$((${#pending_indexes[@]} - next_task)), running=${#gpu_pids[@]}"
            sleep "$POLL_SECONDS"
        fi
    done

    if [[ "$task_failed" -ne 0 ]]; then
        echo "$stage_label has failed tasks. Inspect $RUNS_ROOT/*/stdout.log and resume with:" >&2
        echo "  bash $SCRIPT_PATH --suite-name $SUITE_NAME --gpus $GPU_IDS_CSV" >&2
        exit 1
    fi
}

prepare_sweep_tasks() {
    local model="$1"
    shift
    local -a lrs=("$@")
    local lr task_id
    clear_tasks
    for lr in "${lrs[@]}"; do
        task_id="${model}_tapmem_seed42_lr$(sanitize_lr "$lr")_sweep"
        add_task "$task_id" sweep "$model" tapmem 42 "$lr" "-"
    done
}

extract_overall_metrics() {
    local method="$1"
    local result_file="$2"
    if [[ "$method" == "icl" || "$method" == "rag" ]]; then
        "$JQ_BIN" -r '
            .metrics
            | [.avg_tool_f1_score, .avg_f1_score, .parse_error_rate, .total_samples]
            | @tsv
        ' "$result_file"
    else
        "$JQ_BIN" -r '
            .rounds[-1].eval_results
            | [.avg_tool_f1_score, .avg_f1_score, .parse_error_rate, .total_examples]
            | @tsv
        ' "$result_file"
    fi
}

collect_sweep_metrics() {
    local model="$1"
    shift
    local -a lrs=("$@")
    local output_file="$SUITE_DIR/${model}_tapmem_lr_sweep_seed42.tsv"
    local lr task_id result_file metrics tool_f1 argument_f1 parse_error total score
    printf 'learning_rate\ttool_f1\targument_f1\tparse_error_rate\tcombined_score\n' \
        > "$output_file"
    for lr in "${lrs[@]}"; do
        task_id="${model}_tapmem_seed42_lr$(sanitize_lr "$lr")_sweep"
        result_file="$RUNS_ROOT/$task_id/evaluation_results.json"
        if [[ ! -f "$RUNS_ROOT/$task_id/SUCCESS" || ! -f "$result_file" ]]; then
            echo "Incomplete sweep task: $task_id" >&2
            exit 1
        fi
        metrics="$(extract_overall_metrics tapmem "$result_file")"
        IFS=$'\t' read -r tool_f1 argument_f1 parse_error total <<< "$metrics"
        if [[ "$total" -ne 500 ]]; then
            echo "Sweep task $task_id evaluated $total examples instead of 500." >&2
            exit 1
        fi
        score="$(awk -v tool="$tool_f1" -v argument="$argument_f1" \
            'BEGIN {printf "%.12f", 0.5 * tool + 0.5 * argument}')"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$lr" "$tool_f1" "$argument_f1" "$parse_error" "$score" \
            >> "$output_file"
    done
}

select_best_lr() {
    local model="$1"
    local sweep_file="$SUITE_DIR/${model}_tapmem_lr_sweep_seed42.tsv"
    local best_lr
    best_lr="$(
        awk -F '\t' '
            NR == 1 {next}
            {
                score = $5 + 0
                argument = $3 + 0
                parse_error = $4 + 0
                if (!seen || score > best_score ||
                    (score == best_score && argument > best_argument) ||
                    (score == best_score && argument == best_argument && parse_error < best_parse)) {
                    seen = 1
                    best_lr = $1
                    best_score = score
                    best_argument = argument
                    best_parse = parse_error
                }
            }
            END {
                if (!seen) exit 1
                print best_lr
            }
        ' "$sweep_file"
    )"
    echo "$best_lr" > "$SUITE_DIR/${model}_best_memory_lr.txt"
    log "Selected $model TapMem memory learning rate: $best_lr"
}

run_model_sweep() {
    local model="$1"
    local -a sweep_lrs=("${INITIAL_SWEEP_LRS[@]}")
    local best_lr expansion_lr expansion_round
    local complete_marker="$SUITE_DIR/${model}_sweep_complete"

    prepare_sweep_tasks "$model" "${sweep_lrs[@]}"
    run_task_queue "$model seed42 TapMem LR sweep"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "5e-3"
        return
    fi
    collect_sweep_metrics "$model" "${sweep_lrs[@]}"
    select_best_lr "$model"
    best_lr="$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")"

    for expansion_round in 1 2; do
        expansion_lr=""
        if [[ "$best_lr" == "1e-3" ]]; then
            expansion_lr="5e-4"
        elif [[ "$best_lr" == "5e-4" ]]; then
            expansion_lr="2e-4"
        elif [[ "$best_lr" == "1e-2" ]]; then
            expansion_lr="2e-2"
        elif [[ "$best_lr" == "2e-2" ]]; then
            expansion_lr="5e-2"
        fi
        if [[ -z "$expansion_lr" ]]; then
            break
        fi
        log "$model best LR remains on a search boundary; adding $expansion_lr"
        sweep_lrs+=("$expansion_lr")
        prepare_sweep_tasks "$model" "$expansion_lr"
        run_task_queue "$model boundary LR extension $expansion_round"
        collect_sweep_metrics "$model" "${sweep_lrs[@]}"
        select_best_lr "$model"
        best_lr="$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")"
    done
    touch "$complete_marker"
}

prepare_final_tasks() {
    local model="$1"
    local best_lr="$2"
    local seed method task_id lora_lr task_memory_lr
    clear_tasks
    for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
            lora_lr="-"
            task_memory_lr="$best_lr"
            if [[ "$method" == "icl" || "$method" == "rag" || "$method" == "lora" ]]; then
                task_memory_lr="-"
            fi
            if [[ "$method" == "lora" || "$method" == "adap_tokmem" ]]; then
                lora_lr="${LORA_LRS[$model]}"
            elif [[ "$method" == "adap_tapmem" ]]; then
                lora_lr="${ADAP_TAPMEM_LORA_LRS[$model]}"
            fi
            task_id="${model}_${method}_seed${seed}_final"
            add_task "$task_id" final "$model" "$method" "$seed" "$task_memory_lr" "$lora_lr"
        done
    done
}

extract_icl_call_metrics() {
    local result_file="$1"
    "$JQ_BIN" -r '
        def rows($count):
            [.detailed_results[] | select((.target_calls | length) == $count)];
        [
            (rows(2) | map(.tool_metrics.tool_f1_score) | add / length),
            (rows(3) | map(.tool_metrics.tool_f1_score) | add / length),
            (rows(4) | map(.tool_metrics.tool_f1_score) | add / length),
            (rows(2) | map(.f1_score) | add / length),
            (rows(3) | map(.f1_score) | add / length),
            (rows(4) | map(.f1_score) | add / length)
        ]
        | @tsv
    ' "$result_file"
}

extract_training_call_metrics() {
    local stdout_file="$1"
    awk '
        /AVERAGE F1 SCORE \(Function Calls\)/ {
            section = "argument"
            next
        }
        /AVERAGE TOOL F1 SCORE/ {
            section = "tool"
            next
        }
        /^==========/ {
            section = ""
            next
        }
        section == "argument" && $1 ~ /^[234]$/ {
            split($3, value, "=")
            argument[$1] = value[2] + 0
        }
        section == "tool" && $1 ~ /^[234]$/ {
            split($4, value, "=")
            tool[$1] = value[2] + 0
        }
        END {
            for (count = 2; count <= 4; count++) {
                if (!(count in tool) || !(count in argument)) exit 1
            }
            printf "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
                tool[2], tool[3], tool[4],
                argument[2], argument[3], argument[4]
        }
    ' "$stdout_file"
}

collect_final_metrics() {
    local model method seed task_id task_dir overall calls
    local tool_avg argument_avg parse_error total
    local tool_2 tool_3 tool_4 argument_2 argument_3 argument_4
    local memory_lr lora_lr

    printf 'model\tmethod\tseed\tmemory_lr\tlora_lr\ttool_2\ttool_3\ttool_4\ttool_avg\targument_2\targument_3\targument_4\targument_avg\tparse_error_rate\n' \
        > "$METRICS_FILE"

    for model in "${MODEL_KEYS[@]}"; do
        if [[ ! -f "$SUITE_DIR/${model}_final_complete" ]]; then
            continue
        fi
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                task_id="${model}_${method}_seed${seed}_final"
                task_dir="$RUNS_ROOT/$task_id"
                if [[ ! -f "$task_dir/SUCCESS" || ! -f "$task_dir/evaluation_results.json" ]]; then
                    echo "Incomplete final task: $task_id" >&2
                    exit 1
                fi
                overall="$(extract_overall_metrics "$method" "$task_dir/evaluation_results.json")"
                IFS=$'\t' read -r tool_avg argument_avg parse_error total <<< "$overall"
                if [[ "$total" -ne 500 ]]; then
                    echo "Final task $task_id evaluated $total examples instead of 500." >&2
                    exit 1
                fi
                if [[ "$method" == "icl" || "$method" == "rag" ]]; then
                    calls="$(extract_icl_call_metrics "$task_dir/evaluation_results.json")"
                else
                    calls="$(extract_training_call_metrics "$task_dir/stdout.log")"
                fi
                IFS=$'\t' read -r \
                    tool_2 tool_3 tool_4 argument_2 argument_3 argument_4 <<< "$calls"
                lora_lr="-"
                if [[ "$method" == "lora" || "$method" == "adap_tokmem" ]]; then
                    lora_lr="${LORA_LRS[$model]}"
                elif [[ "$method" == "adap_tapmem" ]]; then
                    lora_lr="${ADAP_TAPMEM_LORA_LRS[$model]}"
                fi
                memory_lr="-"
                if [[ "$method" == "tokmem" || "$method" == "tapmem" || \
                    "$method" == "adap_tokmem" || "$method" == "adap_tapmem" ]]; then
                    memory_lr="$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")"
                fi
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$model" "$method" "$seed" "$memory_lr" "$lora_lr" \
                    "$tool_2" "$tool_3" "$tool_4" "$tool_avg" \
                    "$argument_2" "$argument_3" "$argument_4" "$argument_avg" \
                    "$parse_error" \
                    >> "$METRICS_FILE"
            done
        done
    done
}

format_sweep_rows() {
    local model="$1"
    local file="$SUITE_DIR/${model}_tapmem_lr_sweep_seed42.tsv"
    local selected
    selected="$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")"
    awk -F '\t' -v selected="$selected" '
        NR == 1 {next}
        {
            marker = ($1 == selected ? "yes" : "")
            printf "| `%s` | %.2f | %.2f | %.2f | %s |\n",
                $1, 100 * $2, 100 * $3, 100 * $5, marker
        }
    ' "$file"
}

format_stats_row() {
    local model="$1"
    local method="$2"
    awk -F '\t' -v wanted_model="$model" -v wanted_method="$method" '
        function cell(sum, sumsq, n, mean, variance, std) {
            mean = sum / n
            variance = (sumsq - sum * sum / n) / (n - 1)
            if (variance < 0 && variance > -1e-15) variance = 0
            std = sqrt(variance)
            return sprintf("%.1f ± %.1f", 100 * mean, 100 * std)
        }
        NR == 1 {next}
        $1 == wanted_model && $2 == wanted_method {
            n++
            for (column = 6; column <= 13; column++) {
                sum[column] += $column
                sumsq[column] += $column * $column
            }
        }
        END {
            if (n != 3) exit 1
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                cell(sum[6], sumsq[6], n),
                cell(sum[7], sumsq[7], n),
                cell(sum[8], sumsq[8], n),
                cell(sum[9], sumsq[9], n),
                cell(sum[10], sumsq[10], n),
                cell(sum[11], sumsq[11], n),
                cell(sum[12], sumsq[12], n),
                cell(sum[13], sumsq[13], n)
        }
    ' "$METRICS_FILE"
}

write_summary() {
    local model method display_model display_method lr_text stats
    {
        echo "# Qwen3.5 compositional Table 1 rebuttal experiments"
        echo
        echo "- Dataset: frozen APIGen compositional split, tools 51–100, 500 test examples, 2–4 calls."
        echo "- Methods: ICL, RAG, TokMem, TapMem, Fine-Tuning (LoRA), TokMem + adaptation, TapMem + adaptation."
        echo "- Seeds: 40, 41, and 42; cells are mean ± sample standard deviation in percent."
        echo "- TapMem memory-token learning rate is selected independently for each backbone on seed 42."
        echo "- The initial TapMem grid is 1e-3 through 1e-2; boundary winners are extended for at most two rounds, with a hard range of 2e-4 through 5e-2."
        echo "- LoRA rank/targets and LoRA learning rates remain fixed to the agreed Table 1 settings; the sweep changes only the shared synthetic-memory learning rate."
        echo "- LR selection score: equal-weight average of overall Tool F1 and Argument F1; ties prefer Argument F1, then lower parse error."
        echo "- Seed-42 tuning uses the same 500-example test split and is reported transparently as tuning-on-seed42."
        echo "- Aggregation is implemented with Bash, jq, and awk; no Python result summarizer is used."
        echo
        for model in "${MODEL_KEYS[@]}"; do
            if [[ ! -f "$SUITE_DIR/${model}_sweep_complete" ]]; then
                continue
            fi
            display_model="$(model_display_name "$model")"
            echo "## $display_model seed-42 TapMem learning-rate sweep"
            echo
            echo "| Learning rate | Tool F1 | Argument F1 | Combined | Selected |"
            echo "| ---: | ---: | ---: | ---: | :---: |"
            format_sweep_rows "$model"
            echo
        done
        echo "## Three-seed results"
        echo
        echo "| Model | Method | LR (memory / LoRA) | Tool 2c | Tool 3c | Tool 4c | Tool Avg | Arg 2c | Arg 3c | Arg 4c | Arg Avg |"
        echo "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        for model in "${MODEL_KEYS[@]}"; do
            if [[ ! -f "$SUITE_DIR/${model}_final_complete" ]]; then
                continue
            fi
            display_model="$(model_display_name "$model")"
            for method in "${METHODS[@]}"; do
                display_method="$(method_display_name "$method")"
                case "$method" in
                    icl|rag)
                        lr_text="—"
                        ;;
                    tokmem|tapmem)
                        lr_text="\`$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")\` / —"
                        ;;
                    lora)
                        lr_text="— / \`${LORA_LRS[$model]}\`"
                        ;;
                    adap_tokmem)
                        lr_text="\`$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")\` / \`${LORA_LRS[$model]}\`"
                        ;;
                    adap_tapmem)
                        lr_text="\`$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")\` / \`${ADAP_TAPMEM_LORA_LRS[$model]}\`"
                        ;;
                esac
                stats="$(format_stats_row "$model" "$method")"
                echo "| $display_model | $display_method | $lr_text | ${stats//$'\t'/ | } |"
            done
        done
        echo
        echo "Suite directory: \`$SUITE_DIR\`"
    } > "$SUMMARY_FILE"
}

dry_run_all_commands() {
    local model lr best_lr seed method task_id lora_lr
    for model in "${MODEL_KEYS[@]}"; do
        prepare_sweep_tasks "$model" "${INITIAL_SWEEP_LRS[@]}"
        run_task_queue "$model dry-run sweep"
        best_lr="5e-3"
        prepare_final_tasks "$model" "$best_lr"
        run_task_queue "$model dry-run final"
    done
}

main() {
    preflight
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$SUITE_DIR"
    fi
    acquire_suite_lock
    initialize_suite
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry_run_all_commands
        return
    fi

    source /home/shilong/anaconda3/etc/profile.d/conda.sh
    conda activate tokmem
    export HF_HOME="$HF_CACHE_DIR"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TOKENIZERS_PARALLELISM=false

    log "Suite directory: $SUITE_DIR"
    log "Dynamic GPU pool: $GPU_IDS_CSV"
    log "GPU memory eligibility threshold: <= ${GPU_MEMORY_LIMIT_MIB} MiB used"

    local model best_lr
    for model in "${MODEL_KEYS[@]}"; do
        log "Beginning $(model_display_name "$model")"
        if [[ -f "$SUITE_DIR/${model}_sweep_complete" && \
            -f "$SUITE_DIR/${model}_best_memory_lr.txt" ]]; then
            best_lr="$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")"
            log "Reusing selected $model memory LR: $best_lr"
        else
            run_model_sweep "$model"
            best_lr="$(cat "$SUITE_DIR/${model}_best_memory_lr.txt")"
        fi
        prepare_final_tasks "$model" "$best_lr"
        run_task_queue "$model final three-seed matrix"
        touch "$SUITE_DIR/${model}_final_complete"
        collect_final_metrics
        write_summary
        log "Completed $(model_display_name "$model") final matrix"
    done

    collect_final_metrics
    write_summary
    log "Suite completed: $SUITE_DIR"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
