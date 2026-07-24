#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DATA_DIR="$ROOT_DIR/results/compositional/all_methods/data"
RESULTS_DIR="$ROOT_DIR/results/compositional"
SUMMARY_SCRIPT="$ROOT_DIR/scripts/compositional/summarize_paper_table1_seeds.py"
REFERENCE_FILE="$ROOT_DIR/scripts/compositional/table1_seed42_reference.json"

SEEDS=(40 41)
MODELS=(llama1b llama3b llama8b)
METHODS=(icl rag tokmem tapmem lora adap_tokmem adap_tapmem)
IFS=',' read -r -a GPUS <<< "${GPU_IDS-0,1,2,3,4,5,6}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"
GPU_LOCK_DIR="/tmp/tokmem_gpu_locks"

DRY_RUN=0
RUN_MODE="all"
for option in "$@"; do
    case "$option" in
        --dry-run) DRY_RUN=1 ;;
        --inference-1b3b) RUN_MODE="inference-1b3b" ;;
        --remaining) RUN_MODE="remaining" ;;
        *)
            echo "Usage: GPU_IDS=0,1,... bash $0 [--inference-1b3b|--remaining] [--dry-run]" >&2
            exit 2
            ;;
    esac
done

if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
    echo "GPU_IDS must contain at least one GPU ID." >&2
    exit 2
fi
for gpu in "${GPUS[@]}"; do
    if ! [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "GPU_IDS must contain canonical decimal GPU indexes such as 0,1,2." >&2
        exit 2
    fi
done
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
    echo "GPU_IDS must not contain duplicates." >&2
    exit 2
fi
if ! [[ "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_POLL_SECONDS must be a positive integer." >&2
    exit 2
fi
if ! [[ "$GPU_MEMORY_LIMIT_MIB" =~ ^[0-9]+$ ]]; then
    echo "GPU_MEMORY_LIMIT_MIB must be a nonnegative integer." >&2
    exit 2
fi
if ! [[ "$GPU_MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "GPU_MIN_FREE_MIB must be a nonnegative integer." >&2
    exit 2
fi

SUITE_NAME="${SUITE_NAME:-paper_table1_seeds40_41_$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="$RESULTS_DIR/$SUITE_NAME"
RUNS_DIR="$SUITE_DIR/runs"
MANIFEST="$SUITE_DIR/task_manifest.tsv"

TASK_MODELS=()
TASK_METHODS=()
TASK_SEEDS=()

for seed in "${SEEDS[@]}"; do
    for model in "${MODELS[@]}"; do
        for method in "${METHODS[@]}"; do
            is_inference_1b3b=0
            if [[ "$model" != "llama8b" && \
                ( "$method" == "icl" || "$method" == "rag" ) ]]; then
                is_inference_1b3b=1
            fi
            if [[ "$RUN_MODE" == "inference-1b3b" && "$is_inference_1b3b" -eq 0 ]]; then
                continue
            fi
            if [[ "$RUN_MODE" == "remaining" && "$is_inference_1b3b" -eq 1 ]]; then
                continue
            fi
            TASK_MODELS+=("$model")
            TASK_METHODS+=("$method")
            TASK_SEEDS+=("$seed")
        done
    done
done
if [[ "$RUN_MODE" != "inference-1b3b" ]]; then
    TASK_MODELS+=("llama3b")
    TASK_METHODS+=("tokmem")
    TASK_SEEDS+=("42")
fi

configure_model() {
    case "$1" in
        llama1b)
            MODEL_PATH="$ROOT_DIR/models/Llama-3.2-1B-Instruct"
            ICL_BATCH=64
            RAG_BATCH=256
            LORA_BATCH=16
            LORA_EVAL_BATCH=128
            TOKMEM_BATCH=24
            TOKMEM_EVAL_BATCH=256
            ADAPT_BATCH="16,24"
            LORA_LR="5e-5"
            ADAPT_TOKMEM_LORA_LR="5e-5"
            ;;
        llama3b)
            MODEL_PATH="$ROOT_DIR/models/Llama-3.2-3B-Instruct"
            ICL_BATCH=32
            RAG_BATCH=192
            LORA_BATCH=8
            LORA_EVAL_BATCH=96
            TOKMEM_BATCH=16
            TOKMEM_EVAL_BATCH=192
            ADAPT_BATCH="8,16"
            LORA_LR="5e-5"
            ADAPT_TOKMEM_LORA_LR="5e-5"
            ;;
        llama8b)
            MODEL_PATH="$ROOT_DIR/models/Llama-3.1-8B-Instruct"
            ICL_BATCH=24
            RAG_BATCH=128
            LORA_BATCH=4
            LORA_EVAL_BATCH=32
            TOKMEM_BATCH=8
            TOKMEM_EVAL_BATCH=64
            ADAPT_BATCH="4,8"
            LORA_LR="8e-5"
            ADAPT_TOKMEM_LORA_LR="8e-5"
            ;;
    esac

    if [[ "$RUN_MODE" == "inference-1b3b" ]]; then
        if [[ "$1" == "llama1b" ]]; then
            ICL_BATCH=8
            RAG_BATCH=32
        elif [[ "$1" == "llama3b" ]]; then
            ICL_BATCH=4
            RAG_BATCH=16
        fi
    fi
}

build_command() {
    local method="$1"
    local seed="$2"
    local task_name="$3"

    case "$method" in
        icl)
            COMMAND=(
                python -u icl_baseline.py
                --test_data "$DATA_DIR/test/function_calling_test_tools51-100_4calls.json"
                --tool_descriptions "$DATA_DIR/tool_descriptions_tools51-100.json"
                --model_name "$MODEL_PATH"
                --batch_size "$ICL_BATCH"
                --seed "$seed"
                --run_root_dir "$RUNS_DIR"
                --run_name "$task_name"
            )
            ;;
        rag)
            COMMAND=(
                python -u icl_baseline.py
                --test_data "$DATA_DIR/test/function_calling_test_tools51-100_4calls.json"
                --tool_descriptions "$DATA_DIR/tool_descriptions_tools51-100.json"
                --model_name "$MODEL_PATH"
                --retriever_model_name "$ROOT_DIR/models/all-MiniLM-L6-v2"
                --batch_size "$RAG_BATCH"
                --seed "$seed"
                --use_rag
                --retrieval_k 5
                --run_root_dir "$RUNS_DIR"
                --run_name "$task_name"
            )
            ;;
        lora)
            COMMAND=(
                python -u lora_sequential.py
                --training_rounds "51-100:3"
                --batch_size "$LORA_BATCH"
                --train_max_function_calls 4
                --test_max_function_calls 4
                --model_name "$MODEL_PATH"
                --lora_r 8
                --lora_alpha 32
                --lora_dropout 0.1
                --lora_target_modules "q_proj,v_proj"
                --eval_after_each_round
                --save_checkpoints
                --data_dir "$DATA_DIR"
                --lr "$LORA_LR"
                --eval_batch_size "$LORA_EVAL_BATCH"
                --max_length 512
                --seed "$seed"
                --run_root_dir "$RUNS_DIR"
                --run_name "$task_name"
            )
            ;;
        tokmem|tapmem)
            COMMAND=(
                python -u main_sequential.py
                --training_rounds "51-100:1"
                --epochs 3
                --batch_size "$TOKMEM_BATCH"
                --train_max_function_calls 4
                --test_max_function_calls 4
                --model_name "$MODEL_PATH"
                --eval_after_each_round
                --save_checkpoints
                --data_dir "$DATA_DIR"
                --lr "5e-3"
                --eval_batch_size "$TOKMEM_EVAL_BATCH"
                --max_length 512
                --max_new_tokens 512
                --seed "$seed"
                --tensorboard
                --run_root_dir "$RUNS_DIR"
                --run_name "$task_name"
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
            local adaptation_lora_lr="$ADAPT_TOKMEM_LORA_LR"
            if [[ "$method" == "adap_tapmem" ]]; then
                adaptation_lora_lr="8e-5"
            fi
            COMMAND=(
                python -u main_sequential.py
                --training_rounds "1-50:1,51-100:3"
                --batch_size_per_round "$ADAPT_BATCH"
                --train_max_function_calls_per_round "4,4"
                --test_max_function_calls_per_round "4,4"
                --model_name "$MODEL_PATH"
                --use_lora
                --lora_r 8
                --lora_alpha 32
                --lora_dropout 0.1
                --lora_target_modules "q_proj,v_proj"
                --freeze_lora_after_first
                --eval_after_each_round
                --save_checkpoints
                --data_dir "$DATA_DIR"
                --lr "5e-3"
                --lora_lr "$adaptation_lora_lr"
                --eval_batch_size "$TOKMEM_EVAL_BATCH"
                --max_length 512
                --max_new_tokens 512
                --seed "$seed"
                --tensorboard
                --run_root_dir "$RUNS_DIR"
                --run_name "$task_name"
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
    esac
}

run_one() {
    local model="$1"
    local method="$2"
    local seed="$3"
    local gpu="$4"
    local task_name="${model}_${method}_seed${seed}"
    local task_dir="$RUNS_DIR/$task_name"

    configure_model "$model"
    build_command "$method" "$seed" "$task_name"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run] GPU %s: ' "$gpu"
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
        return
    fi

    mkdir -p "$task_dir"
    {
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    } > "$task_dir/expected_command.txt"

    if [[ -f "$task_dir/SUCCESS" && -f "$task_dir/evaluation_results.json" ]]; then
        if ! cmp -s "$task_dir/command.txt" "$task_dir/expected_command.txt"; then
            echo "Refusing to reuse $task_name because its command changed." >&2
            return 2
        fi
        rm "$task_dir/expected_command.txt"
        echo "Reusing $task_name"
        return
    fi

    mv "$task_dir/expected_command.txt" "$task_dir/command.txt"
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    } > "$task_dir/command.sh"
    rm -f "$task_dir/SUCCESS" "$task_dir/FAILED"

    echo "Starting $task_name on GPU $gpu"
    if (
        cd "$ROOT_DIR/compositional"
        CUDA_VISIBLE_DEVICES="$gpu" "${COMMAND[@]}"
    ) > "$task_dir/stdout.log" 2>&1; then
        touch "$task_dir/SUCCESS"
        echo "Completed $task_name"
    else
        local exit_code=$?
        touch "$task_dir/FAILED"
        echo "Failed $task_name with exit code $exit_code" >&2
        return "$exit_code"
    fi
}

gpu_memory_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "$1" 2>/dev/null | awk 'NR == 1 {gsub(/^[ \t]+|[ \t]+$/, ""); print}'
}

gpu_memory_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        -i "$1" 2>/dev/null | awk 'NR == 1 {gsub(/^[ \t]+|[ \t]+$/, ""); print}'
}

gpu_is_free() {
    local memory_free memory_used
    if [[ "$RUN_MODE" == "inference-1b3b" ]]; then
        memory_free="$(gpu_memory_free_mib "$1")" || return 2
        [[ "$memory_free" =~ ^[0-9]+$ ]] || return 2
        (( memory_free >= GPU_MIN_FREE_MIB ))
        return
    fi
    memory_used="$(gpu_memory_used_mib "$1")" || return 2
    [[ "$memory_used" =~ ^[0-9]+$ ]] || return 2
    (( memory_used <= GPU_MEMORY_LIMIT_MIB ))
}

gpu_lock_file() {
    local lock_name="${1//[^A-Za-z0-9_.-]/_}"
    echo "$GPU_LOCK_DIR/gpu_${lock_name}.lock"
}

run_one_locked() {
    local model="$1"
    local method="$2"
    local seed="$3"
    local gpu="$4"

    echo "Waiting for GPU lock: GPU $gpu"
    flock -x 200
    echo "GPU lock acquired: GPU $gpu"
    while ! gpu_is_free "$gpu"; do
        sleep "$GPU_POLL_SECONDS"
    done
    run_one "$model" "$method" "$seed" "$gpu"
}

if [[ "$DRY_RUN" -eq 0 && ! -d "$SUITE_DIR" ]]; then
    mkdir -p "$RUNS_DIR" "$SUITE_DIR/hf-cache"
    cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"
    printf 'model\tmethod\tseed\tgpu\ttask_dir\n' > "$MANIFEST"
    for seed in "${SEEDS[@]}"; do
        for model in "${MODELS[@]}"; do
            for method in "${METHODS[@]}"; do
                task_name="${model}_${method}_seed${seed}"
                printf '%s\t%s\t%s\tdynamic\t%s\n' \
                    "$model" "$method" "$seed" "$RUNS_DIR/$task_name" >> "$MANIFEST"
            done
        done
    done
    printf 'llama3b\ttokmem\t42\tdynamic\t%s\n' \
        "$RUNS_DIR/llama3b_tokmem_seed42" >> "$MANIFEST"
    sha256sum \
        "$DATA_DIR/training/function_calling_train_tools1-50_4calls.json" \
        "$DATA_DIR/training/function_calling_train_tools51-100_4calls.json" \
        "$DATA_DIR/test/function_calling_test_tools1-50_4calls.json" \
        "$DATA_DIR/test/function_calling_test_tools51-100_4calls.json" \
        "$DATA_DIR/tool_descriptions_tools51-100.json" \
        "$REFERENCE_FILE" \
        "$SUMMARY_SCRIPT" \
        "$ROOT_DIR/scripts/compositional/summarize_completed_trials.py" \
        "$ROOT_DIR"/compositional/*.py \
        > "$SUITE_DIR/input_sha256.txt"
    git -C "$ROOT_DIR" rev-parse HEAD > "$SUITE_DIR/source_commit.txt"
    git -C "$ROOT_DIR" status --short > "$SUITE_DIR/source_worktree_status.txt"
elif [[ "$DRY_RUN" -eq 0 ]] && \
    ! cmp -s "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"; then
    echo "Refusing to resume because the launcher differs from the suite snapshot." >&2
    exit 2
elif [[ "$DRY_RUN" -eq 0 ]] && \
    [[ "$(git -C "$ROOT_DIR" rev-parse HEAD)" != "$(cat "$SUITE_DIR/source_commit.txt")" ]]; then
    echo "Refusing to resume because the Git commit changed." >&2
    exit 2
elif [[ "$DRY_RUN" -eq 0 ]] && \
    ! sha256sum --check --status "$SUITE_DIR/input_sha256.txt"; then
    echo "Refusing to resume because code, data, or the reference changed." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    source /home/shilong/anaconda3/etc/profile.d/conda.sh
    conda activate tokmem
    export HF_HOME="$SUITE_DIR/hf-cache"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TOKENIZERS_PARALLELISM=false
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    for index in "${!TASK_MODELS[@]}"; do
        gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
        run_one \
            "${TASK_MODELS[$index]}" \
            "${TASK_METHODS[$index]}" \
            "${TASK_SEEDS[$index]}" \
            "$gpu"
    done
    exit 0
fi

for gpu in "${GPUS[@]}"; do
    if [[ "$RUN_MODE" == "inference-1b3b" ]]; then
        if ! gpu_query="$(gpu_memory_free_mib "$gpu")"; then
            gpu_query=""
        fi
    else
        if ! gpu_query="$(gpu_memory_used_mib "$gpu")"; then
            gpu_query=""
        fi
    fi
    if ! [[ "$gpu_query" =~ ^[0-9]+$ ]]; then
        echo "Cannot query GPU $gpu with nvidia-smi." >&2
        exit 2
    fi
done
if ! command -v flock >/dev/null; then
    echo "flock is required for GPU scheduling." >&2
    exit 2
fi
mkdir -p "$GPU_LOCK_DIR"

declare -A GPU_PIDS=()
next_task=0
task_failed=0

while [[ "$next_task" -lt "${#TASK_MODELS[@]}" || "${#GPU_PIDS[@]}" -gt 0 ]]; do
    made_progress=0

    for gpu in "${GPUS[@]}"; do
        pid="${GPU_PIDS[$gpu]:-}"
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            if ! wait "$pid"; then
                task_failed=1
            fi
            unset 'GPU_PIDS[$gpu]'
            made_progress=1
        fi
    done

    for gpu in "${GPUS[@]}"; do
        if [[ "$next_task" -ge "${#TASK_MODELS[@]}" ]]; then
            break
        fi
        if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then
            continue
        fi
        if ! gpu_is_free "$gpu"; then
            continue
        fi

        echo "Dispatching ${TASK_MODELS[$next_task]}_${TASK_METHODS[$next_task]}_seed${TASK_SEEDS[$next_task]} to GPU $gpu"
        run_one_locked \
            "${TASK_MODELS[$next_task]}" \
            "${TASK_METHODS[$next_task]}" \
            "${TASK_SEEDS[$next_task]}" \
            "$gpu" 200>"$(gpu_lock_file "$gpu")" &
        GPU_PIDS["$gpu"]=$!
        next_task=$((next_task + 1))
        made_progress=1
        break
    done

    if [[ "$made_progress" -eq 0 ]]; then
        echo "Waiting for a free GPU: pending=$((${#TASK_MODELS[@]} - next_task)), running=${#GPU_PIDS[@]}"
        sleep "$GPU_POLL_SECONDS"
    fi
done

if [[ "$task_failed" -ne 0 ]]; then
    echo "At least one task failed. Rerun with SUITE_NAME=$SUITE_NAME after inspection." >&2
    exit 1
fi

completed_tasks="$(find "$RUNS_DIR" -mindepth 2 -maxdepth 2 -name SUCCESS | wc -l)"
if [[ "$completed_tasks" -eq 43 ]]; then
    (
        flock -x 201
        python "$SUMMARY_SCRIPT" "$SUITE_DIR"
    ) 201>"$SUITE_DIR/summary.lock"
    echo "Suite completed: $SUITE_DIR"
else
    echo "$RUN_MODE queue completed: $completed_tasks/43 total tasks are ready."
fi
