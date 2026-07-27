#!/usr/bin/env bash
set -uo pipefail

repo_dir=/data/shilong/tokmem
checkpoint_dir="$repo_dir/compositional_intercode_bash/runs/tapmem_llama8b_eocmask_t1_e50_seed42_v1"
output_dir="$repo_dir/compositional_intercode_bash/evaluations/tapmem_llama8b_eocmask_t1_e50_seed42_v1_official_agent_parse_preserve_full200_v1"
python_bin=/home/shilong/anaconda3/envs/tokmem/bin/python
runtime_script="$repo_dir/compositional_intercode_bash/single_uid_rootless_runtime.sh"
shard_count=4
gpus=(3 3 5 5)

mkdir -p "$output_dir/logs"
exec >>"$output_dir/coordinator.log" 2>&1

echo "[$(date --iso-8601=seconds)] starting rootless Docker"
"$runtime_script" start
eval "$("$runtime_script" env)"
export DOCKER_BUILDKIT=0
export PYTHONUNBUFFERED=1

evaluate_shard() {
    local shard_index=$1
    local gpu=$2
    local phase=$3
    local log_path="$output_dir/logs/shard_${shard_index}_${phase}.log"

    echo "[$(date --iso-8601=seconds)] starting shard $shard_index on GPU $gpu ($phase)"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        -m compositional_intercode_bash.evaluate \
        --checkpoint "$checkpoint_dir" \
        --output-dir "$output_dir" \
        --max-turns 10 \
        --shard-count "$shard_count" \
        --shard-index "$shard_index" \
        --resume \
        --allow-exploratory-checkpoint \
        >>"$log_path" 2>&1
}

cd "$repo_dir"

declare -a shard_pids
declare -a shard_status
for shard_index in 0 1 2 3; do
    evaluate_shard "$shard_index" "${gpus[$shard_index]}" parallel &
    shard_pids[$shard_index]=$!
done

for shard_index in 0 1 2 3; do
    status=0
    wait "${shard_pids[$shard_index]}" || status=$?
    shard_status[$shard_index]=$status
    echo "[$(date --iso-8601=seconds)] shard $shard_index parallel exit=$status"
done

for shard_index in 0 1 2 3; do
    if [[ "${shard_status[$shard_index]}" -eq 0 ]]; then
        continue
    fi
    recovered=0
    for attempt in 1 2 3; do
        echo "[$(date --iso-8601=seconds)] recovering shard $shard_index attempt=$attempt"
        if evaluate_shard "$shard_index" "${gpus[$shard_index]}" "recovery_${attempt}"; then
            recovered=1
            break
        fi
    done
    if [[ "$recovered" -eq 0 ]]; then
        echo "[$(date --iso-8601=seconds)] shard $shard_index recovery exhausted; final resume will fill missing episodes"
    fi
done

merge_status=1
for attempt in 1 2 3; do
    echo "[$(date --iso-8601=seconds)] starting full resume and strict merge attempt=$attempt"
    if CUDA_VISIBLE_DEVICES=3 "$python_bin" \
        -m compositional_intercode_bash.evaluate \
        --checkpoint "$checkpoint_dir" \
        --output-dir "$output_dir" \
        --max-turns 10 \
        --resume \
        --allow-exploratory-checkpoint \
        >>"$output_dir/logs/final_resume_${attempt}.log" 2>&1; then
        merge_status=0
        break
    fi
done

if [[ "$merge_status" -ne 0 ]]; then
    echo "[$(date --iso-8601=seconds)] FAILED: final resume did not complete"
    exit "$merge_status"
fi

echo "[$(date --iso-8601=seconds)] COMPLETED: 200-task summary is ready at $output_dir/summary_10_turn.json"
