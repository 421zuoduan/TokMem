#!/usr/bin/env bash
set -uo pipefail

repo_dir=/data/shilong/tokmem
python_bin=/home/shilong/anaconda3/envs/tokmem/bin/python
runtime_script="$repo_dir/compositional_intercode_bash/single_uid_rootless_runtime.sh"
e80=tokmem_llama8b_current_e80_seed42_v1
e90=tokmem_llama8b_current_e90_seed42_v1
eval_suffix=official_agent_parse_preserve_full200_v1

checkpoint_dir() {
    printf '%s/compositional_intercode_bash/runs/%s' "$repo_dir" "$1"
}

evaluation_dir() {
    printf '%s/compositional_intercode_bash/evaluations/%s_%s' "$repo_dir" "$1" "$eval_suffix"
}

"$runtime_script" start
eval "$("$runtime_script" env)"
export DOCKER_BUILDKIT=0
export PYTHONUNBUFFERED=1

run_shard_until_complete() {
    local run_name=$1
    local shard_index=$2
    local gpu=$3
    local checkpoint
    local output
    local summary
    local attempt=0

    checkpoint=$(checkpoint_dir "$run_name")
    output=$(evaluation_dir "$run_name")
    summary=$(printf '%s/shard_summary_10_turn_%03d_of_004.json' "$output" "$shard_index")
    mkdir -p "$output/logs"

    while [[ ! -f "$summary" ]]; do
        attempt=$((attempt + 1))
        echo "[$(date --iso-8601=seconds)] run=$run_name shard=$shard_index attempt=$attempt gpu=$gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
            -m compositional_intercode_bash.evaluate \
            --checkpoint "$checkpoint" \
            --output-dir "$output" \
            --max-turns 10 \
            --shard-count 4 \
            --shard-index "$shard_index" \
            --resume \
            --allow-exploratory-checkpoint \
            >>"$output/logs/coordinator_shard_${shard_index}.log" 2>&1 || true
        sleep 2
    done
}

echo "[$(date --iso-8601=seconds)] launching missing shards"
run_shard_until_complete "$e80" 3 6 &
missing_e80_pid=$!
sleep 3
run_shard_until_complete "$e90" 1 3 &
missing_e90_pid=$!

wait "$missing_e80_pid"
wait "$missing_e90_pid"
echo "[$(date --iso-8601=seconds)] initially missing shards complete"

while pgrep -f 'python -m compositional_intercode_bash.evaluate.*tokmem_llama8b_current_e(80|90)_seed42_v1' >/dev/null; do
    sleep 15
done

echo "[$(date --iso-8601=seconds)] checking every shard for a strict shard summary"
for shard_index in 0 1 2 3; do
    run_shard_until_complete "$e80" "$shard_index" 6
done
for shard_index in 0 1 2 3; do
    run_shard_until_complete "$e90" "$shard_index" 3
done

finalize_until_complete() {
    local run_name=$1
    local gpu=$2
    local checkpoint
    local output
    local summary
    local attempt=0

    checkpoint=$(checkpoint_dir "$run_name")
    output=$(evaluation_dir "$run_name")
    summary="$output/summary_10_turn.json"
    while [[ ! -f "$summary" ]]; do
        attempt=$((attempt + 1))
        echo "[$(date --iso-8601=seconds)] finalizing run=$run_name attempt=$attempt gpu=$gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
            -m compositional_intercode_bash.evaluate \
            --checkpoint "$checkpoint" \
            --output-dir "$output" \
            --max-turns 10 \
            --resume \
            --allow-exploratory-checkpoint \
            >>"$output/logs/coordinator_final.log" 2>&1 || true
        sleep 2
    done
}

finalize_until_complete "$e80" 6
finalize_until_complete "$e90" 3
echo "[$(date --iso-8601=seconds)] COMPLETED e80 and e90"
