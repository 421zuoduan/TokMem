#!/usr/bin/env bash
set -uo pipefail

repo_dir=/data/shilong/tokmem
python_bin=/home/shilong/anaconda3/envs/tokmem/bin/python
run_name=tokmem_llama8b_current_e90_seed42_v1
checkpoint="$repo_dir/compositional_intercode_bash/runs/$run_name"
output="$repo_dir/compositional_intercode_bash/evaluations/${run_name}_official_agent_parse_preserve_full200_v1"
runtime_script="$repo_dir/compositional_intercode_bash/single_uid_rootless_runtime.sh"

"$runtime_script" start
eval "$("$runtime_script" env)"
export DOCKER_BUILDKIT=0
export PYTHONUNBUFFERED=1

recover_shard() {
    local shard_index=$1
    local gpu=$2
    local summary
    local attempt=0

    summary=$(printf '%s/shard_summary_10_turn_%03d_of_004.json' "$output" "$shard_index")
    while [[ ! -f "$summary" ]]; do
        attempt=$((attempt + 1))
        echo "[$(date --iso-8601=seconds)] shard=$shard_index attempt=$attempt gpu=$gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
            -m compositional_intercode_bash.evaluate \
            --checkpoint "$checkpoint" \
            --output-dir "$output" \
            --max-turns 10 \
            --shard-count 4 \
            --shard-index "$shard_index" \
            --resume \
            --allow-exploratory-checkpoint \
            >>"$output/logs/parallel_recovery_shard_${shard_index}.log" 2>&1 || true
        sleep 2
    done
}

recover_shard 2 5 &
pid2=$!
sleep 3
recover_shard 3 7 &
pid3=$!
wait "$pid2"
wait "$pid3"
echo "[$(date --iso-8601=seconds)] e90 shards 2 and 3 complete"
