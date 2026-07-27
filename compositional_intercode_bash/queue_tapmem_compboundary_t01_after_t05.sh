#!/usr/bin/env bash
set -euo pipefail

repo_dir=/data/shilong/tokmem
dependency_run=tapmem_llama8b_compboundary_onehot_t05_e100_seed42_v1
queued_run=tapmem_llama8b_compboundary_onehot_t01_e100_seed42_v1
dependency_pid_file="/tmp/${dependency_run}.pid"
dependency_summary="$repo_dir/compositional_intercode_bash/evaluations/${dependency_run}_official_agent_parse_preserve_full200_v1/summary_10_turn.json"
queued_launcher="$repo_dir/compositional_intercode_bash/run_tapmem_compboundary_onehot_t01_e100_seed42_pipeline.sh"

if [[ ! -s "$dependency_pid_file" ]]; then
    echo "[$(date --iso-8601=seconds)] missing dependency PID file: $dependency_pid_file"
    exit 1
fi

dependency_pid=$(<"$dependency_pid_file")
echo "[$(date --iso-8601=seconds)] waiting for $dependency_run PID $dependency_pid"

while kill -0 "$dependency_pid" 2>/dev/null; do
    if [[ -f "$dependency_summary" ]]; then
        echo "[$(date --iso-8601=seconds)] dependency summary exists; waiting for pipeline cleanup"
    else
        echo "[$(date --iso-8601=seconds)] dependency still running"
    fi
    sleep 60
done

if [[ ! -f "$dependency_summary" ]]; then
    echo "[$(date --iso-8601=seconds)] dependency stopped without summary: $dependency_summary"
    exit 1
fi

if [[ -e "$repo_dir/compositional_intercode_bash/runs/$queued_run/checkpoint.json" ]]; then
    echo "[$(date --iso-8601=seconds)] queued checkpoint already exists"
    exit 1
fi

echo "[$(date --iso-8601=seconds)] dependency complete; starting $queued_run"
exec bash "$queued_launcher"
