#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRACK_DIR/.." && pwd)"
AGENT_PYTHON="${TOOLATHLON_AGENT_PYTHON:-/home/shilong/anaconda3/envs/tokmem/bin/python}"
AGENT_BIN="$(dirname "$AGENT_PYTHON")"
HOST_RUNTIME_BIN="${TOOLATHLON_HOST_RUNTIME_BIN:-/home/shilong/anaconda3/envs/tokmem/bin}"
MANIFEST="$TRACK_DIR/data/llm_v1/manifests/tool_manifest.json"
IMAGE="127.0.0.1:5055/lockon0927/toolathlon-task-image:1016beta-singleuid"
GPU="${TOOLATHLON_SUITE_GPU:-1}"
LOCK_ID="${TOOLATHLON_SUITE_LOCK_ID:-default}"
CHECKPOINT_MODEL="${TOOLATHLON_SUITE_CHECKPOINT_MODEL:-llama31_8b}"
EPOCH_TAG="${TOOLATHLON_SUITE_EPOCH_TAG:-e10}"
OUTPUT_TAG="${TOOLATHLON_SUITE_OUTPUT_TAG:-e10}"
for identifier in "$LOCK_ID" "$CHECKPOINT_MODEL" "$EPOCH_TAG" "$OUTPUT_TAG"; do
    if [[ ! "$identifier" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Suite identifiers must contain only letters, digits, dot, underscore, or hyphen" >&2
        exit 2
    fi
done
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/compositional-toolathlon-seed42-${OUTPUT_TAG}-${LOCK_ID}.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another seed42/e10 official suite is already running: $LOCK_FILE" >&2
    exit 2
fi

run_is_complete() {
    local rollout_path="$1"
    local evaluator_path="$2"
    local expected_method="$3"
    local expected_task="$4"
    "$AGENT_PYTHON" -c '
import json
import sys

rollout_path, evaluator_path, method, task = sys.argv[1:]
try:
    outer = json.load(open(rollout_path, encoding="utf-8"))
    evaluator = json.load(open(evaluator_path, encoding="utf-8"))
    rollout = outer["rollout"]
    complete = (
        outer.get("failure") is None
        and isinstance(rollout, dict)
        and rollout.get("method") == method
        and rollout.get("task_dir") == f"finalpool/{task}"
        and isinstance(rollout.get("events"), list)
        and evaluator.get("pass") in (True, False, None)
        and "pass" in evaluator
    )
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    complete = False
raise SystemExit(0 if complete else 1)
' "$rollout_path" "$evaluator_path" "$expected_method" "$expected_task"
}

if [[ "$#" -eq 0 ]]; then
    TASKS=(
        arrange-workspace
        courses-ta-hws
        detect-revised-terms
        excel-data-transformation
        excel-market-research
        imagenet
        paper-checker
        privacy-desensitization
        reimbursement-form-filler
        university-course-selection
    )
else
    TASKS=("$@")
fi

for task in "${TASKS[@]}"; do
    task_slug="${task//-/_}"
    for method in tapmem tokmem; do
        run_dir="$TRACK_DIR/runs/mixed_v1_${CHECKPOINT_MODEL}_${method}_seed42_${EPOCH_TAG}"
        output_root="$TRACK_DIR/runs/exploratory_singleuid_derived/${method}_${OUTPUT_TAG}_${task_slug}_trial0"
        task_output="$output_root/finalpool/$task"
        if run_is_complete \
            "$task_output/tokmem_rollout.json" \
            "$task_output/eval_res.json" \
            "$method" \
            "$task"; then
            echo "SKIP method=$method task=$task complete=$task_output"
            continue
        fi
        echo "RUN method=$method task=$task output=$task_output"
        env \
            PATH="$AGENT_BIN:$HOST_RUNTIME_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
            CUDA_VISIBLE_DEVICES="$GPU" \
            TOKMEM_PROJECT_ROOT="$PROJECT_ROOT" \
            TOKMEM_AGENT_PYTHON="$AGENT_PYTHON" \
            TOKMEM_RUN_DIR="$run_dir" \
            TOKMEM_TOOL_MANIFEST="$MANIFEST" \
            TOKMEM_EPISODE_ID="${method}_${OUTPUT_TAG}_${task_slug}_trial0" \
            bash "$SCRIPT_DIR/with_rootless_docker.sh" \
            bash "$SCRIPT_DIR/run_pinned_official_task.sh" \
            "finalpool/$task" \
            quickstart \
            "$output_root" \
            "$method" \
            unified \
            100 \
            scripts/formal_run_v0.json \
            "$IMAGE" \
            tokmem_runtime
        exit_code="$?"
        echo "DONE method=$method task=$task exit_code=$exit_code"
    done
done
