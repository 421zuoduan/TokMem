#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOOLATHLON_AGENT_PYTHON="/home/shilong/anaconda3/envs/tokmem-qwen35/bin/python"
export TOOLATHLON_SUITE_CHECKPOINT_MODEL="qwen35_9b"
export TOOLATHLON_SUITE_EPOCH_TAG="e20"
export TOOLATHLON_SUITE_OUTPUT_TAG="qwen35_9b_e20"

exec bash "$SCRIPT_DIR/run_seed42_e10_official_suite.sh" "$@"
