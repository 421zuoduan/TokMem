#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="paper_compositional_head_8gpu_${SUITE_TIMESTAMP}"
SUITE_DIR="$ROOT_DIR/results/compositional/$SUITE_NAME"
LOG_FILE="$SUITE_DIR/nohup.log"
PID_FILE="$SUITE_DIR/nohup.pid"

mkdir -p "$SUITE_DIR"

nohup bash -c '
set -euo pipefail
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
exec bash "$1" --gpus "0,1,2,3,4,5,6,7" --suite-name "$2"
' _ "$ROOT_DIR/scripts/compositional/rerun_paper_compositional_head.sh" "$SUITE_NAME" > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"

echo "suite: $SUITE_NAME"
echo "pid: $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
echo "status: $SUITE_DIR/task_status.json"
echo "summary: $SUITE_DIR/summary.md"
