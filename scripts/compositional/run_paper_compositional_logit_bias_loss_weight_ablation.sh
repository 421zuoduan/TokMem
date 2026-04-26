#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="paper_compositional_logit_bias_loss_weight_ablation_llama1b_4calls_8gpu_${SUITE_TIMESTAMP}"
SUITE_DIR="$ROOT_DIR/results/compositional/$SUITE_NAME"
LOG_FILE="$SUITE_DIR/nohup.log"
PID_FILE="$SUITE_DIR/nohup.pid"

mkdir -p "$SUITE_DIR"

nohup bash -c '
set -euo pipefail
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
exec bash "$1" --gpus "0,1,2,3,4,5,6,7" --suite-name "$2"
' _ "$ROOT_DIR/scripts/compositional/llama_1b/rerun_paper_compositional_logit_bias_loss_weight_ablation.sh" "$SUITE_NAME" > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"

echo "suite: $SUITE_NAME"
echo "pid: $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
echo "manifest: $SUITE_DIR/manifest.tsv"
echo "summary: $SUITE_DIR/summary.md"
echo "results: $SUITE_DIR/results.json"
