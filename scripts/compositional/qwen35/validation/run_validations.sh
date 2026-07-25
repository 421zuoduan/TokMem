#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VALIDATION_DIR="$ROOT_DIR/scripts/compositional/qwen35/validation"
ENV_PYTHON="/home/shilong/anaconda3/envs/tokmem-qwen35/bin/python"
OUTPUT_DIR="${QWEN35_VALIDATION_OUTPUT_DIR:-$ROOT_DIR/results/compositional/qwen35_environment_artifacts/validation/20260725}"
GPU_CAUSAL_AND_SMOKE="${1:-4}"
GPU_9B_FAST="${2:-5}"
GPU_4B_FAST="${3:-6}"
GPU_9B_COMPARISON="${4:-7}"

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "Missing tokmem-qwen35 Python: $ENV_PYTHON" >&2
    exit 2
fi
if [[ "$(printf '%s\n' \
    "$GPU_CAUSAL_AND_SMOKE" \
    "$GPU_9B_FAST" \
    "$GPU_4B_FAST" \
    "$GPU_9B_COMPARISON" | sort -u | wc -l)" -ne 4 ]]; then
    echo "Validation requires four distinct GPU IDs." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

(
    CUDA_VISIBLE_DEVICES="$GPU_CAUSAL_AND_SMOKE" \
        "$ENV_PYTHON" "$VALIDATION_DIR/validate_causal_conv1d.py" \
        > "$OUTPUT_DIR/causal_conv1d.log" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_CAUSAL_AND_SMOKE" \
        "$ENV_PYTHON" "$VALIDATION_DIR/smoke_tapmem_step.py" \
        > "$OUTPUT_DIR/tapmem_step_9b.log" 2>&1
) &
pid_causal_and_smoke=$!

CUDA_VISIBLE_DEVICES="$GPU_9B_FAST" \
    "$ENV_PYTHON" "$VALIDATION_DIR/validate_fast_model.py" \
    --model-path "$ROOT_DIR/models/Qwen3.5-9B" \
    > "$OUTPUT_DIR/fast_model_9b.log" 2>&1 &
pid_9b_fast=$!

CUDA_VISIBLE_DEVICES="$GPU_4B_FAST" \
    "$ENV_PYTHON" "$VALIDATION_DIR/validate_fast_model.py" \
    --model-path "$ROOT_DIR/models/Qwen3.5-4B" \
    > "$OUTPUT_DIR/fast_model_4b.log" 2>&1 &
pid_4b_fast=$!

CUDA_VISIBLE_DEVICES="$GPU_9B_COMPARISON" \
    "$ENV_PYTHON" "$VALIDATION_DIR/compare_fast_fallback.py" \
    --model-path "$ROOT_DIR/models/Qwen3.5-9B" \
    > "$OUTPUT_DIR/fast_fallback_9b.log" 2>&1 &
pid_9b_comparison=$!

failed=0
for validation_pid in \
    "$pid_causal_and_smoke" \
    "$pid_9b_fast" \
    "$pid_4b_fast" \
    "$pid_9b_comparison"; do
    if ! wait "$validation_pid"; then
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "One or more Qwen3.5 validations failed; inspect $OUTPUT_DIR" >&2
    exit 1
fi

echo "Qwen3.5 fast-environment validations passed: $OUTPUT_DIR"
