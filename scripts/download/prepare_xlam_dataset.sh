#!/bin/bash
set -e

cd "$(dirname "$0")/../.."/compositional || exit 1
export HF_HOME="$(pwd)/../.hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

python xlam_datasets.py \
  --top_k 1-50 \
  --max_samples_per_tool 50 \
  --train_size 5000 \
  --test_size 500 \
  --train_max_function_calls 4 \
  --test_max_function_calls 4 \
  --train_multi_tool_ratios 0.5,0.5 \
  --test_multi_tool_ratios 0.5,0.5 \
  --output_dir data

python xlam_datasets.py \
  --top_k 51-100 \
  --max_samples_per_tool 50 \
  --train_size 5000 \
  --test_size 500 \
  --train_max_function_calls 4 \
  --test_max_function_calls 4 \
  --train_multi_tool_ratios 0.5,0.5 \
  --test_multi_tool_ratios 0.5,0.5 \
  --output_dir data
