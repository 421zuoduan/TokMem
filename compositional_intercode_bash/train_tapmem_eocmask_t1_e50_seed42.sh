#!/usr/bin/env bash
set -euo pipefail

cd /data/shilong/tokmem

exec /home/shilong/anaconda3/envs/tokmem/bin/python -u \
  -m compositional_intercode_bash.train \
  --method tapmem \
  --model-name models/Llama-3.1-8B-Instruct \
  --procedure-lexicon compositional_intercode_bash/artifacts/validation_canonical_100e_frozen/procedure_lexicon.json \
  --views compositional_intercode_bash/artifacts/validation_canonical_100e_frozen/views.sampled.1729.jsonl \
  --output-dir compositional_intercode_bash/runs/tapmem_llama8b_eocmask_t1_e50_seed42_v1 \
  --model-seed 42 \
  --expected-epochs 100 \
  --epoch-limit 50 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 0.005 \
  --routing-learning-rate 0.005 \
  --route-loss-weight 0.1 \
  --logit-bias-scale 1.0 \
  --memory-bank-probability-threshold 1.0 \
  --max-length 1024 \
  --dtype bfloat16 \
  --device cuda:0
