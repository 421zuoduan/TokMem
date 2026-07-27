#!/usr/bin/env bash
set -euo pipefail

repo_dir=/data/shilong/tokmem
run_name=tokmem_llama8b_current_e90_seed42_v1
checkpoint_dir="$repo_dir/compositional_intercode_bash/runs/$run_name"
evaluation_dir="$repo_dir/compositional_intercode_bash/evaluations/${run_name}_official_agent_parse_preserve_full200_v1"
python_bin=/home/shilong/anaconda3/envs/tokmem/bin/python
runtime_script="$repo_dir/compositional_intercode_bash/single_uid_rootless_runtime.sh"
train_gpu=7
shard_count=4
gpus=(1 3 5 7)

mkdir -p "$checkpoint_dir" "$evaluation_dir/logs"
if [[ -e "$checkpoint_dir/checkpoint.json" ]]; then
    echo "Refusing to overwrite completed checkpoint: $checkpoint_dir/checkpoint.json"
    exit 1
fi

cd "$repo_dir"
echo "[$(date --iso-8601=seconds)] starting current-framework TokMem 90-epoch training on GPU $train_gpu"
CUDA_VISIBLE_DEVICES="$train_gpu" "$python_bin" -u \
  -m compositional_intercode_bash.train \
  --method tokmem \
  --model-name models/Llama-3.1-8B-Instruct \
  --procedure-lexicon compositional_intercode_bash/artifacts/validation_canonical_100e_frozen/procedure_lexicon.json \
  --views compositional_intercode_bash/artifacts/validation_canonical_100e_frozen/views.sampled.1729.jsonl \
  --output-dir "$checkpoint_dir" \
  --model-seed 42 \
  --expected-epochs 100 \
  --epoch-limit 90 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 0.005 \
  --max-length 1024 \
  --dtype bfloat16 \
  --device cuda:0 \
  >"$checkpoint_dir/training.log" 2>&1

echo "[$(date --iso-8601=seconds)] training complete; starting rootless Docker"
"$runtime_script" start
eval "$("$runtime_script" env)"
export DOCKER_BUILDKIT=0
export PYTHONUNBUFFERED=1

evaluate_shard() {
    local shard_index=$1
    local gpu=$2
    local phase=$3
    local log_path="$evaluation_dir/logs/shard_${shard_index}_${phase}.log"

    echo "[$(date --iso-8601=seconds)] starting shard $shard_index on GPU $gpu ($phase)"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        -m compositional_intercode_bash.evaluate \
        --checkpoint "$checkpoint_dir" \
        --output-dir "$evaluation_dir" \
        --max-turns 10 \
        --shard-count "$shard_count" \
        --shard-index "$shard_index" \
        --resume \
        --allow-exploratory-checkpoint \
        >>"$log_path" 2>&1
}

declare -a shard_pids
declare -a shard_status
for shard_index in 0 1 2 3; do
    evaluate_shard "$shard_index" "${gpus[$shard_index]}" parallel &
    shard_pids[$shard_index]=$!
done

for shard_index in 0 1 2 3; do
    status=0
    wait "${shard_pids[$shard_index]}" || status=$?
    shard_status[$shard_index]=$status
    echo "[$(date --iso-8601=seconds)] shard $shard_index parallel exit=$status"
done

for shard_index in 0 1 2 3; do
    if [[ "${shard_status[$shard_index]}" -eq 0 ]]; then
        continue
    fi
    for attempt in 1 2 3; do
        echo "[$(date --iso-8601=seconds)] recovering shard $shard_index attempt=$attempt"
        if evaluate_shard "$shard_index" "${gpus[$shard_index]}" "recovery_${attempt}"; then
            break
        fi
    done
done

for attempt in 1 2 3; do
    echo "[$(date --iso-8601=seconds)] full resume and strict merge attempt=$attempt"
    if CUDA_VISIBLE_DEVICES="${gpus[0]}" "$python_bin" \
        -m compositional_intercode_bash.evaluate \
        --checkpoint "$checkpoint_dir" \
        --output-dir "$evaluation_dir" \
        --max-turns 10 \
        --resume \
        --allow-exploratory-checkpoint \
        >>"$evaluation_dir/logs/final_resume_${attempt}.log" 2>&1; then
        echo "[$(date --iso-8601=seconds)] COMPLETED: $evaluation_dir/summary_10_turn.json"
        exit 0
    fi
done

echo "[$(date --iso-8601=seconds)] FAILED: final resume did not complete"
exit 1
