SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python "${SCRIPT_DIR}/main_in_domain_fixed_split.py" \
    --num_tasks 1000 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "${REPO_ROOT}/models/Llama-3.2-3B-Instruct" \
    --device_map "balanced" \
    --split_cache_path "${SCRIPT_DIR}/cached_splits/tokmem_atomic_fixed_split.pt" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --eval_batch_size 4 \
    --validate_every_n_steps 1000
