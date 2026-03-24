ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_DIR="$ROOT_DIR/datasets/natural-instructions-2.8/tasks"

cd "$ROOT_DIR/atomic"

python main_in_domain.py \
    --tasks_dir "$TASKS_DIR" \
    --num_tasks 1000 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --eval_batch_size 32 \
    --validate_every_n_steps 1000 \
    --seed 42
