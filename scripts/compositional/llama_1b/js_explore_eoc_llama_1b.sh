ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/compositional/runs/js_explore/js_explore_eoc_llama_1b_${RUN_ID}"

mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/$(basename "$SCRIPT_PATH")"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES=6

cd "$ROOT_DIR"

python compositional/utils/js_explore.py \
    --checkpoint_path "$ROOT_DIR/compositional/runs/compositional_tokmem_eoc_only_llama_1b_50tools_20260412_105149/round_1_tools_51_100.pt" \
    --model_name "$ROOT_DIR/models/Llama-3.2-1B-Instruct" \
    --test_data_path "$ROOT_DIR/compositional/data/test/function_calling_test_tools51-100_4calls.json" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 100 \
    --batch_size 4 \
    --max_new_tokens 256 \
    --max_plot_samples_per_group 20 \
    --device cuda \
    --seed 42
