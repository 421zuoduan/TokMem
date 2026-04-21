# Atomic Memory Recall

Atomic task learning using reserved special tokens as task identifiers. Each token encodes a specific task's procedural knowledge.

The maintained atomic TokMem surface is the restored upstream atomic baseline plus two local additions:

- fixed cached split loading directly in `main_in_domain.py` through `--split_cache_path`
- first-step reserved-task logit bias through `--use_logit_bias`, `--logit_bias_loss_weight`, `--logit_bias_network`, and `--logit_bias_scale`

Older local atomic implementations now live under `atomic/archive/current_local/`.

## Dataset

The project uses the Natural Instructions (Super-NaturalInstructions) dataset. For instructions on how to download and set up the dataset, please see the [Dataset README](natural-instructions-2.8/README.md).

## Configuration

The primary maintained entrypoints are `main_tokmem.sh` for runtime sampling and `main_tokmem_fixed_split.sh` for a cached split. You can configure training and evaluation with these key arguments in `main_in_domain.py`:

- `--num_tasks`: Total number of tasks to load from the Natural Instructions dataset for training and testing.
- `--model_name`: The transformer model to use. Supports Llama 3 and Qwen 2.5 models (e.g., `meta-llama/Llama-3.2-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`).
- `--device_map`: Optional Hugging Face sharding mode such as `balanced` for multi-GPU frozen-backbone loading.
- `--train_size`, `--val_size`, `--test_size`: Number of instances per task for training, validation, and testing.
- `--split_cache_path`: Load `train_data`, `val_data`, `test_data`, and `task_names` from a cached split file instead of runtime sampling.
- `--use_logit_bias`: Enable the first-step task logit-bias head over reserved task tokens.
- `--logit_bias_loss_weight`: Weight on the detached hidden-state bias-head supervision loss.
- `--logit_bias_network`: Bias-head architecture, `linear` or `mlp`.
- `--logit_bias_scale`: Decode-time scale applied to the bias logits on the first generated token.

## Usage

### Maintained TokMem runtime split
```bash
bash main_tokmem.sh
```

### Maintained TokMem fixed split
This path reuses a precomputed cached split under `atomic/cached_splits/` and enables the first-step logit-bias head:
```bash
bash main_tokmem_fixed_split.sh
```

The Qwen 0.5B fixed-split baseline launcher under `scripts/atomic/qwen_0_5b/` uses `--device_map balanced` and a smaller validation batch to match the archived multi-GPU memory layout.

### Older local atomic path
Archived local experiments, scripts, and docs live under `atomic/archive/current_local/`.

### LoRA baseline
```bash
bash main_lora_baseline.sh
```
