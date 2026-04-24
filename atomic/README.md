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
- `--shuffle_train` / `--no-shuffle_train`: Control whether the training dataloader shuffles samples. The maintained default is `--shuffle_train`.
- `--split_cache_path`: Load `train_data`, `val_data`, `test_data`, and `task_names` from a cached split file instead of runtime sampling.
- `--use_logit_bias`: Enable the first-step task logit-bias head over reserved task tokens.
- `--logit_bias_loss_weight`: Weight on the detached hidden-state bias-head supervision loss.
- `--logit_bias_network`: Bias-head architecture, `linear` or `mlp`.
- `--logit_bias_scale`: Decode-time scale applied to the bias logits on the first generated token.

## Fairness And Batch Size

For current `atomic` fairness guidance and single-GPU memory-based batch recommendations, see:

- [docs/atomic/2026-04-24-atomic-fairness-and-memory-batch-size.md](/data/shilong/tokmem/docs/atomic/2026-04-24-atomic-fairness-and-memory-batch-size.md)

## Usage

### Maintained TokMem runtime split
```bash
bash main_tokmem.sh
```

### Maintained TokMem fixed split
This path reuses a precomputed cached split and enables the first-step logit-bias head:

- model directory: `models/Qwen2.5-0.5B-Instruct`
- split cache: `atomic/cached_splits/task50-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt`

```bash
bash main_tokmem_fixed_split.sh
```

The Qwen 0.5B fixed-split baseline launcher under `scripts/atomic/qwen_0_5b/` uses `--device_map balanced` and a smaller validation batch to match the archived multi-GPU memory layout.

### Raw base-model evaluation
This path loads the base model directly and evaluates with an `instruction + query` prompt without few-shot examples:

```bash
bash ../scripts/atomic/qwen_0_5b/run_atomic_qwen_0_5b_fixed_split_50task_test_base_model.sh
```

The Python entrypoint is `main_base_model.py`. It writes `run_config.json`, `evaluation_results.json`, `evaluation_predictions.jsonl`, and `run_summary.json` into one folder under `atomic/runs/`.

### SBERT RAG baseline
This path uses the raw base model for generation, retrieves demonstrations with `Sentence-BERT`, and formats the prompt using the repo's few-shot conversational layout:

```bash
bash ../scripts/atomic/qwen_0_5b/run_atomic_qwen_0_5b_fixed_split_50task_test_rag.sh
```

The Python entrypoint is `main_rag_baseline.py`. It can reuse a saved SBERT corpus through `--corpus_cache_path` and records both generation metrics and retrieval top-1 / top-k accuracy in `evaluation_results.json`.

### Paper suite launcher
This launcher schedules the maintained fixed-split atomic comparison suite across the local `Qwen 0.5B`, `Llama 3B`, and `Llama 8B` checkpoints. The default scope is the existing `700-task` cached split with five methods: `base`, `rag`, `lora`, `tokmem`, and `tokmem_logit_bias`.

```bash
bash ../scripts/atomic/run_paper_atomic_suite.sh --gpus 0,1,2,3
```

The suite writes all artifacts under `results/atomic/<suite-name>/`:

- per-task runs: `results/atomic/<suite-name>/runs/`
- task manifest: `results/atomic/<suite-name>/task_manifest.tsv`
- task status: `results/atomic/<suite-name>/task_status.json`
- summary: `results/atomic/<suite-name>/summary.md`

Default `700-task` batch settings in the launcher:

| Model | LoRA train | LoRA eval | TokMem train | TokMem eval | base test | rag test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen 0.5B` | `16` | `64` | `16` | `256` | `1024` | `512` |
| `Llama 3B` | `16` | `64` | `32` | `128` | `512` | `256` |
| `Llama 8B` | `8` | `48` | `16` | `64` | `256` | `128` |

Use `--suite-name <existing-suite> --rerun-failed` to rerun only failed tasks inside an existing suite directory.

### Older local atomic path
Archived local experiments, scripts, and docs live under `atomic/archive/current_local/`.

### LoRA baseline
```bash
bash main_lora_baseline.sh
```
