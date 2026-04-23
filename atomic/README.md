# Atomic Memory Recall

Atomic task learning using reserved special tokens as task identifiers. Each token encodes a specific task's procedural knowledge.

The maintained atomic TokMem surface is the restored upstream atomic baseline plus two local additions:

- fixed cached split loading directly in `main_in_domain.py` through `--split_cache_path`
- first-step reserved-task logit bias through `--use_logit_bias`, `--logit_bias_loss_weight`, `--logit_bias_network`, and `--logit_bias_scale`
- maintained launchers now use `accelerate launch` as the default entrypoint wrapper, with single-process bf16 as the baseline launch shape and optional FSDP multi-GPU scaling

Older local atomic implementations now live under `atomic/archive/current_local/`.

## Dataset

The project uses the Natural Instructions (Super-NaturalInstructions) dataset. For instructions on how to download and set up the dataset, please see the [Dataset README](natural-instructions-2.8/README.md).

## Configuration

The primary maintained entrypoints are `main_tokmem.sh` for runtime sampling and `main_tokmem_fixed_split.sh` for a cached split. You can configure training and evaluation with these key arguments in `main_in_domain.py`:

- `--num_tasks`: Total number of tasks to load from the Natural Instructions dataset for training and testing.
- `--model_name`: The transformer model to use. Supports Llama 3 and Qwen 2.5 models (e.g., `meta-llama/Llama-3.2-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`).
- `--train_size`, `--val_size`, `--test_size`: Number of instances per task for training, validation, and testing.
- `--shuffle_train` / `--no-shuffle_train`: Control whether the training dataloader shuffles samples. The maintained default is `--shuffle_train`.
- `--split_cache_path`: Load `train_data`, `val_data`, `test_data`, and `task_names` from a cached split file instead of runtime sampling.
- `--run_dir`: Save run-scoped logs and structured outputs such as `run_config.json`, `train_results.json`, and `evaluation_results.json` into one directory.
- `--use_logit_bias`: Enable the first-step task logit-bias head over reserved task tokens.
- `--logit_bias_loss_weight`: Weight on the detached hidden-state bias-head supervision loss.
- `--logit_bias_network`: Bias-head architecture, `linear` or `mlp`.
- `--logit_bias_scale`: Decode-time scale applied to the bias logits on the first generated token.

The maintained launchers now expose the user-visible launcher surface through `accelerate launch`:

- `--num_processes 1` is the maintained default in both shell entrypoints.
- `--num_machines 1` and `--dynamo_backend no` are passed explicitly so the launcher stays single-node and quiet without requiring a saved Accelerate config.
- `--mixed_precision bf16` is passed at launch time instead of through a Python flag.
- `--config_file` and `--multi_gpu` stay on the Accelerate launcher side.
- `--use_fsdp`, `--mixed_precision`, and the `--fsdp_*` family are parsed by `main_in_domain.py` and control how the script builds its internal `Accelerator`.

## Accelerate Launch Expectations

Activate the repository's `tokmem` conda environment before running the maintained scripts:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem
```

The maintained atomic scripts always run through `accelerate launch`, including the default one-GPU case. The training path is now unified around `Accelerator`, with optional FSDP for multi-process multi-GPU runs.

Current multi-GPU implementation details:

- The FSDP path now builds the `Accelerator` before `TaskCallingModel` so Transformers can use FSDP CPU-RAM-efficient loading during `from_pretrained`.
- `--use_fsdp` now requires successful decoder-block class inference and always uses `transformer_based_wrap`.
- In the `--use_fsdp` path, the trainable `task_token_module` and optional `logit_bias_head` stay replicated as FSDP ignored modules, are moved onto each rank's device before wrapping, and have their gradients manually synchronized before each optimizer step.
- Coupled task embeddings stay as one shared trainable parameter in coupled mode.
- Checkpoint save/load stores a single coupled embedding tensor in coupled mode and separate input/output tensors in decoupled mode, with fail-fast shape validation before writing task-token checkpoints.
- The maintained `--use_logit_bias` decode path now pre-fills the prompt once, reads only the final token hidden state for the bias head, and continues generation from the first-pass KV cache.
- In the multi-process FSDP path, the final comprehensive evaluation now stays distributed across all ranks and aggregates predictions on the main process through Accelerate metric gathering. `--test_batch_size` therefore remains a per-process evaluation batch size in that launch shape.

For an optional Accelerate FSDP launch, prepare an Accelerate config and move the placement control to the launcher:

```bash
accelerate config
accelerate launch --config_file /path/to/accelerate_fsdp.yaml --num_processes 2 --multi_gpu main_in_domain.py --use_fsdp
```

In that FSDP launch shape, the launcher owns device placement and sharding. Final evaluation follows the same distributed launch shape and no longer rebuilds a separate rank-0 single-process inference model.

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

The maintained fixed-split launcher stays on the single-process Accelerate path by default. Copy its command and add an Accelerate FSDP config when you move to a distributed-aware atomic launch. In that FSDP launch shape, both validation and final evaluation batch sizes are interpreted per process. The multi-GPU baseline launchers under `scripts/atomic/qwen_0_5b/` and `scripts/atomic/llama_3b/` reuse the cached fixed split and run `main_in_domain.py` through a 3-process Accelerate FSDP launch.

Standalone `700-task / Qwen2.5-0.5B / seed=42` launchers:

```bash
bash ../scripts/atomic/qwen_0_5b/baseline_qwen_0_5b_700task.sh
bash ../scripts/atomic/qwen_0_5b/logit_bias_qwen_0_5b_700task.sh
```

### 700-task mean comparison
This launcher runs the `700-task / Qwen2.5-0.5B / seed=42` baseline and `logit_bias` settings three times each, then appends the mean `Task Prediction Accuracy` and `ROUGE-L` to `results/atomic_mean_results.md`:

```bash
bash ../scripts/atomic/qwen_0_5b/mean_baseline_logit_bias_qwen_0_5b_700task.sh
```

Each run writes `run_config.json`, `train_results.json`, `evaluation_results.json`, `stdout.log`, `gpu_monitor.log`, and `exit_code.txt` into its own folder under `atomic/runs/`.

### 700-task mean comparison on GPUs 3/4/5
This launcher runs all six `700-task / Qwen2.5-0.5B / seed=42` jobs sequentially. Every run uses `GPU 3,4,5` for 3-card joint training with `batch_size=8`. The launcher writes its own aggregate log to `results/atomic_mean_results_3gpu_345_bs8_<timestamp>.log` and appends the mean metrics to `results/atomic_mean_results_3gpu_345_bs8.md`.

Launch it with `nohup`:

```bash
nohup bash ../scripts/atomic/qwen_0_5b/mean_baseline_logit_bias_qwen_0_5b_700task_3gpu.sh > /dev/null 2>&1 &
```

### Older local atomic path
Archived local experiments, scripts, and docs live under `atomic/archive/current_local/`.

### LoRA baseline
```bash
bash main_lora_baseline.sh
```
