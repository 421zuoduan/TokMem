# Compositional Memory Recall

This directory contains the compositional memory recall experiments.

## EOC / Gate Modes

The sequential TokMem path now separates enabling EOC tokens from adding the EOC loss:

| Mode | `--use_eoc` | `--use_eoc_loss` | `--use_gate` | Behavior |
| --- | --- | --- | --- | --- |
| Baseline | off | off | off | Original TokMem decoding and training |
| EOC token only | on | off | off | Inserts explicit `eoc` tokens, but does not add EOC loss |
| EOC loss | on | on | off | Inserts explicit `eoc` tokens and adds EOC loss |
| EOC + gate | on | on/off | on | Uses `eoc` tokens for gating; EOC loss is only added when `--use_eoc_loss` is on |

`--use_gate` requires `--use_eoc`.
`--use_eoc_loss` requires `--use_eoc`.
`--use_tool_loss` also requires `--use_eoc`.

Useful flags:

- `--use_eoc`
- `--use_eoc_loss`
- `--use_gate`
- `--use_tool_loss`
- `--eoc_loss_weight` default `0.1`
- `--tool_loss_weight` default `0.1`
- `--gate_loss_weight` default `0.1`
- `--gate_threshold` default `0.5`
- `--gate_network` default `mlp`, choices: `mlp`, `linear`
- `--max_length` default `1024`

## Experimental Setup

The experiments use tools extracted from the **XLAM** aka. APIGen dataset. A total of 100 tools are used:
- **Tools 1-50**: Used as **adaptation tools**. In the TokMem method, these tools are used in the first round to adapt the model to the tool-calling format and environment.
- **Tools 51-100**: Used for **evaluation** across all methods. Performance is measured on these tools to assess how well the model learns and retains new tool-calling capabilities.

## Methods & Scripts

### 1. TokMem (Main Method)
`scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
This launcher implements the **TokMem** approach. It performs sequential training where the first round (tools 1-50) serves as an adaptation phase.
- **Key Feature**: Uses a specialized training loop in `main_sequential.py` that can freeze adapters after the initial adaptation to maintain stable memory while learning new tool distributions.
- **Usage**:
  ```bash
  bash scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh
  ```

### 2. LoRA Baseline
`scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
This launcher provides a standard sequential fine-tuning baseline using LoRA.
- **Key Feature**: It uses standard LoRA fine-tuning (with optional reinitialization or replay buffers) via `lora_sequential.py`. 
- **Usage**:
  ```bash
  bash scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh
  ```

### 3. ICL Baseline
`scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`
This launcher evaluates the model's zero-shot or few-shot capabilities using In-Context Learning (ICL).
- **Key Feature**: Evaluates the model directly on tools 51-100 without any fine-tuning. It supports RAG-based tool retrieval to fit relevant tool descriptions into the context window.
- **Usage**:
  ```bash
  bash scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh
  ```

## Run Layout

All maintained compositional runs now write artifacts to:

```bash
compositional/runs/<run_name>/
```

Maintained runs keep the following artifacts:

- `run_config.json`
- `evaluation_results.json`
- `evaluation.log`
- round checkpoints
- launcher script snapshot
- optional `tensorboard/`

Maintained runs do not keep:

- `train_results.json`
- `training_summary.json`
- `training.log`
- `run_summary.json`
- `gpu_monitor.log`
- `call_count_breakdown` inside `evaluation_results.json`

The old `compositional/log/`, `checkpoints_*`, and root-level result JSON layout is legacy-only and should not be used for new runs.

## Legacy Migration

To migrate older compositional outputs into the unified run layout:

```bash
python compositional/utils/migrate_legacy_runs.py --dry-run
python compositional/utils/migrate_legacy_runs.py
```

## Deprecated Entry Points

These older shell entrypoints remain in the repository for reference but are no longer the maintained path:

- `compositional/run_n_rounds_main.sh`
- `compositional/run_n_rounds_lora.sh`
- `compositional/icl_baseline.sh`

## Key Components

- `xlam_datasets.py`: Handles data generation, tool extraction, and multi-tool call composition from the XLAM source.
- `model.py` & `training.py`: Core logic for model initialization and the training loops.
- `tool_retrieval.py`: Implementation of RAG for selecting relevant tools during ICL or evaluation.
- `replay_buffer.py`: Management of historical samples to mitigate catastrophic forgetting during sequential learning.
