# Compositional Memory Recall

This directory contains the compositional memory recall experiments.

## EOC / Gate Modes

The sequential TokMem path now separates enabling EOC tokens from adding the EOC loss:

| Mode | `--use_eoc` | `--use_eoc_loss` | `--use_gate` | `--use_toolmix` | Behavior |
| --- | --- | --- | --- | --- | --- |
| Baseline | off | off | off | off | Original TokMem decoding and training |
| EOC token only | on | off | off | off | Inserts explicit `eoc` tokens, but does not add EOC loss |
| EOC loss | on | on | off | off | Inserts explicit `eoc` tokens and adds EOC loss |
| EOC + gate | on | on/off | on | off | Uses the shared `routing_probe` for gating; `--probe_from eoc` keeps the original boundary-state behavior and `--probe_from tool` probes the current token state |
| EOC + toolmix | on | on/off | on/off | on | Uses the shared `routing_probe` on BOS and gold `eoc` decision sites to mix tool-token CE with tool-selection loss; `--probe_from` changes which hidden state feeds that shared probe |

`--use_gate` requires `--use_eoc`.
`--use_eoc_loss` requires `--use_eoc`.
`--use_tool_loss` also requires `--use_eoc`.
`--use_toolmix` also requires `--use_eoc`.

Useful flags:

- `--use_eoc`
- `--use_eoc_loss`
- `--use_gate`
- `--use_tool_loss`
- `--use_toolmix`
- `--eoc_loss_weight` default `0.1`
- `--tool_loss_weight` default `0.1`
- `--gate_loss_weight` default `0.1`
- `--toolmix_loss_weight` default `0.1`
- `--gate_threshold` default `0.5`
- `--gate_network` default `linear`, choices: `mlp`, `linear`
- `--probe_from` default `eoc`, choices: `eoc`, `tool`
- `--max_length` default `1024`

`--probe_from` applies to the shared `routing_probe` used by both `--use_gate` and `--use_toolmix`:

- `eoc`: keep the current implementation, where the probe reads the boundary token hidden state and predicts whether the next token should be a tool token
- `tool`: move the probe input to the current token hidden state while keeping the same current-token routing target

When `--use_toolmix` is enabled, training keeps the existing `eoc` target format and standard teacher forcing, then:

- collects candidate routing sites from BOS and every gold `eoc`
- predicts whether the next gold token is a tool token with the shared `routing_probe`
- adds an auxiliary BCE loss weighted by `--toolmix_loss_weight`
- replaces the old additive `CE + tool_loss_weight * tool_loss` behavior on gold tool-token positions with
  `loss_t = (1 - toolmix_prob) * ce_t + toolmix_prob * toolmix_alpha * tool_sel_t`
- computes `toolmix_alpha` automatically as `log(|V|) / log(|T|)` and prints it at training start

When `--use_gate` and `--use_toolmix` are enabled together, they still share one `routing_probe`. Training adds the shared routing BCE once through `--toolmix_loss_weight`. `--gate_loss_weight` applies to the pure gate path.

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
- `training_summary.json`
- `evaluation.log`
- round checkpoints
- launcher script snapshot
- optional `loss_step.png`
- optional `lr_step.png`

Top-level metrics in `evaluation_results.json`, including `tool_accuracy`, `tool_selection_accuracy`, `arguments_accuracy`, and `full_correctness`, are overall values over the full evaluation set. Per-call-count breakdowns are printed in `evaluation.log`.

Maintained runs do not keep:

- `train_results.json`
- `training.log`
- `run_summary.json`
- `gpu_monitor.log`
- `call_count_breakdown` inside `evaluation_results.json`

`training_summary.json` is intentionally compact: it only keeps final per-round average losses such as `avg_total_loss`, `avg_ar_loss`, `avg_eoc_loss`, `avg_tool_loss`, `avg_gate_loss`, and when enabled `avg_toolmix_aux_loss`, `avg_toolmix_prob`, and `toolmix_alpha`. It does not keep step-level or batch-level training traces.

Passing `--tensorboard` on the maintained TokMem path now saves two static PNG trend plots directly under the run directory after training finishes:

- `loss_step.png`
- `lr_step.png`

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
