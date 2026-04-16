# Compositional Memory Recall

This directory contains the compositional memory recall experiments.

## EOC / Gate Modes

The sequential TokMem path now separates enabling EOC tokens from adding the EOC loss:

| Mode | `--use_eoc` | `--use_eoc_loss` | `--use_gate` | `--use_toolmix` | `--use_logit_bias` | Behavior |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | off | off | off | off | off | Original TokMem decoding and training |
| EOC token only | on | off | off | off | off | Inserts explicit `eoc` tokens, but does not add EOC loss |
| EOC loss | on | on | off | off | off | Inserts explicit `eoc` tokens and adds EOC loss |
| EOC + gate | on | on/off | on | off | off | Uses the shared `routing_probe` for gating; code default `--probe_from tool` probes the current token state, while `--probe_from eoc` uses the boundary state |
| EOC + toolmix | on | on/off | on/off | on | off | Uses the shared `routing_probe` on BOS and gold `eoc` decision sites to mix tool-token CE with tool-selection loss; `--probe_from` changes which hidden state feeds that shared probe |
| EOC + logit bias | on | on/off | on/off | on/off | on | Trains a detached external tool prior on assistant-start and gold-`eoc` boundary states, then adds its tool-only log-probabilities back to the main logits as a soft decode-time bias |

`--use_gate` requires `--use_eoc`.
`--use_eoc_loss` requires `--use_eoc`.
`--use_tool_loss` also requires `--use_eoc`.
`--use_toolmix` also requires `--use_eoc`.
`--use_logit_bias` also requires `--use_eoc`.

Useful flags:

- `--use_eoc`
- `--use_eoc_loss`
- `--use_gate`
- `--use_tool_loss`
- `--use_toolmix`
- `--use_logit_bias`
- `--eoc_loss_weight` default `0.1`
- `--tool_loss_weight` default `0.1`
- `--gate_loss_weight` default `0.1`
- `--toolmix_loss_weight` default `0.1`
- `--logit_bias_loss_weight` default `0.1`
- `--gate_threshold` default `0.5`
- `--gate_network` default `linear`, choices: `mlp`, `linear`
- `--probe_from` default `tool`, choices: `eoc`, `tool`
- `--logit_bias_network` default `linear`, choices: `mlp`, `linear`
- `--logit_bias_scale` default `1.0`
- `--max_length` default `1024`

`--probe_from` applies to the shared `routing_probe` used by both `--use_gate` and `--use_toolmix`:

- `eoc`: keep the current implementation, where the probe reads the boundary token hidden state and predicts whether the next token should be a tool token
- `tool`: move the probe input to the current token hidden state and predict whether that current token belongs to the tool-token subset; ordinary text tokens and `<|eot_id|>` both map to target `0`

When `--use_toolmix` is enabled, training keeps the existing `eoc` target format and standard teacher forcing, then:

- collects candidate routing sites from BOS and every gold `eoc`
- predicts whether the next gold token is a tool token with the shared `routing_probe`
- adds an auxiliary BCE loss weighted by `--toolmix_loss_weight`
- replaces the old additive `CE + tool_loss_weight * tool_loss` behavior on gold tool-token positions with
  `loss_t = (1 - toolmix_prob) * ce_t + toolmix_prob * toolmix_alpha * tool_sel_t`
- computes `toolmix_alpha` automatically as `log(|V|) / log(|T|)` and prints it at training start

When `--use_gate` and `--use_toolmix` are enabled together, they still share one `routing_probe`. Training adds the shared routing BCE once through `--toolmix_loss_weight`. `--gate_loss_weight` applies to the pure gate path.

Training detaches the hidden states before feeding them into the shared `routing_probe`, so the routing BCE updates the probe itself while backbone states, TokMem embeddings, and the main autoregressive path keep their gradients from the other active losses.

Batch logs and `training_summary.json` now compute `GateProb` / `avg_gate_prob` and `ToolmixProb` / `avg_toolmix_prob` after excluding boundary sites whose next gold token is `<|eot_id|>`. The BCE losses still use the full boundary set.

When `--use_logit_bias` is enabled, training and decoding use a separate external tool selector:

- candidate boundary positions stay fixed to the assistant-start boundary and every gold/generated `eoc`
- the selected boundary hidden state is always the boundary state itself
- training detaches each boundary hidden state before sending it into `logit_bias_head`
- the auxiliary CE runs only on boundary sites whose next gold token is a tool token
- the auxiliary loss updates only `logit_bias_head`, while backbone, TokMem embeddings, and the main autoregressive path keep their own gradients
- decoding takes the prior head's tool logits, applies `log_softmax` within the tool subset, subtracts the uniform-tool baseline, scales by `--logit_bias_scale`, and adds the result only to tool-token columns in the full vocabulary logits

`--use_logit_bias` is compatible with `--use_gate` and `--use_toolmix`. The decode order is:

1. compute full-vocab logits
2. if `--use_gate --probe_from tool` is active, sample the provisional routing token from the raw logits first
3. add the soft tool-only bias on boundary rows for standalone sampling, JS truncation, and any post-gate tool-only resampling
4. run any existing gate or JS truncation masking logic

## Gate Trace Utility

To inspect gate probabilities per sample instead of the training-log average, use:

```bash
python compositional/utils/gate_trace_samples.py \
  --checkpoint_path compositional/runs/<run_name>/round_1_tools_51_100.pt
```

The script:

- restores the checkpoint and the matching data split from `run_config.json`
- reruns sample-level generation to separate `full_correct` and `not_full_correct` examples
- exports teacher-forced gate traces on BOS and every gold `eoc` boundary
- writes one JSON file with per-sample `gate_sites`, where each site includes its boundary type, source token, next gold token, routing logit, routing probability, thresholded prediction, and correctness

For a sample with `k` gold tool calls, the exported trace contains `k + 1` gate sites: one BOS site and one site after each gold `eoc`.

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

Single-round maintained launchers for direct comparisons on tools `51-100` live in the same directory:

- `scripts/compositional/llama_1b/tokmem_baseline_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_gate_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_toolmix_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh`

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

Top-level metrics in `evaluation_results.json`, including `tool_accuracy`, `tool_selection_accuracy`, `arguments_accuracy`, and `full_correctness`, are overall values over the full evaluation set. Per-call-count breakdowns are printed in `evaluation.log`. `parse_error_rate` follows the code path in `training.py` and `lora_sequential.py`: it is `parse_errors / total_examples`, where `parse_errors` counts output-call parse failures, so the value is the average number of parse errors per example and can exceed `1.0`.

Maintained runs do not keep:

- `train_results.json`
- `training.log`
- `run_summary.json`
- `gpu_monitor.log`
- `call_count_breakdown` inside `evaluation_results.json`

`training_summary.json` is intentionally compact: it only keeps final per-round average losses such as `avg_total_loss`, `avg_ar_loss`, `avg_eoc_loss`, `avg_tool_loss`, `avg_gate_loss`, and when enabled `avg_toolmix_aux_loss`, `avg_toolmix_prob`, `toolmix_alpha`, and `avg_logit_bias_loss`. Its gate/toolmix probability means follow the same non-`<|eot_id|>` boundary filter used by the batch logs. It does not keep step-level or batch-level training traces.

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
