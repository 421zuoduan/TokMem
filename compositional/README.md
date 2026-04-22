# Compositional Memory Recall

This directory contains the compositional TokMem experiments on XLAM/APIGen.

## Current Maintained Method Surface

The maintained compositional path now keeps only three method switches:

- `--use_eoc`
- `--use_js_trunc`
- `--use_logit_bias`

Historical archived runs still exist under `compositional/runs/`. Current code, launchers, and docs only describe the maintained `eoc/js_trunc/logit_bias` family.

## Modes

| Mode | `--use_eoc` | `--use_js_trunc` | `--use_logit_bias` | Behavior |
| --- | --- | --- | --- | --- |
| Baseline | off | off | off | Original TokMem decoding and training |
| EOC token only | on | off | off | Inserts explicit `eoc` boundary tokens between tool-controlled spans |
| EOC + JS truncation | on | on | off | Uses boundary-time JS divergence to decide whether the next step should enter tool-only decoding |
| EOC + logit bias | on | off | on | Trains a detached tool-prior head on boundary states and adds centered tool-only bias back to decode logits |
| EOC + JS truncation + logit bias | on | on | on | Applies logit-bias reweighting first, then runs JS-based tool-only truncation on active boundary rows |

Constraint summary:

- `--use_js_trunc` requires `--use_eoc`
- `--use_logit_bias` requires `--use_eoc`

Useful flags:

- `--use_eoc`
- `--use_js_trunc`
- `--use_logit_bias`
- `--logit_bias_loss_weight` default `0.1`
- `--logit_bias_network` default `linear`, choices: `mlp`, `linear`
- `--logit_bias_scale` default `1.0`
- `--max_length` default `1024`
- `--max_new_tokens` default `512`

## Method Notes

### EOC

When `--use_eoc` is enabled, each gold tool span becomes:

```text
<tool_token> <json_args> <eoc>
```

`eoc` is a reserved special token that marks the end of one tool-controlled span and the next boundary decision point.

### JS truncation

`--use_js_trunc` is a decode-time routing method. At assistant-start and after each generated `eoc`, the model:

1. reads hidden states from all transformer layers for the current boundary step
2. compares each layer's tool-token distribution against the final-layer distribution with JS divergence
3. averages that JS curve per example
4. if the mean JS exceeds the fixed threshold, masks logits to the tool-token subset for that step

Training stays plain autoregressive teacher forcing. `js_trunc` changes decoding only.

`--max_new_tokens` controls only the training-time evaluation and demo decode budget in `main_sequential.py`. It does not change teacher-forcing supervision length during training.

### Logit bias

`--use_logit_bias` adds a detached auxiliary head and a soft decode-time bias.

Training:

1. collect assistant-start and gold-`eoc` boundary states
2. detach those boundary hidden states
3. predict the next gold tool id with `logit_bias_head`
4. add `logit_bias_loss_weight * CE` to the autoregressive loss

Decoding:

1. compute tool-only logits from the boundary state
2. convert them to `tool_log_probs`
3. center them against the uniform-tool prior `1 / K`
4. multiply by `logit_bias_scale`
5. add the result back only to tool-token columns

This means an informative prior only reweights relative preference among tool tokens and leaves non-tool logits unchanged.

## Maintained Launchers

Single-round maintained launchers for tools `51-100`:

- `scripts/compositional/llama_1b/tokmem_baseline_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_js_trunc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_js_trunc_logit_bias_llama_1b.sh`
- `scripts/compositional/llama_3b/tokmem_baseline_llama_3b.sh`
- `scripts/compositional/llama_3b/tokmem_eoc_llama_3b.sh`
- `scripts/compositional/llama_3b/tokmem_eoc_logit_bias_llama_3b.sh`

README 汇总复现实验的 maintained launcher:

- `scripts/compositional/llama_3b/run_readme_myself_3methods_llama_3b.sh`
- `scripts/compositional/llama_1b/run_readme_myself_7settings_llama_1b.sh`
- `scripts/compositional/llama_1b/run_readme_myself_allmethods_llama_1b.sh`
- `scripts/compositional/llama_1b/run_readme_myself_logit_bias_methods_llama_1b.sh`
- `scripts/compositional/llama_1b/run_readme_myself_3methods_top1_100_10calls_llama_1b.sh`

Current maintained comparison tables only include:

- `baseline`
- `eoc-only`
- `eoc+js_trunc`
- `eoc+logit_bias`
- `eoc+js_trunc+logit_bias`

## Run Layout

All maintained compositional runs write artifacts to:

```bash
compositional/runs/<run_name>/
```

Maintained runs keep:

- `run_config.json`
- `evaluation_results.json`
- `training_summary.json`
- `evaluation.log`
- round checkpoints
- launcher script snapshot
- optional `loss_step.png`
- optional `lr_step.png`

Top-level metrics in `evaluation_results.json`, including `tool_accuracy`, `tool_exact_match_acc`, `arguments_accuracy`, and `full_correctness`, are overall values over the full evaluation set. `tool_accuracy` is the binary per-sample-tool accuracy over TP/TN/FP/FN, `tool_exact_match_acc` is the sample rate where the full tool set is predicted exactly, and `avg_tool_f1_score` keeps the existing Tool F1 definition. Per-call-count breakdowns are printed in `evaluation.log`.

`training_summary.json` is intentionally compact. It keeps:

- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss` when `use_logit_bias=true`

It also keeps lightweight counters such as total supervised positions and logit-bias boundary counts. It does not keep step-level or batch-level traces.

Passing `--tensorboard` on the maintained TokMem path saves two static PNG trend plots under the run directory:

- `loss_step.png`
- `lr_step.png`

## Legacy Entry Points

These older shell entrypoints remain in the repository for reference but are not the maintained path:

- `compositional/run_n_rounds_main.sh`
- `compositional/run_n_rounds_lora.sh`
- `compositional/icl_baseline.sh`

## Dataset Inspection

For quick manual inspection of newly synthesized compositional data over tools `1-100`, use:

```bash
bash scripts/compositional/datasets/inspect_synth_data_tools1_100.sh
```

This script regenerates the synthetic data files under `compositional/data/` with fixed settings and prints:

- train/test file locations
- training-set `function_calls` distribution
- training-set `unique_tools` distribution
- single-tool training-sample call-count distribution
- a few representative training samples

`xlam_datasets.py` accepts variable-length `--train_multi_tool_ratios` / `--test_multi_tool_ratios`. A ratio list of length `k` maps to `2-tool` through `(k+1)-tool`, and output filenames keep following `..._{max_function_calls}calls.json`, such as `..._4calls.json` or `..._10calls.json`.

## Key Components

- `main_sequential.py`: maintained TokMem entrypoint
- `model.py`: reserved-tool-token model, EOC boundary logic, JS truncation, logit-bias decoding
- `training.py`: autoregressive training loop plus detached logit-bias auxiliary loss
- `dataset.py`: XLAM/APIGen data loading and EOC target formatting
- `tool_retrieval.py`: RAG tool retrieval for ICL and related baselines
