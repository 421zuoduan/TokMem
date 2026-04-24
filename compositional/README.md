# Compositional Memory Recall

This directory contains the compositional TokMem experiments on XLAM/APIGen.

## Current Maintained Method Surface

The maintained compositional path now keeps two method switches:

- `--use_eoc`
- `--use_logit_bias`

Historical archived runs still exist under `compositional/runs/`. Current code, launchers, and docs describe the maintained `eoc/logit_bias` family.

## Modes

| Mode | `--use_eoc` | `--use_logit_bias` | Behavior |
| --- | --- | --- | --- |
| Baseline | off | off | Original TokMem decoding and training |
| EOC token only | on | off | Inserts explicit `eoc` boundary tokens between tool-controlled spans |
| EOC + logit bias | on | on | Trains a detached tool-prior head on boundary states and adds centered tool-only bias back to decode logits |

Constraint summary:

- `--use_logit_bias` requires `--use_eoc`

Useful flags:

- `--use_eoc`
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

- `scripts/compositional/llama_1b/tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh`

Single-round comparison launchers with the same Llama-1B data split settings:

- `scripts/compositional/llama_1b/baseline_llama_1b.sh`: direct tool-description prompting over all 50 benchmark tools through `icl_baseline.py`
- `scripts/compositional/llama_1b/icl_llama_1b.sh`: explicit ICL launcher over all 50 benchmark tools through `icl_baseline.py`
- `scripts/compositional/llama_1b/rag_llama_1b.sh`: ICL launcher with top-5 tool retrieval through `--use_rag --retrieval_k 5`
- `scripts/compositional/llama_1b/lora_llama_1b.sh`: standard LoRA finetuning through `lora_sequential.py`

These comparison launchers use tools `51-100`, `train_size=5000`, `test_size=500`, max calls `4`, multi-tool ratios `0.5,0.5`, seed `42`, model `models/Llama-3.2-1B-Instruct`, and write under `compositional/runs/`.

Python entrypoints now expect explicit local model paths through `--model_name`. The RAG launcher also passes a local sentence-transformer path through `--retriever_model_name`, with the retriever model stored at `models/all-MiniLM-L6-v2`.

README 汇总复现实验的 maintained launcher:

- `scripts/compositional/llama_1b/run_readme_myself_3methods_llama_1b.sh`
- `scripts/compositional/llama_1b/run_readme_myself_3methods_10calls_llama_1b.sh`

Paper-level compositional suite launcher:

- `scripts/compositional/run_paper_compositional_suite.sh`

This suite launcher is the maintained entrypoint for the `51-100 / 4 calls` paper comparison sweep across `llama1b`, `llama3b`, `llama8b` and methods `icl`, `rag`, `lora`, `tokmem`, `tokmem_eoc`, `tokmem_eoc_logit_bias`.

It uses one scheduling unit per `model × method × trial`, keeps the 5 trials independently schedulable across GPUs, writes archived outputs under `results/compositional/<suite_name>/`, and produces:

- `runs/<task_name>/` per trial
- `task_manifest.tsv`
- `task_status.json`
- `summary.md`
- `summary.json`
- `scheduler.log`

When the suite is resumed with `--rerun-failed --suite-name <existing-suite>`, the launcher keeps the existing suite-level `run_paper_compositional_suite.sh`, `suite_config.json`, and `task_manifest.tsv` in place. The current rerun invocation is snapshotted under:

- `reruns/<rerun_id>/run_paper_compositional_suite.sh`
- `reruns/<rerun_id>/suite_config.json`
- `reruns/<rerun_id>/task_manifest.tsv`

Current maintained comparison tables only include:

- `baseline`
- `eoc-only`
- `eoc+logit_bias`

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
- `round_<round>_tools_<range>.pt` when `--save_checkpoints` is passed
- launcher script snapshot when the shell launcher copies itself into the run directory
- `loss_step.png` when `--tensorboard` is passed
- `lr_step.png` when `--tensorboard` is passed

Training methods keep metrics in `evaluation_results.json` under the latest round payload:

```text
rounds[-1].eval_results
```

ICL/RAG baselines save the same maintained metric names under the top-level `metrics` field in `evaluation_results.json`.

For ICL/RAG, `function_calls` on the maintained compositional dataset store argument JSON only. The saved tool metrics therefore first map each predicted argument object back to a tool id from the active prompt tool schemas, and ambiguous calls stay unresolved. Fields including `tool_accuracy`, `tool_exact_match_acc`, `avg_tool_f1_score`, `avg_f1_score`, `exact_accuracy`, and `parse_error_rate` are overall values over the full evaluation set. `tool_accuracy` is the binary per-sample-tool accuracy over TP/TN/FP/FN, `tool_exact_match_acc` is the sample rate where the full tool set is predicted exactly, and `avg_tool_f1_score` keeps the existing Tool F1 definition. Per-call-count breakdowns are printed in `evaluation.log`.

`training_summary.json` is intentionally compact. It keeps:

- `round`
- `tools`
- `epochs`
- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss` when `use_logit_bias=true`

Detailed position counters are available in the per-round training `results` payload and training logs. The saved `training_summary.json` stays a run-level loss summary.

Passing `--tensorboard` on the maintained TokMem path saves two static PNG trend plots under the run directory:

- `loss_step.png`
- `lr_step.png`

## Legacy Entry Points

These older shell entrypoints remain in the repository for reference but are not the maintained path:

- `compositional/run_n_rounds_main.sh`
- `compositional/run_n_rounds_lora.sh`
- `compositional/icl_baseline.sh`

## Dataset Inspection

For quick manual inspection of newly synthesized compositional data over tools `1-100`, run the generator directly:

```bash
cd compositional
python xlam_datasets.py \
  --top_k "1-100" \
  --max_samples_per_tool 50 \
  --train_size 12000 \
  --test_size 1200 \
  --train_max_function_calls 10 \
  --test_max_function_calls 10 \
  --train_multi_tool_ratios "0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125" \
  --test_multi_tool_ratios "0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125" \
  --output_dir data
```

This regenerates the synthetic data files under `compositional/data/` with fixed settings and prints:

- train/test file locations
- training-set `function_calls` distribution
- training-set `unique_tools` distribution
- single-tool training-sample call-count distribution
- a few representative training samples

`xlam_datasets.py` accepts variable-length `--train_multi_tool_ratios` / `--test_multi_tool_ratios`. A ratio list of length `k` maps to `2-tool` through `(k+1)-tool`, and output filenames keep following `..._{max_function_calls}calls.json`, such as `..._4calls.json` or `..._10calls.json`.

## Key Components

- `main_sequential.py`: maintained TokMem entrypoint
- `model.py`: reserved-tool-token model, EOC boundary logic, logit-bias decoding
- `training.py`: autoregressive training loop plus detached logit-bias auxiliary loss
- `dataset.py`: XLAM/APIGen data loading and EOC target formatting
- `tool_retrieval.py`: RAG tool retrieval for ICL and related baselines
