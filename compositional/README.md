# Compositional Memory Recall

This directory contains the compositional TokMem experiments on XLAM/APIGen.

## Current Maintained Method Surface

The maintained compositional path keeps these method switches:

- `--use_eoc`
- `--use_logit_bias`
- `--use_tool_head_replacement`

Historical archived runs still exist under `compositional/runs/`. Current code and docs describe the maintained `eoc/logit_bias/tool_head_replacement` family.

## Modes

| Mode | `--use_eoc` | `--use_logit_bias` | `--use_tool_head_replacement` | Behavior |
| --- | --- | --- | --- | --- |
| Baseline | off | off | off | Original TokMem decoding and training |
| EOC token only | on | off | off | Inserts explicit `eoc` boundary tokens between tool-controlled spans |
| EOC + logit bias | on | on | off | Trains a detached tool-prior head on boundary states and adds centered tool-only bias back to decode logits |
| EOC + tool-head replacement | on | off | on | Trains the same detached tool-prior head and replaces boundary-time tool triggers with tool ids sampled from that head |

Constraint summary:

- `--use_logit_bias` requires `--use_eoc`
- `--use_tool_head_replacement` requires `--use_eoc`
- `--use_logit_bias` and `--use_tool_head_replacement` are decode-time alternatives

Useful flags:

- `--use_eoc`
- `--use_logit_bias`
- `--use_tool_head_replacement`
- `--logit_bias_loss_weight` default `0.1`
- `--logit_bias_network` default `linear`, choices: `mlp`, `linear`
- `--logit_bias_scale` default `1.0`
- `--max_length` default `512`
- `--max_new_tokens` default `512`

## Method Notes

### EOC

When `--use_eoc` is enabled, each gold tool span becomes:

```text
<tool_token> <json_args> <eoc>
```

`eoc` is a reserved special token that marks the end of one tool-controlled span and the next boundary decision point.

`--max_new_tokens` controls only the training-time evaluation and demo decode budget in `main_sequential.py`. It does not change teacher-forcing supervision length during training.

### Auxiliary tool head

`--use_logit_bias` and `--use_tool_head_replacement` both train a detached auxiliary tool-prior head on boundary states.

Training:

1. collect assistant-start and gold-`eoc` boundary states
2. detach those boundary hidden states
3. predict the next gold tool id with `logit_bias_head`
4. add `logit_bias_loss_weight * CE` to the autoregressive loss

### Logit bias

`--use_logit_bias` uses the auxiliary head as a soft decode-time bias.

Decoding:

1. compute tool-only logits from the boundary state
2. convert them to `tool_log_probs`
3. center them against the uniform-tool prior `1 / K`
4. multiply by `logit_bias_scale`
5. add the result back only to tool-token columns

This means an informative prior only reweights relative preference among tool tokens and leaves non-tool logits unchanged.

### Tool-head replacement

`--use_tool_head_replacement` uses the auxiliary head as a hard replacement policy at EOC decision sites.

Decoding:

1. let the base LM sample the next token at assistant-start or `eoc` boundary rows
2. check whether that sampled token is one of the reserved tool tokens
3. for rows where the base LM already triggered a tool token, sample a replacement tool id from the auxiliary head
4. write the replacement reserved tool token into the generated sequence

This makes replacement trigger-gated: the auxiliary head chooses which tool token to emit only after the base LM has already decided to emit a tool token at a boundary. Non-tool continuations from the base LM pass through unchanged.

## Maintained Launchers

Single-round maintained launchers for tools `51-100`:

- `scripts/compositional/llama_1b/tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh`

Additional Llama-1B adaptation launchers over `1-50 -> 51-100`:

- `scripts/compositional/llama_1b/adap_tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/adap_tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/adap_tokmem_eoc_logit_bias_llama_1b.sh`

These launchers use `1-50:1,51-100:3` with `--use_lora --freeze_lora_after_first`, so the first round adapts LoRA on held-out tools and later rounds continue TokMem-side training on `51-100`.

`main_sequential.py` now accepts `--batch_size_per_round` for multi-round TokMem runs. The current Llama-1B adaptation launchers use `16,24`, so the adaptation round matches the suite LoRA train batch size and the later TokMem round matches the suite TokMem train batch size. The shell launchers keep `BATCH_SIZE_PER_ROUND` overridable for smoke tests.

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

This suite launcher is the maintained entrypoint for the `51-100 / 4 calls` paper comparison sweep across `llama1b`, `llama3b`, `llama8b` and methods `icl`, `rag`, `lora`, `tokmem`, `tokmem_eoc`, `tokmem_eoc_logit_bias`, `adap_tokmem`, `adap_tokmem_eoc`, `adap_tokmem_eoc_logit_bias`.

The same suite also schedules a separate `51-100 / 10 calls` TokMem-family stress test for `tokmem`, `tokmem_eoc_logit_bias`, `adap_tokmem`, and `adap_tokmem_eoc_logit_bias`. It synthesizes `8000` train and `800` test examples with 10-call filenames and uses smaller TokMem train/eval batches:

- `llama1b`: `4/16`
- `llama3b`: `2/8`
- `llama8b`: `1/4`

For the two 10-call adaptation methods, the first round stays aligned with the suite adaptation setup over `1-50 / 4 calls`, and the second round trains on `51-100 / 10 calls`:

- `llama1b`: `16,4`
- `llama3b`: `8,2`
- `llama8b`: `4,1`

The suite passes max length by call scope: `4calls=512` and `10calls=1024`.

The three adaptation methods use `1-50:1,51-100:3` inside the same suite, with round-wise train batch sizes aligned to the maintained suite defaults:

- `llama1b`: `16,24`
- `llama3b`: `8,16`
- `llama8b`: `4,8`

It uses one scheduling unit per `call scope × model × method × trial`, keeps the 5 trials independently schedulable across GPUs, writes archived outputs under `results/compositional/<suite_name>/`, and produces:

- `runs/<task_name>/` per trial
- `task_manifest.tsv`
- `task_status.json`
- `summary.md`
- `summary.json`
- `scheduler.log`
- `gpu_availability.log`

The suite scheduler only monitors and schedules the GPUs listed by `--gpus`. A task starts on a GPU after that GPU reports `memory.used <= 2048 MiB` for 300 consecutive seconds; `--poll-seconds` controls the sampling interval.

When the suite is rerun with `--suite-name <existing-suite>`, the launcher reconciles the existing `task_manifest.tsv` with the method set defined in the current script. Existing successful tasks stay skipped, newly introduced methods are appended into the same manifest, and the refreshed `summary.md` and `summary.json` cover the combined old and new tasks.

When the suite is resumed with `--rerun-failed --suite-name <existing-suite>`, the launcher reuses the recorded `task_manifest.tsv` as-is and only retries unfinished tasks already listed in that manifest. The current rerun invocation is snapshotted under:

- `reruns/<rerun_id>/run_paper_compositional_suite.sh`
- `reruns/<rerun_id>/suite_config.json`
- `reruns/<rerun_id>/task_manifest.tsv`

Suite-level metadata keeps evaluation scopes and adaptation training scopes separately. `summary.md` writes separate result tables for `tools 51-100 / 4 calls` and `tools 51-100 / 10 calls`; adaptation entries record `1-50:1,51-100:3` plus their per-round call limits.

Use these labels when reporting TokMem-family method comparisons:

- `baseline`
- `eoc-only`
- `eoc+logit_bias`
- `eoc+tool_head_replacement`

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
- `avg_logit_bias_loss` when `use_logit_bias=true` or `use_tool_head_replacement=true`

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
- `model.py`: reserved-tool-token model, EOC boundary logic, logit-bias decoding, tool-head replacement decoding
- `training.py`: autoregressive training loop plus detached auxiliary tool-head loss
- `dataset.py`: XLAM/APIGen data loading and EOC target formatting
- `tool_retrieval.py`: RAG tool retrieval for ICL and related baselines
