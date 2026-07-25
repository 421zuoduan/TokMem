# Compositional Memory Recall

This directory contains the compositional TokMem experiments on XLAM/APIGen.

## Current Maintained Method Surface

The maintained compositional path keeps these method switches:

- `--use_eoc`
- `--use_logit_bias`
- `--use_logit_train_add`
- `--detach_head_from_ar_loss`
- `--use_tool_head_replacement`
- `--use_memory_bank_constraint`

Historical archived runs still exist under `compositional/runs/`. Current code and docs describe the maintained `eoc/logit_bias/tool_head_replacement` family.

## Modes

| Mode | `--use_eoc` | `--use_logit_bias` | `--use_logit_train_add` | `--use_tool_head_replacement` | `--use_memory_bank_constraint` | Behavior |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | off | off | off | off | off | Original TokMem decoding and training |
| EOC token only | on | off | off | off | off | Inserts explicit `eoc` boundary tokens between tool-controlled spans |
| EOC + logit bias | on | on | on | off | off | Trains a detached tool-prior head on boundary states, adds centered tool-only bias back to decode logits, and applies the same transform during boundary-site training |
| EOC + logit bias without train add | on | on | off | off | off | Uses `--no-use_logit_train_add` to keep the head out of teacher-forced AR logits |
| EOC + tool-head replacement | on | off | off | on | off | Trains the same detached tool-prior head and replaces boundary-time tool triggers with tool ids sampled from that head |
| TapMem + bank constraint | on | on | on | off | on | Applies the TapMem/TCRA bias first, then selects from memory tokens plus EOS at assistant-start and generated-EOC boundaries |

Constraint summary:

- `--use_logit_bias` requires `--use_eoc`
- `--use_logit_train_add` defaults enabled and is active with `--use_logit_bias`
- explicit `--use_logit_train_add` requires `--use_logit_bias`
- `--detach_head_from_ar_loss` requires `--use_logit_bias --use_logit_train_add`
- `--detach_head_from_ar_loss` does not apply to `--use_tool_head_replacement`
- `--use_tool_head_replacement` requires `--use_eoc`
- `--use_logit_bias` and `--use_tool_head_replacement` are decode-time alternatives
- `--use_memory_bank_constraint` can be combined with `--use_logit_bias`; the fused logits are constrained after TCRA reweighting
- `--use_memory_bank_constraint` cannot be combined with `--use_tool_head_replacement`
- memory-bank constrained candidates always contain all memory tokens plus `tokenizer.eos_token_id`

Useful flags:

- `--use_eoc`
- `--use_logit_bias`
- `--use_logit_train_add` default enabled; use `--no-use_logit_train_add` to disable boundary-site train add
- `--detach_head_from_ar_loss` default disabled; enable it to block the train-add AR-loss gradient to the head without changing the forward logits
- `--use_tool_head_replacement`
- `--use_memory_bank_constraint` enables constrained decoding, either as a no-adapter control or on top of TapMem logit bias
- `--memory_bank_probability_threshold` defaults to `0.5` and is used only by TokMem without EOC
- `--logit_bias_loss_weight` default `0.1`
- `--logit_bias_network` default `linear`, choices: `mlp`, `linear`
- `--logit_bias_scale` default `1.0`
- `--detach` default enabled; use `--no-detach` to let prior-head CE gradients update the boundary-state path
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

### Memory-bank constraint

`--use_memory_bank_constraint` changes free decoding only. It does not create a routing head, add a routing loss, or alter teacher-forcing training. At every active constraint site, greedy or sampled selection is performed over the compact candidate set consisting of every memory token plus `tokenizer.eos_token_id`; including the ending token lets the model terminate instead of forcing another procedure. It can be used alone for the TokMem/EOC-only controls or layered on an existing TapMem logit-bias checkpoint.

Without `--use_eoc`, the assistant-start position is constrained and later positions are gated by the full-vocabulary normalized memory-token mass:

```text
P(memory bank) = sum(softmax(full_vocab_logits)[memory_token_ids])
```

The constraint fires after at least one ordinary response token has followed the previous memory token and `P(memory bank) >= memory_bank_probability_threshold`. The default threshold is `0.5`. The bank-conditional normalized entropy is recorded for analysis but is not part of the gate.

With `--use_eoc`, the constraint fires deterministically at assistant start and after every actually generated `eoc`; the probability threshold is recorded but does not control activation. When `--use_logit_bias` is also enabled, the TCRA bias is added to the LM logits first and both constrained selection and diagnostics use that fused distribution. In all modes, argument and EOC generation positions retain the full vocabulary. Evaluation results record trigger counts, how often the constrained choice differs from full-vocabulary decoding, mean trigger mass, and mean bank-conditional entropy.

### Tool head

`--use_logit_bias` and `--use_tool_head_replacement` both train a tool-prior head on boundary states. The default `--detach` setting detaches those states before the head classification loss, so classification gradients update the head parameters without updating tool embeddings or LoRA through the boundary state. Use `--no-detach` to let the classification loss also backpropagate through the boundary hidden states into trainable tool embeddings and LoRA parameters.

Training:

1. collect assistant-start and gold-`eoc` boundary states
2. apply the `detach` setting to those boundary hidden states
3. predict the next gold tool id with `logit_bias_head`
4. add `logit_bias_loss_weight * CE` to the autoregressive loss

With `--use_logit_bias --use_logit_train_add`, training also applies the decode-time logit-bias transform to the main autoregressive CE logits at those same gathered boundary sites. The flag defaults enabled for logit-bias runs; use `--no-use_logit_train_add` for classification-only head training. The transform is:

```text
(log_softmax(tool_logits) + log(num_tools)) * logit_bias_scale
```

The resulting bias is added to the full-vocab columns for tool tokens. By default, its computation graph stays intact, so AR loss can update `logit_bias_head` through the train-add path. With `--detach_head_from_ar_loss`, the transformed bias is detached immediately before it is added: forward logits and loss values stay unchanged, but AR loss no longer updates the head through train-add. The classification loss still uses non-detached head logits and continues to train the head.

The three gradient controls are separate:

- `--use_logit_train_add` controls whether the transformed head result is added to tool-token logits during training.
- `--detach` controls whether the head classification loss can update tool embeddings or LoRA through boundary hidden states. The existing train-add implementation also feeds the head a boundary state using this same detach setting.
- `--detach_head_from_ar_loss` controls whether AR loss can update the head through the train-add path.

To train the head only with classification loss while still applying its result to training and inference logits, use:

```bash
--use_eoc \
--use_logit_bias \
--use_logit_train_add \
--detach \
--detach_head_from_ar_loss
```

### Logit bias

`--use_logit_bias` uses the head as a soft decode-time bias.

Decoding:

1. compute tool-only logits from the boundary state
2. convert them to `tool_log_probs`
3. center them against the uniform-tool prior `1 / K`
4. multiply by `logit_bias_scale`
5. add the result back only to tool-token columns

This means an informative prior only reweights relative preference among tool tokens and leaves non-tool logits unchanged.

### Tool-head replacement

`--use_tool_head_replacement` uses the head as a hard replacement policy at EOC decision sites.

Decoding:

1. let the base LM sample the next token at assistant-start or `eoc` boundary rows
2. check whether that sampled token is one of the reserved tool tokens
3. for rows where the base LM already triggered a tool token, sample a replacement tool id from the head
4. write the replacement reserved tool token into the generated sequence

This makes replacement trigger-gated: the head chooses which tool token to emit only after the base LM has already decided to emit a tool token at a boundary. Non-tool continuations from the base LM pass through unchanged.

## Maintained Launchers

Single-round maintained launchers for tools `51-100`:

- `scripts/compositional/run_rebuttal_compositional_pure_classification_head.sh`
- `scripts/compositional/run_memory_bank_constraint_eval.sh`
- `scripts/compositional/llama_1b/tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_train_add_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_train_add_no_detach_llama_1b.sh`
- `scripts/compositional/llama_1b/rerun_paper_compositional_logit_bias_scale_ablation.sh`
- `scripts/compositional/llama_1b/rerun_paper_compositional_logit_bias_loss_weight_ablation.sh`
- `scripts/compositional/qwen_0_5b/tokmem_eoc_logit_bias_scale_ablation_qwen_0_5b_4calls_seed42_3x.sh`
- `scripts/compositional/qwen_9b/tokmem_eoc_logit_bias_qwen_9b.sh`
- `scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh`
- `scripts/compositional/run_paper_compositional_logit_bias_scale_ablation_8gpu_nohup.sh`
- `scripts/compositional/run_paper_compositional_logit_bias_loss_weight_ablation.sh`

`run_rebuttal_compositional_pure_classification_head.sh` runs three trials concurrently as three independent processes on three distinct GPUs, with seed `42` for every trial. Set the GPU assignment with `TOKMEM_GPUS=0,1,2`; the launcher rejects missing or duplicate GPU identifiers. For a one-process capacity check, set matching single entries such as `TOKMEM_TRIALS=1 TOKMEM_GPUS=0`; the default remains trials `1,2,3`. `TOKMEM_MODEL_KEY=llama1b` is the default and uses `batch_size=24`, `eval_batch_size=256`; `TOKMEM_MODEL_KEY=llama8b` selects Llama-3.1-8B and uses `batch_size=8`, `eval_batch_size=64`. Both retain the corresponding 4-call TapMem paper settings (`max_length=512`, three epochs, and the same EOC/logit-bias settings) and add only `--detach_head_from_ar_loss`, so the routing head is trained by routing loss while its output is still added to memory-token logits. After the requested processes finish, the parent launcher writes per-trial artifacts plus `manifest.tsv`, `summary.md`, and `results.json` under `results/compositional/`; the summary reports the mean and standard deviation across successful trials and marks the suite incomplete unless every requested trial succeeds.

`run_memory_bank_constraint_eval.sh` performs eval-only inference on the paper TokMem checkpoints and matched EOC-only checkpoints. It produces `tokmem_bank_constraint` and `eoc_only_bank_constraint` predictions without retraining, defaults to threshold `0.5`, and writes `manifest.json`, per-trial JSONL, `summary.json`, and `summary.md` under `compositional/rebuttal/results/memory_bank_constraint/`. Use `--models llama1b --trial-ids 1 --limit 8` for a narrow smoke run.

The Qwen-0.5B scale-ablation launcher runs `tokmem_eoc_logit_bias` on `models/Qwen2.5-0.5B-Instruct` with tools `51-100`, 4-call data, `training_rounds=51-100:1`, `epochs=3`, `batch_size=16`, `eval_batch_size=64`, `max_length=512`, and `lr=5e-3`. It fixes `seed=42`, runs three trials per scale, assigns `logit_bias_scale=0.1` to GPU `5`, `0.5` to GPU `6`, and `2` to GPU `7`, then writes `manifest.tsv`, `summary.md`, and `results.json` under `results/compositional/<suite_name>/`.

`scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh` runs the complete
seven-method compositional Table 1 matrix for Qwen3.5-9B followed by
Qwen3.5-4B. Each backbone first performs an independent seed-42 TapMem
learning-rate sweep, then runs seeds 40/41/42 with the selected learning rate.
The scheduler dynamically consumes all empty GPUs from the requested pool,
does not save sweep checkpoints, and saves only memory/head/LoRA deltas for
the final trained runs. The final three-seed mean and sample standard
deviation are written with Bash, `jq`, and `awk`. Pass
`--conda-env tokmem-qwen35` to use the validated Qwen3.5 FLA/causal-conv1d
environment. The default remains `tokmem` for compatibility, and each suite
records the environment name, package manifest, and critical implementation
hashes in `suite_config.txt` to prevent mixed-environment resumes.

Qwen3.5 uses a dedicated `Qwen35FunctionCallingModel`, selected internally by
`backbone_registry.py` when the local Hugging Face config reports
`model_type=qwen3_5`. All other backbones continue to use the existing
`FunctionCallingModel`, and their prompt strings, response targets, and
generation settings remain unchanged. Qwen3.5 uses its native chat template
with thinking disabled, supervises `<|im_end|>` as the response-ending token,
and explicitly keeps `<|endoftext|>` as padding while both native TokMem and
custom TapMem decoding stop on `<|im_end|>`. The standalone Qwen3.5-9B
launcher uses the same `main_sequential.py` EOC + logit-bias path with
conservative default capacity settings (`batch_size=1`, `eval_batch_size=4`,
`max_length=512`);
`TOKMEM_GPU`, `TOKMEM_MODEL_PATH`, `TOKMEM_BATCH_SIZE`,
`TOKMEM_EVAL_BATCH_SIZE`, and `TOKMEM_MAX_LENGTH` are overridable.

The Llama-1B paper-style scale-ablation nohup launcher starts `scripts/compositional/llama_1b/rerun_paper_compositional_logit_bias_scale_ablation.sh` on GPUs `0,1,2,3,4,5,6,7`. It runs `tokmem_eoc_logit_bias` with `--detach --use_logit_train_add` over `logit_bias_scale=0.1,0.5,0.8,1,1.1,1.2,1.3,1.4,1.5,2,3,5`, three trials per scale, tools `51-100`, 4-call data, `batch_size=24`, `eval_batch_size=256`, and writes `nohup.log`, `nohup.pid`, `manifest.tsv`, `summary.md`, `results.json`, and `gpu_availability.log` under `results/compositional/<suite_name>/`. The runner treats the GPU list as worker slots, takes a per-GPU `flock` under `/tmp/tokmem_gpu_locks`, and starts each GPU worker after `memory.used <= 2048 MiB` for 180 consecutive seconds. Extra scales queue round-robin onto the available worker slots. `manifest.tsv` records every attempted trial with `status` and `exit_code`; failed trials remain visible in `summary.md` and `results.json`. Mean metrics and loss summaries include trials whose manifest status is `success`. The summary table labels `avg_f1_score` as `Arguments F1`.

The Llama-1B loss-weight ablation launcher starts `scripts/compositional/llama_1b/rerun_paper_compositional_logit_bias_loss_weight_ablation.sh` on GPUs `0,1,2,3,4,5,6,7`. It keeps `logit_bias_scale=1.0` and sweeps `logit_bias_loss_weight=0.01,0.05,0.1,0.15,0.2,0.3,0.5,1`, three trials per weight, with the same tools `51-100`, 4-call data, `batch_size=24`, `eval_batch_size=256`, status-aware manifest, GPU lock scheduling, and output files as the scale ablation. Its GPU workers start after `memory.used <= 2048 MiB` for 240 consecutive seconds. Mean metrics and loss summaries include trials whose manifest status is `success`.

Tokenizer note: Llama 3.x tokenizers expose native `reserved_special_token_*` entries. Qwen2.5-0.5B does not expose those entries in its tokenizer config, so `FunctionCallingModel` adds synthetic `<|reserved_special_token_N|>` special tokens at runtime when the reserved-token budget is short. If the added tokenizer length exceeds the base model embedding table, the model resizes token embeddings before creating the trainable TokMem tool/eoc embeddings.

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
- `scripts/compositional/rerun_paper_compositional_head.sh`
- `scripts/compositional/run_paper_compositional_head_8gpu_nohup.sh`
- `scripts/compositional/run_paper_compositional_logit_bias_scale_ablation_8gpu_nohup.sh`
- `scripts/compositional/run_paper_compositional_logit_bias_loss_weight_ablation.sh`
- `scripts/compositional/launch_paper_compositional_llama3b8b_lora_vs_adap_logit_bias_6trials_nohup.sh`
- `scripts/compositional/run_paper_compositional_llama3b8b_lora_vs_adap_logit_bias_6trials_suite.sh`

This suite launcher is the maintained entrypoint for the `51-100 / 4 calls` paper comparison sweep across `llama1b`, `llama3b`, `llama8b` and methods `icl`, `rag`, `lora`, `tokmem`, `tokmem_eoc`, `tokmem_eoc_logit_bias`, `tokmem_eoc_replace_head`, `adap_tokmem`, `adap_tokmem_eoc`, `adap_tokmem_eoc_logit_bias`, `adap_tokmem_eoc_replace_head`.

`rerun_paper_compositional_head.sh` keeps the same scheduler, datasets, model set, artifact layout, and `--rerun-failed` workflow, while scheduling three trials of the logit-bias head methods `tokmem_eoc_logit_bias` and `adap_tokmem_eoc_logit_bias` for both `4calls` and `10calls`. It keeps the tool-token embedding LR at `5e-3`, uses adaptation LoRA LR `8e-5`, and runs all logit-bias methods with `--detach --use_logit_train_add`. When pointed at an existing suite directory, its task loading, status JSON, and summaries are filtered to that logit-bias method set.

`run_paper_compositional_head_8gpu_nohup.sh` starts that logit-bias head suite through `nohup` on GPUs `0,1,2,3,4,5,6,7`, activates the `tokmem` conda environment before launching, and writes `nohup.log` plus `nohup.pid` under the generated suite directory.

`launch_paper_compositional_llama3b8b_lora_vs_adap_logit_bias_6trials_nohup.sh` starts the Llama-3B/8B 4-call rerun through `nohup` on GPUs `0,1,2,3,4,5,6,7`. It calls `run_paper_compositional_llama3b8b_lora_vs_adap_logit_bias_6trials_suite.sh`, which schedules `lora` and `adap_tokmem_eoc_logit_bias` over `llama3b` and `llama8b`, six trials per model/method, using the maintained paper-suite batch settings. The logit-bias adaptation method runs with `--use_eoc --use_logit_bias --detach --use_logit_train_add`, `training_rounds=1-50:1,51-100:3`, `lr=5e-3`, and `lora_lr=8e-5`; LoRA runs with `training_rounds=51-100:3`, `lr=5e-5`, and `q_proj,v_proj` target modules. The summary focuses on `Tool F1` from `avg_tool_f1_score` and `Arguments F1` from `avg_f1_score`, with means reported across complete six-trial groups. The runner marks GPUs that just finished suite-owned tasks as immediately reusable before continuing the scheduling loop.

The same suite also schedules a separate `51-100 / 10 calls` TokMem-family stress test for `tokmem`, `tokmem_eoc_logit_bias`, `tokmem_eoc_replace_head`, `adap_tokmem`, `adap_tokmem_eoc_logit_bias`, and `adap_tokmem_eoc_replace_head`. It synthesizes `8000` train and `800` test examples with 10-call filenames and uses smaller TokMem train/eval batches:

- `llama1b`: `4/16`
- `llama3b`: `2/8`
- `llama8b`: `1/4`

For the three 10-call adaptation methods, the first round stays aligned with the suite adaptation setup over `1-50 / 4 calls`, and the second round trains on `51-100 / 10 calls`:

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

To summarize already completed suite trials after a partial stop, or to compare suites with different completed trial counts, use:

```bash
python scripts/compositional/summarize_completed_trials.py \
  results/compositional/all_methods \
  results/compositional/paper_compositional_mainline_bs \
  --output-dir results/compositional/completed_trials_summary
```

This script averages every successful trial within each `call_scope x model x method` group and writes `completed_trials_summary.md` plus `completed_trials_summary.json`. It keeps `4calls` and `10calls` separate, reports the regular suite metrics, and adds Table-3-style Tool Selection F1 / Argument F1 by target function-call count. The call-count F1 parser first uses `evaluation_results.json` breakdowns, then falls back to `stdout.log`, and finally recomputes ICL/RAG breakdowns from `detailed_results` when available.

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

`--save_checkpoints` retains the historical full `model_state_dict` format by
default. Add `--checkpoint_format trainable_only` to save only the TokMem
memory embeddings, optional TapMem/TCRA head, and optional PEFT LoRA adapter.
Trainable-only checkpoints also record the complete ordered tool list and
reserved-token mapping. Loading first reconstructs the wrapper from the
original pretrained model and then applies these deltas. Existing full
checkpoints continue to load with the original strict state-dict path.

Training methods keep metrics in `evaluation_results.json` under the latest round payload:

```text
rounds[-1].eval_results
```

ICL/RAG baselines save the same maintained metric names under the top-level `metrics` field in `evaluation_results.json`.

Their saved `config.generation_params` records the actual evaluation decoding settings: `max_new_tokens=256`, `temperature=0.6`, `do_sample=false`, and `top_p=0.9`.

For ICL/RAG, `function_calls` on the maintained compositional dataset store argument JSON only. The saved tool metrics therefore first map each predicted argument object back to a tool id from the active prompt tool schemas, and ambiguous calls stay unresolved. Fields including `tool_accuracy`, `tool_exact_match_acc`, `avg_tool_f1_score`, `avg_f1_score`, `exact_accuracy`, and `parse_error_rate` are overall values over the full evaluation set. `tool_accuracy` is the binary per-sample-tool accuracy over TP/TN/FP/FN, `tool_exact_match_acc` is the sample rate where the full tool set is predicted exactly, and `avg_tool_f1_score` keeps the existing Tool F1 definition. Per-call-count breakdowns are printed in `evaluation.log`.

`training_summary.json` is intentionally compact. It keeps:

- `round`
- `tools`
- `epochs`
- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss` when `use_logit_bias=true` or `use_tool_head_replacement=true`
- `use_logit_train_add`
- `detach_head_from_ar_loss`
- `detach`

Detailed position counters are available in the per-round training `results` payload and training logs. The saved `training_summary.json` stays a run-level loss summary.

Passing `--tensorboard` on the maintained TokMem path saves two static PNG trend plots under the run directory:

- `loss_step.png`
- `lr_step.png`

## Per-Sample Prediction Comparison

Use `scripts/compositional/generate_checkpoint_predictions.py` to run a saved
TokMem-family checkpoint over every example in a compositional test split and
write one JSONL record per sample. The script reads `run_config.json` for the
model path, data path, and maintained mode flags such as `use_eoc`,
`use_logit_bias`, and adaptation LoRA configuration. Both historical full
checkpoints and versioned trainable-only checkpoints are supported.
Unless overridden with `--max-new-tokens`, generation reuses the training
run's `max_new_tokens` value; older run configs without that field fall back
to 256.

Example for the Llama-1B TokMem and EOC+logit-bias checkpoints:

```bash
python scripts/compositional/generate_checkpoint_predictions.py \
  --run-config results/compositional/all_methods/runs/llama1b_tokmem_trial1_seed42/run_config.json \
  --checkpoint results/compositional/all_methods/runs/llama1b_tokmem_trial1_seed42/round_1_tools_51_100.pt \
  --method tokmem \
  --output results/compositional/checkpoint_predictions/llama1b/tokmem_predictions.jsonl

python scripts/compositional/generate_checkpoint_predictions.py \
  --run-config results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/run_config.json \
  --checkpoint results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/round_1_tools_51_100.pt \
  --method tokmem_eoc_logit_bias \
  --output results/compositional/checkpoint_predictions/llama1b/eoc_logit_bias_predictions.jsonl
```

Each JSONL record keeps the sample index, user input, expected tools/calls, predicted tools/calls, reserved tool tokens, `tool_sequence_exact`, `call_exact`, F1, Tool F1, and parse-error counts.

Standalone Fine-Tuning checkpoints are PEFT adapter directories rather than
TokMem `.pt` files. Evaluate one by loading its original backbone plus adapter:

```bash
python scripts/compositional/evaluate_lora_checkpoint.py \
  --run-config results/compositional/<suite>/runs/<run>/run_config.json \
  --adapter-checkpoint results/compositional/<suite>/runs/<run>/round_1_tools_51_100 \
  --output results/compositional/<suite>/runs/<run>/adapter_evaluation.json
```

The evaluator derives the final-round test split from `run_config.json`,
loads the base model locally, applies the saved adapter with PEFT, and then
uses the same maintained LoRA evaluation routine.

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
- `training.py`: autoregressive training loop plus tool-head classification loss
- `dataset.py`: XLAM/APIGen data loading and EOC target formatting
- `tool_retrieval.py`: RAG tool retrieval for ICL and related baselines
