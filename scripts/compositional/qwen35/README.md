# Qwen3.5 Table 1 rebuttal suite

`run_qwen35_table1_rebuttal.sh` runs the complete compositional Table 1 method
set for Qwen3.5-9B and then Qwen3.5-4B:

The isolated fast-kernel environment and its validation requirements are
documented in
[`docs/compositional/qwen35-fast-environment.md`](../../../docs/compositional/qwen35-fast-environment.md).
The reproducible validation command and recorded outputs are in
[`docs/compositional/qwen35-fast-validation-20260725.md`](../../../docs/compositional/qwen35-fast-validation-20260725.md).
The earlier `tokmem` fallback suite was stopped and retained as a partial
archive. New formal runs should pass `--conda-env tokmem-qwen35`; omitting the
flag preserves the previous `tokmem` default.

- ICL
- RAG with top-5 MiniLM retrieval
- TokMem
- TapMem
- Fine-Tuning (LoRA)
- TokMem with adaptation
- TapMem with adaptation

For each backbone, the suite first searches the TapMem memory-token learning
rate on seed 42. Only after fixing that learning rate does it run the final
seeds 40, 41, and 42 matrix. The two backbones are tuned independently.
LoRA rank, target modules, and LoRA learning rates retain the previously
agreed Table 1 settings; the sweep covers the shared synthetic-memory
learning rate used by TokMem, TapMem, and their adaptation variants.

The launcher uses conservative batches that fit the longest 512-token
training examples with the Transformers PyTorch fallback:

| Backbone | ICL | RAG | TokMem/TapMem train/eval | LoRA train/eval | Adaptation rounds/eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 4 | 16 | 4 / 16 | 2 / 16 | 2,4 / 16 |
| Qwen3.5-4B | 8 | 32 | 4 / 32 | 4 / 32 | 4,4 / 32 |

The launcher dynamically polls every requested GPU and uses all currently
empty cards. Tasks hold the shared `/tmp/tokmem_gpu_locks/gpu_<id>.lock`
throughout execution, so they can coexist with the repository's other
maintained suites.

Learning-rate sweeps do not save checkpoints. Final TokMem, TapMem, and
adaptation runs pass
`--save_checkpoints --checkpoint_format trainable_only`; pure LoRA uses
PEFT's adapter-only `save_pretrained()`. At inference time the original
Qwen3.5 backbone is loaded first, followed by the saved memory embeddings,
TCRA head, and/or LoRA adapter. No final run duplicates the frozen 4B/9B
backbone weights.

For a saved pure-LoRA run, use
`scripts/compositional/evaluate_lora_checkpoint.py` with its
`run_config.json` and `round_<round>_tools_<range>/` adapter directory. The
script validates the adapter's recorded base model before running inference.

The completed
`results/compositional/qwen35_table1_rebuttal_20260725_fast_v1` suite predates
this checkpoint behavior and contains no recoverable trained weights. Its
metrics remain valid, but producing checkpoints requires rerunning the trained
methods (TokMem, TapMem, Fine-Tuning, and both adaptation variants); ICL and
RAG do not train parameters and therefore need no checkpoint rerun.

For that checkpoint-only rerun, use `--trained-only`. This reuses the selected
memory learning rates (`5e-3` for 9B and `3e-3` for 4B), skips ICL, RAG, and
the learning-rate sweep, and requires the expected `.pt` or PEFT adapter files
before a task can be marked successful:

```bash
bash scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh \
    --suite-name qwen35_table1_checkpoint_rerun_20260725 \
    --gpus 1,3,5,6 \
    --conda-env tokmem-qwen35 \
    --trained-only
```

Run in the foreground:

```bash
bash scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh \
    --suite-name qwen35_table1_rebuttal_20260725 \
    --gpus 0,1,2,3,4,5,6,7 \
    --conda-env tokmem-qwen35
```

Run in the background:

```bash
mkdir -p results/compositional/qwen35_table1_rebuttal_20260725
nohup bash scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh \
    --suite-name qwen35_table1_rebuttal_20260725 \
    --gpus 0,1,2,3,4,5,6,7 \
    --conda-env tokmem-qwen35 \
    > results/compositional/qwen35_table1_rebuttal_20260725/launcher.log 2>&1 &
```

Re-running the same command with the same suite name reuses successful tasks
whose command is unchanged. Results are written to `metrics.tsv` and
`summary.md` using Bash, `jq`, and `awk`; no Python result-summary script is
used. The selected conda environment, dependency manifest, and critical
implementation hashes are recorded in `suite_config.txt`; a suite cannot be
resumed after that runtime fingerprint changes. After all runs finish, inspect
those files and manually add the verified table to
`results/rebuttal/rebuttal-needs-exps.md`.
