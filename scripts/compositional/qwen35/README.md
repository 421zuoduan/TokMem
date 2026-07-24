# Qwen3.5 Table 1 rebuttal suite

`run_qwen35_table1_rebuttal.sh` runs the complete compositional Table 1 method
set for Qwen3.5-9B and then Qwen3.5-4B:

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

The launcher dynamically polls every requested GPU and uses all currently
empty cards. Tasks hold the shared `/tmp/tokmem_gpu_locks/gpu_<id>.lock`
throughout execution, so they can coexist with the repository's other
maintained suites.

Run in the foreground:

```bash
bash scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh \
    --suite-name qwen35_table1_rebuttal_20260725 \
    --gpus 0,1,2,3,4,5,6,7
```

Run in the background:

```bash
mkdir -p results/compositional/qwen35_table1_rebuttal_20260725
nohup bash scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh \
    --suite-name qwen35_table1_rebuttal_20260725 \
    --gpus 0,1,2,3,4,5,6,7 \
    > results/compositional/qwen35_table1_rebuttal_20260725/launcher.log 2>&1 &
```

Re-running the same command with the same suite name reuses successful tasks
whose command is unchanged. Results are written to `metrics.tsv` and
`summary.md` using Bash, `jq`, and `awk`; no Python result-summary script is
used. After all runs finish, inspect those files and manually add the verified
table to `results/rebuttal/rebuttal-needs-exps.md`.
