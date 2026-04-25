# Llama 3B 4-call TokMem family scripts

Run the three maintained TokMem-family methods for the compositional 4-call setting:

```bash
cd /data/shilong/tokmem
bash scripts/compositional/llama_3b/multi_runs/run_tokmem_family_4calls_seed42_3x.sh
```

The multi-run launcher uses GPU 4 for `tokmem`, GPU 5 for `tokmem_eoc`, and GPU 6 for `tokmem_eoc_logit_bias`. Each method runs three trials with `seed=42`.

The experiment settings match `scripts/compositional/run_paper_compositional_suite.sh` for `llama3b` / `4calls` / TokMem-family runs:

- dataset: tools `51-100`, train size `5000`, test size `500`, max function calls `4`
- training: `--training_rounds 51-100:1`, `--epochs 3`, `--lr 5e-3`
- batching: `--batch_size 16`, `--eval_batch_size 192`
- context: `--max_length 512`
- model: `models/Llama-3.2-3B-Instruct`

Aggregate outputs are written under:

```text
results/compositional/llama3b_tokmem_family_4calls_seed42_3x_<timestamp>/
```

Individual training run directories are written under:

```text
compositional/runs/
```
