# Compositional Utils

This directory keeps temporary or maintenance-oriented compositional utilities. Experiment launchers belong under `scripts/compositional/`; reusable model, data, and training code belongs directly under `compositional/`.

## Files

- `migrate_legacy_runs.py`: migrates older loose compositional logs into the maintained run directory layout with `run_config.json` and `run_summary.json` metadata.
- `summarize_readme_myself_runs.py`: summarizes a completed README-style compositional run manifest into a compact markdown comparison table.
- `run_train4_checkpoint_eval_10calls.py`: reads the final 4-call checkpoint groups from a compositional suite `summary.md` and adjacent `task_status.json`, runs the 1B/3B/8B TokMem/TapMem checkpoints on the tools51-100 10-call test split, then writes per-trial prediction JSONL files plus `summary.json` and `summary.md` under `compositional/rebuttal/results/`.
- `analyze_generation_eoc_boundaries.py`: reruns free generation for non-adaptation `tokmem_eoc` and `tokmem_eoc_logit_bias` checkpoints, keeps raw generated token IDs so EOC tokens remain visible, and summarizes generated EOC boundary precision/recall/F1 under `compositional/rebuttal/results/eoc_boundary_accuracy/`. For Llama-3B it uses the final checkpoint suite referenced by `completed_trials_summary.md`.
- `run_memory_bank_constraint_eval.py`: loads the paper TokMem, matched EOC-only, and TapMem 4-call checkpoints, applies memory-bank constrained decoding with the ending token retained, and writes aligned predictions plus trigger diagnostics and aggregate metrics under `compositional/rebuttal/results/memory_bank_constraint/`. For TapMem, TCRA bias is fused before constrained candidate selection.
- `analyze_procedure_routing_prefix.py`: regenerates aligned per-sample predictions for the Llama-1B 10-call TokMem and TapMem runs behind TapMem Figure 4, then groups samples by their gold procedure count and reports the `count=0..J` distribution of consecutive correct routing decisions before the first wrong or missing tool. The default checkpoint policy uses four TokMem trials from `all_methods` and three TapMem trials from `paper_compositional_head_8gpu`; counts and rates are aggregated per trial rather than pooling the unequal trial totals.

Run the 4-call checkpoint to 10-call evaluation through the script wrapper:

```bash
bash scripts/compositional/run_train4_checkpoint_eval_10calls.sh
```

For parallel runs, pass `--models llama1b`, `--models llama3b`, or `--models llama8b` in separate shells with the same `--output-dir`, then run once with `--summarize-only` after all prediction JSONL files exist.

Run generated EOC boundary analysis through:

```bash
bash scripts/compositional/analyze_generation_eoc_boundaries.sh
```

Run the memory-bank constraint evaluation through:

```bash
bash scripts/compositional/run_memory_bank_constraint_eval.sh
```

TokMem uses a full-vocabulary memory-mass threshold of `0.5`; EOC-only and TapMem use generated EOC boundaries and do not threshold boundary activation. For TapMem, run with `--methods tapmem_bank_constraint`; the model applies TCRA bias before restricting selection to memory tokens plus EOS. Use `--models llama1b --trial-ids 1 --limit 8` for a smoke run, and `--summarize-only` after all desired prediction files exist.

Inspect the Llama-1B 10-call routing-prefix inputs without loading a model:

```bash
python compositional/utils/analyze_procedure_routing_prefix.py --dry-run
```

Run a narrow GPU smoke test in a separate output directory:

```bash
python compositional/utils/analyze_procedure_routing_prefix.py \
  --methods tokmem,tapmem \
  --tokmem-trial-ids 1 \
  --tapmem-trial-ids 1 \
  --limit 16 \
  --output-dir /tmp/llama1b_10calls_routing_prefix_smoke
```

Run the complete Figure 4 comparison:

```bash
python compositional/utils/analyze_procedure_routing_prefix.py
```

To split generation across two inference workers while retaining one shared
provenance and result directory, keep the full default method/trial selection in
both commands and assign only the worker tasks:

```bash
CUDA_VISIBLE_DEVICES=2 python compositional/utils/analyze_procedure_routing_prefix.py \
  --generate-only \
  --generate-tasks tokmem:1,tokmem:2,tapmem:1,tapmem:2

CUDA_VISIBLE_DEVICES=3 python compositional/utils/analyze_procedure_routing_prefix.py \
  --generate-only \
  --generate-tasks tokmem:3,tokmem:4,tapmem:3

python compositional/utils/analyze_procedure_routing_prefix.py --summarize-only
```

`--generate-tasks` only controls which prediction files a worker produces; it
requires `--generate-only` and does not alter the full experiment provenance.
Therefore the workers can safely populate distinct files in the same run
directory, followed by one summary process after both workers finish.

The default output root is `results/compositional/llama1b_10calls_routing_prefix/`.
Each distinct data/checkpoint/generation/code provenance is stored in its own short,
readable run directory, for example
`runs/figure4_full_tokmem-tapmem_p3a7c91d2e/`. The readable prefix only keeps the
scope and methods; detailed settings live in `manifest.json`, while the short hash
changes whenever those settings or inputs change. `run_index.json` at the output
root lists every run directory, status, method, trial, and update time for lookup.

Each run directory contains a prediction JSONL for every selected checkpoint,
`manifest.json`, `per_trial.jsonl`, `summary.json`, and `summary.md`. Every prediction
record embeds the provenance ID, data SHA, and checkpoint fingerprint. Completed
files and partial resumes are rejected if those fields do not match the run
manifest, so changing data, checkpoints, decoding parameters, or relevant code
automatically uses a different run path instead of silently reusing stale results.

Generation uses the maintained left-padding batch path with greedy decoding,
`max_new_tokens=512`, and batch size `16`. The summary includes an audit against
each archived run's Tool F1; use `--strict-metric-audit` when a mismatch larger than
`--metric-tolerance` should fail the command. Recompute only the summaries from
complete prediction files with the same provenance configuration:

```bash
python compositional/utils/analyze_procedure_routing_prefix.py --summarize-only
```

The direct unit test lives outside this utility directory:

```bash
python compositional/tests/test_analyze_procedure_routing_prefix.py
```
