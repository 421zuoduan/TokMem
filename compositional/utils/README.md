# Compositional Utils

This directory keeps temporary or maintenance-oriented compositional utilities. Experiment launchers belong under `scripts/compositional/`; reusable model, data, and training code belongs directly under `compositional/`.

## Files

- `migrate_legacy_runs.py`: migrates older loose compositional logs into the maintained run directory layout with `run_config.json` and `run_summary.json` metadata.
- `summarize_readme_myself_runs.py`: summarizes a completed README-style compositional run manifest into a compact markdown comparison table.
- `run_train4_checkpoint_eval_10calls.py`: reads the final 4-call checkpoint groups from a compositional suite `summary.md` and adjacent `task_status.json`, runs the 1B/3B/8B TokMem/TapMem checkpoints on the tools51-100 10-call test split, then writes per-trial prediction JSONL files plus `summary.json` and `summary.md` under `compositional/rebuttal/results/`.
- `analyze_generation_eoc_boundaries.py`: reruns free generation for non-adaptation `tokmem_eoc` and `tokmem_eoc_logit_bias` checkpoints, keeps raw generated token IDs so EOC tokens remain visible, and summarizes generated EOC boundary precision/recall/F1 under `compositional/rebuttal/results/eoc_boundary_accuracy/`. For Llama-3B it uses the final checkpoint suite referenced by `completed_trials_summary.md`.

Run the 4-call checkpoint to 10-call evaluation through the script wrapper:

```bash
bash scripts/compositional/run_train4_checkpoint_eval_10calls.sh
```

For parallel runs, pass `--models llama1b`, `--models llama3b`, or `--models llama8b` in separate shells with the same `--output-dir`, then run once with `--summarize-only` after all prediction JSONL files exist.

Run generated EOC boundary analysis through:

```bash
bash scripts/compositional/analyze_generation_eoc_boundaries.sh
```
