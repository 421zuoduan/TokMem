# Repository Guidelines

## Project Structure & Module Organization

This repository is organized by experiment track rather than by a shared package:

- `atomic/`: atomic task-memory experiments on Natural Instructions. Main entrypoints are `main_in_domain.py`, `main_tokmem.sh`, and `main_lora_baseline.sh`.
- `compositional/`: sequential tool-calling experiments on XLAM/APIGen. Main entrypoints are `main_sequential.py`, `run_n_rounds_main.sh`, `run_n_rounds_lora.sh`, and `icl_baseline.sh`.
- `memorization/`: embedding-capacity and memorization ablations, including GSM8K and long-text stress tests.
- `paper.pdf`: the paper corresponding to this repository. Treat it as the baseline method and results reference; future changes are expected to build on or modify the approaches described there.
- `requirements.txt`: minimal Python dependency list.

Tests are limited. The main standalone test-like script is `atomic/test_sbert_retriever.py`.

## Current Experiment Focus

Unless the user explicitly says otherwise, treat the current working scope as:

- prioritize `compositional/` work, scripts, experiments, and code changes by default
- start from `compositional/` entrypoints and datasets unless the task is clearly about another track
- keep `atomic/` workflows and documentation available; they still apply when the user explicitly asks for `atomic/` work
- the main metrics of interest are `routing acc` (`Task Prediction Accuracy`) and `Rouge-L`

## Result Analysis Workflow

For archived `atomic` runs, use `scripts/analyze_atomic_run.sh` as the default entrypoint for low-routing-task analysis.

- treat the script as a general archived-run analysis tool, not something tied only to baseline runs
- prefer the shell wrapper over calling the Python module directly:

```bash
bash scripts/analyze_atomic_run.sh results/<run_folder>
```

- the script assumes the standard archived evaluation format and does not expose a separate prompt-mode switch
- the main output focus is tasks whose `routing acc` falls below a threshold, plus the top confused target tasks derived from final evaluation results
- the script also attaches task definitions, task-form summaries, and representative misrouted examples for each high-confusion target
- if archived confusion-memory summaries are present in `train_results.json`, keep them in the report as run-level context; otherwise rely on result-derived confusion only

Useful optional flags:

- `--threshold`: routing-accuracy cutoff, default `0.9`
- `--top-k`: number of confused target tasks to report per failing task, default `3`
- `--max-examples`: number of representative misrouted examples to keep per confused target, default `3`
- `--tasks-dir`: override the NI task directory if needed

The script writes these files into the analyzed run directory:

- `routing_failure_analysis.json`
- `routing_failure_analysis.md`

When changing analysis behavior:

- do not create repository test files for this workflow
- if validation is needed, run the script directly in the sandbox and inspect its outputs
- do not leave `.codex` files or other Codex-specific scratch artifacts in the repository

## Environment Requirement

Use the `tokmem` conda environment for experiments in this repository.

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem
```

Do not switch to `.venv` or another ad hoc environment for experiment runs unless the user explicitly asks for it.

## Build, Test, and Development Commands

Install dependencies in the active environment when needed:

```bash
pip install -r requirements.txt
```

Run the main experiments from each subdirectory:

```bash
cd atomic && bash main_tokmem.sh
cd compositional && bash run_n_rounds_main.sh
cd memorization && bash run_memorization_comparison.sh
```

Run the retriever evaluation script directly:

```bash
cd atomic && python test_sbert_retriever.py
```

Most experiments require a CUDA-capable GPU and model access through Hugging Face.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep changes consistent with the surrounding file style. Prefer descriptive snake_case for functions, variables, and script names. Keep experiment-specific logic inside its own directory rather than introducing cross-directory coupling unless it is clearly reusable.

For Bash scripts, keep them flat and direct:

- prefer a single straightforward execution path over helper functions and layered control flow
- avoid environment-variable-based parameterization unless it is genuinely necessary
- do not introduce batches of config variables for model hyperparameters; write fixed paper or experiment settings directly in the command when possible
- only define variables for long or reusable paths such as the repository root or dataset directory
- avoid unnecessary input validation, help text, wrappers, and mode switches when the script is meant for one fixed experiment
- prefer one explicit command over generic reusable shell abstractions

There is no enforced formatter or linter in this repository. If you use one locally, avoid large style-only diffs.

## Testing Guidelines

There is no formal coverage target. For contributions, run the narrowest relevant script and document what you validated. Examples:

- model/data changes in `atomic/`: run `python test_sbert_retriever.py` or a reduced `main_in_domain.py` configuration
- sequential training changes in `compositional/`: run a small-round local config
- memorization changes: run a single reduced experiment instead of the full sweep
- do not create dedicated test-only files just for validation unless the user explicitly asks for a file to be added

Name new validation scripts as `test_*.py` when they are meant to be run directly.

## Experiment Archival

Successful experiment runs should be archived under `results/` in a dedicated run folder so future comparisons do not depend on files still living under `atomic/logs/` or other working directories.

- create one subdirectory per archived run under `results/`
- keep the archived layout consistent with existing runs in `results/`
- include the main artifacts for that run when available:
  - structured training log
  - structured evaluation log
  - stdout log
  - GPU monitor log
  - best checkpoint / saved weights
  - exact script snapshot used for the run
  - split cache or other run-specific cached data used by the experiment
- add a `run_summary.md` file inside the run folder
- write `run_summary.md` in concise Chinese and include at least:
  - what experiment was run
  - the main parameter settings
  - the main results
  - what was materially different from earlier archived runs
  - any important caveats such as reused split cache, missing watchdog log, or nonzero outer exit code
- update `results/README.md` after each newly archived successful run
- keep `results/README.md` in concise Chinese and use it as a chronological index of archived experiments, focusing on the main differences between runs rather than duplicating full logs
- if a run did not produce a file that normally exists, note that explicitly in `run_summary.md` instead of silently omitting the fact

## Commit & Pull Request Guidelines

Recent history uses short, direct messages such as `Update README.md`, `Update task_dataset.py`, and `docs: update ...`. Follow that style: concise, imperative, and scoped.

When pushing this repository to the local bare remote, use:

```bash
git push server main
```

PRs should include:

- a short summary of the change
- affected experiment track(s)
- exact commands used for validation
- any dataset/model prerequisites
- result snippets or logs when behavior changes materially
