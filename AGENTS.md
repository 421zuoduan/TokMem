# Repository Guidelines

## Project Structure & Module Organization

This repository is organized by experiment track rather than by a shared package:

- `atomic/`: atomic task-memory experiments on Natural Instructions. Main entrypoints are `main_in_domain.py`, `main_tokmem.sh`, and `main_lora_baseline.sh`.
- `compositional/`: sequential tool-calling experiments on XLAM/APIGen. Main entrypoints are `main_sequential.py`, `run_n_rounds_main.sh`, `run_n_rounds_lora.sh`, and `icl_baseline.sh`.
- `memorization/`: embedding-capacity and memorization ablations, including GSM8K and long-text stress tests.
- `paper.pdf`: the paper corresponding to this repository. Treat it as the baseline method and results reference; future changes are expected to build on or modify the approaches described there.
- `requirements.txt`: minimal Python dependency list.

Tests are limited. The main standalone test-like script is `atomic/test_sbert_retriever.py`.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
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

PRs should include:

- a short summary of the change
- affected experiment track(s)
- exact commands used for validation
- any dataset/model prerequisites
- result snippets or logs when behavior changes materially
