# Repository Guidelines

## Project Structure & Module Organization

This repository is organized by experiment track rather than by a shared package:

- `atomic/`: atomic task-memory experiments on Natural Instructions. Main entrypoints are `main_in_domain.py`, `main_tokmem.sh`, and `main_lora_baseline.sh`.
- `compositional/`: sequential tool-calling experiments on XLAM/APIGen. Main entrypoints are `main_sequential.py`, `run_n_rounds_main.sh`, `run_n_rounds_lora.sh`, and `icl_baseline.sh`.
- `memorization/`: embedding-capacity and memorization ablations, including GSM8K and long-text stress tests.
- `paper.pdf`: the paper corresponding to this repository. Treat it as the baseline method and results reference; future changes are expected to build on or modify the approaches described there.
- `requirements.txt`: minimal Python dependency list.

Execution scripts belong under `scripts/`. When adding or updating runnable experiment shell entrypoints, place them in `scripts/` rather than scattering them across track directories.

Tests are limited. The main standalone test-like script is `atomic/test_sbert_retriever.py`.

## Current Experiment Focus

Unless the user explicitly says otherwise, treat the current working scope as:

- prioritize `compositional/` work, scripts, experiments, and code changes by default
- start from `compositional/` entrypoints and datasets unless the task is clearly about another track
- for `compositional/`, treat the maintained method surface as the `use_eoc`, `use_js_trunc`, and `use_logit_bias` family
- frame tool use around explicit `eoc` boundary decisions, optional JS-based tool-only truncation, and optional detached logit-bias reweighting over tool tokens
- prefer experiments, analyses, and docs that stay inside the maintained `eoc/js_trunc/logit_bias` path unless the user explicitly asks for archived method families
- keep `atomic/` workflows and documentation available; they still apply when the user explicitly asks for `atomic/` work
- the main metrics of interest are `routing acc` (`Task Prediction Accuracy`) and `Rouge-L`

More specifically, when working on `compositional/` by default:

- favor the maintained boundary-time formulation where `eoc` defines the decision sites, `js_trunc` decides tool-only truncation at those sites, and `logit_bias` softly reweights tool-token logits at those same sites
- when referring to the `baseline` in `compositional/`, interpret it as the TokMem setting without the adaptation stage; do not treat `baseline` as a non-TokMem method unless the user explicitly says so
- treat gate, `eoc loss`, `tool loss`, and `toolmix` as archived method families that may appear in older runs or git history
- when adding experiments, analyses, or documentation, make the maintained `eoc/js_trunc/logit_bias` assumptions explicit

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
cd compositional && python main_sequential.py --training_rounds 51-100:3 --use_eoc --use_js_trunc
cd compositional && bash ../scripts/compositional/llama_1b/tokmem_eoc_js_trunc_logit_bias_llama_1b.sh
cd memorization && bash run_memorization_comparison.sh
```

For `compositional/`, treat `main_sequential.py` and the maintained `scripts/compositional/llama_1b/tokmem_*.sh` launchers as the default entrypoints.

Treat these compositional shell entrypoints as legacy:

- `run_n_rounds_main.sh`
- `run_n_rounds_lora.sh`
- `icl_baseline.sh`

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

## Error Handling Philosophy

Prefer fail-fast behavior over defensive fallback logic when working in this repository.

- keep control flow direct and let real errors surface
- raise exceptions when assumptions are violated or required data is missing
- add fallback or silent recovery only when the experiment contract explicitly requires it
- avoid wrapping code just to keep execution going after an invalid state
- prefer fixing the underlying bug over masking it with extra branches

## Testing Guidelines

There is no formal coverage target. For contributions, run the narrowest relevant script and document what you validated. Examples:

- model/data changes in `atomic/`: run `python test_sbert_retriever.py` or a reduced `main_in_domain.py` configuration
- sequential training changes in `compositional/`: run a small-round local config
- memorization changes: run a single reduced experiment instead of the full sweep
- do not create dedicated test-only files just for validation unless the user explicitly asks for a file to be added

Name new validation scripts as `test_*.py` when they are meant to be run directly.

## Documentation Update Rules

When code changes affect how people run, interpret, analyze, or compare experiments, update the relevant documentation in the same change. Do not leave behavior changes only in code when existing docs would become misleading.

Documentation updates are required when changes affect any of the following:

- training or evaluation entrypoints, launcher scripts, CLI flags, default values, or required environment setup
- run artifact layout, checkpoint naming, log files, saved JSON outputs, or the meaning of files under `compositional/runs/`, `atomic/logs/`, or `results/`
- dataset sources, filtering rules, split construction, sample-count assumptions, generated file names, or cached-data behavior
- metric definitions, evaluation logic, analysis workflow, or the interpretation of reported numbers such as `routing acc`, `Task Prediction Accuracy`, `Rouge-L`, exact match, tool accuracy, or parse-error-related metrics
- model behavior that changes the research meaning of an experiment, such as new gating logic, decoding constraints, loss terms, adaptation behavior, memory-token semantics, or tool-token generation rules
- analysis or visualization tooling assumptions, especially when scripts rely on specific run files, config fields, checkpoint structure, or result JSON schema

Documentation updates are usually not required for:

- pure refactors that do not change external behavior or experiment interpretation
- local cleanup, renaming, comments, or internal helper extraction with no effect on usage or outputs
- bug fixes whose observable behavior exactly matches what the docs already say

Choose documentation emphasis based on the type of change:

- if the change is primarily a model-method change, prioritize documents that explain the research idea and evaluation meaning
  - update method/design docs first
  - make the changed hypothesis, supervision, decoding rule, or metric interpretation explicit
  - explain what is different from the previous method and how comparisons should be read
- if the change is primarily a code-implementation or workflow change, prioritize documents that explain how to run and inspect the system
  - update run-layout docs, README sections, launcher usage, artifact descriptions, and analysis-tool assumptions first
  - make file-level output changes, CLI changes, and operational caveats explicit
- if a change affects both method and implementation, update both layers instead of collapsing everything into one brief note

When deciding where to document:

- use `docs/compositional/` or `docs/atomic/` for experiment-track-specific design, method, data, and workflow notes
- use nearby `README.md` files for stable usage instructions, maintained entrypoints, and artifact-layout summaries
- use `results/README.md` and per-run `run_summary.md` only for archived experiment records, not as the primary place to document new code behavior

If a code change removes, renames, or stops producing an artifact, the docs should say that explicitly rather than silently dropping the file from examples.

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

# --- talk-normal BEGIN ---
<!-- talk-normal 0.6.2 -->

Be direct and informative. No filler, no fluff, but give enough to be useful.

Your single hardest constraint: prefer direct positive claims. Do not use negation-based contrastive phrasing in any language or position — neither "reject then correct" (不是X，而是Y) nor "correct then reject" (X，而不是Y). If you catch yourself writing a sentence where a negative adverb sets up or follows a positive claim, restructure and state only the positive.

Examples:
BAD:  真正的创新者不是"有创意的人"，而是五种特质同时拉满的人
GOOD: 真正的创新者是五种特质同时拉满的人

BAD:  真正的创新者是五种特质同时拉满的人，而不是单纯"聪明"的人
GOOD: 真正的创新者是五种特质同时拉满的人

BAD:  这更像创始人筛选框架，不是交易信号
GOOD: 这是一个创始人筛选框架

BAD:  It's not about intelligence, it's about taste
GOOD: Taste is what matters

Rules:
- Lead with the answer, then add context only if it genuinely helps
- Do not use negation-based contrastive phrasing in any position. This covers any sentence structure where a negative adverb rejects an alternative to set up or append to a positive claim: in any order ("reject then correct" or "correct then reject"), chained ("不是A，不是B，而是C"), symmetric ("适合X，不适合Y"), or with or without an explicit "but / 而 / but rather" conjunction. Just state the positive claim directly. If a genuine distinction needs both sides, name them as parallel positive clauses. Narrow exception: technical statements about necessary or sufficient conditions in logic, math, or formal proofs.
- End with a concrete recommendation or next step when relevant. Do not use summary-stamp closings — any closing phrase or label that announces "here comes my one-line summary" before delivering it. This covers "In conclusion", "In summary", "Hope this helps", "Feel free to ask", "一句话总结", "一句话落地", "一句话讲", "一句话概括", "一句话说", "一句话收尾", "总结一下", "简而言之", "概括来说", "总而言之", and any structural variant like "一句话X：" or "X一下：" that labels a summary before delivering it. If you have a final punchy claim, just state it as the last sentence without a summary label.
- Kill all filler: "I'd be happy to", "Great question", "It's worth noting", "Certainly", "Of course", "Let me break this down", "首先我们需要", "值得注意的是", "综上所述", "让我们一起来看看"
- Never restate the question
- Yes/no questions: answer first, one sentence of reasoning
- Comparisons: give your recommendation with brief reasoning, not a balanced essay
- Code: give the code + usage example if non-trivial. No "Certainly! Here is..."
- Explanations: 3-5 sentences max for conceptual questions. Cover the essence, not every subtopic. If the user wants more, they will ask.
- Use structure (numbered steps, bullets) only when the content has natural sequential or parallel structure. Do not use bullets as decoration.
- Match depth to complexity. Simple question = short answer. Complex question = structured but still tight.
- Do not end with hypothetical follow-up offers or conditional next-step menus. This includes "If you want, I can also...", "如果你愿意，我还可以...", "If you tell me...", "如果你告诉我...", "如果你说X，我就Y", "我下一步可以...", "If you'd like, my next step could be...". Do not stage menus where the user has to say a magic phrase to unlock the next action. Answer what was asked, give the recommendation, stop. If a real next action is needed, just take it or name it directly without the conditional wrapper.
- Do not restate the same point in "plain language" or "in human terms" after already explaining it. Say it once clearly. No "翻成人话", "in other words", "简单来说" rewording blocks.
- When listing pros/cons or comparing options: max 3-4 points per side, pick the most important ones
# --- talk-normal END ---
