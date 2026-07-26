# Repository Guidelines

## Repository Map

- `compositional/`: sequential tool-calling experiments; main entrypoint is `main_sequential.py`.
- `atomic/`: Natural Instructions task-memory experiments; standalone check is `test_sbert_retriever.py`.
- `scripts/`: launcher scripts for experiment suites and model/data preparation.
- `docs/`: experiment design, workflow, and result notes.
- `results/`: archived successful runs with Chinese summaries.
- `paper.pdf`: baseline method and results reference.

## Method Terminology

- By default, `TokMem` means the plain method with no adaptation, no EOC, and no logit bias, as described in `paper_tokmem.pdf`.
- `TapMem` means the method with no adaptation, with EOC, and with logit bias, as described in `paper_tapmem.pdf`.
- The TCRA module in the paper is the logit-bias mechanism (`logit bias`) used by TapMem.

## Environment

Use the `tokmem` conda environment for experiment runs:

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
```

Install dependencies inside that environment when needed:

```bash
pip install -r requirements.txt
```

Most experiments require a CUDA-capable GPU and Hugging Face model access.

## Common Commands

```bash
cd compositional && python main_sequential.py --training_rounds 51-100:3 --use_eoc --use_logit_bias
cd compositional && bash ../scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh
cd atomic && python test_sbert_retriever.py
```

For `compositional/`, prefer `main_sequential.py` and maintained `scripts/compositional/llama_1b/tokmem_*.sh` launchers. Treat `run_n_rounds_main.sh`, `run_n_rounds_lora.sh`, and `icl_baseline.sh` as legacy entrypoints.

## Code Style

- Python uses 4-space indentation and descriptive `snake_case`.
- Keep experiment-specific logic inside its track directory.
- Match surrounding style; large style-only diffs are out of scope.
- Bash scripts should stay flat and explicit: fixed experiment settings, direct commands, and variables mainly for long reusable paths.

## Validation

- Run the narrowest relevant command and report it.
- For `compositional/` training changes, use a reduced local run.
- For `atomic/` model/data changes, use `python test_sbert_retriever.py` or a reduced `main_in_domain.py` run.
- Add `test_*.py` files only when they are meant to be kept and run directly.

## Checkpoints

- For new training runs, checkpoint files should contain only parameters that are
  updated during training. Do not save an additional copy of a frozen LLM
  backbone.
- Preserve compatibility with existing checkpoints that already include frozen
  LLM-backbone parameters. Load those checkpoints in full rather than filtering
  out the frozen-backbone entries.

## Documentation

Update docs in the same change when behavior affects:

- entrypoints, launcher scripts, CLI flags, defaults, or environment setup
- batch size in paper-suite launchers or experiment defaults
- artifact layout, checkpoints, logs, saved JSON, or run directory meaning
- dataset sources, filters, splits, sample counts, generated files, or caches
- metric definitions and result interpretation
- decoding, gating, adaptation, memory-token semantics, or tool-token generation
- analysis or visualization assumptions

Use `docs/compositional/` or `docs/atomic/` for track-specific notes, nearby `README.md` files for stable usage, and `results/README.md` plus per-run `run_summary.md` for archived experiment records.

Rebuttal 实验文档的口径：

- `results/rebuttal/rebuttal-needs-exps.md` 记录论文出分后，针对评审意见补充开展的实验。
- `results/rebuttal/summary.md` 记录论文出分前自行设计和完成的实验。

## Experiment Archival

Archive successful runs under `results/<run_name>/` with the available logs, checkpoints, script snapshot, caches, and a concise Chinese `run_summary.md`.

Each `run_summary.md` should include:

- experiment purpose
- main parameters
- main results
- material differences from earlier archived runs
- caveats such as reused split caches, missing logs, or nonzero outer exit code

Update `results/README.md` after archiving a successful run.

## Git

- Use short, scoped commit messages such as `Update README.md` or `docs: update ...`.
- Push to the local bare remote with:

```bash
git push server main
```

PRs should include the summary, affected track, validation commands, dataset/model prerequisites, and result snippets when behavior changes materially.

## Response Style

- Be direct and informative.
- Lead with the answer, then add only useful context.
- Keep conceptual explanations to 3-5 sentences by default.
- Use structure when it matches the content.
- End with a concrete recommendation or next action when relevant.
