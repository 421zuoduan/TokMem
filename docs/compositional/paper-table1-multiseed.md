# TapMem Table 1 Multi-Seed Runs

`scripts/compositional/run_paper_table1_seeds40_41.sh` runs the complete
compositional Table 1 matrix for seeds 40 and 41:

- models: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B
- methods: ICL, RAG, TokMem, TapMem, Fine-Tuning, TokMem with adaptation,
  TapMem with adaptation
- scope: tools 51-100, 5000 training samples, 500 test samples, 2-to-4 calls
- all trainable runs use `max_length=512`, including Llama-3B TokMem

The launcher also runs one Llama-3B TokMem seed-42 calibration at
`max_length=512`. The published row used 1024, so this extra task is required
to keep that row's error bar seed-only.

The launcher keeps one FIFO task queue and polls only the GPU indexes supplied
in `GPU_IDS`. Each scheduler iteration starts at most one task: it finds one
free GPU, removes one task from the front of the queue, starts it, and then
begins the next scheduler iteration. Already started experiments may run in
parallel on different GPUs, but launch decisions are made serially.

A GPU is considered free when `nvidia-smi` reports
`memory.used <= 2048 MiB`. When no GPU is free, the scheduler waits and polls
again. When any running task exits, its GPU can receive the next queued task
without waiting for tasks on other GPUs.

The separate 1B/3B inference queue instead checks
`memory.free >= GPU_MIN_FREE_MIB` so it can use otherwise occupied GPUs with a
controlled amount of remaining memory.

The defaults can be changed with:

```bash
GPU_POLL_SECONDS=30
GPU_MEMORY_LIMIT_MIB=2048
GPU_MIN_FREE_MIB=16000
```

`GPU_IDS` should contain a nonempty set of unique, canonical decimal GPU
indexes such as `0,2,5`, without a trailing comma. UUIDs, negative values,
leading-zero aliases such as `00`, and values containing spaces are rejected.
Every selected GPU must be large enough to run every selected configuration.

Before launching, each task obtains the shared per-GPU `flock` under
`/tmp/tokmem_gpu_locks/gpu_<id>.lock`, then checks GPU memory again. The lock is
held until the task exits, preventing other TokMem launchers that use the same
lock convention from starting on that GPU at the same time. External programs
that do not use these locks are still handled by the memory check.

## Dry run

```bash
GPU_IDS=0,1,2,3,4,5,6 \
bash scripts/compositional/run_paper_table1_seeds40_41.sh --dry-run
```

Dry-run mode prints all 43 commands and does not create a result directory or
start a model process. Its displayed GPU assignment is round-robin for command
inspection only; formal runs use dynamic assignment.

## Formal run

```bash
GPU_IDS=0,1,2,3,4,5,6 \
bash scripts/compositional/run_paper_table1_seeds40_41.sh
```

The 1B/3B ICL and RAG tasks can run as a separate eight-task inference queue
on partially occupied GPUs:

```bash
SUITE_NAME=<shared-suite-name> \
GPU_IDS=0,1 \
GPU_MIN_FREE_MIB=16000 \
bash scripts/compositional/run_paper_table1_seeds40_41.sh --inference-1b3b
```

This queue uses smaller evaluation batches to coexist with other GPU jobs:
1B ICL/RAG use `8/32`, and 3B ICL/RAG use `4/16`. Decoding remains greedy
(`do_sample=False`). The other 35 tasks use the paper settings and run with:

```bash
SUITE_NAME=<shared-suite-name> \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/compositional/run_paper_table1_seeds40_41.sh --remaining
```

The reduced-batch inference mode is an opportunistic resource-sharing mode,
not a paper-batch-aligned run. In the completed 2026-07-24 suite, its raw
predictions differed from the full-batch main-table runs. Do not combine those
eight reduced-batch results with the published seed-42 values for a strict
seed-only error bar; rerun them with the paper batches (`64/256` for 1B
ICL/RAG and `32/192` for 3B ICL/RAG) before the final paper summary.

Both queues share the same 43-row manifest and task directories. The queue
that observes all 43 `SUCCESS` markers writes the final error-bar report.

Set `SUITE_NAME` to resume a previously created suite. Resume is rejected if
the launcher, Git commit, input/code hashes, or saved task command changed:

```bash
SUITE_NAME=paper_table1_seeds40_41_YYYYMMDD_HHMMSS \
GPU_IDS=0,1,2,3,4,5,6 \
bash scripts/compositional/run_paper_table1_seeds40_41.sh
```

Successful task directories contain the command, stdout log, normal run
artifacts, and a `SUCCESS` marker. The suite stores data/reference hashes, Git
revision and dirty-state records, and a script snapshot.

If one task fails, the scheduler still attempts the rest of the queue and exits
nonzero after all scheduled tasks finish. Rerun with the same `SUITE_NAME` to
reuse successful tasks and retry incomplete or failed tasks.

Keep the launcher process alive until the queue finishes. This deliberately
simple scheduler does not manage worker process groups or force-stop already
running experiments when the parent launcher is terminated.

## Error bars

After all 43 tasks succeed, the launcher calls
`scripts/compositional/summarize_paper_table1_seeds.py`. It combines the new
seed 40/41 metrics with the published seed 42 values in
`scripts/compositional/table1_seed42_reference.json`.

The Llama-3B TokMem row instead uses the fresh seed-42 calibration at
`max_length=512`.

For each Table 1 cell, the output reports the equally weighted three-seed mean
and sample standard deviation:

```text
table1_error_bars.json
table1_error_bars.md
```

The reference JSON is transcribed from `paper_tapmem.pdf` Table 1 because the
older completed-trials JSON and the final PDF do not contain the same TapMem
rows.

The published seed-42 cells are rounded Table 1 values and some are themselves
averages of repeated runs that reused seed 42. They are combined at equal
weight with one seed-40 and one seed-41 run. Report the result as a
published-reference error bar, not as a fully balanced three-seed study.
