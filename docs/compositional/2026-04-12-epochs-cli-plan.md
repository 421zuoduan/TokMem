# Epochs CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `--epochs` CLI flag to `compositional/main_sequential.py` for single-round no-adaptation runs.

**Architecture:** Keep `--training_rounds` as the source of tool-range selection, and add a narrow override path that replaces the parsed epoch count only when exactly one round is requested. Reject multi-round use to avoid colliding with adaptation semantics.

**Tech Stack:** Python, argparse, existing compositional sequential training entrypoint

---

### Task 1: Add `--epochs` parsing and single-round override

**Files:**
- Modify: `/data/ruochen/tokmem/compositional/main_sequential.py`

- [ ] **Step 1: Reproduce the missing flag behavior**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --training_rounds 51-100:1 --epochs 3
```

Expected: argument parsing fails because `--epochs` does not exist yet.

- [ ] **Step 2: Add the new CLI argument**

Add an argparse option:

```python
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override the epoch count for a single no-adaptation training round",
)
```

- [ ] **Step 3: Validate the new argument**

After parsing args, add:

```python
if args.epochs is not None and args.epochs <= 0:
    parser.error("--epochs must be positive")
```

- [ ] **Step 4: Apply the override only for single-round runs**

After `rounds = parse_training_rounds(...)`, add:

```python
if args.epochs is not None:
    if len(rounds) != 1:
        parser.error("--epochs only supports single-round no-adaptation runs")
    rounds[0]["epochs"] = args.epochs
```

- [ ] **Step 5: Verify the new flag is visible**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --help
```

Expected: help output includes `--epochs`.

- [ ] **Step 6: Verify single-round override works**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --training_rounds 51-100:1 --epochs 3 --help
```

Expected: parsing succeeds and the flag is accepted.

- [ ] **Step 7: Verify multi-round rejection**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --training_rounds 1-50:1,51-100:3 --epochs 3
```

Expected: parser exits with `--epochs only supports single-round no-adaptation runs`.
