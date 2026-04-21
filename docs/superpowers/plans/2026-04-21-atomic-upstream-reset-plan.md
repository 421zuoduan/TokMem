# Atomic Upstream Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive the current local `atomic` TokMem implementation, replace the maintained `atomic` surface with upstream TokMem atomic code, then add fixed-split support and logit-bias support on top of that upstream baseline.

**Architecture:** Treat the current local `atomic` tree as legacy and move it into archive locations. Copy the upstream `atomic` files into the maintained `atomic/` surface, then make minimal additive changes to that upstream baseline: one fixed-split loading path and one first-step logit-bias method. Keep LoRA baseline in the top-level `atomic/` directory. Keep launcher compatibility centered on fixed-split shell entrypoints.

**Tech Stack:** Python, Bash, Hugging Face Transformers, PyTorch, git-based upstream snapshot import

---

### Task 1: Archive Current Local Atomic Surface

**Files:**
- Create: `atomic/archive/`
- Create: `scripts/atomic/archive/`
- Modify: `atomic/`
- Modify: `scripts/atomic/`

- [ ] **Step 1: Capture the pre-move inventory**

Run:

```bash
find atomic -maxdepth 2 -type f | sort > /tmp/atomic_before.txt
find scripts/atomic -maxdepth 3 -type f | sort > /tmp/scripts_atomic_before.txt
```

Expected: Two inventory files exist and list the current local atomic files.

- [ ] **Step 2: Move superseded atomic TokMem files into archive**

Move the current locally evolved TokMem entrypoints and method-heavy training surface into archive while keeping LoRA baseline available at the top level.

Run shape:

```bash
mkdir -p atomic/archive/current_local scripts/atomic/archive/current_local
# move superseded tokmem entrypoints and legacy scripts here
```

Expected: Files such as the current local TokMem `main_in_domain*`, legacy TokMem shell entrypoints, and old method-heavy launchers live under archive paths.

- [ ] **Step 3: Verify LoRA baseline remains top-level**

Run:

```bash
test -f atomic/main_lora_baseline.py
test -f atomic/main_lora_baseline.sh
```

Expected: Both commands exit successfully.

- [ ] **Step 4: Verify archive move completed**

Run:

```bash
find atomic/archive -maxdepth 3 -type f | sort | sed -n '1,120p'
find scripts/atomic/archive -maxdepth 4 -type f | sort | sed -n '1,160p'
```

Expected: The moved local TokMem code and legacy scripts are visible under archive locations.

### Task 2: Import Upstream Atomic Code Into Maintained Surface

**Files:**
- Modify: `atomic/README.md`
- Create or Replace: `atomic/main_in_domain.py`
- Create or Replace: `atomic/main_tokmem.sh`
- Create or Replace: `atomic/task_model.py`
- Create or Replace: `atomic/task_training.py`
- Create or Replace: `atomic/task_dataset.py`
- Create or Replace: `atomic/natural_instructions_eval.py`
- Create or Replace: `atomic/test_sbert_retriever.py`
- Create or Replace: `atomic/analyze_task_similarity.py`

- [ ] **Step 1: Clone the upstream TokMem repository into a temporary path**

Run:

```bash
rm -rf /tmp/tokmem_upstream
git clone --depth 1 https://github.com/MANGA-UOFA/TokMem /tmp/tokmem_upstream
```

Expected: `/tmp/tokmem_upstream/atomic/` exists with the upstream baseline files.

- [ ] **Step 2: Copy the upstream atomic baseline into the maintained atomic surface**

Run shape:

```bash
# copy upstream atomic baseline files into ./atomic/
```

Expected: The top-level maintained atomic files now match the upstream baseline before local additions.

- [ ] **Step 3: Keep local-only maintained utilities that still belong in the new surface**

Retain or reintroduce only the utilities still needed for maintained runs, such as local run-layout helpers or fixed-split support scaffolding.

Expected: The maintained atomic surface is upstream-first, with only necessary local support files kept.

- [ ] **Step 4: Verify maintained files now align with upstream shape**

Run:

```bash
diff -u /tmp/tokmem_upstream/atomic/main_in_domain.py atomic/main_in_domain.py | sed -n '1,120p'
```

Expected: Either no diff or a small diff limited to intentional local support changes.

### Task 3: Add Fixed-Split Support To The Upstream-Based Main Path

**Files:**
- Modify: `atomic/main_in_domain.py`
- Modify: `atomic/task_dataset.py`
- Modify: `atomic/README.md`
- Modify: `atomic/main_tokmem.sh`
- Create or Replace: `atomic/main_tokmem_fixed_split.sh`

- [ ] **Step 1: Write a failing fixed-split smoke check**

Run:

```bash
python atomic/main_in_domain.py --help | rg "split_cache_path"
```

Expected: This initially fails because the upstream baseline does not expose fixed-split support.

- [ ] **Step 2: Add `--split_cache_path` to the maintained main entrypoint**

Implementation target:

```python
parser.add_argument(
    "--split_cache_path",
    type=str,
    default=None,
    help="Optional path to a cached fixed split. When provided, training and evaluation load this split directly.",
)
```

Expected: The main entrypoint accepts a fixed-split cache path.

- [ ] **Step 3: Add fixed-split loading logic**

Implementation target:

```python
if args.split_cache_path:
    split_payload = torch.load(args.split_cache_path, map_location="cpu")
    train_data = split_payload["train_data"]
    val_data = split_payload["val_data"]
    test_data = split_payload["test_data"]
    task_names = split_payload["task_names"]
else:
    train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(...)
```

Expected: The maintained `main_in_domain.py` supports both upstream runtime sampling and fixed cached splits.

- [ ] **Step 4: Re-run the fixed-split smoke check**

Run:

```bash
python atomic/main_in_domain.py --help | rg "split_cache_path"
```

Expected: The flag is present.

### Task 4: Add First-Step Logit Bias On Top Of The Upstream Atomic Model

**Files:**
- Modify: `atomic/task_model.py`
- Modify: `atomic/task_training.py`
- Modify: `atomic/main_in_domain.py`
- Modify: `atomic/README.md`

- [ ] **Step 1: Write a failing interface check for logit-bias flags**

Run:

```bash
python atomic/main_in_domain.py --help | rg "use_logit_bias|logit_bias_loss_weight|logit_bias_network|logit_bias_scale"
```

Expected: This initially fails because the upstream baseline does not expose logit-bias flags.

- [ ] **Step 2: Add logit-bias config flags to the main entrypoint**

Implementation target:

```python
parser.add_argument("--use_logit_bias", action="store_true")
parser.add_argument("--logit_bias_loss_weight", type=float, default=0.1)
parser.add_argument("--logit_bias_network", type=str, default="linear", choices=["linear", "mlp"])
parser.add_argument("--logit_bias_scale", type=float, default=1.0)
```

Expected: The maintained main path exposes the new method knobs.

- [ ] **Step 3: Extend `TaskCallingModel` with a first-step logit-bias head**

Implementation target:

```python
def build_logit_bias_head(hidden_size, num_tasks, network_type):
    ...

self.use_logit_bias = use_logit_bias
self.logit_bias_head = build_logit_bias_head(...)
```

Expected: The model can learn task-prior logits over reserved task tokens at the first routing step.

- [ ] **Step 4: Add detached auxiliary loss in training**

Implementation target:

```python
if use_logit_bias:
    logit_bias_loss = ...
    loss = loss + logit_bias_loss_weight * logit_bias_loss
```

Expected: Train and validation summaries include `avg_logit_bias_loss`.

- [ ] **Step 5: Apply first-step bias during decoding**

Implementation target:

```python
if self.use_logit_bias:
    logits = self._apply_logit_bias_to_first_step(...)
```

Expected: The first task-token decision is softly reweighted while non-task logits remain unchanged.

- [ ] **Step 6: Re-run the interface check**

Run:

```bash
python atomic/main_in_domain.py --help | rg "use_logit_bias|logit_bias_loss_weight|logit_bias_network|logit_bias_scale"
```

Expected: All four flags are present.

### Task 5: Update Launchers And Docs For The New Maintained Surface

**Files:**
- Modify: `README.md`
- Modify: `atomic/README.md`
- Modify: `atomic/main_tokmem.sh`
- Modify: `atomic/main_tokmem_fixed_split.sh`
- Modify: maintained scripts under `scripts/atomic/`

- [ ] **Step 1: Point maintained launchers at the new main entrypoint**

Run shape:

```bash
rg -n "main_in_domain.py|main_in_domain_fixed_split.py" atomic scripts/atomic
```

Expected: Maintained launchers target the upstream-based `atomic/main_in_domain.py` path.

- [ ] **Step 2: Rewrite `atomic/README.md` for the new maintained surface**

Required content:

```markdown
- maintained TokMem path: upstream atomic baseline + fixed split + logit bias
- maintained baseline: LoRA
- archived local atomic code lives under archive paths
```

- [ ] **Step 3: Rewrite the root `README.md` atomic summary**

Required content:

```markdown
Atomic now uses an upstream-based maintained path with fixed-split support and logit bias.
Older local atomic method families are archived.
```

- [ ] **Step 4: Verify docs and launchers reference the new surface**

Run:

```bash
rg -n "logit bias|fixed split|archive" README.md atomic/README.md atomic scripts/atomic
```

Expected: The maintained entrypoints and docs align with the new design.

### Task 6: Run Focused Verification

**Files:**
- Verify: `atomic/main_in_domain.py`
- Verify: `atomic/task_model.py`
- Verify: `atomic/task_training.py`
- Verify: `atomic/main_tokmem_fixed_split.sh`

- [ ] **Step 1: Verify the Python files parse**

Run:

```bash
python -m py_compile atomic/main_in_domain.py atomic/task_model.py atomic/task_training.py atomic/task_dataset.py
```

Expected: Exit code 0.

- [ ] **Step 2: Verify the fixed-split launcher still resolves**

Run:

```bash
bash -n atomic/main_tokmem_fixed_split.sh
```

Expected: Exit code 0.

- [ ] **Step 3: Verify help output exposes the new maintained knobs**

Run:

```bash
python atomic/main_in_domain.py --help | rg "split_cache_path|use_logit_bias|logit_bias_loss_weight|logit_bias_network|logit_bias_scale"
```

Expected: All maintained flags appear.

- [ ] **Step 4: Run a reduced fixed-split dry path if a local split cache is available**

Run shape:

```bash
python atomic/main_in_domain.py \
  --split_cache_path atomic/cached_splits/.../tokmem_atomic_fixed_split_maxlen1024.pt \
  --num_epochs 0
```

Expected: The script loads the split path and reaches the maintained execution path without flag or wiring errors.
