# Compositional TensorBoard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional TensorBoard logging to the existing `compositional/main_sequential.py` training path so training losses can be inspected over time.

**Architecture:** `main_sequential.py` owns CLI/configuration and the run-scoped TensorBoard directory, while `training.py` owns step-level and epoch-level scalar logging inside the existing training loop. The feature remains opt-in through a single CLI flag.

**Tech Stack:** Python, PyTorch, TensorBoard, existing compositional training pipeline

---

## File Structure

- Modify: `requirements.txt`
  Add the missing TensorBoard package dependency.

- Modify: `compositional/main_sequential.py`
  Add the CLI switch, create the writer under the run directory, pass logging context into the training function, and include the TensorBoard artifact path in run metadata.

- Modify: `compositional/training.py`
  Accept the writer and logging offsets, emit step-level and epoch-level scalars, and return enough information for round-level summaries.

## Task 1: Add the CLI Surface and Run Artifact Wiring

**Files:**
- Modify: `compositional/main_sequential.py`

- [ ] Confirm current CLI has no TensorBoard flag
- [ ] Add `--tensorboard` as a default-off flag
- [ ] Create `run_dir/tensorboard/` only when the flag is enabled
- [ ] Initialize `SummaryWriter` lazily with a clear error if `tensorboard` is missing
- [ ] Record the TensorBoard directory in run metadata and summaries

## Task 2: Add Training-Loop Scalar Logging

**Files:**
- Modify: `compositional/training.py`

- [ ] Extend `train_native_function_calling_model(...)` to accept an optional writer plus step and epoch offsets
- [ ] Emit per-step loss, lr, position-count, and round scalars
- [ ] Accumulate epoch metrics and emit per-epoch averages
- [ ] Return updated global step / epoch counters for multi-round training

## Task 3: Validate the Integration

**Files:**
- Modify: `requirements.txt`

- [ ] Run `python compositional/main_sequential.py --help` in the `tokmem` environment and confirm the new flag is exposed
- [ ] Run `python -m py_compile compositional/main_sequential.py compositional/training.py`
- [ ] Inspect the diff to ensure only the intended code paths changed
