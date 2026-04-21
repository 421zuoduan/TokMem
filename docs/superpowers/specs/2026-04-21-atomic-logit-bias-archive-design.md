# Atomic Logit Bias And Archive Design

## Context

The current `atomic/` track mixes several concerns in the maintained surface:

- the original official `main_in_domain.py` workflow from the upstream TokMem repository
- later local `runtime_split` and `fixed_split` entrypoints
- multiple method families that are no longer expected to be used
- many launcher scripts that still point at older methods

The next round of `atomic` work should narrow the maintained surface to one TokMem line built from the official upstream `main_in_domain.py`, extended with `logit bias`, while preserving the practical `fixed_split` workflow used by existing experiments.

The user also wants old method families physically moved out of the main surface. At the same time, existing fixed-split scripts should keep working as much as possible, and the LoRA baseline should stay in the top-level `atomic/` directory.

## Goals

- Make `atomic/` have a clear maintained path centered on official-style `main_in_domain` plus `logit bias`
- Preserve the `fixed_split` workflow where later runs consume an already prepared split cache
- Keep existing fixed-split launcher usage compatible where practical
- Physically move older method implementations and launcher scripts into archive locations
- Keep the LoRA baseline in the main `atomic/` surface

## Non-Goals

- Reproducing every old method variant as part of the maintained surface
- Cleaning up every historical result artifact under `atomic/runs/` or `results/`
- Refactoring `atomic/` into a package-level architecture beyond what is needed for the new maintained path

## Approaches Considered

### Approach 1: Replace the current main files in place

Use the official upstream `main_in_domain.py` as the new content of the existing maintained entrypoints and move older logic aside.

Pros:

- Fewer top-level files
- Existing filenames stay familiar

Tradeoffs:

- Harder to see which file is the new maintained implementation and which behavior is compatibility glue
- Higher chance of conflating the official-derived path with the later local path

### Approach 2: Add one new maintained implementation and keep compatibility wrappers

Create a new maintained entrypoint derived from the official upstream file, then keep top-level compatibility wrappers for fixed-split usage and existing scripts.

Pros:

- Clean separation between the new implementation and compatibility entrypoints
- Easier to reason about archive boundaries
- Existing fixed-split scripts can continue to call a familiar filename

Tradeoffs:

- One more top-level maintained file
- Requires wrapper logic and clear README guidance

### Approach 3: Create a new subdirectory for the maintained atomic method

Move the new maintained TokMem implementation under a new `atomic/logit_bias/` subtree and keep thin wrappers in `atomic/`.

Pros:

- Strong isolation
- Very explicit code ownership

Tradeoffs:

- Larger path migration
- More script and import churn than the problem needs

## Recommended Approach

Use Approach 2.

This keeps the maintained path explicit, gives old methods a clean physical archive destination, and preserves the `fixed_split` execution style without forcing broad script churn.

## File And Directory Design

### Main `atomic/` surface after the change

These stay in the main `atomic/` directory:

- `main_in_domain_logit_bias.py`
- `main_in_domain_fixed_split.py`
- `main_lora_baseline.py`
- `main_lora_baseline.sh`
- `main_tokmem_fixed_split.sh`
- the shared modules required by the maintained path, including `task_model.py`, `task_training.py`, `task_dataset.py`, `natural_instructions_eval.py`, and `run_layout.py`
- `README.md`

### Archive destinations

Older TokMem method entrypoints and method-specific scripts move to archive locations:

- `atomic/archive/`
- `scripts/atomic/archive/`

The archive area will hold superseded TokMem entrypoints and legacy launcher scripts that target older method families. This is a physical move, not only a documentation label.

## Maintained Entry Points

### `main_in_domain_logit_bias.py`

This is the new maintained TokMem entrypoint for `atomic/`.

It will be created from the official upstream `MANGA-UOFA/TokMem` version of `atomic/main_in_domain.py`, then extended locally with:

- `logit bias` support
- `split_cache_path` support for fixed cached splits
- current local run-layout writing where needed for reproducible outputs

Behavior:

- if `--split_cache_path` is provided, load the cached split and run on that split
- otherwise, keep the official-style runtime sampling path

This makes one file support both official-like runtime behavior and the maintained fixed-split workflow.

### `main_in_domain_fixed_split.py`

This remains as a top-level compatibility entrypoint.

Its purpose is compatibility, not independent method ownership. It will accept the existing fixed-split calling pattern used by current scripts and forward into the maintained `main_in_domain_logit_bias.py` path.

This preserves the operational habit of:

```bash
python -u main_in_domain_fixed_split.py --split_cache_path ...
```

### `main_tokmem_fixed_split.sh`

This stays in the main `atomic/` directory and continues to target the fixed-split maintained path.

It may be simplified so it clearly launches the new maintained implementation or the compatibility fixed-split entrypoint.

## Logit Bias Method Shape For Atomic

The maintained `atomic` version of `logit bias` applies at the first task-routing decision site.

### Training

- collect the hidden state immediately before the first generated task token
- detach that hidden state for the auxiliary head
- predict the gold task id over the reserved task-token set
- add `logit_bias_loss_weight * CE` to the main LM loss

### Decoding

- at the first generation step, compute task-prior logits from the same hidden state
- convert to task log-probabilities
- center the bias against a uniform prior over task tokens
- scale by `logit_bias_scale`
- add the bias only to reserved task-token columns

This matches the routing structure of `atomic`, where the key decision is the first task token.

## Model And Training Changes

### `task_model.py`

The maintained shared model module will absorb the new atomic `logit bias` support:

- add `use_logit_bias`
- add `logit_bias_network`
- add `logit_bias_scale`
- add a `logit_bias_head`
- extend trainable-parameter collection so the head trains with the task embeddings
- extend save/load to persist the head and related config
- apply first-step bias during task-token generation

The save format should remain backward-tolerant where possible. Older task-token checkpoints may load without a bias head. New checkpoints should save enough metadata to restore the new behavior cleanly.

### `task_training.py`

The maintained training module will:

- gather first-step routing examples
- compute detached `logit_bias_loss`
- add the weighted auxiliary loss in both train and validation passes
- report `avg_logit_bias_loss` in returned summaries

Older method losses that are now archived in product direction do not need to remain a focus of the maintained documentation. Compatibility handling is acceptable where existing scripts still pass those flags.

## Compatibility Strategy

### Fixed split

`fixed_split` remains a maintained feature.

The new maintained implementation will support:

- loading precomputed split caches
- writing the same kinds of run metadata needed for later analysis
- continued use of existing fixed-split shell scripts with minimal or zero changes

### Existing script parameters

Many older scripts still pass flags for superseded methods.

Compatibility policy:

- preserve existing fixed-split path usage
- preserve old CLI parameters when they are cheap to tolerate
- method flags that no longer matter may be accepted and ignored rather than causing immediate failures

This keeps launcher compatibility high while the maintained surface becomes much narrower.

### LoRA baseline

The LoRA baseline stays in the top-level `atomic/` directory and remains documented as the maintained baseline comparison path.

## Documentation Plan

### `atomic/README.md`

Rewrite `atomic/README.md` so it clearly distinguishes:

- maintained paths
- archived paths

The maintained section should say:

- the main TokMem path is the official-style atomic entrypoint extended with `logit bias`
- fixed split is maintained
- LoRA baseline is maintained

The archived section should say:

- older TokMem method families and older launcher collections live under archive locations

### Root `README.md`

Update the repository root summary so the `atomic` track describes the new maintained surface and no longer presents older atomic method families as current work.

## Error Handling And Testing

### Error handling

- fail clearly when `split_cache_path` is provided and the file is missing or malformed
- fail clearly when checkpoint config and current model shape are fundamentally incompatible
- keep older checkpoint loading tolerant when the only missing part is the new bias head

### Validation

Run the narrowest practical checks:

1. a reduced fixed-split run that exercises the new maintained path
2. a checkpoint save/load round-trip including the bias head
3. a compatibility invocation through `main_in_domain_fixed_split.py`

No new dedicated test-only files are needed.

## Implementation Sequence

1. Create archive directories and move superseded atomic TokMem entrypoints and legacy scripts there
2. Add `main_in_domain_logit_bias.py` from the official upstream `main_in_domain.py`
3. Extend that maintained entrypoint with `split_cache_path` and `logit bias` configuration
4. Update `task_model.py` and `task_training.py` for first-step atomic `logit bias`
5. Turn `main_in_domain_fixed_split.py` into a compatibility entrypoint over the maintained implementation
6. Keep `main_lora_baseline.py` and `main_lora_baseline.sh` in place
7. Update `atomic/README.md` and root `README.md`

## Risks And Mitigations

### Risk: current scripts depend on old method flags

Mitigation:

- preserve old flag names where cheap
- accept ignorable superseded flags in compatibility entrypoints

### Risk: checkpoint format drift

Mitigation:

- make bias-head loading backward tolerant
- keep checkpoint metadata explicit

### Risk: archive move breaks hard-coded paths

Mitigation:

- keep top-level compatibility entrypoints for fixed-split workflows
- only move scripts whose paths are clearly legacy or method-specific

## Success Criteria

- `atomic/` has one clear maintained TokMem path derived from the official upstream atomic entrypoint
- `logit bias` exists on that maintained path
- `fixed_split` remains supported
- LoRA baseline remains in the top-level `atomic/` surface
- old TokMem method families are physically moved into archive locations
- current fixed-split launcher usage continues to work with minimal disruption
