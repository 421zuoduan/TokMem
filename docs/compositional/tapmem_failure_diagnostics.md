# TapMem failure diagnostics

This workflow diagnoses residual procedure-selection failures without retraining
the paper checkpoints. It complements the existing six-class output analysis in
`docs/compositional/error_type_transition_analysis.md`.

## Scope

The diagnostic uses the APIGen tools 51-100 4-call test split and the paper
TapMem checkpoints. It produces two aligned views:

1. **Free generation** records the actual tool tokens, EOC tokens, parsed calls,
   boundary-local base/TCRA/fused rankings, and end-to-end outcomes.
2. **Oracle-boundary probes** teacher-force the gold procedure/call prefix and
   EOC, then score the next decision. These probes isolate routing under a
   correct boundary and history from errors propagated by free generation.

No model parameters are updated.

## Failure taxonomy

The end-to-end six-class outcome remains the broad coverage view:

- correct;
- argument-only error;
- length error;
- order-only error;
- initial-involved routing error;
- later-only routing error.

For causal inspection, the first residual procedure failure is assigned one
mutually exclusive cause:

- initial procedure missing or initial routing error;
- missing EOC / boundary failure;
- premature sequence stop after a valid EOC;
- malformed boundary before the next procedure;
- query/schema ambiguity, based on an explicit audit record;
- intrinsic routing error when the oracle-prefix probe is also wrong;
- generated-context error propagation when the oracle-prefix probe recovers;
- over-generation after the terminal EOC;
- malformed boundary before an extra procedure.

Argument realization and parse failures remain visible in the end-to-end
taxonomy even when the complete procedure sequence is correct.

## Metrics

- **EOC F1** is micro precision/recall/F1 over valid generated procedure
  closures. It measures whether explicit termination signals are emitted.
- **Exact EOC count** is the fraction of samples whose generated EOC count
  equals the number of gold procedures.
- **Malformed boundary rate** is the fraction of samples with an orphan,
  repeated, or structurally misplaced EOC/tool boundary.
- **Oracle transition full-vocabulary top-1 accuracy** requires the actual
  fused next token at a gold transition boundary to equal the gold memory
  token. It tests routing while preserving competition with the full
  vocabulary.
- **Oracle transition candidate top-1 accuracy** ranks only memory-token
  candidates after TCRA fusion. It isolates tool discrimination.
- **Terminal stop accuracy** checks whether the full-vocabulary top-1 after the
  final gold EOC is the native response-ending token.
- **First-failure attribution** conditions on a correct initial procedure and
  assigns each residual sample to exactly one cause, preventing later
  consequences from being counted as independent root causes.

Wilson 95% confidence intervals are saved for reported binomial proportions.

## Commands

Activate the project environment:

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
```

Run diagnostics for a selected checkpoint:

```bash
CUDA_VISIBLE_DEVICES=GPU_ID python \
  compositional/utils/run_tapmem_failure_diagnostics.py \
  --models llama1b \
  --trial-id 1 \
  --output-dir compositional/rebuttal/results/tapmem_failure_diagnostics
```

Run the offline analysis:

```bash
python compositional/utils/analyze_tapmem_failure_diagnostics.py \
  --trial-ids llama1b:1+2+3,llama3b:3,llama8b:1 \
  --ambiguity-annotations \
    compositional/rebuttal/results/tapmem_failure_diagnostics/analysis/ambiguity_annotations.jsonl
```

Run focused tests:

```bash
python compositional/utils/test_analyze_tapmem_failure_diagnostics.py
```

## Artifacts

The default directory is
`compositional/rebuttal/results/tapmem_failure_diagnostics/`:

```text
free_generation/<model>/trial<trial>.jsonl
oracle_probes/<model>/trial<trial>.jsonl
manifests/<model>/trial<trial>.json
analysis/summary.json
analysis/summary.md
analysis/ambiguity_audit_template.jsonl
analysis/ambiguity_annotations.jsonl
analysis/first_failures/<model>.jsonl
analysis/transition_events/<model>.jsonl
```

For checkpoint groups whose archived trials are byte-identical, report one
canonical checkpoint and sample-level confidence intervals rather than treating
copied artifacts as independent training runs.
