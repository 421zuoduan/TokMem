# TapMem Failure-Case Analysis and Reviewer Response

## Reviewer comment

> The ablations are useful, but the paper could provide more analysis of failure cases. For example, when TapMem still selects the wrong next procedure, is the error mainly due to incorrect termination, poor routing, or ambiguity in the query?

## Ready-to-use response

Thank you for this suggestion. We added a first-divergence analysis of TapMem on the 4-call APIGen test set. We first condition on executions in which the initial procedure is correct, and then locate the first subsequent divergence from the gold procedure sequence. Assigning only the first divergence to each execution avoids counting downstream consequences of the same error multiple times.

We separate the residual failures into three groups. **Boundary/stopping** includes a missing or malformed `<EOC>`, premature sequence stopping, and generation of an extra procedure after the terminal boundary. **Routing** includes an incorrect next procedure after a valid boundary; a gold-prefix probe further distinguishes an intrinsic routing error, which remains incorrect under the gold prefix, from error propagation, which recovers under the gold prefix. **Query/schema ambiguity** is assigned when both the gold and predicted tools plausibly satisfy the same query clause according to their schemas. We manually inspected all 73 unique query-position and gold/predicted-tool combinations exposed by the wrong-transition audit; 20 unique cases were marked ambiguous.

| Model | Checkpoint executions | Residual transition failures | Boundary / stopping | Routing | Query / schema ambiguity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-1B | 1,500 | 84 | 41.7% (35/84) | 44.0% (37/84) | 14.3% (12/84) |
| Llama-3B | 500 | 38 | 63.2% (24/38) | 21.1% (8/38) | 15.8% (6/38) |
| Llama-8B | 500 | 17 | 5.9% (1/17) | 29.4% (5/17) | 64.7% (11/17) |
| Pooled, descriptive | 2,500 | 139 | 43.2% (60/139) | 36.0% (50/139) | 20.9% (29/139) |

The answer is therefore model-dependent rather than a single dominant cause. Boundary/stopping is the largest pooled category, but only narrowly: it accounts for 43.2% of the residual transition failures, compared with 36.0% for routing and 20.9% for ambiguity. For Llama-1B, routing and boundary/stopping are comparable. Llama-3B is dominated by boundary/stopping errors, especially malformed boundaries. In contrast, Llama-8B has only 17 residual transition failures, most of which involve overlapping schemas such as `is_palindrome` versus `is_valid_palindrome`, or `integrate` versus `trapezoidal_integration`. The 8B percentage should therefore be read descriptively because of its small denominator and the manual ambiguity judgment.

The auxiliary diagnostics support this interpretation:

| Model | EOC F1 | Exact EOC count | Malformed boundary rate | Gold-prefix next-procedure top-1 | Gold-prefix terminal-stop accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-1B | 97.2% | 90.4% (1,356/1,500) | 2.1% (31/1,500) | 97.1% (2,308/2,376) | 99.3% (1,489/1,500) |
| Llama-3B | 91.4% | 82.6% (413/500) | 7.4% (37/500) | 98.1% (777/792) | 99.8% (499/500) |
| Llama-8B | 98.5% | 93.6% (468/500) | 6.0% (30/500) | 97.2% (770/792) | 100.0% (500/500) |

EOC F1 measures boundary detection over all generated calls, while exact EOC count is a stricter sample-level measure. Gold-prefix next-procedure top-1 evaluates routing after supplying the correct preceding calls and boundary, thereby removing errors inherited from free generation. Terminal-stop accuracy tests whether the model stops, rather than selecting another procedure, after a complete gold sequence. The high overall boundary and oracle-routing scores are compatible with the conditional failure table: most transitions are handled correctly, while the table characterizes the comparatively small set on which a later transition actually fails.

This analysis also reveals a limitation of exact-match evaluation. For example, a request to integrate with the trapezoidal rule can be satisfied by either the generic `integrate` tool, whose `method` argument supports `trapezoid`, or the dedicated `trapezoidal_integration` tool. Similarly, the two palindrome schemas overlap substantially. We will report these cases separately instead of treating every deviation from the single annotated tool as a routing failure. We will add the aggregate analysis, definitions, and representative examples to the revised manuscript and supplementary material.

## Supporting end-to-end analysis

The first-divergence analysis above directly answers the reviewer's question. We retain the following six-way analysis to cover the full output failure surface, including errors that are not wrong-next-procedure decisions.

| Model | Correct | Argument-only | Length | Order-only | Initial-involved routing | Later-only routing | Later mismatch, first correct | Tool-sequence exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-1B | 48.00% | 22.87% | 10.07% | 0.60% | 17.20% | 1.27% | 6.06% | 70.87% |
| Llama-3B | 34.40% | 15.20% | 19.00% | 1.20% | 28.40% | 1.80% | 8.49% | 49.60% |
| Llama-8B | 44.80% | 12.00% | 10.60% | 4.60% | 25.80% | 2.20% | 5.18% | 56.80% |

The first six columns are mutually exclusive sample-level outcomes:

- **Correct:** the complete serialized call sequence is exact.
- **Argument-only:** the procedure sequence is correct, but at least one argument or call serialization is wrong.
- **Length:** predicted and gold procedure counts differ.
- **Order-only:** procedure counts and multisets match, but their order differs.
- **Initial-involved routing:** a same-length routing error includes the first procedure.
- **Later-only routing:** the first procedure is correct, but a later procedure is wrong in a same-length sequence.

`Later mismatch, first correct` is a separate position-level metric over gold positions 2 through \(J\), conditioned on a correct first procedure. Missing predictions count as mismatches. It is more sensitive to next-procedure failures than the mutually exclusive `Later-only` category, because `Later-only` excludes cases already assigned to Length or Order-only.

Length is not labeled as a termination metric. A length mismatch can be caused by a missing boundary, early stopping, an extra procedure, a duplicated procedure, routing to a neighboring clause, or malformed serialization. The first-divergence analysis is required to turn this observable symptom into a causal category.

## Fine-grained first-divergence counts

| First cause | Llama-1B, n=84 | Llama-3B, n=38 | Llama-8B, n=17 | Pooled, n=139 |
| --- | ---: | ---: | ---: | ---: |
| Missing EOC / boundary | 10 (11.9%) | 5 (13.2%) | 0 | 15 (10.8%) |
| Premature stop after EOC | 10 (11.9%) | 0 | 0 | 10 (7.2%) |
| Malformed boundary before next procedure | 7 (8.3%) | 9 (23.7%) | 0 | 16 (11.5%) |
| Malformed boundary before extra procedure | 2 (2.4%) | 5 (13.2%) | 0 | 7 (5.0%) |
| Over-generation after terminal EOC | 6 (7.1%) | 5 (13.2%) | 1 (5.9%) | 12 (8.6%) |
| Generated-context error propagation | 14 (16.7%) | 3 (7.9%) | 2 (11.8%) | 19 (13.7%) |
| Intrinsic routing under gold prefix | 23 (27.4%) | 5 (13.2%) | 3 (17.6%) | 31 (22.3%) |
| Query / schema ambiguity | 12 (14.3%) | 6 (15.8%) | 11 (64.7%) | 29 (20.9%) |

## Representative cases

| Type | Gold procedure sequence | TapMem prediction | Diagnostic interpretation |
| --- | --- | --- | --- |
| Missing boundary | `place_safeway_order -> generate_random_string` | `place_safeway_order` | The next procedure is absent because the first call is not followed by a usable boundary. |
| Premature stop | `integrate -> get_pokemon_move_info` | `integrate` | A valid first call is completed, but decoding stops before the next procedure. |
| Error propagation | `flatten_list -> integrate -> find_max_subarray_sum` | `flatten_list -> find_max_subarray_sum` | The free-running sequence skips `integrate`, while the gold-prefix probe recovers it. |
| Intrinsic routing | `generate_password -> find_longest_word -> cagr` | `generate_password -> generate_password -> find_longest_word -> cagr` | The next-tool error remains under the gold-prefix probe, indicating a routing decision error rather than a missing boundary. |
| Schema ambiguity | `get_products_in_category -> integrate -> can_attend_all_meetings` | `get_products_in_category -> trapezoidal_integration -> can_attend_all_meetings` | Both integration tools explicitly support the requested trapezoidal rule; the single gold label is not uniquely justified. |
| Terminal over-generation | `polygon_area_shoelace -> binary_addition` | `polygon_area_shoelace -> binary_addition -> binary_addition` | The correct sequence is produced, followed by an unnecessary repeated procedure. |

## What constitutes a complete failure case?

A complete analysis should cover more than the three examples in the reviewer comment. We recommend two complementary views.

### 1. Outcome view

This view asks what is observably wrong in the final output:

1. query or annotation ambiguity;
2. missing or wrong initial procedure;
3. wrong procedure count;
4. wrong procedure order;
5. wrong later procedure;
6. correct procedure sequence but incorrect arguments;
7. malformed or unparsable call serialization;
8. fully correct output.

These categories describe evaluation outcomes and are useful for reporting the overall error surface.

### 2. First-divergence causal view

This view asks why the first error occurred:

1. **input/schema ambiguity:** multiple available procedures plausibly satisfy the same clause;
2. **initial routing:** the first procedure is wrong or missing;
3. **boundary detection:** `<EOC>` is missing, premature, duplicated, or malformed;
4. **terminal stopping:** generation stops early or continues after the final procedure;
5. **intrinsic next-procedure routing:** the gold next procedure is not selected even under a gold prefix and boundary;
6. **generated-context propagation:** routing is correct under the gold prefix but fails after an earlier generated call;
7. **argument realization:** the route is correct, but arguments, values, types, or serialization are incorrect;
8. **ordering/dependency:** the right set of procedures is selected in the wrong order.

Only the first divergence should be counted in the causal table. For example, one missing `<EOC>` may cause every subsequent position to shift; counting all shifted positions as routing errors would inflate the routing category.

Each released failure record should contain:

- sample ID, model, checkpoint, decoding configuration, and query;
- gold and predicted procedure sequences and complete calls;
- raw memory-token, `<EOC>`, and EOS trace;
- first divergent position and assigned cause;
- gold-prefix routing result, gold rank, and top-1 margin where available;
- argument exactness and parse status;
- ambiguity label with the two relevant schemas and a short rationale.

This record is complete enough to reproduce both the outcome taxonomy and the first-divergence diagnosis.

## Metric-to-question mapping

| Metric | Meaning | Question answered |
| --- | --- | --- |
| Correct / tool-sequence exact | End-to-end call correctness / exact procedure order | How often does the complete workflow succeed? |
| Argument-only error | Correct route but incorrect arguments or serialization | Are failures caused by call realization rather than selection? |
| Length error | Predicted and gold procedure counts differ | Where are control-flow symptoms present? This is an alarm, not a termination diagnosis. |
| Order-only error | Correct procedure multiset in an incorrect order | Does dependency or clause ordering fail? |
| Initial-involved routing | The first procedure participates in the route mismatch | Is the failure already present before a transition occurs? |
| Later mismatch, first correct | Position error after a correct first procedure | How often does next-procedure selection still fail? |
| EOC precision / recall / F1 | Accuracy of generated procedure boundaries | Does TapMem detect procedure completion reliably overall? |
| Exact EOC count | Sample has exactly the expected number of boundaries | Is boundary control correct for the entire sequence? |
| Malformed boundary rate | Sample contains an unusable call-boundary transition | Are syntax or parsing failures causing apparent routing errors? |
| Gold-prefix next-procedure top-1 | Next procedure under the correct history and boundary | Does routing itself fail after termination and context are controlled? |
| Gold-prefix terminal-stop accuracy | EOS after the complete gold sequence | Does final stopping fail independently of free-running errors? |
| Ambiguity adjudication | Both schemas plausibly satisfy the same query clause | Is exact-match scoring penalizing a semantically valid alternative? |
| First-divergence cause share | One mutually exclusive root cause per failed execution | Which cause mainly explains residual failures without cascade double counting? |

## Do the experiments need to be rerun?

No retraining or full benchmark rerun is needed for this response. The saved free-generation outputs already support the six-way outcome analysis, later-step mismatch, EOC statistics, representative examples, and the schema-ambiguity audit. Separating intrinsic routing from generated-context propagation requires only a teacher-forced gold-prefix diagnostic on the frozen checkpoints; it does not change model parameters or regenerate the benchmark outputs. The tables above use the completed diagnostic records.

## Statistical and reporting notes

- The test set contains 500 queries. Llama-1B is pooled over three distinct checkpoint executions, giving 1,500 query-checkpoint executions over the same 500 queries.
- The archived Llama-3B TapMem checkpoints are byte-identical to one another, as are the archived Llama-8B TapMem checkpoints. We therefore use one canonical checkpoint for each of these model sizes instead of presenting duplicate files as independent trials.
- The pooled row is descriptive and weights checkpoint executions, not 2,500 independent queries.
- Ambiguity was adjudicated by inspecting the query clause and both schemas. This is a manual, single-pass audit. The ambiguity percentages should be reported as descriptive; a revision that treats ambiguity as a headline benchmark statistic should add a second annotator and agreement measure.
- The 8B residual denominator is only 17, so its cause distribution is useful for identifying schema overlap but not for a broad scaling claim.

## Suggested manuscript additions

1. Add the grouped first-divergence table to the main failure-analysis subsection.
2. Add metric definitions, the fine-grained causes, and representative traces to the appendix.
3. State explicitly that Length is an observable sequence-count mismatch, not a synonym for termination error.
4. Report overlapping-schema cases separately from strict routing errors, or accept a set of valid tools when the annotation permits multiple equivalent procedures.
5. Release the per-case fields listed above so the failure attribution is auditable.
