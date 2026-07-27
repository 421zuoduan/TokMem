#!/usr/bin/env python3
"""Summarize TapMem failure diagnostics and export ambiguity audit cases."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
UTILS_DIR = COMPOSITIONAL_DIR / "utils"
for import_dir in (COMPOSITIONAL_DIR, UTILS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from analyze_error_type_transitions import label_record  # noqa: E402


DEFAULT_INPUT_DIR = (
    COMPOSITIONAL_DIR / "rebuttal" / "results" / "tapmem_failure_diagnostics"
)
DEFAULT_TOOL_DESCRIPTIONS = (
    COMPOSITIONAL_DIR / "data" / "tool_descriptions_tools51-100.json"
)
MODEL_LABELS = {
    "llama1b": "Llama-1B",
    "llama3b": "Llama-3B",
    "llama8b": "Llama-8B",
}
FULL_OUTCOME_KEYS = (
    "correct",
    "argument_only_error",
    "length_error",
    "order_only_error",
    "initial_involved_routing_error",
    "later_only_routing_error",
)
CAUSE_LABELS = {
    "initial_missing": "Initial procedure missing",
    "initial_routing": "Initial routing error",
    "missing_eoc_boundary": "Missing EOC / boundary failure",
    "premature_sequence_stop": "Premature sequence stop after EOC",
    "malformed_boundary_before_tool": "Malformed boundary before next procedure",
    "query_schema_ambiguity": "Query/schema ambiguity",
    "intrinsic_routing": "Intrinsic routing error under oracle prefix",
    "context_propagation": "Generated-context error propagation",
    "over_generation_after_terminal_eoc": "Over-generation after terminal EOC",
    "malformed_boundary_before_extra": "Malformed boundary before extra procedure",
    "procedure_sequence_correct": "Procedure sequence correct",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument(
        "--models",
        default="llama1b,llama3b,llama8b",
    )
    parser.add_argument("--trial-ids", default="llama1b:1,llama3b:3,llama8b:1")
    parser.add_argument(
        "--tool-descriptions",
        default=str(DEFAULT_TOOL_DESCRIPTIONS),
    )
    parser.add_argument("--ambiguity-annotations", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_trial_ids(value):
    result = {}
    for item in split_csv(value):
        model, trial = item.split(":", 1)
        result[model] = [int(value) for value in trial.split("+")]
    return result


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson_interval(numerator, denominator, z=1.959963984540054):
    if denominator <= 0:
        return [None, None]
    proportion = numerator / denominator
    z2 = z * z
    denominator_adjustment = 1 + z2 / denominator
    center = (proportion + z2 / (2 * denominator)) / denominator_adjustment
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / denominator
            + z2 / (4 * denominator * denominator)
        )
        / denominator_adjustment
    )
    return [center - half_width, center + half_width]


def rate_payload(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": safe_rate(numerator, denominator),
        "wilson_95ci": wilson_interval(numerator, denominator),
    }


def ambiguity_key(index, position, gold_tool, predicted_tool):
    return f"{int(index)}|{int(position)}|{gold_tool}|{predicted_tool}"


def load_annotations(path):
    if path is None:
        return {}
    annotations = {}
    for record in load_jsonl(Path(path)):
        label = record.get("label")
        if label not in {"ambiguous", "unambiguous", "uncertain"}:
            raise ValueError(f"Invalid ambiguity label: {label!r}")
        annotations[record["key"]] = record
    return annotations


def oracle_index(records):
    return {
        (
            int(record.get("trial", 1)),
            int(record["index"]),
            int(record["boundary_ordinal"]),
        ): record
        for record in records
    }


def oracle_probe(oracle, record, position):
    return oracle.get(
        (
            int(record.get("trial", 1)),
            int(record["index"]),
            int(position),
        )
    )


def tool_is_immediately_after_eoc(record, predicted_position):
    tool_positions = record.get("tool_positions") or []
    raw_ids = record.get("raw_generated_token_ids") or []
    if predicted_position >= len(tool_positions):
        return False
    raw_position = int(tool_positions[predicted_position])
    return (
        raw_position > 0
        and raw_ids[raw_position - 1] == int(record["eoc_token_id"])
    )


def last_predicted_span_closed(record):
    tool_positions = record.get("tool_positions") or []
    eoc_positions = record.get("eoc_positions") or []
    if not tool_positions:
        return False
    return any(
        int(eoc_position) > int(tool_positions[-1])
        for eoc_position in eoc_positions
    )


def ambiguity_label_for_event(
    record,
    position,
    predicted_tool,
    annotations,
):
    gold_tool = record["expected_tools"][position]
    key = ambiguity_key(record["index"], position, gold_tool, predicted_tool)
    annotation = annotations.get(key)
    return annotation.get("label") if annotation else None


def classify_first_procedure_failure(record, oracle, annotations):
    gold = list(record.get("expected_tools") or [])
    predicted = list(record.get("predicted_tools") or [])
    if not gold:
        return {
            "cause": "procedure_sequence_correct",
            "position": None,
            "gold_tool": None,
            "predicted_tool": None,
        }
    if not predicted:
        return {
            "cause": "initial_missing",
            "position": 0,
            "gold_tool": gold[0],
            "predicted_tool": None,
        }
    if predicted[0] != gold[0]:
        return {
            "cause": "initial_routing",
            "position": 0,
            "gold_tool": gold[0],
            "predicted_tool": predicted[0],
        }

    for position in range(1, len(gold)):
        if position >= len(predicted):
            cause = (
                "premature_sequence_stop"
                if last_predicted_span_closed(record)
                else "missing_eoc_boundary"
            )
            return {
                "cause": cause,
                "position": position,
                "gold_tool": gold[position],
                "predicted_tool": None,
            }
        if predicted[position] == gold[position]:
            continue
        if not tool_is_immediately_after_eoc(record, position):
            cause = "malformed_boundary_before_tool"
        else:
            label = ambiguity_label_for_event(
                record,
                position,
                predicted[position],
                annotations,
            )
            if label == "ambiguous":
                cause = "query_schema_ambiguity"
            else:
                probe = oracle_probe(oracle, record, position)
                cause = (
                    "context_propagation"
                    if probe and probe.get("selected_matches_expected")
                    else "intrinsic_routing"
                )
        return {
            "cause": cause,
            "position": position,
            "gold_tool": gold[position],
            "predicted_tool": predicted[position],
        }

    if len(predicted) > len(gold):
        extra_position = len(gold)
        cause = (
            "over_generation_after_terminal_eoc"
            if tool_is_immediately_after_eoc(record, extra_position)
            else "malformed_boundary_before_extra"
        )
        return {
            "cause": cause,
            "position": extra_position,
            "gold_tool": None,
            "predicted_tool": predicted[extra_position],
        }
    return {
        "cause": "procedure_sequence_correct",
        "position": None,
        "gold_tool": None,
        "predicted_tool": None,
    }


def transition_events(record, oracle, annotations):
    events = []
    gold = list(record.get("expected_tools") or [])
    predicted = list(record.get("predicted_tools") or [])
    first_correct = bool(predicted) and bool(gold) and predicted[0] == gold[0]
    if not first_correct:
        return events
    for position in range(1, len(gold)):
        if position >= len(predicted):
            cause = (
                "premature_sequence_stop"
                if last_predicted_span_closed(record)
                else "missing_eoc_boundary"
            )
            predicted_tool = None
        elif predicted[position] == gold[position]:
            continue
        else:
            predicted_tool = predicted[position]
            if not tool_is_immediately_after_eoc(record, position):
                cause = "malformed_boundary_before_tool"
            else:
                label = ambiguity_label_for_event(
                    record,
                    position,
                    predicted_tool,
                    annotations,
                )
                if label == "ambiguous":
                    cause = "query_schema_ambiguity"
                else:
                    probe = oracle_probe(oracle, record, position)
                    cause = (
                        "context_propagation"
                        if probe and probe.get("selected_matches_expected")
                        else "intrinsic_routing"
                    )
        events.append(
            {
                "index": int(record["index"]),
                "trial": int(record.get("trial", 1)),
                "position": position,
                "gold_tool": gold[position],
                "predicted_tool": predicted_tool,
                "cause": cause,
                "cause_label": CAUSE_LABELS[cause],
            }
        )
    for position in range(len(gold), len(predicted)):
        cause = (
            "over_generation_after_terminal_eoc"
            if tool_is_immediately_after_eoc(record, position)
            else "malformed_boundary_before_extra"
        )
        events.append(
            {
                "index": int(record["index"]),
                "trial": int(record.get("trial", 1)),
                "position": position,
                "gold_tool": None,
                "predicted_tool": predicted[position],
                "cause": cause,
                "cause_label": CAUSE_LABELS[cause],
            }
        )
    return events


def summarize_eoc(records):
    tp = sum(int(record.get("tp", 0)) for record in records)
    fp = sum(int(record.get("fp", 0)) for record in records)
    fn = sum(int(record.get("fn", 0)) for record in records)
    precision = safe_rate(tp, tp + fp) or 0.0
    recall = safe_rate(tp, tp + fn) or 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_count": rate_payload(
            sum(bool(record.get("eoc_count_exact")) for record in records),
            len(records),
        ),
        "malformed_sample": rate_payload(
            sum(bool(record.get("malformed_boundary")) for record in records),
            len(records),
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize_oracle(records):
    phases = {}
    for phase in ("initial", "transition", "terminal"):
        phase_records = [
            record for record in records if record["boundary_type"] == phase
        ]
        selected_correct = sum(
            bool(record.get("selected_matches_expected"))
            for record in phase_records
        )
        phase_summary = {
            "cases": len(phase_records),
            "full_vocab_top1_accuracy": rate_payload(
                selected_correct,
                len(phase_records),
            ),
        }
        if phase != "terminal":
            phase_summary.update(
                {
                    "base_candidate_top1_accuracy": rate_payload(
                        sum(record["base"]["gold_rank"] == 1 for record in phase_records),
                        len(phase_records),
                    ),
                    "fused_candidate_top1_accuracy": rate_payload(
                        sum(record["fused"]["gold_rank"] == 1 for record in phase_records),
                        len(phase_records),
                    ),
                    "fused_candidate_top5_recall": rate_payload(
                        sum(
                            record["fused"]["gold_rank"] is not None
                            and record["fused"]["gold_rank"] <= 5
                            for record in phase_records
                        ),
                        len(phase_records),
                    ),
                    "tcra_wrong_to_correct": sum(
                        record["base"]["gold_rank"] != 1
                        and record["fused"]["gold_rank"] == 1
                        for record in phase_records
                    ),
                    "tcra_correct_to_wrong": sum(
                        record["base"]["gold_rank"] == 1
                        and record["fused"]["gold_rank"] != 1
                        for record in phase_records
                    ),
                }
            )
        phases[phase] = phase_summary
    return phases


def summarize_model(free_records, oracle_records, annotations):
    labeled = [label_record(record, recompute_exact=True) for record in free_records]
    full_counts = Counter(record["error_type"] for record in labeled)
    oracle = oracle_index(oracle_records)
    first_failures = []
    all_events = []
    for record in labeled:
        failure = classify_first_procedure_failure(record, oracle, annotations)
        failure.update(
            {
                "index": int(record["index"]),
                "trial": int(record.get("trial", 1)),
                "user_input": record["user_input"],
                "expected_tools": record["expected_tools"],
                "predicted_tools": record["predicted_tools"],
                "call_exact": bool(record["call_exact"]),
                "eoc_count_exact": bool(record.get("eoc_count_exact")),
                "malformed_boundary": bool(record.get("malformed_boundary")),
            }
        )
        first_failures.append(failure)
        for event in transition_events(record, oracle, annotations):
            event.update(
                {
                    "user_input": record["user_input"],
                    "expected_tools": record["expected_tools"],
                    "predicted_tools": record["predicted_tools"],
                }
            )
            all_events.append(event)

    procedure_failures = [
        failure
        for failure in first_failures
        if failure["cause"] != "procedure_sequence_correct"
    ]
    transition_failures = [
        failure
        for failure in procedure_failures
        if failure["cause"] not in {"initial_missing", "initial_routing"}
    ]
    first_cause_counts = Counter(
        failure["cause"] for failure in transition_failures
    )
    event_counts = Counter(event["cause"] for event in all_events)
    return {
        "samples": len(labeled),
        "full_outcome_counts": {
            key: int(full_counts.get(key, 0)) for key in FULL_OUTCOME_KEYS
        },
        "full_outcome_rates": {
            key: safe_rate(full_counts.get(key, 0), len(labeled))
            for key in FULL_OUTCOME_KEYS
        },
        "procedure_sequence_failure": rate_payload(
            len(procedure_failures),
            len(labeled),
        ),
        "first_correct_transition_failure": rate_payload(
            len(transition_failures),
            len(labeled),
        ),
        "first_transition_failure_causes": {
            cause: {
                **rate_payload(count, len(transition_failures)),
                "label": CAUSE_LABELS[cause],
            }
            for cause, count in sorted(first_cause_counts.items())
        },
        "transition_event_causes": {
            cause: {
                **rate_payload(count, len(all_events)),
                "label": CAUSE_LABELS[cause],
            }
            for cause, count in sorted(event_counts.items())
        },
        "eoc": summarize_eoc(labeled),
        "oracle": summarize_oracle(oracle_records),
        "first_failure_records": first_failures,
        "transition_event_records": all_events,
    }


def ambiguity_candidates(model_records, tool_descriptions):
    grouped = {}
    for model, records in model_records.items():
        for record in records:
            gold = list(record.get("expected_tools") or [])
            predicted = list(record.get("predicted_tools") or [])
            if not predicted or not gold or predicted[0] != gold[0]:
                continue
            for position in range(1, min(len(gold), len(predicted))):
                if predicted[position] == gold[position]:
                    continue
                key = ambiguity_key(
                    record["index"],
                    position,
                    gold[position],
                    predicted[position],
                )
                candidate = grouped.setdefault(
                    key,
                    {
                        "key": key,
                        "index": int(record["index"]),
                        "position": position,
                        "user_input": record["user_input"],
                        "gold_tool": gold[position],
                        "predicted_tool": predicted[position],
                        "gold_schema": tool_descriptions.get(gold[position]),
                        "predicted_schema": tool_descriptions.get(predicted[position]),
                        "models": [],
                        "model_trials": [],
                        "label": "REVIEW",
                        "rationale": "",
                    },
                )
                if model not in candidate["models"]:
                    candidate["models"].append(model)
                model_trial = f"{model}:trial{int(record.get('trial', 1))}"
                if model_trial not in candidate["model_trials"]:
                    candidate["model_trials"].append(model_trial)
    return sorted(grouped.values(), key=lambda row: (row["index"], row["position"], row["predicted_tool"]))


def compact_summary(summary):
    result = {}
    for model, payload in summary.items():
        result[model] = {
            key: value
            for key, value in payload.items()
            if key not in {"first_failure_records", "transition_event_records"}
        }
    return result


def fmt_rate(payload):
    if payload["rate"] is None:
        return "N/A"
    return (
        f"{100 * payload['rate']:.1f}% "
        f"({payload['numerator']}/{payload['denominator']})"
    )


def write_quick_summary(path, summary):
    lines = [
        "# TapMem Failure Diagnostics",
        "",
        "## Boundary and oracle-routing metrics",
        "",
        "| Model | EOC F1 | Exact EOC count | Malformed boundary | Oracle transition top-1 | Oracle transition candidate top-1 | Terminal stop accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, payload in summary.items():
        eoc = payload["eoc"]
        oracle = payload["oracle"]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS.get(model, model),
                    f"{100 * eoc['f1']:.1f}%",
                    fmt_rate(eoc["exact_count"]),
                    fmt_rate(eoc["malformed_sample"]),
                    fmt_rate(oracle["transition"]["full_vocab_top1_accuracy"]),
                    fmt_rate(oracle["transition"]["fused_candidate_top1_accuracy"]),
                    fmt_rate(oracle["terminal"]["full_vocab_top1_accuracy"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## First residual transition failure after a correct initial procedure",
            "",
            "| Model | Cause | Count / transition-failure samples |",
            "| --- | --- | ---: |",
        ]
    )
    for model, payload in summary.items():
        for cause, metrics in payload["first_transition_failure_causes"].items():
            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {CAUSE_LABELS[cause]} | "
                f"{fmt_rate(metrics)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    models = split_csv(args.models)
    trials = parse_trial_ids(args.trial_ids)
    annotations = load_annotations(
        Path(args.ambiguity_annotations) if args.ambiguity_annotations else None
    )
    tool_descriptions = load_json(Path(args.tool_descriptions))

    model_free_records = {}
    model_oracle_records = {}
    summary = {}
    for model in models:
        free_records = []
        oracle_records = []
        for trial in trials[model]:
            free_path = input_dir / "free_generation" / model / f"trial{trial}.jsonl"
            oracle_path = input_dir / "oracle_probes" / model / f"trial{trial}.jsonl"
            free_records.extend(load_jsonl(free_path))
            oracle_records.extend(load_jsonl(oracle_path))
        model_free_records[model] = free_records
        model_oracle_records[model] = oracle_records
        summary[model] = summarize_model(
            free_records,
            oracle_records,
            annotations,
        )
        write_jsonl(
            output_dir / "first_failures" / f"{model}.jsonl",
            summary[model]["first_failure_records"],
        )
        write_jsonl(
            output_dir / "transition_events" / f"{model}.jsonl",
            summary[model]["transition_event_records"],
        )

    candidates = ambiguity_candidates(model_free_records, tool_descriptions)
    write_jsonl(output_dir / "ambiguity_audit_template.jsonl", candidates)
    compact = compact_summary(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_quick_summary(output_dir / "summary.md", compact)
    print(f"Wrote diagnostics summary to {output_dir}")
    print(f"Ambiguity cases requiring audit: {len(candidates)}")


if __name__ == "__main__":
    main()
