#!/usr/bin/env python3
"""Compute rebuttal error types and later-step transition metrics from predictions."""

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from eval import compare_function_calls_advanced  # noqa: E402


CATEGORY_FIELDS = (
    ("correct", "Correct"),
    ("argument_only_error", "Argument-only error"),
    ("length_error", "Length error"),
    ("order_only_error", "Order-only error"),
    ("initial_involved_routing_error", "Initial-involved routing error"),
    ("later_only_routing_error", "Later-only routing error"),
)
CATEGORY_KEYS = tuple(key for key, _label in CATEGORY_FIELDS)
RATE_FIELDS = tuple(f"{key}_rate" for key in CATEGORY_KEYS) + (
    "later_step_mismatch_all",
    "later_step_mismatch_first_correct",
    "tool_sequence_exact",
)
METHOD_LABELS = {
    "tokmem": "TokMem",
    "tapmem": "TapMem",
    "eoc_only": "EOC-only",
}
MODEL_LABELS = {
    "llama1b": "Llama-1B",
    "llama3b": "Llama-3B",
    "llama8b": "Llama-8B",
}
COMPARISON_SPECS = {
    "tokmem_vs_tapmem": ("tokmem", "tapmem"),
    "tokmem_vs_eoc_only": ("tokmem", "eoc_only"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify aligned compositional prediction JSONL records into the six "
            "REBUTTAL.md error types and compute later-step mismatch rates."
        )
    )
    parser.add_argument("--manifest", required=True, help="Prediction manifest JSON produced by the runner.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to the manifest's output_dir field.",
    )
    parser.add_argument(
        "--no-labeled-predictions",
        action="store_true",
        help="Do not write per-sample JSONL records with the assigned error type.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    seen_indices = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "index" not in record:
                raise ValueError(f"Missing index in {path}:{line_number}")
            index = int(record["index"])
            if index in seen_indices:
                raise ValueError(f"Duplicate index {index} in {path}")
            seen_indices.add(index)
            records.append(record)
    records.sort(key=lambda record: int(record["index"]))
    return records


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else None


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def classify_error_type(expected_tools, predicted_tools, call_exact):
    """Return exactly one of the six ordered sample-level outcome categories."""
    expected_tools = list(expected_tools or [])
    predicted_tools = list(predicted_tools or [])

    if predicted_tools == expected_tools:
        return "correct" if call_exact else "argument_only_error"
    if len(predicted_tools) != len(expected_tools):
        return "length_error"
    if Counter(predicted_tools) == Counter(expected_tools):
        return "order_only_error"
    if expected_tools and predicted_tools and predicted_tools[0] != expected_tools[0]:
        return "initial_involved_routing_error"
    return "later_only_routing_error"


def recompute_call_exact(record):
    evaluation = compare_function_calls_advanced(
        record.get("predicted_calls") or [],
        record.get("expected_calls") or [],
        ignore_order=True,
    )
    return bool(evaluation.exact_match), evaluation


def label_record(record, recompute_exact=True):
    labeled = dict(record)
    if recompute_exact:
        call_exact, evaluation = recompute_call_exact(record)
        labeled["call_exact"] = call_exact
        labeled["call_exact_source"] = "compare_function_calls_advanced(ignore_order=True)"
        labeled["argument_parse_errors"] = evaluation.details.get("parse_errors", {})
    else:
        call_exact = bool(record.get("call_exact"))

    expected_tools = list(record.get("expected_tools") or [])
    predicted_tools = list(record.get("predicted_tools") or [])
    labeled["tool_sequence_exact"] = predicted_tools == expected_tools
    labeled["error_type"] = classify_error_type(expected_tools, predicted_tools, call_exact)
    return labeled


def analyze_records(records, recompute_exact=True):
    labeled_records = [label_record(record, recompute_exact=recompute_exact) for record in records]
    counts = Counter(record["error_type"] for record in labeled_records)
    samples = len(labeled_records)

    later_all_mismatches = 0
    later_all_positions = 0
    later_first_mismatches = 0
    later_first_positions = 0
    first_correct_samples = 0

    for record in labeled_records:
        expected_tools = list(record.get("expected_tools") or [])
        predicted_tools = list(record.get("predicted_tools") or [])
        if len(expected_tools) < 2:
            continue

        later_positions = len(expected_tools) - 1
        later_mismatches = sum(
            1
            for position in range(1, len(expected_tools))
            if position >= len(predicted_tools) or predicted_tools[position] != expected_tools[position]
        )
        later_all_positions += later_positions
        later_all_mismatches += later_mismatches

        first_correct = bool(predicted_tools) and predicted_tools[0] == expected_tools[0]
        if first_correct:
            first_correct_samples += 1
            later_first_positions += later_positions
            later_first_mismatches += later_mismatches

    metrics = {
        "samples": samples,
        "category_counts": {key: counts.get(key, 0) for key in CATEGORY_KEYS},
        "later_step_mismatch_all_numerator": later_all_mismatches,
        "later_step_mismatch_all_denominator": later_all_positions,
        "later_step_mismatch_all": safe_rate(later_all_mismatches, later_all_positions),
        "first_correct_samples": first_correct_samples,
        "later_step_mismatch_first_correct_numerator": later_first_mismatches,
        "later_step_mismatch_first_correct_denominator": later_first_positions,
        "later_step_mismatch_first_correct": safe_rate(
            later_first_mismatches,
            later_first_positions,
        ),
        "tool_sequence_exact": safe_rate(
            sum(1 for record in labeled_records if record["tool_sequence_exact"]),
            samples,
        ),
        "call_exact": safe_rate(
            sum(1 for record in labeled_records if record["call_exact"]),
            samples,
        ),
    }
    for key in CATEGORY_KEYS:
        metrics[f"{key}_rate"] = safe_rate(counts.get(key, 0), samples)

    if sum(metrics["category_counts"].values()) != samples:
        raise AssertionError("Six error-type counts do not cover every sample exactly once")
    return labeled_records, metrics


def alignment_signature(record):
    return {
        "index": int(record["index"]),
        "user_input": record.get("user_input"),
        "expected_tools": record.get("expected_tools") or [],
        "expected_calls": record.get("expected_calls") or [],
    }


def validate_alignment(entries_with_records):
    """Require identical gold examples for all methods within each model/trial pair."""
    grouped = {}
    for entry, records in entries_with_records:
        grouped.setdefault((entry["model"], int(entry["trial"])), []).append((entry, records))

    for (model, trial), group in grouped.items():
        reference_entry, reference_records = group[0]
        reference = [alignment_signature(record) for record in reference_records]
        for entry, records in group[1:]:
            candidate = [alignment_signature(record) for record in records]
            if candidate != reference:
                raise ValueError(
                    "Prediction files are not aligned for "
                    f"{model}/trial{trial}: {reference_entry['method']} vs {entry['method']}"
                )


def aggregate_trials(trials):
    aggregate = {
        "trials": len(trials),
        "samples_total": sum(trial["metrics"]["samples"] for trial in trials),
        "category_counts_total": {
            key: sum(trial["metrics"]["category_counts"][key] for trial in trials)
            for key in CATEGORY_KEYS
        },
    }
    for metric_key in RATE_FIELDS + ("call_exact",):
        values = [trial["metrics"].get(metric_key) for trial in trials]
        values = [value for value in values if value is not None]
        aggregate[metric_key] = mean(values)
        aggregate[f"{metric_key}_std"] = stdev(values)
    return aggregate


def build_comparisons(groups):
    comparisons = {}
    for comparison_name, (baseline_method, variant_method) in COMPARISON_SPECS.items():
        comparison_models = {}
        for model, model_groups in groups.items():
            if baseline_method not in model_groups or variant_method not in model_groups:
                continue
            baseline = model_groups[baseline_method]["aggregate"]
            variant = model_groups[variant_method]["aggregate"]
            delta = {}
            for metric_key in RATE_FIELDS:
                baseline_value = baseline.get(metric_key)
                variant_value = variant.get(metric_key)
                delta[metric_key] = (
                    variant_value - baseline_value
                    if baseline_value is not None and variant_value is not None
                    else None
                )
            comparison_models[model] = {
                "baseline": baseline_method,
                "variant": variant_method,
                "delta_variant_minus_baseline": delta,
            }
        if comparison_models:
            comparisons[comparison_name] = comparison_models
    return comparisons


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def aggregate_row(model, method, aggregate, fields):
    return [
        MODEL_LABELS.get(model, model),
        METHOD_LABELS.get(method, method),
        str(aggregate["trials"]),
        *(fmt(aggregate.get(field)) for field in fields),
    ]


def delta_row(model, label, delta, fields):
    return [MODEL_LABELS.get(model, model), label, "-", *(fmt(delta.get(field)) for field in fields)]


def append_error_table(lines, title, groups, comparison_models, baseline_method, variant_method):
    fields = tuple(f"{key}_rate" for key in CATEGORY_KEYS)
    lines.extend(
        [
            f"## {title}",
            "",
            "| Model | Method | Trials | Correct | Argument-only | Length | Order-only | Initial-involved routing | Later-only routing |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, comparison in comparison_models.items():
        for method in (baseline_method, variant_method):
            lines.append(
                "| "
                + " | ".join(aggregate_row(model, method, groups[model][method]["aggregate"], fields))
                + " |"
            )
        label = f"Delta ({METHOD_LABELS[variant_method]} - {METHOD_LABELS[baseline_method]})"
        lines.append(
            "| "
            + " | ".join(
                delta_row(model, label, comparison["delta_variant_minus_baseline"], fields)
            )
            + " |"
        )
    lines.append("")


def append_transition_table(lines, title, groups, comparison_models, baseline_method, variant_method):
    fields = (
        "later_step_mismatch_all",
        "later_step_mismatch_first_correct",
        "tool_sequence_exact",
    )
    lines.extend(
        [
            f"## {title}",
            "",
            "| Model | Method | Trials | Later mismatch (all) | Later mismatch (first-correct) | Tool Sequence Exact |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, comparison in comparison_models.items():
        for method in (baseline_method, variant_method):
            lines.append(
                "| "
                + " | ".join(aggregate_row(model, method, groups[model][method]["aggregate"], fields))
                + " |"
            )
        label = f"Delta ({METHOD_LABELS[variant_method]} - {METHOD_LABELS[baseline_method]})"
        lines.append(
            "| "
            + " | ".join(
                delta_row(model, label, comparison["delta_variant_minus_baseline"], fields)
            )
            + " |"
        )
    lines.append("")


def write_summary_md(path, manifest, groups, comparisons):
    lines = [
        "# Error Type and Later-step Transition Analysis",
        "",
        f"- test split: `{manifest.get('data_path', '')}`",
        f"- sample limit: `{manifest.get('limit')}`",
        f"- decoding: greedy (`do_sample=False`), max new tokens `{manifest.get('max_new_tokens')}`",
        "- aggregate: mean across trials; JSON contains per-trial standard deviations and raw numerators/denominators",
    ]
    duplicate_groups = manifest.get("duplicate_checkpoint_groups") or []
    if duplicate_groups:
        duplicate_text = "; ".join(
            f"{MODEL_LABELS.get(group['model'], group['model'])}/{METHOD_LABELS.get(group['method'], group['method'])} "
            f"trials {','.join(str(trial) for trial in group['trials'])}"
            for group in duplicate_groups
        )
        lines.append(
            "- checkpoint identity caveat: the following trial files have identical PyTorch ZIP member "
            f"size/CRC fingerprints and therefore are not independent checkpoints: {duplicate_text}"
        )
    lines.append("")

    main = comparisons.get("tokmem_vs_tapmem")
    if main:
        append_error_table(lines, "TokMem vs TapMem: six mutually exclusive outcomes", groups, main, "tokmem", "tapmem")
        append_transition_table(lines, "TokMem vs TapMem: later-step mismatch", groups, main, "tokmem", "tapmem")

    eoc = comparisons.get("tokmem_vs_eoc_only")
    if eoc:
        append_error_table(lines, "TokMem vs EOC-only: six mutually exclusive outcomes", groups, eoc, "tokmem", "eoc_only")
        append_transition_table(lines, "TokMem vs EOC-only: EOC usefulness", groups, eoc, "tokmem", "eoc_only")

    lines.extend(
        [
            "## Interpretation",
            "",
            "For Correct, a positive delta is better. For every error or mismatch rate, a negative delta is better. "
            "Later-step mismatch is position-weighted over gold positions 2..J; missing predictions count as mismatches and predictions beyond J are ignored.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_manifest(manifest_path, output_dir=None, write_labeled_predictions=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json(manifest_path)
    resolved_output_dir = Path(output_dir or manifest.get("output_dir") or manifest_path.parent).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    entries = manifest.get("entries") or []
    if not entries:
        raise ValueError(f"No prediction entries found in manifest: {manifest_path}")

    entries_with_records = []
    for entry in entries:
        prediction_path = Path(entry["prediction_path"])
        if not prediction_path.is_absolute():
            prediction_path = (manifest_path.parent / prediction_path).resolve()
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction JSONL not found: {prediction_path}")
        entries_with_records.append((entry, load_jsonl(prediction_path)))

    validate_alignment(entries_with_records)

    groups = {}
    per_trial = []
    for entry, records in entries_with_records:
        labeled_records, metrics = analyze_records(records, recompute_exact=True)
        model = entry["model"]
        method = entry["method"]
        trial = int(entry["trial"])
        labeled_path = None
        if write_labeled_predictions:
            prediction_stem = Path(entry["prediction_path"]).stem
            labeled_path = (
                resolved_output_dir
                / "labeled_predictions"
                / model
                / method
                / f"{prediction_stem}_labeled.jsonl"
            )
            write_jsonl(labeled_path, labeled_records)

        trial_payload = {
            "model": model,
            "method": method,
            "trial": trial,
            "run_name": entry.get("run_name"),
            "checkpoint_path": entry.get("checkpoint_path"),
            "prediction_path": entry["prediction_path"],
            "labeled_prediction_path": str(labeled_path) if labeled_path else None,
            "metrics": metrics,
        }
        per_trial.append(trial_payload)
        groups.setdefault(model, {}).setdefault(method, {"trials": []})["trials"].append(trial_payload)

    for model_groups in groups.values():
        for method_payload in model_groups.values():
            method_payload["trials"].sort(key=lambda item: item["trial"])
            method_payload["aggregate"] = aggregate_trials(method_payload["trials"])

    comparisons = build_comparisons(groups)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path),
        "definitions_source": str(REPO_ROOT / "REBUTTAL.md"),
        "groups": groups,
        "comparisons": comparisons,
    }

    summary_json = resolved_output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_trial_path = resolved_output_dir / "per_trial.jsonl"
    write_jsonl(per_trial_path, sorted(per_trial, key=lambda item: (item["model"], item["method"], item["trial"])))
    summary_md = resolved_output_dir / "summary.md"
    write_summary_md(summary_md, manifest, groups, comparisons)
    return summary_json, per_trial_path, summary_md


def main():
    args = parse_args()
    summary_json, per_trial_path, summary_md = analyze_manifest(
        args.manifest,
        output_dir=args.output_dir,
        write_labeled_predictions=not args.no_labeled_predictions,
    )
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote per-trial JSONL: {per_trial_path}")
    print(f"Wrote summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
