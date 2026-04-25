#!/usr/bin/env python
import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METRIC_FIELDS = (
    ("tool_accuracy", "Tool Acc"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("tool_exact_match_acc", "Tool Exact Match Acc"),
    ("exact_accuracy", "Exact Match Acc"),
    ("parse_error_rate", "Parse Error Rate"),
)

TRAINABLE_PARAMS = {
    "llama1b": {
        "lora": "0.85M",
        "tokmem": "0.10M",
    },
    "llama3b": {
        "lora": "2.29M",
        "tokmem": "0.15M",
    },
    "llama8b": {
        "lora": "3.41M",
        "tokmem": "0.20M",
    },
}

CALL_SCOPE_LABELS = {
    "4calls": "tools 51-100 / 4 calls",
    "10calls": "tools 51-100 / 10 calls",
}

CALL_COUNT_COLUMNS = {
    "4calls": list(range(2, 5)),
    "10calls": list(range(2, 11)),
}

STDOUT_ARGUMENT_MARKER = "AVERAGE F1 SCORE (Function Calls)"
STDOUT_TOOL_MARKER = "AVERAGE TOOL F1 SCORE"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize successful compositional suite trials across one or more suite directories. "
            "Unlike run_paper_compositional_suite.sh, groups with different trial counts are compared "
            "by averaging all available successful trials."
        )
    )
    parser.add_argument(
        "suite_dirs",
        nargs="+",
        help="Suite directories under results/compositional, each containing task_status.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/compositional/completed_trials_summary",
        help="Directory for completed_trials_summary.md/json.",
    )
    return parser.parse_args()


def fmt(value):
    return "" if value is None else f"{value:.3f}"


def fmt_percent(value):
    return "" if value is None else f"{value * 100:.1f}"


def fmt_seconds(seconds):
    if seconds is None:
        return ""
    return f"{seconds / 3600:.2f}h"


def mean_or_none(values):
    return statistics.mean(values) if values else None


def stdev_or_none(values):
    return statistics.stdev(values) if len(values) > 1 else None


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_path(path_value, suite_dir):
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = suite_dir / path
        if candidate.exists():
            return candidate
    return path


def infer_call_scope(task):
    call_scope = task.get("call_scope")
    if call_scope:
        return call_scope
    task_name = task.get("task_name", "")
    if "_10calls_" in task_name or task_name.endswith("_10calls"):
        return "10calls"
    return "4calls"


def normalize_metrics(metrics):
    exact_accuracy = metrics.get("exact_accuracy", metrics.get("exact_match_accuracy"))
    tool_f1 = metrics.get("avg_tool_f1_score", metrics.get("tool_selection_f1"))
    arguments_f1 = metrics.get("avg_f1_score", metrics.get("average_f1_score"))
    return {
        "tool_accuracy": as_float(metrics.get("tool_accuracy")),
        "avg_tool_f1_score": as_float(tool_f1),
        "avg_f1_score": as_float(arguments_f1),
        "tool_exact_match_acc": as_float(metrics.get("tool_exact_match_acc")),
        "exact_accuracy": as_float(exact_accuracy),
        "parse_error_rate": as_float(metrics.get("parse_error_rate")),
    }


def load_eval_payload(evaluation_path):
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("rounds"):
        last_round = payload["rounds"][-1]
        metrics = last_round.get("eval_results") or {}
        detailed_results = (
            last_round.get("detailed_results")
            or last_round.get("results")
            or payload.get("detailed_results")
        )
    elif isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
        detailed_results = payload.get("detailed_results") or payload.get("results")
    else:
        metrics = payload if isinstance(payload, dict) else {}
        detailed_results = None
    return payload, metrics, detailed_results


def list_mean(values):
    numeric = [as_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    return mean_or_none(numeric)


def normalize_breakdown_entry(stats):
    total = as_float(stats.get("total") or stats.get("total_samples"))
    if total is None and isinstance(stats.get("f1_scores"), list):
        total = len(stats["f1_scores"])

    argument_f1 = as_float(
        stats.get("avg_generation_f1")
        or stats.get("avg_f1_score")
        or stats.get("argument_f1")
        or stats.get("generation_f1")
    )
    if argument_f1 is None and isinstance(stats.get("f1_scores"), list):
        argument_f1 = list_mean(stats["f1_scores"])
    if argument_f1 is None and total:
        argument_f1 = as_float(stats.get("generation_f1_sum"))
        if argument_f1 is not None:
            argument_f1 /= total

    argument_precision = as_float(stats.get("avg_precision") or stats.get("precision"))
    if argument_precision is None and isinstance(stats.get("precision_scores"), list):
        argument_precision = list_mean(stats["precision_scores"])

    argument_recall = as_float(stats.get("avg_recall") or stats.get("recall"))
    if argument_recall is None and isinstance(stats.get("recall_scores"), list):
        argument_recall = list_mean(stats["recall_scores"])

    tool_f1 = as_float(
        stats.get("avg_tool_f1")
        or stats.get("avg_tool_f1_score")
        or stats.get("tool_f1")
        or stats.get("tool_f1_score")
    )
    if tool_f1 is None and isinstance(stats.get("tool_f1_scores"), list):
        tool_f1 = list_mean(stats["tool_f1_scores"])
    if tool_f1 is None and total:
        tool_f1 = as_float(stats.get("tool_f1_sum"))
        if tool_f1 is not None:
            tool_f1 /= total

    tool_precision = as_float(stats.get("avg_tool_precision") or stats.get("tool_precision"))
    if tool_precision is None and isinstance(stats.get("tool_precision_scores"), list):
        tool_precision = list_mean(stats["tool_precision_scores"])

    tool_recall = as_float(stats.get("avg_tool_recall") or stats.get("tool_recall"))
    if tool_recall is None and isinstance(stats.get("tool_recall_scores"), list):
        tool_recall = list_mean(stats["tool_recall_scores"])

    exact_match = as_float(stats.get("accuracy") or stats.get("exact_accuracy"))
    if exact_match is None and total:
        correct = as_float(stats.get("correct") or stats.get("exact_matches"))
        if correct is not None:
            exact_match = correct / total

    tool_exact_match = as_float(stats.get("avg_tool_exact_match_acc"))
    if tool_exact_match is None and total:
        tool_exact_matches = as_float(stats.get("tool_exact_matches"))
        if tool_exact_matches is not None:
            tool_exact_match = tool_exact_matches / total

    parse_error_rate = as_float(stats.get("parse_error_rate"))
    if parse_error_rate is None and total:
        parse_errors = as_float(stats.get("parse_errors"))
        if parse_errors is not None:
            parse_error_rate = parse_errors / total

    return {
        "total_samples": int(total) if total is not None else None,
        "argument_f1": argument_f1,
        "argument_precision": argument_precision,
        "argument_recall": argument_recall,
        "tool_f1": tool_f1,
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "exact_match": exact_match,
        "tool_exact_match_acc": tool_exact_match,
        "parse_error_rate": parse_error_rate,
    }


def call_count_breakdown_from_metrics(metrics):
    raw = metrics.get("call_count_breakdown")
    if not isinstance(raw, dict):
        return {}

    breakdown = {}
    for call_count, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        try:
            call_count_int = int(call_count)
        except (TypeError, ValueError):
            continue
        breakdown[call_count_int] = normalize_breakdown_entry(stats)
    return breakdown


def parse_last_stdout_section(text, marker, line_pattern):
    matches = list(re.finditer(re.escape(marker), text))
    if not matches:
        return {}

    section = text[matches[-1].end() :]
    end_match = re.search(r"\n={10,}", section)
    if end_match:
        section = section[: end_match.start()]

    values = {}
    for line in section.splitlines():
        match = line_pattern.search(line)
        if not match:
            continue
        call_count = int(match.group("call_count"))
        values[call_count] = {
            "f1": float(match.group("f1")),
            "precision": float(match.group("precision")),
            "recall": float(match.group("recall")),
        }
    return values


def call_count_breakdown_from_stdout(stdout_path):
    if stdout_path is None or not stdout_path.exists():
        return {}

    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    argument_pattern = re.compile(
        r"^\s*(?P<call_count>\d+) call\(s\): F1=(?P<f1>[0-9.]+), "
        r"P=(?P<precision>[0-9.]+), R=(?P<recall>[0-9.]+)"
    )
    tool_pattern = re.compile(
        r"^\s*(?P<call_count>\d+) call\(s\): Tool F1=(?P<f1>[0-9.]+), "
        r"P=(?P<precision>[0-9.]+), R=(?P<recall>[0-9.]+)"
    )

    argument_values = parse_last_stdout_section(text, STDOUT_ARGUMENT_MARKER, argument_pattern)
    tool_values = parse_last_stdout_section(text, STDOUT_TOOL_MARKER, tool_pattern)
    call_counts = sorted(set(argument_values) | set(tool_values))

    breakdown = {}
    for call_count in call_counts:
        argument = argument_values.get(call_count, {})
        tool = tool_values.get(call_count, {})
        breakdown[call_count] = {
            "total_samples": None,
            "argument_f1": argument.get("f1"),
            "argument_precision": argument.get("precision"),
            "argument_recall": argument.get("recall"),
            "tool_f1": tool.get("f1"),
            "tool_precision": tool.get("precision"),
            "tool_recall": tool.get("recall"),
            "exact_match": None,
            "tool_exact_match_acc": None,
            "parse_error_rate": None,
        }
    return breakdown


def call_count_breakdown_from_detailed_results(detailed_results):
    if not isinstance(detailed_results, list):
        return {}

    grouped = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(int)
    for row in detailed_results:
        target_calls = row.get("target_calls") or []
        call_count = len(target_calls)
        if call_count <= 0:
            continue

        totals[call_count] += 1
        grouped[call_count]["argument_f1"].append(as_float(row.get("f1_score")))
        grouped[call_count]["argument_precision"].append(as_float(row.get("precision")))
        grouped[call_count]["argument_recall"].append(as_float(row.get("recall")))

        tool_metrics = row.get("tool_metrics") or {}
        grouped[call_count]["tool_f1"].append(as_float(tool_metrics.get("tool_f1_score")))
        grouped[call_count]["tool_precision"].append(as_float(tool_metrics.get("tool_precision")))
        grouped[call_count]["tool_recall"].append(as_float(tool_metrics.get("tool_recall")))
        grouped[call_count]["tool_exact_match_acc"].append(
            as_float(tool_metrics.get("tool_exact_match_acc"))
        )
        grouped[call_count]["exact_match"].append(1.0 if row.get("exact_match") else 0.0)

    breakdown = {}
    for call_count, values_by_metric in grouped.items():
        breakdown[call_count] = {
            "total_samples": totals[call_count],
            "argument_f1": mean_or_none(
                [value for value in values_by_metric["argument_f1"] if value is not None]
            ),
            "argument_precision": mean_or_none(
                [value for value in values_by_metric["argument_precision"] if value is not None]
            ),
            "argument_recall": mean_or_none(
                [value for value in values_by_metric["argument_recall"] if value is not None]
            ),
            "tool_f1": mean_or_none(
                [value for value in values_by_metric["tool_f1"] if value is not None]
            ),
            "tool_precision": mean_or_none(
                [value for value in values_by_metric["tool_precision"] if value is not None]
            ),
            "tool_recall": mean_or_none(
                [value for value in values_by_metric["tool_recall"] if value is not None]
            ),
            "exact_match": mean_or_none(
                [value for value in values_by_metric["exact_match"] if value is not None]
            ),
            "tool_exact_match_acc": mean_or_none(
                [value for value in values_by_metric["tool_exact_match_acc"] if value is not None]
            ),
            "parse_error_rate": None,
        }
    return breakdown


def merge_breakdown_sources(metrics_breakdown, stdout_breakdown, detailed_breakdown):
    if metrics_breakdown:
        return metrics_breakdown, "evaluation_results.call_count_breakdown"
    if stdout_breakdown:
        return stdout_breakdown, "stdout.log"
    if detailed_breakdown:
        return detailed_breakdown, "detailed_results"
    return {}, None


def load_successful_trials(suite_dir):
    status_path = suite_dir / "task_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"task_status.json not found: {status_path}")

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    trials = []
    skipped = []

    for task in payload.get("tasks", []):
        if task.get("status") != "success":
            skipped.append(
                {
                    "suite": suite_dir.name,
                    "task_name": task.get("task_name"),
                    "status": task.get("status", "pending"),
                }
            )
            continue

        evaluation_path = resolve_path(task.get("evaluation_results"), suite_dir)
        if evaluation_path is None or not evaluation_path.exists():
            skipped.append(
                {
                    "suite": suite_dir.name,
                    "task_name": task.get("task_name"),
                    "status": "missing_evaluation_results",
                }
            )
            continue

        stdout_path = resolve_path(task.get("stdout_log"), suite_dir)
        _, raw_metrics, detailed_results = load_eval_payload(evaluation_path)
        metrics = normalize_metrics(raw_metrics)

        metrics_breakdown = call_count_breakdown_from_metrics(raw_metrics)
        stdout_breakdown = call_count_breakdown_from_stdout(stdout_path)
        detailed_breakdown = call_count_breakdown_from_detailed_results(detailed_results)
        breakdown, breakdown_source = merge_breakdown_sources(
            metrics_breakdown,
            stdout_breakdown,
            detailed_breakdown,
        )

        trials.append(
            {
                "suite": suite_dir.name,
                "task_name": task.get("task_name"),
                "call_scope": infer_call_scope(task),
                "model": task.get("model"),
                "method": task.get("method"),
                "trial": task.get("trial"),
                "seed": task.get("seed"),
                "duration_seconds": as_float(task.get("duration_seconds")),
                "metrics": metrics,
                "call_count_breakdown": breakdown,
                "breakdown_source": breakdown_source,
                "evaluation_results": str(evaluation_path),
                "stdout_log": str(stdout_path) if stdout_path else None,
            }
        )

    return trials, skipped


def trainable_params_for(model, method):
    if method in {"icl", "rag"}:
        return "-"
    family = "lora" if method == "lora" else "tokmem"
    return TRAINABLE_PARAMS.get(model, {}).get(family, "")


def summarize_groups(trials):
    grouped = defaultdict(list)
    for trial in trials:
        key = (trial["call_scope"], trial["model"], trial["method"])
        grouped[key].append(trial)

    groups = []
    for (call_scope, model, method), group_trials in sorted(grouped.items()):
        metric_summary = {}
        for key, _ in METRIC_FIELDS:
            values = [
                trial["metrics"][key]
                for trial in group_trials
                if trial["metrics"].get(key) is not None
            ]
            metric_summary[key] = {
                "mean": mean_or_none(values),
                "std": stdev_or_none(values),
                "n": len(values),
            }

        runtime_values = [
            trial["duration_seconds"]
            for trial in group_trials
            if trial.get("duration_seconds") is not None
        ]

        call_count_summary = {}
        observed_call_counts = sorted(
            {
                call_count
                for trial in group_trials
                for call_count in trial["call_count_breakdown"].keys()
            }
        )
        for call_count in observed_call_counts:
            metric_names = (
                "argument_f1",
                "argument_precision",
                "argument_recall",
                "tool_f1",
                "tool_precision",
                "tool_recall",
                "exact_match",
                "tool_exact_match_acc",
                "parse_error_rate",
            )
            call_count_summary[str(call_count)] = {}
            for metric_name in metric_names:
                values = []
                for trial in group_trials:
                    stats = trial["call_count_breakdown"].get(call_count, {})
                    value = stats.get(metric_name)
                    if value is not None:
                        values.append(value)
                call_count_summary[str(call_count)][metric_name] = {
                    "mean": mean_or_none(values),
                    "std": stdev_or_none(values),
                    "n": len(values),
                }

            sample_counts = [
                trial["call_count_breakdown"].get(call_count, {}).get("total_samples")
                for trial in group_trials
                if trial["call_count_breakdown"].get(call_count, {}).get("total_samples") is not None
            ]
            call_count_summary[str(call_count)]["sample_count"] = {
                "mean": mean_or_none(sample_counts),
                "n": len(sample_counts),
            }

        groups.append(
            {
                "call_scope": call_scope,
                "model": model,
                "method": method,
                "trainable_params": trainable_params_for(model, method),
                "n_trials": len(group_trials),
                "source_suites": sorted({trial["suite"] for trial in group_trials}),
                "metrics": metric_summary,
                "avg_runtime_seconds": mean_or_none(runtime_values),
                "call_count_breakdown": call_count_summary,
                "missing_call_breakdown_trials": [
                    trial["task_name"] for trial in group_trials if not trial["call_count_breakdown"]
                ],
                "breakdown_sources": sorted(
                    {trial["breakdown_source"] for trial in group_trials if trial["breakdown_source"]}
                ),
            }
        )
    return groups


def metric_mean(group, key):
    return group["metrics"].get(key, {}).get("mean")


def call_metric_mean(group, call_count, key):
    return (
        group["call_count_breakdown"]
        .get(str(call_count), {})
        .get(key, {})
        .get("mean")
    )


def render_overall_table(groups, call_scope):
    rows = [group for group in groups if group["call_scope"] == call_scope]
    if not rows:
        return []

    lines = [
        f"## Mean Results - {CALL_SCOPE_LABELS.get(call_scope, call_scope)}",
        "",
        "| Model | Method | #Params | Trials | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate | Avg Runtime | Sources |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for group in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    group["model"],
                    group["method"],
                    group["trainable_params"],
                    str(group["n_trials"]),
                    fmt(metric_mean(group, "tool_accuracy")),
                    fmt(metric_mean(group, "avg_tool_f1_score")),
                    fmt(metric_mean(group, "avg_f1_score")),
                    fmt(metric_mean(group, "tool_exact_match_acc")),
                    fmt(metric_mean(group, "exact_accuracy")),
                    fmt(metric_mean(group, "parse_error_rate")),
                    fmt_seconds(group["avg_runtime_seconds"]),
                    ", ".join(group["source_suites"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_table3_scope(groups, call_scope):
    rows = [group for group in groups if group["call_scope"] == call_scope]
    if not rows:
        return []

    call_counts = CALL_COUNT_COLUMNS.get(call_scope)
    if call_counts is None:
        call_counts = sorted(
            {
                int(call_count)
                for group in rows
                for call_count in group["call_count_breakdown"].keys()
            }
        )

    tool_headers = [f"Tool {call_count}c" for call_count in call_counts]
    arg_headers = [f"Arg {call_count}c" for call_count in call_counts]
    lines = [
        f"## Table 3 Style F1 - {CALL_SCOPE_LABELS.get(call_scope, call_scope)}",
        "",
        "| Model | Method | #Params | Trials | "
        + " | ".join(tool_headers)
        + " | Tool Avg | "
        + " | ".join(arg_headers)
        + " | Arg Avg | Breakdown Sources |",
        "| --- | --- | ---: | ---: | "
        + " | ".join(["---:"] * len(tool_headers))
        + " | ---: | "
        + " | ".join(["---:"] * len(arg_headers))
        + " | ---: | --- |",
    ]

    for group in rows:
        tool_values = [fmt_percent(call_metric_mean(group, count, "tool_f1")) for count in call_counts]
        arg_values = [fmt_percent(call_metric_mean(group, count, "argument_f1")) for count in call_counts]
        lines.append(
            "| "
            + " | ".join(
                [
                    group["model"],
                    group["method"],
                    group["trainable_params"],
                    str(group["n_trials"]),
                    *tool_values,
                    fmt_percent(metric_mean(group, "avg_tool_f1_score")),
                    *arg_values,
                    fmt_percent(metric_mean(group, "avg_f1_score")),
                    ", ".join(group["breakdown_sources"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_missing_breakdown(groups):
    missing = [
        group
        for group in groups
        if group["missing_call_breakdown_trials"]
    ]
    lines = ["## Call-Count Breakdown Coverage", ""]
    if not missing:
        lines.append("All successful trials had call-count F1 breakdowns.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Scope | Model | Method | Missing Trials |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for group in missing:
        lines.append(
            f"| {CALL_SCOPE_LABELS.get(group['call_scope'], group['call_scope'])} | "
            f"{group['model']} | {group['method']} | "
            f"{len(group['missing_call_breakdown_trials'])}/{group['n_trials']} |"
        )
    lines.append("")
    return lines


def render_markdown(groups, suite_dirs, skipped_tasks):
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Completed Compositional Trial Summary",
        "",
        f"- generated_at: `{generated_at}`",
        "- trial policy: all successful trials in the input suites are averaged within each `call_scope x model x method` group.",
        "- Table 3 style values are percentages; overall mean tables use decimal metrics to match suite summaries.",
        "- call-count F1 source priority: `evaluation_results.call_count_breakdown`, then `stdout.log`, then `detailed_results`.",
        "",
        "## Input Suites",
        "",
    ]
    for suite_dir in suite_dirs:
        lines.append(f"- `{suite_dir}`")
    lines.append("")

    for call_scope in ("4calls", "10calls"):
        lines.extend(render_overall_table(groups, call_scope))

    for call_scope in ("4calls", "10calls"):
        lines.extend(render_table3_scope(groups, call_scope))

    lines.extend(render_missing_breakdown(groups))

    lines.extend(["## Skipped Tasks", ""])
    if skipped_tasks:
        lines.extend(
            [
                "| Suite | Task | Status |",
                "| --- | --- | --- |",
            ]
        )
        for item in skipped_tasks:
            lines.append(
                f"| {item.get('suite', '')} | {item.get('task_name', '')} | {item.get('status', '')} |"
            )
    else:
        lines.append("No failed, pending, or missing-evaluation tasks were encountered.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    suite_dirs = [Path(path).resolve() for path in args.suite_dirs]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trials = []
    skipped_tasks = []
    for suite_dir in suite_dirs:
        trials, skipped = load_successful_trials(suite_dir)
        all_trials.extend(trials)
        skipped_tasks.extend(skipped)

    groups = summarize_groups(all_trials)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_suites": [str(path) for path in suite_dirs],
        "trial_policy": "mean over all successful trials per call_scope x model x method",
        "successful_trial_count": len(all_trials),
        "skipped_tasks": skipped_tasks,
        "groups": groups,
    }

    markdown_path = output_dir / "completed_trials_summary.md"
    json_path = output_dir / "completed_trials_summary.json"
    markdown_path.write_text(render_markdown(groups, suite_dirs, skipped_tasks), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    print(f"Successful trials summarized: {len(all_trials)}")
    print(f"Groups summarized: {len(groups)}")


if __name__ == "__main__":
    main()
