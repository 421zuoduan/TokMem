#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from summarize_completed_trials import (
    call_count_breakdown_from_detailed_results,
    call_count_breakdown_from_metrics,
    call_count_breakdown_from_stdout,
    load_eval_payload,
    merge_breakdown_sources,
    normalize_metrics,
)


METHOD_ORDER = (
    "icl",
    "rag",
    "tokmem",
    "tapmem",
    "lora",
    "adap_tokmem",
    "adap_tapmem",
)
MODEL_ORDER = ("llama1b", "llama3b", "llama8b")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine Table 1 seed 40/41 runs with the published seed 42 values."
    )
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().with_name("table1_seed42_reference.json"),
    )
    return parser.parse_args()


def mean(values):
    return statistics.mean(values)


def sample_std(values):
    return statistics.stdev(values)


def read_trial(task):
    task_dir = Path(task["task_dir"])
    evaluation_path = task_dir / "evaluation_results.json"
    stdout_path = task_dir / "stdout.log"
    if not (task_dir / "SUCCESS").exists() or not evaluation_path.exists():
        raise RuntimeError(f"Incomplete task: {task_dir}")

    _, raw_metrics, detailed_results = load_eval_payload(evaluation_path)
    total_examples = raw_metrics.get(
        "total_examples", raw_metrics.get("total_samples")
    )
    if total_examples != 500:
        raise RuntimeError(
            f"Expected 500 evaluation examples for {task_dir}, got {total_examples}"
        )
    metrics = normalize_metrics(raw_metrics)
    breakdown, _ = merge_breakdown_sources(
        call_count_breakdown_from_metrics(raw_metrics),
        call_count_breakdown_from_stdout(stdout_path),
        call_count_breakdown_from_detailed_results(detailed_results),
    )

    values = {
        "tool_avg": metrics["avg_tool_f1_score"],
        "argument_avg": metrics["avg_f1_score"],
    }
    for call_count in (2, 3, 4):
        call_values = breakdown.get(call_count, {})
        values[f"tool_{call_count}"] = call_values.get("tool_f1")
        values[f"argument_{call_count}"] = call_values.get("argument_f1")

    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(f"Missing metrics for {task_dir}: {', '.join(missing)}")
    invalid = [
        name
        for name, value in values.items()
        if not math.isfinite(value) or not 0.0 <= value <= 1.0
    ]
    if invalid:
        raise RuntimeError(f"Invalid metrics for {task_dir}: {', '.join(invalid)}")
    return values


def load_tasks(suite_dir):
    manifest_path = suite_dir / "task_manifest.tsv"
    with manifest_path.open(encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))

    expected = {
        (model, method, str(seed))
        for model in MODEL_ORDER
        for method in METHOD_ORDER
        for seed in (40, 41)
    }
    expected.add(("llama3b", "tokmem", "42"))
    observed = {(task["model"], task["method"], task["seed"]) for task in tasks}
    if len(tasks) != len(expected) or len(observed) != len(tasks):
        raise RuntimeError("Manifest must contain 43 unique tasks")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Manifest mismatch. Missing={missing}, extra={extra}")
    return tasks


def format_cell(stats):
    return f"{stats['mean'] * 100:.1f} ± {stats['std'] * 100:.1f}"


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_rows = {
        (row["model"], row["method"]): row for row in reference["rows"]
    }

    trial_values = {}
    for task in load_tasks(suite_dir):
        key = (task["model"], task["method"], int(task["seed"]))
        trial_values[key] = read_trial(task)

    rows = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            reference_row = reference_rows[(model, method)]
            seed42_source = "published_table1"
            seed42_values = {
                name: value / 100.0
                for name, value in reference_row["metrics"].items()
            }
            if model == "llama3b" and method == "tokmem":
                seed42_source = "fresh_max_length_512"
                seed42_values = trial_values[(model, method, 42)]
            per_seed = {
                "40": trial_values[(model, method, 40)],
                "41": trial_values[(model, method, 41)],
                "42": seed42_values,
            }
            stats = {}
            for metric in per_seed["40"]:
                values = [per_seed[str(seed)][metric] for seed in (40, 41, 42)]
                stats[metric] = {
                    "mean": mean(values),
                    "std": sample_std(values),
                    "values": values,
                }
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "seed42_source": seed42_source,
                    "published_seed42": reference_row["metrics"],
                    "per_seed": per_seed,
                    "stats": stats,
                }
            )

    output = {
        "suite_dir": str(suite_dir),
        "seeds": [40, 41, 42],
        "error_bar": "sample standard deviation across three equally weighted seed-level values",
        "seed42_reference": str(args.reference.resolve()),
        "published_seed42_reference_values_are_percentages_rounded_to_one_decimal": True,
        "llama3b_tokmem_seed42_is_fresh_max_length_512": True,
        "rows": rows,
    }
    (suite_dir / "table1_error_bars.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TapMem Table 1: Seeds 40/41/42",
        "",
        "- Values are `mean ± sample standard deviation` in percent.",
        "- Seeds 40 and 41 come from this suite.",
        "- Seed 42 is the published Table 1 value, except Llama-3B TokMem, which is rerun at `max_length=512`.",
        "- Each seed has equal weight. Published seed 42 values are rounded to one decimal.",
        "- All trainable runs use `max_length=512`, including Llama-3B TokMem.",
        "",
        "| Model | Method | Tool 2c | Tool 3c | Tool 4c | Tool Avg | Arg 2c | Arg 3c | Arg 4c | Arg Avg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        stats = row["stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["method"],
                    format_cell(stats["tool_2"]),
                    format_cell(stats["tool_3"]),
                    format_cell(stats["tool_4"]),
                    format_cell(stats["tool_avg"]),
                    format_cell(stats["argument_2"]),
                    format_cell(stats["argument_3"]),
                    format_cell(stats["argument_4"]),
                    format_cell(stats["argument_avg"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Statistical note",
            "",
            "The published seed-42 cells are rounded Table 1 values and can themselves be "
            "aggregates of repeated runs that used seed 42. They are combined with one fresh "
            "seed-40 run and one fresh seed-41 run at equal weight, so these should be reported "
            "as published-reference error bars rather than a fully balanced three-seed study. "
            "The Llama-3B TokMem row uses a fresh seed-42 run at 512 to avoid mixing sequence lengths.",
            "",
        ]
    )
    (suite_dir / "table1_error_bars.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
