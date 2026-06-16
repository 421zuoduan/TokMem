#!/usr/bin/env python3
"""Aggregate atomic NI Rouge-L by task category from existing evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_CATEGORIES = ("Question Answering", "Program Execution")


def resolve_results_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "evaluation_results.json"
        if candidate.exists():
            return candidate
    return path


def load_task_category(
    task_name: str,
    tasks_dir: Path,
    cache: dict[str, list[str]],
) -> list[str] | None:
    if task_name in cache:
        return cache[task_name]

    task_path = tasks_dir / f"{task_name}.json"
    if not task_path.exists():
        return None
    try:
        payload = json.loads(task_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse task file: {task_path}") from exc
    cache[task_name] = payload.get("Categories", [])
    return cache[task_name]


def find_default_tasks_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "datasets" / "natural-instructions-2.8" / "tasks",
        repo_root / "atomic" / "natural-instructions-2.8" / "tasks",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find NI tasks dir. Pass --tasks-dir explicitly."
    )


def aggregate_file(
    results_path: Path,
    tasks_dir: Path,
    task_category_cache: dict[str, list[str]],
    target_categories: set[str],
) -> list[dict[str, object]]:
    payload = json.loads(results_path.read_text())
    per_task = payload.get("ni_per_task")
    if not isinstance(per_task, dict):
        raise ValueError(f"{results_path} does not contain an ni_per_task object")

    totals = {
        category: {"weighted_rouge": 0.0, "examples": 0, "tasks": 0}
        for category in target_categories
    }
    missing_tasks: list[str] = []

    for task_name, metrics in per_task.items():
        categories = load_task_category(task_name, tasks_dir, task_category_cache)
        if categories is None:
            missing_tasks.append(task_name)
            continue

        for category in categories:
            if category not in target_categories:
                continue
            num_examples = int(metrics.get("num_examples", 0))
            rouge_l = float(metrics.get("rougeL", 0.0))
            totals[category]["weighted_rouge"] += rouge_l * num_examples
            totals[category]["examples"] += num_examples
            totals[category]["tasks"] += 1

    rows: list[dict[str, object]] = []
    for category in sorted(target_categories):
        total = totals[category]
        examples = int(total["examples"])
        rouge_l = (
            float(total["weighted_rouge"]) / examples
            if examples
            else math.nan
        )
        rows.append(
            {
                "source": str(results_path),
                "run": results_path.parent.name,
                "category": category,
                "tasks": int(total["tasks"]),
                "examples": examples,
                "rouge_l": rouge_l,
                "missing_task_metadata": len(missing_tasks),
            }
        )
    return rows


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_across_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for category, category_rows in sorted(grouped.items()):
        rouge_values = [
            float(row["rouge_l"])
            for row in category_rows
            if not math.isnan(float(row["rouge_l"]))
        ]
        summary_rows.append(
            {
                "source": "MEAN",
                "run": f"n={len(rouge_values)}",
                "category": category,
                "tasks": category_rows[0]["tasks"] if category_rows else 0,
                "examples": category_rows[0]["examples"] if category_rows else 0,
                "rouge_l": sum(rouge_values) / len(rouge_values)
                if rouge_values
                else math.nan,
                "rouge_l_std": sample_std(rouge_values),
                "missing_task_metadata": sum(
                    int(row["missing_task_metadata"]) for row in category_rows
                ),
            }
        )
    return summary_rows


def format_float(value: object) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.4f}"


def print_table(rows: list[dict[str, object]], include_std: bool) -> None:
    headers = ["Run", "Category", "#Tasks", "Examples", "Rouge-L"]
    if include_std:
        headers.append("Rouge-L Std")
    headers.append("Source")
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        values = [
            str(row["run"]),
            str(row["category"]),
            str(row["tasks"]),
            str(row["examples"]),
            format_float(row["rouge_l"]),
        ]
        if include_std:
            values.append(format_float(row.get("rouge_l_std", 0.0)))
        values.append(str(row["source"]))
        print(" | ".join(values))


def write_csv(path: Path, rows: list[dict[str, object]], include_std: bool) -> None:
    fieldnames = [
        "run",
        "category",
        "tasks",
        "examples",
        "rouge_l",
    ]
    if include_std:
        fieldnames.append("rouge_l_std")
    fieldnames.extend(["source", "missing_task_metadata"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Question Answering and Program Execution Rouge-L from "
            "atomic evaluation_results.json files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Run directories or evaluation_results.json files.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="NI tasks directory. Defaults to datasets/natural-instructions-2.8/tasks.",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Target NI category. Can be repeated.",
    )
    parser.add_argument(
        "--no-mean",
        action="store_true",
        help="Only print per-run rows.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    tasks_dir = args.tasks_dir or find_default_tasks_dir(repo_root)
    categories = set(args.categories or DEFAULT_CATEGORIES)
    task_category_cache: dict[str, list[str]] = {}

    rows: list[dict[str, object]] = []
    for raw_path in args.inputs:
        results_path = resolve_results_path(Path(raw_path))
        rows.extend(
            aggregate_file(results_path, tasks_dir, task_category_cache, categories)
        )

    output_rows = rows if args.no_mean else aggregate_across_runs(rows) + rows
    print_table(output_rows, include_std=not args.no_mean)
    if args.csv:
        write_csv(args.csv, output_rows, include_std=not args.no_mean)


if __name__ == "__main__":
    main()
