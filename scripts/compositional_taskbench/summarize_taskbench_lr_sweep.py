#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from pathlib import Path


METRICS = [
    "routing_acc",
    "task_prediction_accuracy",
    "avg_rouge_l",
    "avg_tool_f1_score",
    "avg_argument_f1",
    "transition_error",
    "parse_error_rate",
    "parse_error_example_rate",
]


DISPLAY_METRICS = [
    ("routing_acc", "Routing Acc"),
    ("avg_rouge_l", "Rouge-L"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_argument_f1", "Argument F1"),
    ("transition_error", "Transition Error"),
]


def read_manifest(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean(values):
    return sum(values) / len(values) if values else None


def stdev(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def safe_float(value):
    return None if value is None else float(value)


def sort_lr_key(value):
    try:
        return float(value)
    except ValueError:
        return value


def load_eval_metrics(row):
    eval_path = Path(row["run_dir"]) / "evaluation_results.json"
    if not eval_path.exists():
        return None
    with eval_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(rows):
    grouped = {}
    failures = []

    for row in rows:
        key = (row["method"], row["lr"])
        grouped.setdefault(key, {"trials": [], "failed": []})
        if row["status"] != "success":
            grouped[key]["failed"].append(row)
            failures.append(row)
            continue

        metrics = load_eval_metrics(row)
        if metrics is None:
            missing_row = dict(row)
            missing_row["status"] = "missing_evaluation_results"
            grouped[key]["failed"].append(missing_row)
            failures.append(missing_row)
            continue
        grouped[key]["trials"].append({"manifest": row, "metrics": metrics})

    method_lr = {}
    for (method, lr), payload in grouped.items():
        trials = payload["trials"]
        metric_summary = {}
        for metric in METRICS:
            values = [safe_float(trial["metrics"].get(metric)) for trial in trials]
            values = [value for value in values if value is not None]
            metric_summary[metric] = {
                "mean": mean(values),
                "stdev": stdev(values),
                "values": values,
            }

        method_lr.setdefault(method, {})[lr] = {
            "successful_trials": len(trials),
            "failed_trials": len(payload["failed"]),
            "failed": payload["failed"],
            "metrics": metric_summary,
        }

    deltas = {}
    lrs = sorted({row["lr"] for row in rows}, key=sort_lr_key)
    for lr in lrs:
        tokmem = method_lr.get("tokmem", {}).get(lr)
        tapmem = method_lr.get("tapmem", {}).get(lr)
        if not tokmem or not tapmem:
            continue
        delta_metrics = {}
        for metric in METRICS:
            left = tapmem["metrics"][metric]["mean"]
            right = tokmem["metrics"][metric]["mean"]
            delta_metrics[metric] = None if left is None or right is None else left - right
        deltas[lr] = delta_metrics

    return {"methods": method_lr, "deltas_tapmem_minus_tokmem": deltas, "failures": failures}


def format_value(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_markdown(summary, suite_dir):
    path = suite_dir / "summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# TaskBench DailyLife LR Sweep Summary\n\n")
        f.write(f"Suite: `{suite_dir}`\n\n")
        f.write("Best LR is selected by Tool F1, with Argument F1 as the tie-breaker.\n\n")
        f.write("## Method Results\n\n")
        f.write("| Method | LR | Trials | Routing Acc | Rouge-L | Tool F1 | Argument F1 | Transition Error |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for method in sorted(summary["methods"]):
            for lr in sorted(summary["methods"][method], key=sort_lr_key):
                payload = summary["methods"][method][lr]
                metrics = payload["metrics"]
                f.write(
                    "| {method} | {lr} | {trials} | {routing} | {rouge} | {tool} | {arg} | {transition} |\n".format(
                        method=method,
                        lr=lr,
                        trials=payload["successful_trials"],
                        routing=format_value(metrics["routing_acc"]["mean"]),
                        rouge=format_value(metrics["avg_rouge_l"]["mean"]),
                        tool=format_value(metrics["avg_tool_f1_score"]["mean"]),
                        arg=format_value(metrics["avg_argument_f1"]["mean"]),
                        transition=format_value(metrics["transition_error"]["mean"]),
                    )
                )

        best_rows = select_best_lrs(summary)
        if best_rows:
            f.write("\n## Best LR by Tool/Argument F1\n\n")
            f.write("| Method | LR | Trials | Tool F1 | Argument F1 | Routing Acc | Rouge-L | Transition Error |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for method in sorted(best_rows):
                lr, payload = best_rows[method]
                metrics = payload["metrics"]
                f.write(
                    "| {method} | {lr} | {trials} | {tool} | {arg} | {routing} | {rouge} | {transition} |\n".format(
                        method=method,
                        lr=lr,
                        trials=payload["successful_trials"],
                        tool=format_value(metrics["avg_tool_f1_score"]["mean"]),
                        arg=format_value(metrics["avg_argument_f1"]["mean"]),
                        routing=format_value(metrics["routing_acc"]["mean"]),
                        rouge=format_value(metrics["avg_rouge_l"]["mean"]),
                        transition=format_value(metrics["transition_error"]["mean"]),
                    )
                )

        f.write("\n## TapMem - TokMem Delta\n\n")
        f.write("| LR | Routing Acc | Rouge-L | Tool F1 | Argument F1 | Transition Error |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for lr in sorted(summary["deltas_tapmem_minus_tokmem"], key=sort_lr_key):
            metrics = summary["deltas_tapmem_minus_tokmem"][lr]
            f.write(
                "| {lr} | {routing} | {rouge} | {tool} | {arg} | {transition} |\n".format(
                    lr=lr,
                    routing=format_value(metrics["routing_acc"]),
                    rouge=format_value(metrics["avg_rouge_l"]),
                    tool=format_value(metrics["avg_tool_f1_score"]),
                    arg=format_value(metrics["avg_argument_f1"]),
                    transition=format_value(metrics["transition_error"]),
                )
            )

        if summary["failures"]:
            f.write("\n## Failures\n\n")
            f.write("| Method | LR | Trial | Status | Exit Code | Log |\n")
            f.write("| --- | ---: | ---: | --- | ---: | --- |\n")
            for row in summary["failures"]:
                f.write(
                    f"| {row['method']} | {row['lr']} | {row['trial']} | "
                    f"{row['status']} | {row.get('exit_code', '')} | `{row.get('log_file', '')}` |\n"
                )
    return path


def select_best_lrs(summary):
    best_rows = {}
    for method, lr_payloads in summary["methods"].items():
        candidates = []
        for lr, payload in lr_payloads.items():
            metrics = payload["metrics"]
            tool_f1 = metrics["avg_tool_f1_score"]["mean"]
            argument_f1 = metrics["avg_argument_f1"]["mean"]
            if tool_f1 is None or argument_f1 is None or payload["successful_trials"] == 0:
                continue
            candidates.append((tool_f1, argument_f1, lr, payload))
        if candidates:
            _, _, lr, payload = max(candidates, key=lambda item: (item[0], item[1]))
            best_rows[method] = (lr, payload)
    return best_rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_taskbench_lr_sweep.py <suite_dir>")

    suite_dir = Path(sys.argv[1]).resolve()
    manifest_path = suite_dir / "manifest.tsv"
    rows = read_manifest(manifest_path)
    summary = build_summary(rows)
    summary["suite_dir"] = str(suite_dir)

    summary_json = suite_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary_md = write_markdown(summary, suite_dir)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    for method in sorted(summary["methods"]):
        for lr in sorted(summary["methods"][method], key=sort_lr_key):
            payload = summary["methods"][method][lr]
            metrics = payload["metrics"]
            print(
                f"{method} lr={lr} trials={payload['successful_trials']} "
                f"routing_acc={metrics['routing_acc']['mean']} "
                f"rouge_l={metrics['avg_rouge_l']['mean']} "
                f"transition_error={metrics['transition_error']['mean']}"
            )
    for method, (lr, payload) in sorted(select_best_lrs(summary).items()):
        metrics = payload["metrics"]
        print(
            f"best_{method}_lr_by_tool_arg={lr} "
            f"tool_f1={metrics['avg_tool_f1_score']['mean']:.4f} "
            f"argument_f1={metrics['avg_argument_f1']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
