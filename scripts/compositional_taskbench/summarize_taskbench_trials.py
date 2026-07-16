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


def load_manifest(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def metric_mean(values):
    return sum(values) / len(values) if values else None


def metric_stdev(values):
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_taskbench_trials.py <suite_dir>")

    suite_dir = Path(sys.argv[1]).resolve()
    manifest_path = suite_dir / "manifest.tsv"
    rows = load_manifest(manifest_path)
    grouped = {}

    for row in rows:
        method = row["method"]
        grouped.setdefault(method, {"trials": [], "failed": []})
        if row["status"] != "success":
            grouped[method]["failed"].append(row)
            continue

        eval_path = Path(row["run_dir"]) / "evaluation_results.json"
        if not eval_path.exists():
            failed = dict(row)
            failed["status"] = "missing_evaluation_results"
            grouped[method]["failed"].append(failed)
            continue

        with eval_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        grouped[method]["trials"].append({"manifest": row, "metrics": metrics})

    summary = {"suite_dir": str(suite_dir), "methods": {}}
    for method, payload in grouped.items():
        trials = payload["trials"]
        method_summary = {
            "successful_trials": len(trials),
            "failed_trials": len(payload["failed"]),
            "failed": payload["failed"],
            "metrics": {},
        }
        for metric in METRICS:
            values = [trial["metrics"].get(metric) for trial in trials]
            values = [float(value) for value in values if value is not None]
            method_summary["metrics"][metric] = {
                "mean": metric_mean(values),
                "stdev": metric_stdev(values),
                "values": values,
            }
        summary["methods"][method] = method_summary

    summary_json = suite_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_md = suite_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# TaskBench DailyLife 3-Trial Summary\n\n")
        f.write(f"Suite: `{suite_dir}`\n\n")
        f.write("| Method | Trials | Routing Acc | Rouge-L | Tool F1 | Argument F1 | Transition Error |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for method in sorted(summary["methods"]):
            method_summary = summary["methods"][method]
            metrics = method_summary["metrics"]
            f.write(
                "| {method} | {trials} | {routing:.4f} | {rouge:.4f} | {tool:.4f} | {arg:.4f} | {transition:.4f} |\n".format(
                    method=method,
                    trials=method_summary["successful_trials"],
                    routing=metrics["routing_acc"]["mean"] or 0.0,
                    rouge=metrics["avg_rouge_l"]["mean"] or 0.0,
                    tool=metrics["avg_tool_f1_score"]["mean"] or 0.0,
                    arg=metrics["avg_argument_f1"]["mean"] or 0.0,
                    transition=metrics["transition_error"]["mean"] or 0.0,
                )
            )

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    for method in sorted(summary["methods"]):
        method_summary = summary["methods"][method]
        print(
            f"{method}: trials={method_summary['successful_trials']} "
            f"routing_acc={method_summary['metrics']['routing_acc']['mean']} "
            f"rouge_l={method_summary['metrics']['avg_rouge_l']['mean']} "
            f"transition_error={method_summary['metrics']['transition_error']['mean']}"
        )


if __name__ == "__main__":
    main()
