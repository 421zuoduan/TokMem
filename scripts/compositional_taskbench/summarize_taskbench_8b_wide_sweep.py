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
    ("parse_error_example_rate", "Parse Example Err"),
]


GROUP_FIELDS = [
    "method",
    "lr",
    "batch_size",
    "eval_batch_size",
    "logit_bias_scale",
    "logit_bias_loss_weight",
]


def read_manifest(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def safe_float(value):
    if value is None or value == "" or value == "NA":
        return None
    return float(value)


def mean(values):
    return sum(values) / len(values) if values else None


def stdev(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def sort_numeric_or_text(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def sort_key(row):
    return tuple(sort_numeric_or_text(row.get(field, "")) for field in GROUP_FIELDS)


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
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        grouped.setdefault(key, {"rows": [], "trials": [], "failed": []})
        grouped[key]["rows"].append(row)

        if row.get("status") != "success":
            grouped[key]["failed"].append(row)
            failures.append(row)
            continue

        metrics = load_eval_metrics(row)
        if metrics is None:
            missing = dict(row)
            missing["status"] = "missing_evaluation_results"
            grouped[key]["failed"].append(missing)
            failures.append(missing)
            continue
        grouped[key]["trials"].append({"manifest": row, "metrics": metrics})

    configs = []
    for key, payload in grouped.items():
        config = dict(zip(GROUP_FIELDS, key))
        metric_summary = {}
        for metric in METRICS:
            values = [safe_float(trial["metrics"].get(metric)) for trial in payload["trials"]]
            values = [value for value in values if value is not None]
            metric_summary[metric] = {
                "mean": mean(values),
                "stdev": stdev(values),
                "values": values,
            }
        configs.append(
            {
                "config": config,
                "successful_trials": len(payload["trials"]),
                "failed_trials": len(payload["failed"]),
                "failed": payload["failed"],
                "metrics": metric_summary,
            }
        )

    configs.sort(key=lambda item: sort_key(item["config"]))
    return {"configs": configs, "failures": failures}


def format_value(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def best_configs(summary):
    best = {}
    for item in summary["configs"]:
        method = item["config"]["method"]
        if item["successful_trials"] == 0:
            continue
        metrics = item["metrics"]
        tool_f1 = metrics["avg_tool_f1_score"]["mean"]
        argument_f1 = metrics["avg_argument_f1"]["mean"]
        routing = metrics["routing_acc"]["mean"]
        if tool_f1 is None or argument_f1 is None:
            continue
        candidate = (tool_f1, argument_f1, routing or -1, item)
        if method not in best or candidate[:3] > best[method][:3]:
            best[method] = candidate
    return {method: payload[-1] for method, payload in best.items()}


def write_config_row(f, item):
    config = item["config"]
    metrics = item["metrics"]
    values = {
        key: format_value(metrics[key]["mean"])
        for key, _label in DISPLAY_METRICS
    }
    f.write(
        "| {method} | {lr} | {batch} | {eval_batch} | {scale} | {loss_weight} | "
        "{trials} | {routing} | {rouge} | {tool} | {arg} | {transition} | {parse_example} |\n".format(
            method=config["method"],
            lr=config["lr"],
            batch=config["batch_size"],
            eval_batch=config["eval_batch_size"],
            scale=config["logit_bias_scale"],
            loss_weight=config["logit_bias_loss_weight"],
            trials=item["successful_trials"],
            routing=values["routing_acc"],
            rouge=values["avg_rouge_l"],
            tool=values["avg_tool_f1_score"],
            arg=values["avg_argument_f1"],
            transition=values["transition_error"],
            parse_example=values["parse_error_example_rate"],
        )
    )


def write_markdown(summary, suite_dir):
    path = suite_dir / "summary.md"
    best = best_configs(summary)
    with path.open("w", encoding="utf-8") as f:
        f.write("# TaskBench DailyLife Llama-8B Wide Sweep Summary\n\n")
        f.write(f"Suite: `{suite_dir}`\n\n")
        f.write("Best config is selected by Tool F1, then Argument F1, then Routing Acc.\n\n")

        if best:
            f.write("## Best Configs\n\n")
            f.write(
                "| Method | LR | Batch | Eval Batch | Scale | Loss Weight | Trials | Routing Acc | "
                "Rouge-L | Tool F1 | Argument F1 | Transition Error | Parse Example Err |\n"
            )
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for method in sorted(best):
                write_config_row(f, best[method])

        f.write("\n## All Configs\n\n")
        f.write(
            "| Method | LR | Batch | Eval Batch | Scale | Loss Weight | Trials | Routing Acc | "
            "Rouge-L | Tool F1 | Argument F1 | Transition Error | Parse Example Err |\n"
        )
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for item in summary["configs"]:
            write_config_row(f, item)

        if summary["failures"]:
            f.write("\n## Failures\n\n")
            f.write("| Method | LR | Batch | Scale | Loss Weight | Trial | Seed | Status | Exit Code | Log |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
            for row in summary["failures"]:
                f.write(
                    f"| {row.get('method', '')} | {row.get('lr', '')} | {row.get('batch_size', '')} | "
                    f"{row.get('logit_bias_scale', '')} | {row.get('logit_bias_loss_weight', '')} | "
                    f"{row.get('trial', '')} | {row.get('seed', '')} | {row.get('status', '')} | "
                    f"{row.get('exit_code', '')} | `{row.get('log_file', '')}` |\n"
                )
    return path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_taskbench_8b_wide_sweep.py <suite_dir>")

    suite_dir = Path(sys.argv[1]).resolve()
    rows = read_manifest(suite_dir / "manifest.tsv")
    summary = build_summary(rows)
    summary["suite_dir"] = str(suite_dir)

    summary_json = suite_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary_md = write_markdown(summary, suite_dir)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    for method, item in sorted(best_configs(summary).items()):
        config = item["config"]
        metrics = item["metrics"]
        print(
            "best_{method}: lr={lr} batch={batch} scale={scale} loss_weight={loss_weight} "
            "trials={trials} tool_f1={tool:.4f} arg_f1={arg:.4f} routing={routing:.4f}".format(
                method=method,
                lr=config["lr"],
                batch=config["batch_size"],
                scale=config["logit_bias_scale"],
                loss_weight=config["logit_bias_loss_weight"],
                trials=item["successful_trials"],
                tool=metrics["avg_tool_f1_score"]["mean"],
                arg=metrics["avg_argument_f1"]["mean"],
                routing=metrics["routing_acc"]["mean"],
            )
        )


if __name__ == "__main__":
    main()
