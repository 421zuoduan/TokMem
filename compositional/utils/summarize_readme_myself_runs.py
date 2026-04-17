#!/usr/bin/env python
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


TABLE_HEADER = (
    "| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | "
    "toolmix | js trunc | logit bias | Tool Prediction Acc | Tool F1 | "
    "Arguments F1 | Exact Match Acc | Parse Error Rate |"
)
TABLE_RULE = (
    "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | "
    "---: | ---: | ---: | ---: | ---: |"
)

METRIC_FIELDS = (
    ("tool_accuracy", "Tool Prediction Acc"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("exact_accuracy", "Exact Match Acc"),
    ("parse_error_rate", "Parse Error Rate"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize completed README_MYSELF trials into a copy-ready markdown table."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing manifest.tsv")
    parser.add_argument(
        "--output",
        help="Output markdown path. Defaults to <run_dir>/comparison_summary.partial.md",
    )
    return parser.parse_args()


def as_bool_text(value):
    return "√" if str(value) == "1" else "×"


def format_metric(value):
    return "" if value is None else f"{value:.3f}"


def format_hparam(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def mean_or_none(values):
    return None if not values else statistics.mean(values)


def load_eval_metrics(evaluation_path):
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return payload["rounds"][-1]["eval_results"]


def load_hparams(row):
    run_config_path = Path(row["run_dir"]) / "run_config.json"
    epochs = None
    lr = None

    if run_config_path.exists():
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        args = payload.get("args", {})
        epochs = args.get("epochs")
        lr = args.get("lr")

    training_summary_path = Path(row["training_summary"])
    if epochs is None and training_summary_path.exists():
        payload = json.loads(training_summary_path.read_text(encoding="utf-8"))
        rounds = payload.get("rounds") if isinstance(payload, dict) else payload
        if rounds:
            epochs = rounds[-1].get("epochs")

    return format_hparam(epochs), format_hparam(lr)


def load_manifest(run_dir):
    manifest_path = run_dir / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def infer_planned_counts(run_dir, rows):
    default_settings = len({row["setting_id"] for row in rows})
    default_trials = len(rows)
    default_trials_per_setting = {}
    for row in rows:
        default_trials_per_setting.setdefault(row["setting_id"], 0)
        default_trials_per_setting[row["setting_id"]] += 1

    launcher_path = next(run_dir.glob("*.sh"), None)
    if launcher_path is None:
        return default_settings, default_trials, default_trials_per_setting

    text = launcher_path.read_text(encoding="utf-8")

    settings_match = re.search(r"SETTINGS=\((.*?)\)\n", text, re.DOTALL)
    seeds_match = re.search(r"TRIAL_SEEDS=\((.*?)\)", text, re.DOTALL)
    if settings_match is None or seeds_match is None:
        return default_settings, default_trials, default_trials_per_setting

    setting_entries = re.findall(r'"([^"]+)"', settings_match.group(1))
    seed_entries = [token for token in seeds_match.group(1).split() if token.strip()]

    if not setting_entries or not seed_entries:
        return default_settings, default_trials, default_trials_per_setting

    planned_trials_per_setting = {}
    for entry in setting_entries:
        setting_id = entry.split("|", 1)[0]
        planned_trials_per_setting[setting_id] = len(seed_entries)

    return (
        len(setting_entries),
        len(setting_entries) * len(seed_entries),
        planned_trials_per_setting,
    )


def summarize_rows(rows):
    grouped = {}
    completed_trials = 0

    for row in rows:
        evaluation_path = Path(row["evaluation_results"])
        if not evaluation_path.exists():
            continue
        grouped.setdefault(row["setting_id"], []).append(row)
        completed_trials += 1

    settings = []
    for setting_id in sorted(grouped, key=lambda value: int(value)):
        setting_rows = sorted(grouped[setting_id], key=lambda row: int(row["trial"]))
        first_row = setting_rows[0]
        metrics = {field: [] for field, _ in METRIC_FIELDS}
        epochs_values = []
        lr_values = []

        for row in setting_rows:
            eval_results = load_eval_metrics(Path(row["evaluation_results"]))
            for field, _ in METRIC_FIELDS:
                metrics[field].append(float(eval_results[field]))

            epochs, lr = load_hparams(row)
            if epochs and epochs not in epochs_values:
                epochs_values.append(epochs)
            if lr and lr not in lr_values:
                lr_values.append(lr)

        averaged = {field: mean_or_none(values) for field, values in metrics.items()}
        settings.append(
            {
                "setting_id": int(setting_id),
                "mode": first_row["mode"],
                "epochs": ", ".join(epochs_values),
                "lr": ", ".join(lr_values),
                "trial_count": len(setting_rows),
                "flags": {
                    "use_eoc": first_row["use_eoc"],
                    "use_gate": first_row["use_gate"],
                    "use_eoc_loss": first_row["use_eoc_loss"],
                    "use_tool_loss": first_row["use_tool_loss"],
                    "use_toolmix": first_row["use_toolmix"],
                    "use_js_trunc": first_row["use_js_trunc"],
                    "use_logit_bias": first_row["use_logit_bias"],
                },
                "metrics": averaged,
            }
        )

    return settings, completed_trials


def infer_title(run_name):
    if "readme_myself_allmethods" in run_name:
        return "README_MYSELF 全方法部分结果"
    if "readme_myself_logit_bias_methods" in run_name:
        return "README_MYSELF Logit Bias 方法部分结果"
    return f"README_MYSELF 部分结果 `{run_name}`"


def render_summary(run_dir, rows, settings, completed_trials):
    run_name = run_dir.name
    total_settings, total_trials, planned_trials_by_setting = infer_planned_counts(run_dir, rows)
    completed_settings = len(settings)

    lines = [
        f"# {infer_title(run_name)}",
        "",
        f"- 自动生成自 `compositional/runs/{run_name}`。",
        f"- 当前已完成 {completed_settings}/{total_settings} 个设置，{completed_trials}/{total_trials} 个 trial。",
        "- 本表只统计已经写出 `evaluation_results.json` 的 trial。",
        "",
        TABLE_HEADER,
        TABLE_RULE,
    ]

    for setting in settings:
        lines.append(
            f"| `{setting['setting_id']}` | {setting['mode']} | {setting['epochs']} | {setting['lr']} | "
            f"{as_bool_text(setting['flags']['use_eoc'])} | "
            f"{as_bool_text(setting['flags']['use_gate'])} | "
            f"{as_bool_text(setting['flags']['use_eoc_loss'])} | "
            f"{as_bool_text(setting['flags']['use_tool_loss'])} | "
            f"{as_bool_text(setting['flags']['use_toolmix'])} | "
            f"{as_bool_text(setting['flags']['use_js_trunc'])} | "
            f"{as_bool_text(setting['flags']['use_logit_bias'])} | "
            f"{format_metric(setting['metrics']['tool_accuracy'])} | "
            f"{format_metric(setting['metrics']['avg_tool_f1_score'])} | "
            f"{format_metric(setting['metrics']['avg_f1_score'])} | "
            f"{format_metric(setting['metrics']['exact_accuracy'])} | "
            f"{format_metric(setting['metrics']['parse_error_rate'])} |"
        )

    if not settings:
        lines.append("|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")

    lines.extend(["", "## 完成情况", ""])
    if settings:
        for setting in settings:
            planned_trials = planned_trials_by_setting.get(
                str(setting["setting_id"]),
                setting["trial_count"],
            )
            lines.append(
                f"- 设置 `{setting['setting_id']}` `{setting['mode']}`："
                f"{setting['trial_count']}/{planned_trials} 个 trial 已完成。"
            )
    else:
        lines.append("- 当前还没有已完成的 trial。")

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "comparison_summary.partial.md"
    )

    rows = load_manifest(run_dir)
    settings, completed_trials = summarize_rows(rows)
    summary_text = render_summary(run_dir, rows, settings, completed_trials)
    output_path.write_text(summary_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
