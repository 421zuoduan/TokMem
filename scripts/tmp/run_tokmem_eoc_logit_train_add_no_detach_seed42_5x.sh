#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_SCRIPT="$ROOT_DIR/scripts/compositional/llama_1b/tokmem_eoc_logit_train_add_no_detach_llama_1b.sh"
RESULTS_ROOT="$ROOT_DIR/scripts/tmp/results"
SUITE_ID="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="tokmem_eoc_logit_train_add_no_detach_llama_1b_seed42_5x_${SUITE_ID}"
SUITE_DIR="$RESULTS_ROOT/$SUITE_NAME"
MANIFEST_FILE="$SUITE_DIR/manifest.tsv"
SUMMARY_FILE="$SUITE_DIR/summary.md"
ARTIFACTS_FILE="$SUITE_DIR/results.json"

mkdir -p "$SUITE_DIR"

printf "trial\tseed\trun_name\trun_dir\tevaluation_results\ttraining_summary\tstdout_log\n" > "$MANIFEST_FILE"

for trial in 1 2 3 4 5; do
    stdout_log="$SUITE_DIR/trial_${trial}.stdout.log"
    echo "[trial $trial/5] running $SOURCE_SCRIPT"
    bash "$SOURCE_SCRIPT" 2>&1 | tee "$stdout_log"

    run_dir="$(
        find "$ROOT_DIR/compositional/runs" \
            -maxdepth 1 \
            -type d \
            -name "tokmem_eoc_logit_train_add_no_detach_llama_1b_50tools_*" \
            -printf '%T@ %p\n' \
            | sort -nr \
            | head -n 1 \
            | cut -d' ' -f2-
    )"
    run_name="$(basename "$run_dir")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$trial" \
        "42" \
        "$run_name" \
        "$run_dir" \
        "$run_dir/evaluation_results.json" \
        "$run_dir/training_summary.json" \
        "$stdout_log" \
        >> "$MANIFEST_FILE"
done

python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$ARTIFACTS_FILE" "$SUITE_NAME" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
artifacts_path = Path(sys.argv[3])
suite_name = sys.argv[4]

metric_fields = (
    ("tool_accuracy", "Routing Acc"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("tool_exact_match_acc", "Tool Exact Match Acc"),
    ("exact_accuracy", "Exact Match Acc"),
    ("parse_error_rate", "Parse Error Rate"),
)

loss_fields = (
    ("avg_total_loss", "Avg Total Loss"),
    ("avg_ar_loss", "Avg AR Loss"),
    ("avg_logit_bias_loss", "Avg Logit Bias Loss"),
)


def mean_or_none(values):
    return statistics.mean(values) if values else None


def stdev_or_none(values):
    return statistics.stdev(values) if len(values) >= 2 else None


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def load_last_eval_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("rounds"):
        return payload["rounds"][-1].get("eval_results") or {}
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    return payload


def load_last_training_round(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["rounds"][-1]
    if isinstance(payload, list):
        return payload[-1]
    raise TypeError(f"Unsupported training summary type: {type(payload)!r}")


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
metric_values = {key: [] for key, _ in metric_fields}
loss_values = {key: [] for key, _ in loss_fields}
trial_summaries = []

for row in rows:
    eval_path = Path(row["evaluation_results"])
    train_path = Path(row["training_summary"])
    trial_entry = {
        "trial": int(row["trial"]),
        "seed": int(row["seed"]),
        "run_name": row["run_name"],
        "run_dir": row["run_dir"],
        "evaluation_results": row["evaluation_results"],
        "training_summary": row["training_summary"],
        "stdout_log": row["stdout_log"],
        "metrics": {},
        "losses": {},
    }

    if eval_path.exists():
        eval_metrics = load_last_eval_metrics(eval_path)
        for key, _ in metric_fields:
            value = eval_metrics.get(key)
            if value is not None:
                metric_values[key].append(value)
                trial_entry["metrics"][key] = value

    if train_path.exists():
        train_round = load_last_training_round(train_path)
        for key, _ in loss_fields:
            value = train_round.get(key)
            if value is not None:
                loss_values[key].append(value)
                trial_entry["losses"][key] = value

    trial_summaries.append(trial_entry)

metric_stats = {
    key: {
        "mean": mean_or_none(values),
        "stdev": stdev_or_none(values),
        "count": len(values),
    }
    for key, values in metric_values.items()
}
loss_stats = {
    key: {
        "mean": mean_or_none(values),
        "stdev": stdev_or_none(values),
        "count": len(values),
    }
    for key, values in loss_values.items()
}

artifacts = {
    "suite_name": suite_name,
    "source_script": "scripts/compositional/llama_1b/tokmem_eoc_logit_train_add_no_detach_llama_1b.sh",
    "trials_requested": 5,
    "seed": 42,
    "metric_stats": metric_stats,
    "loss_stats": loss_stats,
    "trials": trial_summaries,
}
artifacts_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")

summary_lines = [
    f"# {suite_name}",
    "",
    "- method: `tokmem_eoc_logit_train_add_no_detach_llama_1b`",
    "- trials: `5`",
    "- seed: `42` for every trial",
    "- source script: `scripts/compositional/llama_1b/tokmem_eoc_logit_train_add_no_detach_llama_1b.sh`",
    "",
    "## Mean Metrics",
    "",
    "| Metric | Mean | Std | Count |",
    "| --- | ---: | ---: | ---: |",
]
for key, label in metric_fields:
    stats = metric_stats[key]
    summary_lines.append(
        f"| {label} | {fmt(stats['mean'])} | {fmt(stats['stdev'])} | {stats['count']} |"
    )

summary_lines.extend(
    [
        "",
        "## Mean Losses",
        "",
        "| Loss | Mean | Std | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
)
for key, label in loss_fields:
    stats = loss_stats[key]
    summary_lines.append(
        f"| {label} | {fmt(stats['mean'])} | {fmt(stats['stdev'])} | {stats['count']} |"
    )

summary_lines.extend(
    [
        "",
        "## Trials",
        "",
        "| Trial | Run | Routing Acc | Tool F1 | Arguments F1 | Exact Match Acc |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
)
for trial in trial_summaries:
    metrics = trial["metrics"]
    summary_lines.append(
        "| "
        + " | ".join(
            [
                str(trial["trial"]),
                f"`{trial['run_name']}`",
                fmt(metrics.get("tool_accuracy")),
                fmt(metrics.get("avg_tool_f1_score")),
                fmt(metrics.get("avg_f1_score")),
                fmt(metrics.get("exact_accuracy")),
            ]
        )
        + " |"
    )

summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
PY

echo "Summary written to $SUMMARY_FILE"
echo "JSON results written to $ARTIFACTS_FILE"
