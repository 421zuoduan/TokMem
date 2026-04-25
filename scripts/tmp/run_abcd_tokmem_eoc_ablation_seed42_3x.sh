#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RESULTS_ROOT="$ROOT_DIR/scripts/tmp/results"
SUITE_ID="$(date -u +%Y%m%d_%H%M%S)"
SUITE_NAME="abcd_tokmem_eoc_ablation_llama_1b_seed42_3x_${SUITE_ID}"
SUITE_DIR="$RESULTS_ROOT/$SUITE_NAME"
DATA_DIR="$SUITE_DIR/data"
MANIFEST_FILE="$SUITE_DIR/method_manifest.tsv"
SUMMARY_FILE="$SUITE_DIR/summary.md"
ARTIFACTS_FILE="$SUITE_DIR/results.json"

METHOD_KEYS=(
    A
    B
    C
    D
)

METHOD_NAMES=(
    a_tokmem_eoc_logit_bias_detach_llama_1b
    b_tokmem_eoc_logit_bias_no_detach_llama_1b
    c_tokmem_eoc_logit_train_add_detach_llama_1b
    d_tokmem_eoc_logit_train_add_no_detach_llama_1b
)

METHOD_SCRIPTS=(
    scripts/tmp/run_a_tokmem_eoc_logit_bias_detach_seed42_3x.sh
    scripts/tmp/run_b_tokmem_eoc_logit_bias_no_detach_seed42_3x.sh
    scripts/tmp/run_c_tokmem_eoc_logit_train_add_detach_seed42_3x.sh
    scripts/tmp/run_d_tokmem_eoc_logit_train_add_no_detach_seed42_3x.sh
)

METHOD_GPUS=(
    4
    5
    6
    7
)

mkdir -p "$SUITE_DIR"
cp "$SCRIPT_PATH" "$SUITE_DIR/$(basename "$SCRIPT_PATH")"

printf "method_key\tmethod_name\tscript\tsuite_dir\tresults_json\tsummary_md\tstdout_log\n" > "$MANIFEST_FILE"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

(
    cd "$ROOT_DIR/compositional"

    python xlam_datasets.py \
        --top_k "51-100" \
        --max_samples_per_tool 50 \
        --train_size 5000 \
        --test_size 500 \
        --train_max_function_calls 4 \
        --test_max_function_calls 4 \
        --train_multi_tool_ratios "0.5,0.5" \
        --test_multi_tool_ratios "0.5,0.5" \
        --output_dir "$DATA_DIR"
)

declare -a METHOD_PIDS

for idx in "${!METHOD_KEYS[@]}"; do
    method_key="${METHOD_KEYS[$idx]}"
    method_name="${METHOD_NAMES[$idx]}"
    method_script="${METHOD_SCRIPTS[$idx]}"
    method_gpu="${METHOD_GPUS[$idx]}"
    child_suite_file="$SUITE_DIR/method_${method_key}.suite_dir"
    stdout_log="$SUITE_DIR/method_${method_key}.stdout.log"

    echo "[method $method_key/ABCD] launching $method_script on GPU $method_gpu"
    (
        CUDA_VISIBLE_DEVICES="$method_gpu" \
        TOKMEM_DATA_DIR="$DATA_DIR" \
        TOKMEM_CHILD_SUITE_DIR_FILE="$child_suite_file" \
        SKIP_XLAM_DATASET=1 \
        bash "$ROOT_DIR/$method_script"
    ) 2>&1 | tee "$stdout_log" &
    METHOD_PIDS[$idx]="$!"
done

failed=0
for idx in "${!METHOD_KEYS[@]}"; do
    method_key="${METHOD_KEYS[$idx]}"
    method_name="${METHOD_NAMES[$idx]}"
    method_script="${METHOD_SCRIPTS[$idx]}"
    child_suite_file="$SUITE_DIR/method_${method_key}.suite_dir"
    stdout_log="$SUITE_DIR/method_${method_key}.stdout.log"

    if wait "${METHOD_PIDS[$idx]}"; then
        echo "[method $method_key/ABCD] completed"
    else
        echo "[method $method_key/ABCD] failed; see $stdout_log" >&2
        failed=1
        continue
    fi

    if [[ ! -s "$child_suite_file" ]]; then
        echo "Missing child suite marker for method $method_key: $method_name" >&2
        exit 1
    fi

    child_suite_dir="$(<"$child_suite_file")"

    if [[ -z "$child_suite_dir" || ! -f "$child_suite_dir/results.json" ]]; then
        echo "Missing child results for method $method_key: $method_name" >&2
        exit 1
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$method_key" \
        "$method_name" \
        "$method_script" \
        "$child_suite_dir" \
        "$child_suite_dir/results.json" \
        "$child_suite_dir/summary.md" \
        "$stdout_log" \
        >> "$MANIFEST_FILE"
done

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi

python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$ARTIFACTS_FILE" "$SUITE_NAME" <<'PY'
import csv
import json
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


def fmt(value):
    return "" if value is None else f"{value:.4f}"


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
methods = []

for row in rows:
    results_path = Path(row["results_json"])
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    methods.append(
        {
            "method_key": row["method_key"],
            "method_name": row["method_name"],
            "script": row["script"],
            "suite_dir": row["suite_dir"],
            "results_json": row["results_json"],
            "summary_md": row["summary_md"],
            "stdout_log": row["stdout_log"],
            "method_flags": payload.get("method_flags", ""),
            "metric_stats": payload.get("metric_stats", {}),
            "loss_stats": payload.get("loss_stats", {}),
            "trials": payload.get("trials", []),
        }
    )

artifacts = {
    "suite_name": suite_name,
    "methods": methods,
}
artifacts_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

summary_lines = [
    f"# {suite_name}",
    "",
    "- task: sequential A/B/C/D TokMem EOC ablation",
    "- trials per method: `3`",
    "- seed: `42` for every trial",
    "",
    "## Methods",
    "",
    "| Method | Flags | Child Suite |",
    "| --- | --- | --- |",
]
for method in methods:
    summary_lines.append(
        f"| {method['method_key']} | `{method['method_flags']}` | `{Path(method['suite_dir']).name}` |"
    )

summary_lines.extend(
    [
        "",
        "## Mean Metrics",
        "",
        "| Method | Routing Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
)
for method in methods:
    metric_stats = method["metric_stats"]
    values = [fmt(metric_stats.get(key, {}).get("mean")) for key, _ in metric_fields]
    summary_lines.append(
        "| "
        + " | ".join([method["method_key"], *values])
        + " |"
    )

summary_lines.extend(
    [
        "",
        "## Mean Losses",
        "",
        "| Method | Avg Total Loss | Avg AR Loss | Avg Logit Bias Loss |",
        "| --- | ---: | ---: | ---: |",
    ]
)
for method in methods:
    loss_stats = method["loss_stats"]
    values = [fmt(loss_stats.get(key, {}).get("mean")) for key, _ in loss_fields]
    summary_lines.append(
        "| "
        + " | ".join([method["method_key"], *values])
        + " |"
    )

summary_lines.extend(
    [
        "",
        "## Trial Metrics",
        "",
        "| Method | Trial | Run | Routing Acc | Tool F1 | Arguments F1 | Exact Match Acc |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
)
for method in methods:
    for trial in method["trials"]:
        metrics = trial.get("metrics", {})
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    method["method_key"],
                    str(trial.get("trial", "")),
                    f"`{trial.get('run_name', '')}`",
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

echo "Aggregate summary written to $SUMMARY_FILE"
echo "Aggregate JSON results written to $ARTIFACTS_FILE"
