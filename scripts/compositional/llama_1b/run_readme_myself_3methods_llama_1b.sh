#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="readme_myself_3methods_llama_1b_${RUN_ID}"
RUN_DIR="$ROOT_DIR/compositional/runs/$RUN_NAME"
DATA_DIR="$RUN_DIR/data"
TRIAL_ROOT="$RUN_DIR/trials"
MANIFEST_FILE="$RUN_DIR/manifest.tsv"
SUMMARY_FILE="$RUN_DIR/comparison_summary.md"
ARTIFACTS_FILE="$RUN_DIR/comparison_artifacts.json"
README_FILE="$ROOT_DIR/README_MYSELF.md"
HF_CACHE_DIR="$RUN_DIR/hf-cache"

MODEL_DIR="$ROOT_DIR/models/Llama-3.2-1B-Instruct"
TRIAL_SEEDS=(42 42 42 42 42)
TOP_K="51-100"
TRAINING_ROUNDS="${TOP_K}:1"
MAX_SAMPLES_PER_TOOL=50
TRAIN_SIZE=5000
TEST_SIZE=500
TRAIN_MAX_CALLS=4
TEST_MAX_CALLS=4
BATCH_SIZE=4
EVAL_BATCH_SIZE=16
EPOCHS=3
MAX_LENGTH=1024
LR="0.005"
TRAIN_MULTI_TOOL_RATIOS="0.5,0.5"
TEST_MULTI_TOOL_RATIOS="0.5,0.5"

mkdir -p "$RUN_DIR" "$DATA_DIR" "$TRIAL_ROOT" "$HF_CACHE_DIR"
cp "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"

source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem

export CUDA_VISIBLE_DEVICES=4
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"

cd "$ROOT_DIR/compositional"

python xlam_datasets.py \
    --top_k "$TOP_K" \
    --max_samples_per_tool "$MAX_SAMPLES_PER_TOOL" \
    --train_size "$TRAIN_SIZE" \
    --test_size "$TEST_SIZE" \
    --train_max_function_calls "$TRAIN_MAX_CALLS" \
    --test_max_function_calls "$TEST_MAX_CALLS" \
    --train_multi_tool_ratios "$TRAIN_MULTI_TOOL_RATIOS" \
    --test_multi_tool_ratios "$TEST_MULTI_TOOL_RATIOS" \
    --output_dir "$DATA_DIR" \
    2>&1 | tee "$RUN_DIR/dataset.log"

printf "setting_id\tmode\ttrial\tseed\tuse_eoc\tuse_js_trunc\tuse_logit_bias\trun_name\trun_dir\tevaluation_results\ttraining_summary\n" > "$MANIFEST_FILE"

SETTINGS=(
    "1|baseline|0|0|0"
    "2|eoc-only|1|0|0"
    "3|eoc+logit_bias|1|0|1"
)

for setting_entry in "${SETTINGS[@]}"; do
    IFS='|' read -r setting_id mode use_eoc use_js_trunc use_logit_bias <<< "$setting_entry"

    trial_index=0
    for seed in "${TRIAL_SEEDS[@]}"; do
        trial_index=$((trial_index + 1))
        trial_name="readme_myself_3methods_setting${setting_id}_trial${trial_index}_${RUN_ID}"
        trial_dir="$TRIAL_ROOT/$trial_name"
        mkdir -p "$trial_dir"

        cmd=(
            python -u main_sequential.py
            --training_rounds "$TRAINING_ROUNDS"
            --epochs "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --train_max_function_calls "$TRAIN_MAX_CALLS"
            --test_max_function_calls "$TEST_MAX_CALLS"
            --model_name "$MODEL_DIR"
            --eval_after_each_round
            --save_checkpoints
            --data_dir "$DATA_DIR"
            --lr "$LR"
            --eval_batch_size "$EVAL_BATCH_SIZE"
            --max_length "$MAX_LENGTH"
            --seed "$seed"
            --tensorboard
            --run_root_dir "$TRIAL_ROOT"
            --run_name "$trial_name"
            --run_tag "readme_myself_3methods_setting${setting_id}"
        )

        if [[ "$use_eoc" == "1" ]]; then
            cmd+=(--use_eoc)
        fi
        if [[ "$use_js_trunc" == "1" ]]; then
            cmd+=(--use_js_trunc)
        fi
        if [[ "$use_logit_bias" == "1" ]]; then
            cmd+=(--use_logit_bias)
        fi

        {
            printf '=== setting %s trial %s ===\n' "$setting_id" "$trial_index"
            printf 'mode=%s seed=%s use_eoc=%s use_js_trunc=%s use_logit_bias=%s\n' \
                "$mode" "$seed" "$use_eoc" "$use_js_trunc" "$use_logit_bias"
            "${cmd[@]}"
        } 2>&1 | tee "$trial_dir/stdout.log"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$setting_id" "$mode" "$trial_index" "$seed" "$use_eoc" "$use_js_trunc" "$use_logit_bias" \
            "$trial_name" "$trial_dir" "$trial_dir/evaluation_results.json" "$trial_dir/training_summary.json" \
            >> "$MANIFEST_FILE"
    done
done

python - "$MANIFEST_FILE" "$SUMMARY_FILE" "$ARTIFACTS_FILE" "$README_FILE" "$RUN_NAME" "$EPOCHS" "$LR" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
artifacts_path = Path(sys.argv[3])
readme_path = Path(sys.argv[4])
run_name = sys.argv[5]
launcher_epochs = sys.argv[6]
launcher_lr = sys.argv[7]

metric_fields = (
    ("tool_accuracy", "Tool Acc"),
    ("avg_tool_f1_score", "Tool F1"),
    ("avg_f1_score", "Arguments F1"),
    ("tool_exact_match_acc", "Tool Exact Match Acc"),
    ("exact_accuracy", "Exact Match Acc"),
    ("parse_error_rate", "Parse Error Rate"),
)

loss_fields = (
    ("avg_total_loss", "avg total loss"),
    ("avg_ar_loss", "avg AR loss"),
    ("avg_logit_bias_loss", "avg Logit bias loss"),
)

FALLBACK_HEADER = "| 实验编号 | 模式 | epochs | lr | eoc | js trunc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |"
FALLBACK_SEPARATOR = "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
SEPARATOR_PATTERN = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")


def as_bool_text(value):
    return "√" if value == "1" else "×"


def mean_or_none(values):
    return statistics.mean(values) if values else None


def format_metric(value):
    return "" if value is None else f"{value:.3f}"


def format_hparam(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def last_training_round(training_payload):
    if isinstance(training_payload, dict):
        return training_payload["rounds"][-1]
    if isinstance(training_payload, list):
        return training_payload[-1]
    raise TypeError(f"Unsupported training_summary payload type: {type(training_payload)!r}")


def load_run_hparams(row):
    run_config_path = Path(row["run_dir"]) / "run_config.json"
    if not run_config_path.exists():
        return launcher_epochs, launcher_lr
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    args = run_config.get("args", {})
    return format_hparam(args.get("epochs", launcher_epochs)), format_hparam(args.get("lr", launcher_lr))


def replace_or_append_block(text, begin_marker, end_marker, replacement):
    if begin_marker in text and end_marker in text:
        start = text.index(begin_marker)
        end = text.index(end_marker) + len(end_marker)
        return text[:start] + replacement + text[end:]
    text = text.rstrip() + "\n\n" if text else ""
    return text + replacement + "\n"


def load_last_table_header(readme_text):
    last_pair = None
    lines = readme_text.splitlines()
    for index in range(len(lines) - 1):
        if not lines[index].startswith("|"):
            continue
        if not lines[index + 1].startswith("|"):
            continue
        if not SEPARATOR_PATTERN.fullmatch(lines[index + 1]):
            continue
        last_pair = (lines[index], lines[index + 1])
    if last_pair is None:
        return FALLBACK_HEADER, FALLBACK_SEPARATOR
    return last_pair


rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
grouped = {}
for row in rows:
    grouped.setdefault(row["setting_id"], []).append(row)

trial_count = len({row["trial"] for row in rows if row["setting_id"] == "1"})
artifacts = {
    "run_name": run_name,
    "manifest": rows,
    "defaults": {
        "seed": 42,
        "logit_bias_network": "linear",
        "logit_bias_loss_weight": 0.1,
        "logit_bias_scale": 1.0,
    },
    "settings": [],
}

readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
header, separator = load_last_table_header(readme_text)
summary_lines = [
    f"# README_MYSELF 三种维护方法对比（{trial_count} 次重复均值）",
    "",
    f"- run: `{run_name}`",
    f"- trials per setting: `{trial_count}`",
    "- methods: `baseline`, `eoc-only`, `eoc+logit_bias`",
    "- defaults: `logit_bias_network=linear`, `logit_bias_loss_weight=0.1`, `logit_bias_scale=1.0`",
    "",
    header,
    separator,
]
readme_table_lines = [header, separator]

for setting_id in sorted(grouped, key=lambda value: int(value)):
    setting_rows = sorted(grouped[setting_id], key=lambda row: int(row["trial"]))
    first_row = setting_rows[0]
    epochs, lr = load_run_hparams(first_row)
    metric_values = {key: [] for key, _ in metric_fields}
    loss_values = {key: [] for key, _ in loss_fields}

    for row in setting_rows:
        eval_path = Path(row["evaluation_results"])
        if eval_path.exists():
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("rounds"):
                eval_payload = payload["rounds"][-1].get("eval_results", {})
            else:
                eval_payload = payload
            for key, _ in metric_fields:
                value = eval_payload.get(key)
                if value is not None:
                    metric_values[key].append(value)
        train_path = Path(row["training_summary"])
        if train_path.exists():
            payload = json.loads(train_path.read_text(encoding="utf-8"))
            round_payload = last_training_round(payload)
            for key, _ in loss_fields:
                value = round_payload.get(key)
                if value is not None:
                    loss_values[key].append(value)

    row_text = (
        f"| `{setting_id}` | {first_row['mode']} | {epochs} | {lr} | "
        f"{as_bool_text(first_row['use_eoc'])} | {as_bool_text(first_row['use_js_trunc'])} | {as_bool_text(first_row['use_logit_bias'])} | "
        + " | ".join(format_metric(mean_or_none(metric_values[key])) for key, _ in metric_fields)
        + " |"
    )
    summary_lines.append(row_text)
    readme_table_lines.append(row_text)

    artifacts["settings"].append(
        {
            "setting_id": int(setting_id),
            "mode": first_row["mode"],
            "epochs": epochs,
            "lr": lr,
            "flags": {
                "use_eoc": first_row["use_eoc"] == "1",
                "use_js_trunc": first_row["use_js_trunc"] == "1",
                "use_logit_bias": first_row["use_logit_bias"] == "1",
            },
            "metrics": {key: mean_or_none(metric_values[key]) for key, _ in metric_fields},
            "losses": {key: mean_or_none(loss_values[key]) for key, _ in loss_fields},
        }
    )

summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
artifacts_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")

begin_marker = "<!-- README_MYSELF_3METHODS_TABLE:BEGIN -->"
end_marker = "<!-- README_MYSELF_3METHODS_TABLE:END -->"
replacement_lines = [
    begin_marker,
    f"## Compositional baseline / eoc / logit bias（{trial_count} 次重复均值）",
    "",
    f"- run: `{run_name}`",
    "- 模式只保留 `baseline`、`eoc-only`、`eoc+logit_bias`。",
    "- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。",
    "",
    *readme_table_lines,
    end_marker,
]
replacement = "\n".join(replacement_lines)

updated_readme = replace_or_append_block(readme_text, begin_marker, end_marker, replacement)
readme_path.write_text(updated_readme, encoding="utf-8")
PY
