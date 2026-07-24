#!/usr/bin/env python3
"""Compare TokMem and TapMem procedure-routing prefix lengths on 1B 10-call runs."""

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
UTILS_DIR = COMPOSITIONAL_DIR / "utils"
for import_dir in (COMPOSITIONAL_DIR, UTILS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from run_error_type_transition_analysis import (  # noqa: E402
    checkpoint_archive_fingerprint,
    partial_record_count,
    sha256_file,
    validate_prediction_file,
)
from run_train4_checkpoint_eval_10calls import (  # noqa: E402
    build_model,
    final_checkpoint_name,
    generate_batch,
    load_json,
    prediction_record,
    torch_dtype,
)


MODEL = "llama1b"
METHODS = ("tokmem", "tapmem")
EXPECTED_MODEL_NAME = "Llama-3.2-1B-Instruct"
DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "results"
    / "compositional"
    / "all_methods"
    / "data"
    / "test"
    / "function_calling_test_tools51-100_10calls.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "compositional" / "llama1b_10calls_routing_prefix"
)
ALL_METHODS_RUNS = REPO_ROOT / "results" / "compositional" / "all_methods" / "runs"
PAPER_HEAD_RUNS = (
    REPO_ROOT / "results" / "compositional" / "paper_compositional_head_8gpu" / "runs"
)
CHECKPOINT_RUN_DIRS = {
    "tokmem": tuple(
        ALL_METHODS_RUNS / f"llama1b_tokmem_10calls_trial{trial}_seed42"
        for trial in range(1, 5)
    ),
    "tapmem": tuple(
        PAPER_HEAD_RUNS / f"llama1b_tokmem_eoc_logit_bias_10calls_trial{trial}_seed42"
        for trial in range(1, 4)
    ),
}
DEFAULT_TRIAL_IDS = {
    "tokmem": tuple(range(1, 5)),
    "tapmem": tuple(range(1, 4)),
}
EXPECTED_FLAGS = {
    "tokmem": {
        "use_eoc": False,
        "use_logit_bias": False,
        "use_tool_head_replacement": False,
    },
    "tapmem": {
        "use_eoc": True,
        "use_logit_bias": True,
        "use_tool_head_replacement": False,
        "detach": True,
        "use_logit_train_add": True,
    },
}
PROVENANCE_CODE_PATHS = (
    Path(__file__).resolve(),
    UTILS_DIR / "run_train4_checkpoint_eval_10calls.py",
    UTILS_DIR / "run_error_type_transition_analysis.py",
    COMPOSITIONAL_DIR / "model.py",
    COMPOSITIONAL_DIR / "dataset.py",
    COMPOSITIONAL_DIR / "eval.py",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate aligned per-sample predictions for the Llama-1B 10-call "
            "TokMem/TapMem runs behind TapMem Figure 4, then compare the number "
            "of consecutive correct procedure-routing decisions before the first error."
        )
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output root. Each distinct provenance configuration is written under "
            "<output-dir>/runs/<readable-name>_p<hash>/."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional short label prepended to the provenance run directory name.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated methods: tokmem,tapmem.",
    )
    parser.add_argument(
        "--tokmem-trial-ids",
        default=",".join(str(value) for value in DEFAULT_TRIAL_IDS["tokmem"]),
        help="Comma-separated TokMem trial ids (default: Figure 4 trials 1-4).",
    )
    parser.add_argument(
        "--tapmem-trial-ids",
        default=",".join(str(value) for value in DEFAULT_TRIAL_IDS["tapmem"]),
        help="Comma-separated TapMem trial ids (default: Figure 4 trials 1-3).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix for a smoke run.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Regenerate completed predictions.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate predictions and manifest without writing analysis summaries.",
    )
    parser.add_argument(
        "--generate-tasks",
        default=None,
        help=(
            "Optional comma-separated worker shards such as tokmem:1,tapmem:2. "
            "The full method/trial selection still defines shared provenance. "
            "Requires --generate-only."
        ),
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Require existing predictions and only regenerate analysis summaries.",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=0.005,
        help="Allowed absolute regenerated-vs-archived Tool F1 difference.",
    )
    parser.add_argument(
        "--strict-metric-audit",
        action="store_true",
        help="Fail summary generation when Tool F1 differs by more than the tolerance.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print inputs only.")
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_trial_ids(value, maximum, method):
    try:
        trial_ids = [int(item) for item in split_csv(value)]
    except ValueError as exc:
        raise ValueError(f"{method} trial ids must be integers") from exc
    if not trial_ids:
        raise ValueError(f"{method} trial ids must not be empty")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(f"{method} trial ids must not contain duplicates")
    if any(trial < 1 or trial > maximum for trial in trial_ids):
        raise ValueError(f"{method} trial ids must be between 1 and {maximum}")
    return trial_ids


def select_generation_entries(value, entries):
    if value is None:
        return list(entries)
    requested = split_csv(value)
    if not requested:
        raise ValueError("--generate-tasks must contain at least one method:trial shard")
    if len(set(requested)) != len(requested):
        raise ValueError("--generate-tasks must not contain duplicates")

    available = {
        f"{entry['method']}:{entry['trial']}": entry
        for entry in entries
    }
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            "Unknown or unselected generation tasks: " + ", ".join(unknown)
        )
    return [available[key] for key in requested]


def load_jsonl(path):
    records = []
    seen_indices = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "index" not in record:
                raise ValueError(f"Missing index in {path}:{line_number}")
            index = int(record["index"])
            if index in seen_indices:
                raise ValueError(f"Duplicate index {index} in {path}")
            seen_indices.add(index)
            records.append(record)
    records.sort(key=lambda record: int(record["index"]))
    return records


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else None


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def mean_std(values):
    usable = [value for value in values if value is not None]
    return {
        "mean": mean(usable),
        "std": stdev(usable),
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_fingerprints():
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in PROVENANCE_CODE_PATHS
    }


def build_provenance(args, entries, data_path, data):
    """Build a canonical identity for every input that can change predictions."""
    payload = {
        "schema_version": 1,
        "experiment": "llama1b_10calls_routing_prefix",
        "run_label": sanitize_run_label(args.run_label),
        "data": {
            "sha256": sha256_file(data_path),
            "samples_used": len(data),
            "limit": args.limit,
            "call_length_distribution": call_length_distribution(data),
        },
        "generation": {
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "eval_batch_size": args.eval_batch_size,
            "padding_side": "left",
            "do_sample": False,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        "analysis": {
            "metric_tolerance": args.metric_tolerance,
            "strict_metric_audit": bool(args.strict_metric_audit),
        },
        "checkpoints": [
            {
                "method": entry["method"],
                "trial": int(entry["trial"]),
                "run_name": entry["run_name"],
                "base_model_name": entry["base_model_name"],
                "run_config_sha256": entry["run_config_sha256"],
                "fingerprint": entry["checkpoint_archive_fingerprint"],
            }
            for entry in entries
        ],
        "code_sha256": code_fingerprints(),
        "software_versions": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "id": hashlib.sha256(canonical).hexdigest(),
        "payload": payload,
    }


def sanitize_run_label(value):
    if not value:
        return None
    sanitized = "".join(
        character.lower()
        if character.isalnum() or character in ("-", "_")
        else "-"
        for character in value.strip()
    ).strip("-_")
    if not sanitized:
        raise ValueError("--run-label must contain at least one letter or number")
    return sanitized[:40]


def resolve_run_dir(output_root, args, methods, provenance):
    scope = "full" if args.limit is None else f"smoke{len_from_provenance(provenance)}"
    method_label = "-".join(methods)
    prefix = sanitize_run_label(args.run_label) or "figure4"
    run_name = (
        f"{prefix}_{scope}_{method_label}_p{provenance['id'][:10]}"
    )
    return Path(output_root) / "runs" / run_name


def len_from_provenance(provenance):
    return int(provenance["payload"]["data"]["samples_used"])


def attach_prediction_paths(entries, run_dir, provenance_id):
    for entry in entries:
        run_dir_path = Path(entry["run_dir"])
        entry["prediction_path"] = str(
            prediction_path(run_dir, entry["method"], run_dir_path).resolve()
        )
        entry["provenance_id"] = provenance_id
    return entries


def correct_prefix_length(expected_tools, predicted_tools):
    """Return the number of position-wise correct gold tools before the first mismatch."""
    expected_tools = list(expected_tools or [])
    predicted_tools = list(predicted_tools or [])
    if not expected_tools:
        raise ValueError("expected_tools must contain at least one procedure")

    prefix_length = 0
    for position, expected_tool in enumerate(expected_tools):
        if position >= len(predicted_tools):
            break
        if predicted_tools[position] != expected_tool:
            break
        prefix_length += 1
    return prefix_length


def label_prefix_record(record):
    """Add routing-prefix fields without considering arguments."""
    labeled = dict(record)
    expected_tools = list(record.get("expected_tools") or [])
    predicted_tools = list(record.get("predicted_tools") or [])
    gold_count = len(expected_tools)
    prefix_length = correct_prefix_length(expected_tools, predicted_tools)

    if prefix_length < gold_count:
        first_error_position = prefix_length + 1
        first_error_kind = (
            "missing_prediction"
            if prefix_length >= len(predicted_tools)
            else "wrong_tool"
        )
    elif len(predicted_tools) > gold_count:
        first_error_position = gold_count + 1
        first_error_kind = "extra_prediction"
    else:
        first_error_position = None
        first_error_kind = None

    labeled.update(
        {
            "gold_procedure_count": gold_count,
            "correct_prefix_length": prefix_length,
            "correct_prefix_ratio": prefix_length / gold_count,
            "all_gold_routing_correct": prefix_length == gold_count,
            "routing_sequence_exact": predicted_tools == expected_tools,
            "over_generated_after_gold": (
                prefix_length == gold_count and len(predicted_tools) > gold_count
            ),
            "first_routing_error_position": first_error_position,
            "first_routing_error_kind": first_error_kind,
        }
    )
    return labeled


def analyze_records(records):
    """Compute the requested count=0..J distribution for every gold length J."""
    labeled_records = [label_prefix_record(record) for record in records]
    grouped = {}
    for record in labeled_records:
        grouped.setdefault(record["gold_procedure_count"], []).append(record)

    by_gold_count = {}
    for gold_count in sorted(grouped):
        group = grouped[gold_count]
        counts = Counter(record["correct_prefix_length"] for record in group)
        prefix_counts = {
            str(prefix): counts.get(prefix, 0)
            for prefix in range(gold_count + 1)
        }
        samples = len(group)
        if sum(prefix_counts.values()) != samples:
            raise AssertionError(
                f"Prefix counts for {gold_count} procedures do not sum to {samples}"
            )
        by_gold_count[str(gold_count)] = {
            "gold_procedure_count": gold_count,
            "samples": samples,
            "prefix_counts": prefix_counts,
            "prefix_rates": {
                key: safe_rate(value, samples)
                for key, value in prefix_counts.items()
            },
            "average_prefix_length": mean(
                [record["correct_prefix_length"] for record in group]
            ),
            "average_prefix_ratio": mean(
                [record["correct_prefix_ratio"] for record in group]
            ),
            "all_gold_routing_correct_rate": safe_rate(
                sum(record["all_gold_routing_correct"] for record in group),
                samples,
            ),
            "routing_sequence_exact_rate": safe_rate(
                sum(record["routing_sequence_exact"] for record in group),
                samples,
            ),
            "over_generated_after_gold_count": sum(
                record["over_generated_after_gold"] for record in group
            ),
        }

    samples = len(labeled_records)
    overall = {
        "samples": samples,
        "average_prefix_length": mean(
            [record["correct_prefix_length"] for record in labeled_records]
        ),
        "average_prefix_ratio": mean(
            [record["correct_prefix_ratio"] for record in labeled_records]
        ),
        "all_gold_routing_correct_rate": safe_rate(
            sum(record["all_gold_routing_correct"] for record in labeled_records),
            samples,
        ),
        "routing_sequence_exact_rate": safe_rate(
            sum(record["routing_sequence_exact"] for record in labeled_records),
            samples,
        ),
        "over_generated_after_gold_count": sum(
            record["over_generated_after_gold"] for record in labeled_records
        ),
        "regenerated_tool_f1": mean(
            [float(record.get("tool_f1", 0.0) or 0.0) for record in labeled_records]
        ),
    }
    return labeled_records, {"by_gold_count": by_gold_count, "overall": overall}


def aggregate_trials(trials):
    if not trials:
        raise ValueError("At least one trial is required for aggregation")

    gold_counts = sorted(
        {
            int(gold_count)
            for trial in trials
            for gold_count in trial["metrics"]["by_gold_count"]
        }
    )
    by_gold_count = {}
    for gold_count in gold_counts:
        key = str(gold_count)
        trial_metrics = [trial["metrics"]["by_gold_count"].get(key) for trial in trials]
        if any(metrics is None for metrics in trial_metrics):
            raise ValueError(f"Gold length {gold_count} is missing from at least one trial")
        sample_counts = {metrics["samples"] for metrics in trial_metrics}
        if len(sample_counts) != 1:
            raise ValueError(
                f"Gold length {gold_count} has inconsistent sample counts: {sample_counts}"
            )

        prefix_counts = {}
        prefix_rates = {}
        for prefix in range(gold_count + 1):
            prefix_key = str(prefix)
            prefix_counts[prefix_key] = mean_std(
                [metrics["prefix_counts"][prefix_key] for metrics in trial_metrics]
            )
            prefix_rates[prefix_key] = mean_std(
                [metrics["prefix_rates"][prefix_key] for metrics in trial_metrics]
            )

        by_gold_count[key] = {
            "gold_procedure_count": gold_count,
            "samples_per_trial": sample_counts.pop(),
            "prefix_counts": prefix_counts,
            "prefix_rates": prefix_rates,
            "average_prefix_length": mean_std(
                [metrics["average_prefix_length"] for metrics in trial_metrics]
            ),
            "average_prefix_ratio": mean_std(
                [metrics["average_prefix_ratio"] for metrics in trial_metrics]
            ),
            "all_gold_routing_correct_rate": mean_std(
                [metrics["all_gold_routing_correct_rate"] for metrics in trial_metrics]
            ),
            "routing_sequence_exact_rate": mean_std(
                [metrics["routing_sequence_exact_rate"] for metrics in trial_metrics]
            ),
            "over_generated_after_gold_count": mean_std(
                [metrics["over_generated_after_gold_count"] for metrics in trial_metrics]
            ),
        }

    overall_fields = (
        "average_prefix_length",
        "average_prefix_ratio",
        "all_gold_routing_correct_rate",
        "routing_sequence_exact_rate",
        "over_generated_after_gold_count",
        "regenerated_tool_f1",
    )
    return {
        "trials": len(trials),
        "samples_per_trial": trials[0]["metrics"]["overall"]["samples"],
        "by_gold_count": by_gold_count,
        "overall": {
            field: mean_std(
                [trial["metrics"]["overall"].get(field) for trial in trials]
            )
            for field in overall_fields
        },
    }


def build_comparison(groups):
    if "tokmem" not in groups or "tapmem" not in groups:
        return {}
    tokmem = groups["tokmem"]["aggregate"]
    tapmem = groups["tapmem"]["aggregate"]
    gold_counts = sorted(
        set(tokmem["by_gold_count"]) & set(tapmem["by_gold_count"]),
        key=int,
    )

    by_gold_count = {}
    for key in gold_counts:
        baseline = tokmem["by_gold_count"][key]
        variant = tapmem["by_gold_count"][key]
        prefix_rate_delta = {}
        for prefix in baseline["prefix_rates"]:
            baseline_value = baseline["prefix_rates"][prefix]["mean"]
            variant_value = variant["prefix_rates"][prefix]["mean"]
            prefix_rate_delta[prefix] = variant_value - baseline_value
        by_gold_count[key] = {
            "tapmem_minus_tokmem_prefix_rate": prefix_rate_delta,
            "tapmem_minus_tokmem_average_prefix_length": (
                variant["average_prefix_length"]["mean"]
                - baseline["average_prefix_length"]["mean"]
            ),
            "tapmem_minus_tokmem_average_prefix_ratio": (
                variant["average_prefix_ratio"]["mean"]
                - baseline["average_prefix_ratio"]["mean"]
            ),
            "tapmem_minus_tokmem_all_gold_routing_correct_rate": (
                variant["all_gold_routing_correct_rate"]["mean"]
                - baseline["all_gold_routing_correct_rate"]["mean"]
            ),
        }

    overall_fields = (
        "average_prefix_length",
        "average_prefix_ratio",
        "all_gold_routing_correct_rate",
        "routing_sequence_exact_rate",
    )
    return {
        "by_gold_count": by_gold_count,
        "overall": {
            f"tapmem_minus_tokmem_{field}": (
                tapmem["overall"][field]["mean"]
                - tokmem["overall"][field]["mean"]
            )
            for field in overall_fields
        },
    }


def alignment_signature(record):
    return {
        "index": int(record["index"]),
        "user_input": record.get("user_input"),
        "expected_tools": record.get("expected_tools") or [],
        "expected_calls": record.get("expected_calls") or [],
    }


def validate_alignment(entries_with_records):
    if not entries_with_records:
        return
    reference_entry, reference_records = entries_with_records[0]
    reference = [alignment_signature(record) for record in reference_records]
    for entry, records in entries_with_records[1:]:
        candidate = [alignment_signature(record) for record in records]
        if candidate != reference:
            raise ValueError(
                "Prediction gold examples are not aligned: "
                f"{reference_entry['run_name']} vs {entry['run_name']}"
            )


def archived_eval_metrics(run_dir):
    path = Path(run_dir) / "evaluation_results.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    rounds = payload.get("rounds") or []
    if not rounds:
        return {}
    return dict(rounds[-1].get("eval_results") or {})


def validate_run_config(method, run_config, run_config_path):
    args = run_config.get("args", {})
    rounds = run_config.get("rounds") or []
    errors = []

    if Path(args.get("model_name", "")).name != EXPECTED_MODEL_NAME:
        errors.append(f"model_name={args.get('model_name')!r}")
    if bool(args.get("use_lora", False)):
        errors.append("use_lora must be false")
    if int(args.get("train_max_function_calls", -1)) != 10:
        errors.append(
            f"train_max_function_calls={args.get('train_max_function_calls')!r}"
        )
    if int(args.get("test_max_function_calls", -1)) != 10:
        errors.append(
            f"test_max_function_calls={args.get('test_max_function_calls')!r}"
        )
    if len(rounds) != 1 or rounds[0].get("tools") != "51-100":
        errors.append(f"rounds={rounds!r}")
    for flag, expected in EXPECTED_FLAGS[method].items():
        actual = bool(args.get(flag, False))
        if actual != expected:
            errors.append(f"{flag}={actual}, expected {expected}")

    if errors:
        raise ValueError(
            f"Invalid {method} 1B 10-call run {run_config_path}: {'; '.join(errors)}"
        )


def prediction_path(output_dir, method, run_dir):
    return (
        output_dir
        / "predictions"
        / method
        / f"{run_dir.name}_10calls_predictions.jsonl"
    )


def select_entries(methods, trial_ids_by_method):
    entries = []
    for method in methods:
        run_dirs = CHECKPOINT_RUN_DIRS[method]
        for trial in trial_ids_by_method[method]:
            run_dir = run_dirs[trial - 1]
            run_config_path = run_dir / "run_config.json"
            if not run_config_path.exists():
                raise FileNotFoundError(f"Run config not found: {run_config_path}")
            run_config = load_json(run_config_path)
            validate_run_config(method, run_config, run_config_path)
            checkpoint_path = run_dir / final_checkpoint_name(run_config)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            entries.append(
                {
                    "model": MODEL,
                    "method": method,
                    "trial": trial,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "run_config_path": str(run_config_path.resolve()),
                    "run_config_sha256": sha256_file(run_config_path),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "base_model_name": run_config["args"]["model_name"],
                    "checkpoint_archive_fingerprint": checkpoint_archive_fingerprint(
                        checkpoint_path
                    ),
                    "archived_eval_metrics": archived_eval_metrics(run_dir),
                }
            )
    return entries


def call_length_distribution(data):
    counts = Counter(len(item.get("tools") or []) for item in data)
    if any(gold_count <= 0 for gold_count in counts):
        raise ValueError("Every test sample must contain at least one gold tool")
    return {str(gold_count): counts[gold_count] for gold_count in sorted(counts)}


def expected_record_provenance(entry, provenance):
    return {
        "provenance_id": provenance["id"],
        "data_sha256": provenance["payload"]["data"]["sha256"],
        "checkpoint_fingerprint": entry["checkpoint_archive_fingerprint"]["digest"],
        "method": entry["method"],
        "trial": int(entry["trial"]),
        "run_name": entry["run_name"],
    }


def validate_prediction_provenance(path, entry, provenance):
    expected = expected_record_provenance(entry, provenance)
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            actual = {
                key: record.get(key)
                for key in expected
            }
            if actual != expected:
                raise ValueError(
                    f"Prediction provenance mismatch in {path}:{line_number}: "
                    f"found {actual}, expected {expected}. Use the matching run "
                    "directory or regenerate with --force."
                )


def validate_prediction_artifact(path, expected_samples, entry, provenance):
    validate_prediction_file(path, expected_samples)
    validate_prediction_provenance(path, entry, provenance)


def all_predictions_complete(entries, expected_samples, provenance):
    for entry in entries:
        path = Path(entry["prediction_path"])
        if not path.exists():
            return False
        validate_prediction_artifact(
            path,
            expected_samples,
            entry,
            provenance,
        )
    return True


def run_prediction(args, entry, data, provenance):
    import torch
    from transformers import AutoTokenizer

    out_path = Path(entry["prediction_path"])
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force:
        if out_path.exists():
            out_path.unlink()
        if partial_path.exists():
            partial_path.unlink()

    if partial_path.exists():
        validate_prediction_provenance(partial_path, entry, provenance)
    start_index = partial_record_count(partial_path)
    if start_index > len(data):
        raise ValueError(f"Partial prediction has too many records: {partial_path}")
    if start_index not in (0, len(data)) and start_index % args.eval_batch_size:
        raise ValueError(
            f"Partial prediction stops mid-batch at {start_index}: {partial_path}; "
            "rerun with --force to preserve batch-aligned decoding"
        )

    run_config = load_json(entry["run_config_path"])
    checkpoint = torch.load(
        entry["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        run_config["args"]["model_name"],
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_model(
        run_config,
        checkpoint,
        tokenizer,
        args.device,
        torch_dtype(args.dtype),
    )
    del checkpoint
    gc.collect()
    candidate_tools = list(getattr(model, "tool_names", []))

    mode = "a" if start_index else "w"
    with open(partial_path, mode, encoding="utf-8") as handle:
        for start in range(start_index, len(data), args.eval_batch_size):
            batch = data[start : start + args.eval_batch_size]
            results = generate_batch(
                model,
                tokenizer,
                batch,
                args.device,
                args.max_new_tokens,
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Generation returned {len(results)} results for "
                    f"{len(batch)} examples at index {start}"
                )
            for offset, (sample, result) in enumerate(zip(batch, results)):
                index = start + offset
                record = prediction_record(
                    index,
                    sample,
                    result,
                    tokenizer,
                    entry["method"],
                    candidate_tools,
                )
                record.update(
                    {
                        "model": MODEL,
                        "trial": entry["trial"],
                        "run_name": entry["run_name"],
                        "provenance_id": provenance["id"],
                        "data_sha256": provenance["payload"]["data"]["sha256"],
                        "checkpoint_fingerprint": entry[
                            "checkpoint_archive_fingerprint"
                        ]["digest"],
                    }
                )
                record = label_prefix_record(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            completed = start + len(batch)
            if args.progress_every > 0 and (
                completed == len(data)
                or completed % args.progress_every < args.eval_batch_size
            ):
                print(
                    f"Generated {completed}/{len(data)} for {entry['run_name']}",
                    flush=True,
                )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    validate_prediction_artifact(
        partial_path,
        len(data),
        entry,
        provenance,
    )
    partial_path.replace(out_path)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def analyze_entries(
    entries,
    output_dir,
    metric_tolerance=0.005,
    strict_metric_audit=False,
    provenance=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_with_records = []
    for entry in entries:
        records = load_jsonl(entry["prediction_path"])
        entries_with_records.append((entry, records))
    validate_alignment(entries_with_records)

    groups = {}
    per_trial = []
    metric_audit_failures = []
    for entry, records in entries_with_records:
        _labeled, metrics = analyze_records(records)
        archived_tool_f1 = entry.get("archived_eval_metrics", {}).get(
            "avg_tool_f1_score"
        )
        regenerated_tool_f1 = metrics["overall"]["regenerated_tool_f1"]
        tool_f1_delta = (
            regenerated_tool_f1 - archived_tool_f1
            if archived_tool_f1 is not None and regenerated_tool_f1 is not None
            else None
        )
        audit_within_tolerance = (
            tool_f1_delta is None or abs(tool_f1_delta) <= metric_tolerance
        )
        if not audit_within_tolerance:
            metric_audit_failures.append(
                f"{entry['run_name']}: delta={tool_f1_delta:.6f}"
            )

        trial_payload = {
            "model": MODEL,
            "method": entry["method"],
            "trial": int(entry["trial"]),
            "run_name": entry["run_name"],
            "checkpoint_path": entry["checkpoint_path"],
            "prediction_path": entry["prediction_path"],
            "archived_tool_f1": archived_tool_f1,
            "regenerated_tool_f1": regenerated_tool_f1,
            "regenerated_minus_archived_tool_f1": tool_f1_delta,
            "metric_audit_within_tolerance": audit_within_tolerance,
            "metrics": metrics,
        }
        per_trial.append(trial_payload)
        groups.setdefault(entry["method"], {"trials": []})["trials"].append(
            trial_payload
        )

    if strict_metric_audit and metric_audit_failures:
        raise ValueError(
            "Regenerated Tool F1 does not match archived metrics within tolerance: "
            + "; ".join(metric_audit_failures)
        )

    for payload in groups.values():
        payload["trials"].sort(key=lambda item: item["trial"])
        payload["aggregate"] = aggregate_trials(payload["trials"])

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "provenance_id": provenance["id"] if provenance else None,
        "definition": (
            "correct_prefix_length is the number of position-wise correct gold "
            "procedure tools before the first wrong or missing prediction; arguments "
            "are ignored, and extra predictions after all gold tools do not reduce "
            "the prefix length."
        ),
        "metric_tolerance": metric_tolerance,
        "metric_audit_failures": metric_audit_failures,
        "groups": groups,
        "comparison": build_comparison(groups),
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    per_trial_path = output_dir / "per_trial.jsonl"
    write_jsonl(
        per_trial_path,
        sorted(per_trial, key=lambda item: (item["method"], item["trial"])),
    )
    summary_md = output_dir / "summary.md"
    write_summary_md(summary_md, summary)
    return summary_json, per_trial_path, summary_md


def fmt_mean_std(stat, digits=3, percentage=False):
    if stat is None or stat.get("mean") is None:
        return "N/A"
    scale = 100.0 if percentage else 1.0
    mean_value = stat["mean"] * scale
    std_value = stat.get("std")
    if std_value is None:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {std_value * scale:.{digits}f}"


def fmt_count_rate(count_stat, rate_stat):
    return (
        f"{fmt_mean_std(count_stat, digits=2)} "
        f"({fmt_mean_std(rate_stat, digits=1, percentage=True)}%)"
    )


def write_summary_md(path, summary):
    groups = summary["groups"]
    comparison = summary.get("comparison") or {}
    lines = [
        "# Llama-1B 10-call Procedure-routing Correct-prefix Analysis",
        "",
        "- methods: no-adaptation TokMem vs no-adaptation TapMem (EOC + TCRA/logit bias)",
        "- count definition: position-wise consecutive correct procedure tools before the first error",
        "- wrong tools and missing predictions stop the prefix; argument errors are ignored",
        "- if all gold tools are correct, `count=J`; extra predictions are tracked separately",
        "- aggregation: mean and sample standard deviation across trials, never pooled across methods",
        "",
    ]
    audit_failures = summary.get("metric_audit_failures") or []
    if audit_failures:
        lines.extend(
            [
                "> [!WARNING]",
                "> Regenerated Tool F1 differs from at least one archived evaluation by more "
                "than the configured tolerance. See the metric audit table before interpreting "
                "the prefix comparison.",
                "",
            ]
        )

    if "tokmem" in groups and "tapmem" in groups:
        lines.extend(
            [
                "## Average prefix comparison",
                "",
                "| Gold procedures | Samples/trial | TokMem avg prefix | TapMem avg prefix | Delta | TokMem prefix ratio | TapMem prefix ratio | Delta | TokMem all-gold correct | TapMem all-gold correct | Delta |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        tokmem = groups["tokmem"]["aggregate"]
        tapmem = groups["tapmem"]["aggregate"]
        for key in sorted(comparison["by_gold_count"], key=int):
            baseline = tokmem["by_gold_count"][key]
            variant = tapmem["by_gold_count"][key]
            delta = comparison["by_gold_count"][key]
            lines.append(
                "| "
                + " | ".join(
                    [
                        key,
                        str(baseline["samples_per_trial"]),
                        fmt_mean_std(baseline["average_prefix_length"]),
                        fmt_mean_std(variant["average_prefix_length"]),
                        f"{delta['tapmem_minus_tokmem_average_prefix_length']:+.3f}",
                        fmt_mean_std(
                            baseline["average_prefix_ratio"],
                            percentage=True,
                        )
                        + "%",
                        fmt_mean_std(
                            variant["average_prefix_ratio"],
                            percentage=True,
                        )
                        + "%",
                        f"{100 * delta['tapmem_minus_tokmem_average_prefix_ratio']:+.2f} pp",
                        fmt_mean_std(
                            baseline["all_gold_routing_correct_rate"],
                            percentage=True,
                        )
                        + "%",
                        fmt_mean_std(
                            variant["all_gold_routing_correct_rate"],
                            percentage=True,
                        )
                        + "%",
                        f"{100 * delta['tapmem_minus_tokmem_all_gold_routing_correct_rate']:+.2f} pp",
                    ]
                )
                + " |"
            )
        lines.append("")

    all_gold_counts = sorted(
        {
            int(key)
            for payload in groups.values()
            for key in payload["aggregate"]["by_gold_count"]
        }
    )
    for gold_count in all_gold_counts:
        key = str(gold_count)
        lines.extend(
            [
                f"## Gold procedure count = {gold_count}",
                "",
                "| Method | Trials | "
                + " | ".join(f"count={prefix}" for prefix in range(gold_count + 1))
                + " | Avg prefix | Prefix ratio | All-gold correct | Exact length/sequence | Over-generated after gold |",
                "| --- | ---: | "
                + " | ".join("---:" for _ in range(gold_count + 1))
                + " | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            if method not in groups or key not in groups[method]["aggregate"]["by_gold_count"]:
                continue
            aggregate = groups[method]["aggregate"]
            metrics = aggregate["by_gold_count"][key]
            cells = [
                method,
                str(aggregate["trials"]),
                *(
                    fmt_count_rate(
                        metrics["prefix_counts"][str(prefix)],
                        metrics["prefix_rates"][str(prefix)],
                    )
                    for prefix in range(gold_count + 1)
                ),
                fmt_mean_std(metrics["average_prefix_length"]),
                fmt_mean_std(metrics["average_prefix_ratio"], percentage=True)
                + "%",
                fmt_mean_std(
                    metrics["all_gold_routing_correct_rate"],
                    percentage=True,
                )
                + "%",
                fmt_mean_std(
                    metrics["routing_sequence_exact_rate"],
                    percentage=True,
                )
                + "%",
                fmt_mean_std(metrics["over_generated_after_gold_count"], digits=2),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if (
            "tokmem" in groups
            and "tapmem" in groups
            and key in comparison.get("by_gold_count", {})
        ):
            baseline = groups["tokmem"]["aggregate"]["by_gold_count"][key]
            variant = groups["tapmem"]["aggregate"]["by_gold_count"][key]
            delta = comparison["by_gold_count"][key]
            delta_cells = [
                "TapMem - TokMem",
                "-",
                *(
                    f"{100 * delta['tapmem_minus_tokmem_prefix_rate'][str(prefix)]:+.2f} pp"
                    for prefix in range(gold_count + 1)
                ),
                f"{delta['tapmem_minus_tokmem_average_prefix_length']:+.3f}",
                f"{100 * delta['tapmem_minus_tokmem_average_prefix_ratio']:+.2f} pp",
                (
                    f"{100 * delta['tapmem_minus_tokmem_all_gold_routing_correct_rate']:+.2f} pp"
                ),
                (
                    f"{100 * (variant['routing_sequence_exact_rate']['mean'] - baseline['routing_sequence_exact_rate']['mean']):+.2f} pp"
                ),
                (
                    f"{variant['over_generated_after_gold_count']['mean'] - baseline['over_generated_after_gold_count']['mean']:+.2f}"
                ),
            ]
            lines.append("| " + " | ".join(delta_cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## Overall averages",
            "",
            "| Method | Trials | Samples/trial | Avg prefix | Avg prefix ratio | All-gold correct | Exact length/sequence |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        if method not in groups:
            continue
        aggregate = groups[method]["aggregate"]
        overall = aggregate["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(aggregate["trials"]),
                    str(aggregate["samples_per_trial"]),
                    fmt_mean_std(overall["average_prefix_length"]),
                    fmt_mean_std(
                        overall["average_prefix_ratio"],
                        percentage=True,
                    )
                    + "%",
                    fmt_mean_std(
                        overall["all_gold_routing_correct_rate"],
                        percentage=True,
                    )
                    + "%",
                    fmt_mean_std(
                        overall["routing_sequence_exact_rate"],
                        percentage=True,
                    )
                    + "%",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.extend(
        [
            "## Regeneration metric audit",
            "",
            "| Method | Trial | Archived Tool F1 | Regenerated Tool F1 | Delta | Within tolerance |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for method in METHODS:
        for trial in groups.get(method, {}).get("trials", []):
            archived = trial["archived_tool_f1"]
            regenerated = trial["regenerated_tool_f1"]
            delta = trial["regenerated_minus_archived_tool_f1"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        str(trial["trial"]),
                        "N/A" if archived is None else f"{archived:.4f}",
                        "N/A" if regenerated is None else f"{regenerated:.4f}",
                        "N/A" if delta is None else f"{delta:+.4f}",
                        "yes" if trial["metric_audit_within_tolerance"] else "no",
                    ]
                )
                + " |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(args, entries, data_path, output_dir, data, provenance):
    fingerprint_groups = {}
    for entry in entries:
        fingerprint = entry["checkpoint_archive_fingerprint"]
        key = (entry["method"], fingerprint["kind"], fingerprint["digest"])
        fingerprint_groups.setdefault(key, []).append(int(entry["trial"]))
    duplicate_checkpoint_groups = [
        {
            "method": method,
            "trials": sorted(trials),
            "fingerprint_kind": kind,
            "fingerprint": digest,
        }
        for (method, kind, digest), trials in fingerprint_groups.items()
        if len(trials) > 1
    ]

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "provenance": provenance,
        "experiment": "Llama-1B 10-call procedure-routing correct-prefix analysis",
        "checkpoint_policy": (
            "TapMem Figure 4 no-adaptation 10-call runs: TokMem trials 1-4 "
            "from all_methods and TapMem trials 1-3 from paper_compositional_head_8gpu"
        ),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "samples": len(data),
        "call_length_distribution": call_length_distribution(data),
        "limit": args.limit,
        "generation": {
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "eval_batch_size": args.eval_batch_size,
            "padding_side": "left",
            "do_sample": False,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        "metric_tolerance": args.metric_tolerance,
        "duplicate_checkpoint_groups": duplicate_checkpoint_groups,
        "entries": entries,
    }
    path = output_dir / "manifest.json"
    lock_path = output_dir / ".manifest.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                existing = load_json(path)
                if existing.get("provenance") != provenance:
                    raise ValueError(
                        f"Existing run manifest has different provenance: {path}. "
                        "Use its matching configuration or choose a different --run-label."
                    )
                return path, existing
            temporary_path = output_dir / f".manifest.{os.getpid()}.tmp"
            temporary_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
            return path, manifest
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def update_run_index(output_root, run_dir, manifest, status):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "run_index.json"
    lock_path = output_root / ".run_index.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if index_path.exists():
                index = load_json(index_path)
                if index.get("schema_version") != 1 or not isinstance(
                    index.get("runs"), list
                ):
                    raise ValueError(f"Invalid run index: {index_path}")
            else:
                index = {"schema_version": 1, "runs": []}

            provenance = manifest["provenance"]
            run_id = provenance["id"]
            now = datetime.now().isoformat(timespec="seconds")
            existing = next(
                (
                    item
                    for item in index["runs"]
                    if item.get("provenance_id") == run_id
                ),
                None,
            )
            record = {
                "provenance_id": run_id,
                "short_id": run_id[:10],
                "run_dir": str(Path(run_dir).resolve()),
                "created_at": (
                    existing.get("created_at")
                    if existing
                    else manifest.get("created_at", now)
                ),
                "updated_at": now,
                "status": status,
                "samples": manifest["samples"],
                "scope": (
                    "full"
                    if manifest.get("limit") is None
                    else f"smoke{manifest['samples']}"
                ),
                "methods": sorted(
                    {
                        entry["method"]
                        for entry in manifest.get("entries", [])
                    }
                ),
                "trials": {
                    method: sorted(
                        int(entry["trial"])
                        for entry in manifest.get("entries", [])
                        if entry["method"] == method
                    )
                    for method in sorted(
                        {
                            entry["method"]
                            for entry in manifest.get("entries", [])
                        }
                    )
                },
            }
            index["runs"] = [
                item
                for item in index["runs"]
                if item.get("provenance_id") != run_id
            ]
            index["runs"].append(record)
            index["runs"].sort(
                key=lambda item: (
                    item.get("updated_at", ""),
                    item.get("provenance_id", ""),
                ),
                reverse=True,
            )

            temporary_path = output_root / f".run_index.{os.getpid()}.tmp"
            temporary_path.write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(index_path)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return index_path


def main():
    args = parse_args()
    if args.generate_only and args.summarize_only:
        raise SystemExit("--generate-only and --summarize-only are mutually exclusive")
    if args.generate_tasks and not args.generate_only:
        raise SystemExit("--generate-tasks requires --generate-only")
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.metric_tolerance < 0:
        raise SystemExit("--metric-tolerance must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    methods = split_csv(args.methods)
    invalid_methods = sorted(set(methods) - set(METHODS))
    if invalid_methods:
        raise SystemExit(f"Unknown methods: {', '.join(invalid_methods)}")
    if not methods:
        raise SystemExit("--methods must select at least one method")

    try:
        trial_ids_by_method = {
            "tokmem": parse_trial_ids(
                args.tokmem_trial_ids,
                len(CHECKPOINT_RUN_DIRS["tokmem"]),
                "tokmem",
            ),
            "tapmem": parse_trial_ids(
                args.tapmem_trial_ids,
                len(CHECKPOINT_RUN_DIRS["tapmem"]),
                "tapmem",
            ),
        }
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    data_path = Path(args.data_path)
    output_root = Path(args.output_dir)
    if not data_path.exists():
        raise SystemExit(f"10-call test split not found: {data_path}")
    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    entries = select_entries(methods, trial_ids_by_method)
    provenance = build_provenance(args, entries, data_path, data)
    run_dir = resolve_run_dir(output_root, args, methods, provenance)
    attach_prediction_paths(entries, run_dir, provenance["id"])
    try:
        generation_entries = select_generation_entries(
            args.generate_tasks,
            entries,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Data: {data_path} ({len(data)} samples, "
        f"sha256={sha256_file(data_path)})"
    )
    print(f"Call lengths: {call_length_distribution(data)}")
    print(f"Output root: {output_root}")
    print(f"Run directory: {run_dir}")
    print(f"Provenance: {provenance['id']}")
    if args.generate_tasks:
        print(
            "Worker tasks: "
            + ",".join(
                f"{entry['method']}:{entry['trial']}"
                for entry in generation_entries
            )
        )
    for entry in entries:
        print(
            f"{entry['method']}/trial{entry['trial']}: "
            f"{entry['checkpoint_path']} -> {entry['prediction_path']}"
        )

    if args.dry_run:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest = write_manifest(
        args,
        entries,
        data_path,
        run_dir,
        data,
        provenance,
    )
    run_index_path = update_run_index(
        output_root,
        run_dir,
        manifest,
        status="manifested",
    )

    if not args.summarize_only:
        for entry in generation_entries:
            out_path = Path(entry["prediction_path"])
            if out_path.exists() and not args.force:
                validate_prediction_artifact(
                    out_path,
                    len(data),
                    entry,
                    provenance,
                )
                print(f"Skipping complete predictions: {out_path}")
                continue
            print(
                f"Running {entry['method']}/trial{entry['trial']}: "
                f"{entry['run_name']}",
                flush=True,
            )
            run_prediction(args, entry, data, provenance)
    else:
        for entry in entries:
            validate_prediction_artifact(
                Path(entry["prediction_path"]),
                len(data),
                entry,
                provenance,
            )

    predictions_complete = all_predictions_complete(
        entries,
        len(data),
        provenance,
    )
    update_run_index(
        output_root,
        run_dir,
        manifest,
        status=(
            "predictions_complete"
            if predictions_complete
            else "predictions_partial"
        ),
    )

    print(f"Wrote manifest: {manifest_path}")
    print(f"Updated run index: {run_index_path}")
    if args.generate_only:
        return
    if not predictions_complete:
        raise RuntimeError(
            "Not all selected prediction artifacts are complete; "
            "finish the worker shards before summarizing."
        )

    summary_json, per_trial_path, summary_md = analyze_entries(
        entries,
        run_dir,
        metric_tolerance=args.metric_tolerance,
        strict_metric_audit=args.strict_metric_audit,
        provenance=provenance,
    )
    update_run_index(
        output_root,
        run_dir,
        manifest,
        status="summary_complete",
    )
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote per-trial JSONL: {per_trial_path}")
    print(f"Wrote summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
