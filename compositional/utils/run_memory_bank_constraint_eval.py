#!/usr/bin/env python3
"""Evaluate paper TokMem/EOC-only/TapMem checkpoints with bank constraints."""

import argparse
import gc
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
UTILS_DIR = COMPOSITIONAL_DIR / "utils"
for import_dir in (COMPOSITIONAL_DIR, UTILS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from run_error_type_transition_analysis import (  # noqa: E402
    CHECKPOINT_RUN_DIRS,
    DEFAULT_EVAL_BATCH_SIZES,
    checkpoint_archive_fingerprint,
    parse_trial_ids,
    sha256_file,
    split_csv,
    validate_run_config,
)
from run_train4_checkpoint_eval_10calls import (  # noqa: E402
    build_model,
    final_checkpoint_name,
    generate_batch,
    load_json,
    prediction_record,
    torch_dtype,
)


DEFAULT_DATA_PATH = (
    COMPOSITIONAL_DIR / "data" / "test" / "function_calling_test_tools51-100_4calls.json"
)
DEFAULT_OUTPUT_DIR = (
    COMPOSITIONAL_DIR / "rebuttal" / "results" / "memory_bank_constraint"
)
MODELS = ("llama1b", "llama3b", "llama8b")
METHODS = (
    "tokmem_bank_constraint",
    "eoc_only_bank_constraint",
    "tapmem_bank_constraint",
)
BASE_METHOD = {
    "tokmem_bank_constraint": "tokmem",
    "eoc_only_bank_constraint": "eoc_only",
    "tapmem_bank_constraint": "tapmem",
}
METHOD_LABELS = {
    "tokmem_bank_constraint": "TokMem + bank constraint",
    "eoc_only_bank_constraint": "EOC-only + bank constraint",
    "tapmem_bank_constraint": "TapMem + bank constraint",
}
METRIC_FIELDS = (
    ("tool_f1", "Tool F1"),
    ("argument_f1", "Argument F1"),
    ("tool_sequence_exact", "Tool Sequence Exact"),
    ("call_exact", "Call Exact"),
    ("triggers_per_sample", "Triggers / Sample"),
    ("transition_triggers_per_sample", "Transition Triggers / Sample"),
    ("changed_trigger_rate", "Changed Trigger Rate"),
    ("mean_trigger_memory_mass", "Mean Trigger Memory Mass"),
    ("mean_trigger_normalized_entropy", "Mean Trigger Normalized Entropy"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run memory-bank constrained decoding on the paper-table TokMem, matched "
            "EOC-only, and TapMem 4-call checkpoints."
        )
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial-ids", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--memory-bank-probability-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def eval_batch_size(args, model):
    return args.eval_batch_size or DEFAULT_EVAL_BATCH_SIZES[model]


def prediction_path(output_dir, entry):
    return (
        output_dir
        / "predictions"
        / entry["model"]
        / entry["method"]
        / f"trial{entry['trial']}.jsonl"
    )


def select_entries(models, methods, trial_ids, output_dir):
    entries = []
    for model in models:
        for method in methods:
            base_method = BASE_METHOD[method]
            run_dirs = CHECKPOINT_RUN_DIRS[(model, base_method)]
            for trial in trial_ids:
                run_dir = run_dirs[trial - 1]
                run_config_path = run_dir / "run_config.json"
                if not run_config_path.exists():
                    raise FileNotFoundError(f"Run config not found: {run_config_path}")
                run_config = load_json(run_config_path)
                validate_run_config(model, base_method, run_config, run_config_path)
                checkpoint_path = run_dir / final_checkpoint_name(run_config)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                entry = {
                    "model": model,
                    "method": method,
                    "base_method": base_method,
                    "trial": trial,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "run_config_path": str(run_config_path.resolve()),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_archive_fingerprint": checkpoint_archive_fingerprint(checkpoint_path),
                }
                entry["prediction_path"] = str(prediction_path(output_dir, entry).resolve())
                entries.append(entry)
    return entries


def validate_prediction_file(
    path,
    expected_count,
    expected_method=None,
    expected_mode=None,
    expected_threshold=None,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    indices = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            indices.append(int(record["index"]))
            if expected_method is not None and record.get("method") != expected_method:
                raise ValueError(
                    f"Prediction method in {path}:{line_number} is "
                    f"{record.get('method')!r}, expected {expected_method!r}"
                )
            if expected_mode is not None or expected_threshold is not None:
                diagnostics = record.get("memory_bank_constraint") or {}
                if diagnostics.get("enabled") is not True:
                    raise ValueError(
                        f"Memory-bank constraint diagnostics are missing in "
                        f"{path}:{line_number}"
                    )
                if expected_mode is not None and diagnostics.get("mode") != expected_mode:
                    raise ValueError(
                        f"Constraint mode in {path}:{line_number} is "
                        f"{diagnostics.get('mode')!r}, expected {expected_mode!r}"
                    )
                if expected_threshold is not None:
                    actual_threshold = diagnostics.get("probability_threshold")
                    if actual_threshold is None or abs(
                        float(actual_threshold) - float(expected_threshold)
                    ) > 1e-12:
                        raise ValueError(
                            f"Constraint threshold in {path}:{line_number} is "
                            f"{actual_threshold!r}, expected {expected_threshold!r}"
                        )
    expected_indices = list(range(expected_count))
    if indices != expected_indices:
        raise ValueError(
            f"Prediction indices in {path} are not the continuous range 0..{expected_count - 1}"
        )


def validate_entry_prediction(path, expected_count, entry, threshold):
    expected_mode = (
        "eoc_boundary"
        if entry["base_method"] in {"eoc_only", "tapmem"}
        else "probability_threshold"
    )
    validate_prediction_file(
        path,
        expected_count,
        expected_method=entry["method"],
        expected_mode=expected_mode,
        expected_threshold=threshold,
    )


def run_prediction(args, entry, data):
    import torch
    from transformers import AutoTokenizer

    run_config = load_json(entry["run_config_path"])
    checkpoint = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
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
    path = Path(entry["prediction_path"])
    partial_path = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = eval_batch_size(args, entry["model"])
    use_eoc = entry["base_method"] in {"eoc_only", "tapmem"}
    with partial_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            results = generate_batch(
                model,
                tokenizer,
                batch,
                args.device,
                args.max_new_tokens,
                use_memory_bank_constraint=True,
                memory_bank_probability_threshold=args.memory_bank_probability_threshold,
                use_eoc=use_eoc,
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Generation returned {len(results)} results for {len(batch)} samples"
                )
            for offset, (sample, result) in enumerate(zip(batch, results)):
                record = prediction_record(
                    start + offset,
                    sample,
                    result,
                    tokenizer,
                    entry["method"],
                    candidate_tools,
                )
                record.update(
                    {
                        "model": entry["model"],
                        "trial": entry["trial"],
                        "run_name": entry["run_name"],
                    }
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = start + len(batch)
            if args.progress_every > 0 and (
                completed == len(data) or completed % args.progress_every < batch_size
            ):
                print(
                    f"Generated {completed}/{len(data)} for "
                    f"{entry['model']}/{entry['method']}/trial{entry['trial']}",
                    flush=True,
                )

    validate_entry_prediction(
        partial_path,
        len(data),
        entry,
        args.memory_bank_probability_threshold,
    )
    partial_path.replace(path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_prediction_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize_prediction_file(path):
    records = load_prediction_records(path)
    sample_count = len(records)
    trigger_count = 0
    transition_trigger_count = 0
    changed_count = 0
    memory_mass_sum = 0.0
    entropy_sum = 0.0
    for record in records:
        diagnostics = record.get("memory_bank_constraint") or {}
        sample_triggers = int(diagnostics.get("trigger_count", 0))
        trigger_count += sample_triggers
        transition_trigger_count += int(diagnostics.get("transition_trigger_count", 0))
        changed_count += int(diagnostics.get("changed_token_count", 0))
        memory_mass_sum += float(diagnostics.get("mean_trigger_memory_mass", 0.0)) * sample_triggers
        entropy_sum += (
            float(diagnostics.get("mean_trigger_normalized_entropy", 0.0)) * sample_triggers
        )

    return {
        "samples": sample_count,
        "tool_f1": mean([float(record.get("tool_f1", 0.0)) for record in records]),
        "argument_f1": mean([float(record.get("f1", 0.0)) for record in records]),
        "tool_sequence_exact": mean(
            [1.0 if record.get("tool_sequence_exact") else 0.0 for record in records]
        ),
        "call_exact": mean([1.0 if record.get("call_exact") else 0.0 for record in records]),
        "trigger_count": trigger_count,
        "transition_trigger_count": transition_trigger_count,
        "changed_count": changed_count,
        "triggers_per_sample": trigger_count / sample_count if sample_count else 0.0,
        "transition_triggers_per_sample": (
            transition_trigger_count / sample_count if sample_count else 0.0
        ),
        "changed_trigger_rate": changed_count / trigger_count if trigger_count else 0.0,
        "mean_trigger_memory_mass": memory_mass_sum / trigger_count if trigger_count else 0.0,
        "mean_trigger_normalized_entropy": entropy_sum / trigger_count if trigger_count else 0.0,
    }


def build_summary(entries):
    groups = {}
    per_trial = []
    for entry in entries:
        metrics = summarize_prediction_file(entry["prediction_path"])
        trial_payload = {
            "model": entry["model"],
            "method": entry["method"],
            "trial": entry["trial"],
            "run_name": entry["run_name"],
            "checkpoint_path": entry["checkpoint_path"],
            "prediction_path": entry["prediction_path"],
            "metrics": metrics,
        }
        per_trial.append(trial_payload)
        groups.setdefault(entry["model"], {}).setdefault(entry["method"], []).append(
            trial_payload
        )

    summary_groups = {}
    for model, model_groups in groups.items():
        summary_groups[model] = {}
        for method, trials in model_groups.items():
            trials.sort(key=lambda item: item["trial"])
            aggregate = {"trials": len(trials)}
            for metric, _label in METRIC_FIELDS:
                values = [float(trial["metrics"][metric]) for trial in trials]
                aggregate[metric] = mean(values)
                aggregate[f"{metric}_std"] = stdev(values)
            summary_groups[model][method] = {"trials": trials, "aggregate": aggregate}
    return summary_groups, sorted(
        per_trial,
        key=lambda item: (item["model"], item["method"], item["trial"]),
    )


def fmt(value):
    return f"{float(value):.4f}"


def write_outputs(args, entries, output_dir, data_path, sample_count):
    fingerprint_groups = {}
    for entry in entries:
        fingerprint = entry["checkpoint_archive_fingerprint"]
        key = (
            entry["model"],
            entry["method"],
            fingerprint["kind"],
            fingerprint["digest"],
        )
        fingerprint_groups.setdefault(key, []).append(entry["trial"])
    duplicate_groups = [
        {
            "model": model,
            "method": method,
            "trials": sorted(trials),
            "fingerprint_kind": kind,
            "fingerprint": digest,
        }
        for (model, method, kind, digest), trials in fingerprint_groups.items()
        if len(trials) > 1
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "samples": sample_count,
        "checkpoint_policy": (
            "paper-table TokMem, matched EOC-only, and TapMem 4-call checkpoints"
        ),
        "memory_bank_probability_threshold": args.memory_bank_probability_threshold,
        "ending_token_policy": "tokenizer.eos_token_id is included with all memory tokens",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "duplicate_checkpoint_groups": duplicate_groups,
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    groups, per_trial = build_summary(entries)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path.resolve()),
        "groups": groups,
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_trial_path = output_dir / "per_trial.jsonl"
    with per_trial_path.open("w", encoding="utf-8") as handle:
        for item in per_trial:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# Memory-bank Constraint Evaluation",
        "",
        f"- test split: `{data_path.resolve()}`",
        f"- memory-bank probability threshold: `{args.memory_bank_probability_threshold}`",
        "- constrained candidates: all memory tokens plus `tokenizer.eos_token_id`",
        "- TokMem trigger: full-vocabulary normalized memory-token mass reaches the threshold",
        "- EOC-only/TapMem trigger: assistant start or a generated EOC boundary; the threshold is diagnostic only",
        "- TapMem order: apply TCRA logit bias first, then select from the constrained candidates",
        "- decoding: greedy (`do_sample=False`)",
    ]
    if duplicate_groups:
        caveat = "; ".join(
            f"{group['model']}/{group['method']} trials "
            + ",".join(str(trial) for trial in group["trials"])
            for group in duplicate_groups
        )
        lines.append(f"- duplicate checkpoint caveat: {caveat}")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Model | Method | Trials | Tool F1 | Argument F1 | Tool Sequence Exact | Call Exact | Triggers / Sample | Changed Trigger Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in groups:
        for method in METHODS:
            if method not in groups[model]:
                continue
            aggregate = groups[model][method]["aggregate"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        METHOD_LABELS[method],
                        str(aggregate["trials"]),
                        fmt(aggregate["tool_f1"]),
                        fmt(aggregate["argument_f1"]),
                        fmt(aggregate["tool_sequence_exact"]),
                        fmt(aggregate["call_exact"]),
                        fmt(aggregate["triggers_per_sample"]),
                        fmt(aggregate["changed_trigger_rate"]),
                    ]
                )
                + " |"
            )
    summary_md = output_dir / "summary.md"
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return manifest_path, summary_json, per_trial_path, summary_md


def main():
    args = parse_args()
    if args.trials <= 0 or args.trials > 3:
        raise SystemExit("--trials must be between 1 and 3")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive")
    if not 0.0 <= args.memory_bank_probability_threshold <= 1.0:
        raise SystemExit("--memory-bank-probability-threshold must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    models = split_csv(args.models)
    methods = split_csv(args.methods)
    invalid_models = sorted(set(models) - set(MODELS))
    invalid_methods = sorted(set(methods) - set(METHODS))
    if invalid_models:
        raise SystemExit(f"Unknown models: {', '.join(invalid_models)}")
    if invalid_methods:
        raise SystemExit(f"Unknown methods: {', '.join(invalid_methods)}")
    try:
        trial_ids = parse_trial_ids(args.trial_ids, args.trials)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    if not data_path.exists():
        raise SystemExit(f"Test split not found: {data_path}")
    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    entries = select_entries(models, methods, trial_ids, output_dir)
    print(f"Data: {data_path} ({len(data)} samples)")
    print(f"Output: {output_dir}")
    for entry in entries:
        print(
            f"{entry['model']}/{entry['method']}/trial{entry['trial']}: "
            f"{entry['checkpoint_path']} -> {entry['prediction_path']}"
        )
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for entry in entries:
            path = Path(entry["prediction_path"])
            if path.exists() and not args.force:
                validate_entry_prediction(
                    path,
                    len(data),
                    entry,
                    args.memory_bank_probability_threshold,
                )
                print(f"Skipping complete predictions: {path}")
                continue
            print(
                f"Running {entry['model']}/{entry['method']}/trial{entry['trial']}",
                flush=True,
            )
            run_prediction(args, entry, data)
    else:
        for entry in entries:
            validate_entry_prediction(
                entry["prediction_path"],
                len(data),
                entry,
                args.memory_bank_probability_threshold,
            )

    outputs = write_outputs(args, entries, output_dir, data_path, len(data))
    for path in outputs:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
