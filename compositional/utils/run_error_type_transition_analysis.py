#!/usr/bin/env python3
"""Generate aligned 4-call predictions and run the rebuttal error analysis."""

import argparse
import gc
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
UTILS_DIR = COMPOSITIONAL_DIR / "utils"
for import_dir in (COMPOSITIONAL_DIR, UTILS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from analyze_error_type_transitions import analyze_manifest  # noqa: E402
from run_train4_checkpoint_eval_10calls import (  # noqa: E402
    build_model,
    final_checkpoint_name,
    generate_batch,
    load_json,
    prediction_record,
    torch_dtype,
)


DEFAULT_DATA_PATH = COMPOSITIONAL_DIR / "data" / "test" / "function_calling_test_tools51-100_4calls.json"
DEFAULT_OUTPUT_DIR = COMPOSITIONAL_DIR / "rebuttal" / "results" / "error_type_transition_analysis"
MODELS = ("llama1b", "llama3b", "llama8b")
METHODS = ("tokmem", "tapmem", "eoc_only")
DEFAULT_EVAL_BATCH_SIZES = {
    "llama1b": 8,
    "llama3b": 4,
    "llama8b": 2,
}
EXPECTED_MODEL_NAMES = {
    "llama1b": "Llama-3.2-1B-Instruct",
    "llama3b": "Llama-3.2-3B-Instruct",
    "llama8b": "Llama-3.1-8B-Instruct",
}
EXPECTED_FLAGS = {
    "tokmem": {"use_eoc": False, "use_logit_bias": False, "use_tool_head_replacement": False},
    "tapmem": {"use_eoc": True, "use_logit_bias": True, "use_tool_head_replacement": False},
    "eoc_only": {"use_eoc": True, "use_logit_bias": False, "use_tool_head_replacement": False},
}


def trial_dirs(parent, template):
    return tuple(parent / template.format(trial=trial) for trial in range(1, 4))


ALL_METHODS_RUNS = REPO_ROOT / "results" / "compositional" / "all_methods" / "runs"
PAPER_HEAD_RUNS = REPO_ROOT / "results" / "compositional" / "paper_compositional_head_8gpu" / "runs"
LOCAL_RUNS = COMPOSITIONAL_DIR / "runs"

CHECKPOINT_RUN_DIRS = {
    ("llama1b", "tokmem"): trial_dirs(ALL_METHODS_RUNS, "llama1b_tokmem_trial{trial}_seed42"),
    ("llama3b", "tokmem"): trial_dirs(
        LOCAL_RUNS,
        "tokmem_llama_3b_4calls_seed42_3x_20260425_214658_trial{trial}",
    ),
    ("llama8b", "tokmem"): trial_dirs(ALL_METHODS_RUNS, "llama8b_tokmem_trial{trial}_seed42"),
    ("llama1b", "tapmem"): trial_dirs(
        PAPER_HEAD_RUNS,
        "llama1b_tokmem_eoc_logit_bias_trial{trial}_seed42",
    ),
    ("llama3b", "tapmem"): trial_dirs(
        PAPER_HEAD_RUNS,
        "llama3b_tokmem_eoc_logit_bias_trial{trial}_seed42",
    ),
    ("llama8b", "tapmem"): trial_dirs(
        PAPER_HEAD_RUNS,
        "llama8b_tokmem_eoc_logit_bias_trial{trial}_seed42",
    ),
    ("llama1b", "eoc_only"): trial_dirs(ALL_METHODS_RUNS, "llama1b_tokmem_eoc_trial{trial}_seed42"),
    ("llama3b", "eoc_only"): trial_dirs(
        LOCAL_RUNS,
        "tokmem_eoc_llama_3b_4calls_seed42_3x_20260425_214658_trial{trial}",
    ),
    ("llama8b", "eoc_only"): trial_dirs(ALL_METHODS_RUNS, "llama8b_tokmem_eoc_trial{trial}_seed42"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-checkpoint TokMem, TapMem, and EOC-only inference on the aligned "
            "4-call APIGen split, then compute rebuttal error and transition metrics."
        )
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model keys.")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated method keys.")
    parser.add_argument("--trials", type=int, default=3, help="Trials per model/method group (maximum 3).")
    parser.add_argument(
        "--trial-ids",
        default=None,
        help="Optional comma-separated 1-based trial ids; overrides --trials.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override the per-model defaults (1B=8, 3B=4, 8B=2).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix size for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Regenerate completed predictions.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate selected predictions and manifest without running the analyzer.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Require existing selected predictions and only rerun analysis.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print selected inputs only.")
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_trial_ids(value, trials):
    if value is None:
        return list(range(1, trials + 1))
    trial_ids = [int(item) for item in split_csv(value)]
    if not trial_ids:
        raise ValueError("--trial-ids must contain at least one trial")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("--trial-ids must not contain duplicates")
    if any(trial < 1 or trial > 3 for trial in trial_ids):
        raise ValueError("--trial-ids values must be between 1 and 3")
    return trial_ids


def eval_batch_size(args, model):
    return args.eval_batch_size or DEFAULT_EVAL_BATCH_SIZES[model]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_archive_fingerprint(path):
    """Fingerprint tensor payloads cheaply from PyTorch ZIP member CRC metadata."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        stat = path.stat()
        return {
            "kind": "non_zip_file_stat",
            "digest": hashlib.sha256(
                f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest(),
            "file_size": stat.st_size,
        }

    members = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative_name = member.filename.split("/", 1)[-1]
            members.append((relative_name, member.file_size, member.CRC))
    payload = json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode()
    return {
        "kind": "pytorch_zip_member_size_crc32",
        "digest": hashlib.sha256(payload).hexdigest(),
        "members": len(members),
        "file_size": path.stat().st_size,
    }


def validate_run_config(model, method, run_config, run_config_path):
    args = run_config.get("args", {})
    rounds = run_config.get("rounds") or []
    errors = []

    if Path(args.get("model_name", "")).name != EXPECTED_MODEL_NAMES[model]:
        errors.append(f"model_name={args.get('model_name')!r}")
    if bool(args.get("use_lora", False)):
        errors.append("use_lora must be false")
    if int(args.get("train_max_function_calls", -1)) != 4:
        errors.append(f"train_max_function_calls={args.get('train_max_function_calls')!r}")
    if int(args.get("test_max_function_calls", -1)) != 4:
        errors.append(f"test_max_function_calls={args.get('test_max_function_calls')!r}")
    if len(rounds) != 1 or rounds[0].get("tools") != "51-100":
        errors.append(f"rounds={rounds!r}")
    for flag, expected in EXPECTED_FLAGS[method].items():
        actual = bool(args.get(flag, False))
        if actual != expected:
            errors.append(f"{flag}={actual}, expected {expected}")

    if errors:
        raise ValueError(f"Invalid {model}/{method} run {run_config_path}: {'; '.join(errors)}")


def select_entries(models, methods, trial_ids, output_dir):
    entries = []
    for model in models:
        for method in methods:
            configured_run_dirs = CHECKPOINT_RUN_DIRS[(model, method)]
            for trial in trial_ids:
                run_dir = configured_run_dirs[trial - 1]
                run_config_path = run_dir / "run_config.json"
                if not run_config_path.exists():
                    raise FileNotFoundError(f"Run config not found: {run_config_path}")
                run_config = load_json(run_config_path)
                validate_run_config(model, method, run_config, run_config_path)
                checkpoint_path = run_dir / final_checkpoint_name(run_config)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                prediction_path = (
                    output_dir
                    / "predictions"
                    / model
                    / method
                    / f"{run_dir.name}_4calls_predictions.jsonl"
                )
                entries.append(
                    {
                        "model": model,
                        "method": method,
                        "trial": trial,
                        "run_name": run_dir.name,
                        "run_dir": str(run_dir.resolve()),
                        "run_config_path": str(run_config_path.resolve()),
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "checkpoint_archive_fingerprint": checkpoint_archive_fingerprint(
                            checkpoint_path
                        ),
                        "prediction_path": str(prediction_path.resolve()),
                    }
                )
    return entries


def validate_prediction_file(path, expected_samples):
    indices = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "index" not in record:
                raise ValueError(f"Missing index in {path}:{line_number}")
            indices.append(int(record["index"]))
    expected_indices = list(range(expected_samples))
    if indices != expected_indices:
        raise ValueError(
            f"Incomplete or misordered predictions in {path}: "
            f"found {len(indices)} records, expected {expected_samples}"
        )


def partial_record_count(path):
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("index", -1)) != count:
                raise ValueError(f"Non-contiguous partial predictions at {path}: expected index {count}")
            count += 1
    return count


def run_prediction(args, entry, data):
    import torch
    from transformers import AutoTokenizer

    prediction_path = Path(entry["prediction_path"])
    partial_path = prediction_path.with_suffix(prediction_path.suffix + ".partial")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force and partial_path.exists():
        partial_path.unlink()
    start_index = partial_record_count(partial_path)
    if start_index > len(data):
        raise ValueError(f"Partial prediction has too many records: {partial_path}")

    run_config = load_json(entry["run_config_path"])
    checkpoint = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(run_config["args"]["model_name"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_model(run_config, checkpoint, tokenizer, args.device, torch_dtype(args.dtype))
    del checkpoint
    gc.collect()
    candidate_tools = list(getattr(model, "tool_names", []))

    mode = "a" if start_index else "w"
    with open(partial_path, mode, encoding="utf-8") as handle:
        batch_size = eval_batch_size(args, entry["model"])
        for start in range(start_index, len(data), batch_size):
            batch = data[start : start + batch_size]
            results = generate_batch(model, tokenizer, batch, args.device, args.max_new_tokens)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Generation returned {len(results)} results for a batch of {len(batch)} "
                    f"at index {start}"
                )
            for offset, (sample, result) in enumerate(zip(batch, results)):
                index = start + offset
                output_record = prediction_record(
                    index,
                    sample,
                    result,
                    tokenizer,
                    entry["method"],
                    candidate_tools,
                )
                output_record.update(
                    {
                        "model": entry["model"],
                        "trial": entry["trial"],
                        "run_name": entry["run_name"],
                    }
                )
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = start + len(batch)
            if args.progress_every > 0 and (
                completed == len(data) or completed % args.progress_every < batch_size
            ):
                print(f"Generated {completed}/{len(data)} for {entry['run_name']}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    validate_prediction_file(partial_path, len(data))
    partial_path.replace(prediction_path)


def write_manifest(args, entries, data_path, output_dir, sample_count):
    for entry in entries:
        entry["eval_batch_size"] = eval_batch_size(args, entry["model"])
    fingerprint_groups = {}
    for entry in entries:
        fingerprint = entry["checkpoint_archive_fingerprint"]
        key = (entry["model"], entry["method"], fingerprint["kind"], fingerprint["digest"])
        fingerprint_groups.setdefault(key, []).append(int(entry["trial"]))
    duplicate_checkpoint_groups = [
        {
            "model": model,
            "method": method,
            "trials": sorted(trials),
            "fingerprint_kind": fingerprint_kind,
            "fingerprint": fingerprint,
        }
        for (model, method, fingerprint_kind, fingerprint), trials in fingerprint_groups.items()
        if len(trials) > 1
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_policy": (
            "paper-table no-adaptation TokMem and TapMem checkpoints; "
            "matched no-adaptation EOC-only checkpoints"
        ),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "samples": sample_count,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size_by_model": {
            model: eval_batch_size(args, model)
            for model in sorted({entry["model"] for entry in entries})
        },
        "dtype": args.dtype,
        "do_sample": False,
        "temperature": 0.6,
        "top_p": 0.9,
        "duplicate_checkpoint_groups": duplicate_checkpoint_groups,
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main():
    args = parse_args()
    if args.generate_only and args.summarize_only:
        raise SystemExit("--generate-only and --summarize-only are mutually exclusive")
    if args.trials <= 0 or args.trials > 3:
        raise SystemExit("--trials must be between 1 and 3")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    models = split_csv(args.models)
    methods = split_csv(args.methods)
    invalid_models = sorted(set(models) - set(MODELS))
    invalid_methods = sorted(set(methods) - set(METHODS))
    if invalid_models:
        raise SystemExit(f"Unknown models: {', '.join(invalid_models)}")
    if invalid_methods:
        raise SystemExit(f"Unknown methods: {', '.join(invalid_methods)}")

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    if not data_path.exists():
        raise SystemExit(f"4-call test split not found: {data_path}")
    data = load_json(data_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        data = data[: args.limit]

    try:
        trial_ids = parse_trial_ids(args.trial_ids, args.trials)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    entries = select_entries(models, methods, trial_ids, output_dir)
    print(f"Data: {data_path} ({len(data)} samples, sha256={sha256_file(data_path)})")
    print(f"Output: {output_dir}")
    for entry in entries:
        print(
            f"{entry['model']}/{entry['method']}/trial{entry['trial']}: "
            f"{entry['checkpoint_path']} -> {entry['prediction_path']}"
        )

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(args, entries, data_path, output_dir, len(data))

    if not args.summarize_only:
        for entry in entries:
            prediction_path = Path(entry["prediction_path"])
            if prediction_path.exists() and not args.force:
                validate_prediction_file(prediction_path, len(data))
                print(f"Skipping complete predictions: {prediction_path}")
                continue
            print(
                f"Running {entry['model']}/{entry['method']}/trial{entry['trial']}: "
                f"{entry['run_name']}",
                flush=True,
            )
            run_prediction(args, entry, data)
    else:
        for entry in entries:
            validate_prediction_file(Path(entry["prediction_path"]), len(data))

    print(f"Wrote manifest: {manifest_path}")
    if args.generate_only:
        return

    summary_json, per_trial_path, summary_md = analyze_manifest(manifest_path, output_dir=output_dir)
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote per-trial JSONL: {per_trial_path}")
    print(f"Wrote summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
