#!/usr/bin/env python3
"""Evaluate final 4-call compositional checkpoints on the 10-call test split."""

import argparse
import gc
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

DEFAULT_SOURCE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "compositional"
    / "paper_compositional_head_8gpu"
    / "summary.md"
)
DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "results"
    / "compositional"
    / "all_methods"
    / "data"
    / "test"
    / "function_calling_test_tools51-100_10calls.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "compositional" / "rebuttal" / "results" / "train4_checkpoint_eval_10calls"

MODELS = ("llama1b", "llama3b", "llama8b")
METHODS = ("tokmem_eoc_logit_bias", "adap_tokmem_eoc_logit_bias")
METHOD_LABELS = {
    "tokmem_eoc_logit_bias": "tokmem",
    "adap_tokmem_eoc_logit_bias": "tapmem",
}
METRIC_FIELDS = (
    ("tool_sequence_exact", "Tool Sequence Exact"),
    ("tool_multiset_exact", "Tool Multiset Exact"),
    ("call_exact", "Call Exact"),
    ("f1", "Argument F1"),
    ("tool_f1", "Tool F1"),
    ("parse_error_rate", "Parse Error Rate"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use the final 4-call checkpoints from a compositional suite summary to run "
            "1B/3B/8B TokMem and TapMem inference on the tools51-100 10-call test split."
        )
    )
    parser.add_argument("--source-summary", default=str(DEFAULT_SOURCE_SUMMARY))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model keys.")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated suite method names.")
    parser.add_argument("--trials", type=int, default=3, help="Number of successful 4-call trials per group.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test example limit.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Regenerate predictions even if output JSONL exists.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize existing prediction JSONL files.")
    parser.add_argument("--dry-run", action="store_true", help="List selected checkpoints without running inference.")
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else None


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def torch_dtype(name):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def expand_per_round_values(values_str, fallback, rounds):
    if values_str is None:
        return [fallback] * len(rounds)
    values = [int(item.strip()) for item in values_str.split(",") if item.strip()]
    if len(values) < len(rounds):
        values.extend([values[-1]] * (len(rounds) - len(values)))
    return values[: len(rounds)]


def suite_dir_from_summary(summary_path):
    path = Path(summary_path)
    if path.name != "summary.md":
        raise SystemExit(f"--source-summary must point to a summary.md file: {path}")
    return path.parent


def load_suite_tasks(summary_path):
    suite_dir = suite_dir_from_summary(summary_path)
    status_path = suite_dir / "task_status.json"
    if not status_path.exists():
        raise SystemExit(f"task_status.json not found next to source summary: {status_path}")
    payload = load_json(status_path)
    return payload.get("tasks", []), status_path


def final_checkpoint_name(run_config):
    rounds = run_config.get("rounds") or []
    if not rounds:
        args = run_config.get("args", {})
        training_rounds = args.get("training_rounds", "")
        tools = training_rounds.split(":", 1)[0]
    else:
        tools = rounds[-1]["tools"]
    return f"round_{len(rounds) or 1}_tools_{tools.replace('-', '_')}.pt"


def select_checkpoint_tasks(tasks, models, methods, trials):
    selected = {}
    for model in models:
        for method in methods:
            group = [
                task
                for task in tasks
                if task.get("call_scope") == "4calls"
                and task.get("model") == model
                and task.get("method") == method
                and task.get("status") == "success"
            ]
            group.sort(key=lambda task: int(task.get("trial", 10**9)))
            group = group[:trials]
            if len(group) < trials:
                raise SystemExit(f"Not enough 4-call checkpoints for {model}/{method}: {len(group)}/{trials}")

            resolved = []
            for task in group:
                run_dir = Path(task["task_dir"])
                run_config_path = Path(task.get("run_config") or run_dir / "run_config.json")
                run_config = load_json(run_config_path)
                checkpoint_path = run_dir / final_checkpoint_name(run_config)
                if not checkpoint_path.exists():
                    raise SystemExit(f"Final checkpoint not found: {checkpoint_path}")
                resolved.append(
                    {
                        "model": model,
                        "method": method,
                        "trial": int(task.get("trial", 0)),
                        "run_dir": run_dir,
                        "run_config_path": run_config_path,
                        "checkpoint_path": checkpoint_path,
                    }
                )
            selected[(model, method)] = resolved
    return selected


def discover_all_tool_names(run_config):
    from dataset import discover_available_tools

    args = run_config["args"]
    rounds = run_config.get("rounds") or []
    train_calls = expand_per_round_values(
        args.get("train_max_function_calls_per_round"),
        int(args.get("train_max_function_calls", 4)),
        rounds,
    )
    test_calls = expand_per_round_values(
        args.get("test_max_function_calls_per_round"),
        int(args.get("test_max_function_calls", 4)),
        rounds,
    )

    data_dir = Path(args["data_dir"])
    all_tool_names = []
    seen = set()
    for idx, round_spec in enumerate(rounds):
        tools_range = round_spec["tools"]
        train_path = data_dir / "training" / f"function_calling_train_tools{tools_range}_{train_calls[idx]}calls.json"
        test_path = data_dir / "test" / f"function_calling_test_tools{tools_range}_{test_calls[idx]}calls.json"
        for tool_name in discover_available_tools(str(train_path), str(test_path)):
            if tool_name not in seen:
                all_tool_names.append(tool_name)
                seen.add(tool_name)
    if not all_tool_names:
        raise RuntimeError(f"No tools discovered for run {run_config.get('run_name')}")
    return all_tool_names


def build_lora_config(run_args):
    if not run_args.get("use_lora", False):
        return None
    target_modules = [item.strip() for item in run_args.get("lora_target_modules", "o_proj").split(",")]
    config = {
        "r": int(run_args.get("lora_r", 8)),
        "alpha": int(run_args.get("lora_alpha", 32)),
        "dropout": float(run_args.get("lora_dropout", 0.1)),
        "target_modules": target_modules,
    }
    if run_args.get("lora_layer_indices") is not None:
        config["layer_indices"] = [int(item.strip()) for item in run_args["lora_layer_indices"].split(",")]
    return config


def build_model(run_config, checkpoint, tokenizer, device, dtype):
    from model import FunctionCallingModel

    run_args = run_config["args"]
    tool_names = discover_all_tool_names(run_config)
    model = FunctionCallingModel(
        model_name=run_args["model_name"],
        num_tools=len(tool_names),
        tool_names=tool_names,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        decouple_embeddings=bool(run_args.get("decouple_embeddings", False)),
        lora_config=build_lora_config(run_args),
        use_eoc=bool(run_args.get("use_eoc", False)),
        use_logit_bias=bool(run_args.get("use_logit_bias", False)),
        use_tool_head_replacement=bool(run_args.get("use_tool_head_replacement", False)),
        logit_bias_network=run_args.get("logit_bias_network", "linear"),
        logit_bias_scale=float(run_args.get("logit_bias_scale", 1.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def build_user_text(item):
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{item['user_input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


def generate_batch(model, tokenizer, batch, device, max_new_tokens):
    import torch

    encoded = tokenizer(
        [build_user_text(item) for item in batch],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        return model.generate_with_tool_prediction(
            encoded["input_ids"],
            encoded["attention_mask"],
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=False,
        )


def prediction_record(index, item, result, tokenizer, method, candidate_tools):
    from eval import calculate_tool_metrics, compare_function_calls_advanced

    expected_tools = item.get("tools", [item.get("tool_name", "unknown")])
    expected_calls = item.get("function_calls", [item.get("function_call", "{}")])
    predicted_tools = [tool_info["tool_name"] for tool_info in result.get("predicted_tools", [])]
    predicted_calls = result.get("function_calls", [])
    tool_tokens = [tokenizer.decode([tool_info["token_id"]]) for tool_info in result.get("predicted_tools", [])]

    call_eval = compare_function_calls_advanced(predicted_calls, expected_calls, ignore_order=True)
    tool_metrics = calculate_tool_metrics(
        predicted_tools=predicted_tools,
        expected_tools=expected_tools,
        candidate_tools=candidate_tools,
    )

    return {
        "index": index,
        "method": method,
        "user_input": item["user_input"],
        "expected_tools": expected_tools,
        "expected_calls": expected_calls,
        "predicted_tools": predicted_tools,
        "predicted_calls": predicted_calls,
        "tool_tokens": tool_tokens,
        "full_generated_sequence": result.get("full_generated_sequence", ""),
        "tool_sequence_exact": predicted_tools == expected_tools,
        "tool_multiset_exact": bool(tool_metrics["tool_exact_match_acc"] >= 1.0),
        "call_exact": bool(call_eval.exact_match),
        "f1": call_eval.f1_score,
        "tool_f1": tool_metrics["tool_f1_score"],
        "parse_errors": call_eval.details.get("parse_errors", {}),
    }


def prediction_path(output_dir, item):
    label = METHOD_LABELS.get(item["method"], item["method"])
    return (
        output_dir
        / "predictions"
        / item["model"]
        / label
        / f"{item['run_dir'].name}_train4_on_10calls_predictions.jsonl"
    )


def run_prediction(args, item, out_path, data):
    import torch
    from transformers import AutoTokenizer

    run_config = load_json(item["run_config_path"])
    checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
    dtype = torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(run_config["args"]["model_name"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_model(run_config, checkpoint, tokenizer, args.device, dtype)
    del checkpoint
    gc.collect()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_tools = list(getattr(model, "tool_names", []))
    method_label = METHOD_LABELS.get(item["method"], item["method"])
    with open(out_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(data), args.eval_batch_size):
            batch = data[start : start + args.eval_batch_size]
            results = generate_batch(model, tokenizer, batch, args.device, args.max_new_tokens)
            for offset, (sample, result) in enumerate(zip(batch, results)):
                index = start + offset
                record = prediction_record(index, sample, result, tokenizer, method_label, candidate_tools)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = start + len(batch)
            if args.progress_every > 0 and completed % args.progress_every < args.eval_batch_size:
                print(f"Generated {completed}/{len(data)} predictions for {item['run_dir'].name}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_prediction_file(path):
    records = load_jsonl(path)
    if not records:
        return {"samples": 0}

    parse_error_outputs = []
    for record in records:
        errors = record.get("parse_errors") or {}
        parse_error_outputs.append(float(errors.get("outputs", 0) or 0))

    return {
        "samples": len(records),
        "tool_sequence_exact": mean([1.0 if record.get("tool_sequence_exact") else 0.0 for record in records]),
        "tool_multiset_exact": mean([1.0 if record.get("tool_multiset_exact") else 0.0 for record in records]),
        "call_exact": mean([1.0 if record.get("call_exact") else 0.0 for record in records]),
        "f1": mean([float(record.get("f1") or 0.0) for record in records]),
        "tool_f1": mean([float(record.get("tool_f1") or 0.0) for record in records]),
        "parse_error_rate": sum(parse_error_outputs) / len(records) if records else 0.0,
    }


def summarize_all(selected, output_paths):
    summaries = {}
    for key, items in selected.items():
        trials = []
        for item in items:
            pred_path = output_paths[(item["model"], item["method"], item["trial"])]
            if not pred_path.exists():
                raise SystemExit(f"Prediction file missing for summary: {pred_path}")
            metrics = summarize_prediction_file(pred_path)
            trials.append(
                {
                    "trial": item["trial"],
                    "run_name": item["run_dir"].name,
                    "run_dir": str(item["run_dir"]),
                    "checkpoint_path": str(item["checkpoint_path"]),
                    "prediction_path": str(pred_path),
                    "metrics": metrics,
                }
            )

        aggregate = {"trials": len(trials)}
        for metric_key, _label in METRIC_FIELDS:
            values = [trial["metrics"].get(metric_key) for trial in trials]
            values = [value for value in values if value is not None]
            aggregate[metric_key] = mean(values)
            aggregate[f"{metric_key}_std"] = stdev(values)

        model, method = key
        summaries.setdefault(model, {})[method] = {
            "trials": trials,
            "aggregate": aggregate,
        }
    return summaries


def write_summary_json(output_dir, args, source_status_path, summaries):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary": str(Path(args.source_summary)),
        "source_status": str(source_status_path),
        "data_path": str(Path(args.data_path)),
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "limit": args.limit,
        "results": summaries,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_summary_md(output_dir, args, summaries):
    lines = [
        "# 4-call Checkpoints Evaluated on 10-call Test",
        "",
        f"- source summary: `{Path(args.source_summary)}`",
        f"- test split: `{Path(args.data_path)}`",
        f"- max new tokens: `{args.max_new_tokens}`",
        f"- eval batch size: `{args.eval_batch_size}`",
    ]
    if args.limit is not None:
        lines.append(f"- limit: `{args.limit}`")

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Model | Method | Trials | Tool Sequence Exact | Tool Multiset Exact | Call Exact | Argument F1 | Tool F1 | Parse Error Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for model in summaries:
        for method, payload in summaries[model].items():
            aggregate = payload["aggregate"]
            label = METHOD_LABELS.get(method, method)
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        label,
                        str(aggregate["trials"]),
                        fmt(aggregate["tool_sequence_exact"]),
                        fmt(aggregate["tool_multiset_exact"]),
                        fmt(aggregate["call_exact"]),
                        fmt(aggregate["f1"]),
                        fmt(aggregate["tool_f1"]),
                        fmt(aggregate["parse_error_rate"]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Per Trial", ""])
    for model in summaries:
        for method, payload in summaries[model].items():
            label = METHOD_LABELS.get(method, method)
            lines.extend(
                [
                    f"### {model} / {label}",
                    "",
                    "| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |",
                    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for trial in payload["trials"]:
                metrics = trial["metrics"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(trial["trial"]),
                            trial["run_name"],
                            str(metrics.get("samples", "")),
                            fmt(metrics.get("tool_sequence_exact")),
                            fmt(metrics.get("call_exact")),
                            fmt(metrics.get("f1")),
                            fmt(metrics.get("tool_f1")),
                            f"`{trial['checkpoint_path']}`",
                        ]
                    )
                    + " |"
                )
            lines.append("")

    path = output_dir / "summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    source_summary = Path(args.source_summary)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    if not source_summary.exists():
        raise SystemExit(f"Source summary not found: {source_summary}")
    if not data_path.exists():
        raise SystemExit(f"10-call test split not found: {data_path}")

    models = split_csv(args.models)
    methods = split_csv(args.methods)
    tasks, source_status_path = load_suite_tasks(source_summary)
    selected = select_checkpoint_tasks(tasks, models, methods, args.trials)
    output_paths = {}
    for items in selected.values():
        for item in items:
            output_paths[(item["model"], item["method"], item["trial"])] = prediction_path(output_dir, item)

    print(f"Output directory: {output_dir}")
    for model in models:
        for method in methods:
            label = METHOD_LABELS.get(method, method)
            print(f"\n{model} / {label}:")
            for item in selected[(model, method)]:
                pred_path = output_paths[(item["model"], item["method"], item["trial"])]
                print(f"  trial {item['trial']}: {item['checkpoint_path']} -> {pred_path}")

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        data = load_json(data_path)
        if args.limit is not None:
            data = data[: args.limit]

        for model in models:
            for method in methods:
                for item in selected[(model, method)]:
                    pred_path = output_paths[(item["model"], item["method"], item["trial"])]
                    if pred_path.exists() and not args.force:
                        print(f"Skipping existing predictions: {pred_path}")
                        continue
                    print(
                        f"Running {model}/{METHOD_LABELS.get(method, method)} "
                        f"trial {item['trial']}: {item['run_dir'].name}"
                    )
                    run_prediction(args, item, pred_path, data)

    summaries = summarize_all(selected, output_paths)
    summary_json = write_summary_json(output_dir, args, source_status_path, summaries)
    summary_md = write_summary_md(output_dir, args, summaries)
    print(f"\nWrote summary JSON: {summary_json}")
    print(f"Wrote summary MD: {summary_md}")


if __name__ == "__main__":
    main()
