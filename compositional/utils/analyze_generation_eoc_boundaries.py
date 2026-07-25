#!/usr/bin/env python3
"""Analyze generated EOC boundary events for compositional checkpoints."""

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

from backbone_prompting import format_user_assistant_prompt
from checkpoint_io import checkpoint_tool_names, load_checkpoint_into_model

DEFAULT_OUTPUT_DIR = REPO_ROOT / "compositional" / "rebuttal" / "results" / "eoc_boundary_accuracy"
ALL_METHODS_STATUS = REPO_ROOT / "results" / "compositional" / "all_methods" / "task_status.json"
PAPER_HEAD_STATUS = REPO_ROOT / "results" / "compositional" / "paper_compositional_head_8gpu" / "task_status.json"
LLAMA3B_FINAL_SUITE = (
    REPO_ROOT
    / "results"
    / "compositional"
    / "llama3b_tokmem_family_4calls_seed42_3x_20260425_214658"
)

MODELS = ("llama1b", "llama3b", "llama8b")
DEFAULT_METHODS = ("tokmem_eoc", "tokmem_eoc_logit_bias")
SUPPORTED_METHODS = ("tokmem_eoc", "tokmem_eoc_logit_bias", "adap_tokmem_eoc_logit_bias")
METHOD_LABELS = {
    "tokmem_eoc": "eoc",
    "tokmem_eoc_logit_bias": "eoc_logit_bias",
    "adap_tokmem_eoc_logit_bias": "eoc_logit_bias_adapt",
}
LLAMA3B_METHOD_DIRS = {
    "tokmem_eoc": "tokmem_eoc_llama_3b_4calls_seed42_3x_20260425_214658",
    "tokmem_eoc_logit_bias": "tokmem_eoc_logit_bias_llama_3b_4calls_seed42_3x_20260425_214658",
}
METRIC_FIELDS = (
    ("eoc_count_exact", "EOC Count Exact"),
    ("eoc_precision", "EOC Precision"),
    ("eoc_recall", "EOC Recall"),
    ("eoc_f1", "EOC F1"),
    ("avg_expected_eoc", "Expected EOC"),
    ("avg_predicted_eoc", "Predicted EOC"),
    ("avg_missing_eoc", "Missing EOC"),
    ("avg_extra_eoc", "Extra EOC"),
    ("malformed_boundary_rate", "Malformed Boundary Rate"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run free-generation EOC boundary analysis for TokMem EOC checkpoints."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model keys.")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated method names.")
    parser.add_argument(
        "--selection-policy",
        default="legacy",
        choices=["legacy", "table1_ours"],
        help=(
            "legacy reproduces the original rebuttal mix. table1_ours selects "
            "paper_compositional_head_8gpu checkpoints for the Table 1 Ours rows."
        ),
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials for all_methods EOC-only 1B/8B groups.")
    parser.add_argument("--logit-bias-trials", type=int, default=3, help="Trials for final 1B/8B EOC+logit-bias groups.")
    parser.add_argument("--llama3b-trials", type=int, default=3, help="Trials from the final 3B summary suite.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test example limit.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Regenerate per-sample records if output exists.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize existing per-sample JSONL files.")
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


def final_checkpoint_name(run_config):
    rounds = run_config.get("rounds") or []
    if not rounds:
        training_rounds = run_config.get("args", {}).get("training_rounds", "51-100:1")
        tools = training_rounds.split(":", 1)[0]
    else:
        tools = rounds[-1]["tools"]
    return f"round_{len(rounds) or 1}_tools_{tools.replace('-', '_')}.pt"


def expand_per_round_values(values_str, fallback, rounds):
    if values_str is None:
        return [fallback] * len(rounds)
    values = [int(item.strip()) for item in values_str.split(",") if item.strip()]
    if len(values) < len(rounds):
        values.extend([values[-1]] * (len(rounds) - len(values)))
    return values[: len(rounds)]


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
    from backbone_registry import resolve_function_calling_model_class

    run_args = run_config["args"]
    discovered_tool_names = discover_all_tool_names(run_config)
    tool_names = checkpoint_tool_names(
        checkpoint,
        fallback=discovered_tool_names,
    )
    model_class = resolve_function_calling_model_class(run_args["model_name"])
    model = model_class(
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
        use_memory_bank_constraint=bool(
            run_args.get("use_memory_bank_constraint", False)
        ),
        memory_bank_probability_threshold=float(
            run_args.get("memory_bank_probability_threshold", 0.5)
        ),
        logit_bias_network=run_args.get("logit_bias_network", "linear"),
        logit_bias_scale=float(run_args.get("logit_bias_scale", 1.0)),
    )
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return model


def build_user_text(item, tokenizer, model):
    return format_user_assistant_prompt(
        tokenizer,
        item["user_input"],
        model=model,
    )


def resolve_test_data_path(run_config):
    args = run_config["args"]
    rounds = run_config.get("rounds") or []
    if rounds:
        tools = rounds[-1]["tools"]
        test_calls = expand_per_round_values(
            args.get("test_max_function_calls_per_round"),
            int(args.get("test_max_function_calls", 4)),
            rounds,
        )[-1]
    else:
        tools = args.get("training_rounds", "51-100:1").split(":", 1)[0]
        test_calls = int(args.get("test_max_function_calls", 4))
    return Path(args["data_dir"]) / "test" / f"function_calling_test_tools{tools}_{test_calls}calls.json"


def add_selected_item(selected, model, method, trial, run_dir, source, run_config_path=None):
    run_dir = Path(run_dir)
    run_config_path = Path(run_config_path) if run_config_path else run_dir / "run_config.json"
    if not run_config_path.exists():
        raise SystemExit(f"run_config.json not found: {run_config_path}")
    run_config = load_json(run_config_path)
    checkpoint_path = run_dir / final_checkpoint_name(run_config)
    if not checkpoint_path.exists():
        raise SystemExit(f"Final checkpoint not found: {checkpoint_path}")
    data_path = resolve_test_data_path(run_config)
    if not data_path.exists():
        raise SystemExit(f"Test split not found: {data_path}")
    selected.setdefault((model, method), []).append(
        {
            "model": model,
            "method": method,
            "trial": int(trial),
            "source": source,
            "run_dir": run_dir,
            "run_config_path": run_config_path,
            "checkpoint_path": checkpoint_path,
            "data_path": data_path,
        }
    )


def select_status_items(selected, status_path, source, models, methods, trials_by_method, skip_models=()):
    payload = load_json(status_path)
    tasks = payload.get("tasks", [])
    for model in models:
        if model in skip_models:
            continue
        for method in methods:
            trials = trials_by_method[method]
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
                raise SystemExit(f"Not enough {source} checkpoints for {model}/{method}: {len(group)}/{trials}")
            for task in group:
                add_selected_item(
                    selected,
                    model=model,
                    method=method,
                    trial=task.get("trial", 0),
                    run_dir=task["task_dir"],
                    source=source,
                    run_config_path=task.get("run_config"),
                )


def select_llama3b_items(selected, models, methods, trials):
    if "llama3b" not in models:
        return
    for method in methods:
        method_dir = LLAMA3B_METHOD_DIRS[method]
        results_path = LLAMA3B_FINAL_SUITE / "method_runs" / method_dir / "results.json"
        if not results_path.exists():
            raise SystemExit(f"3B final-suite results missing: {results_path}")
        payload = load_json(results_path)
        group = sorted(payload.get("trials", []), key=lambda item: int(item.get("trial", 10**9)))[:trials]
        if len(group) < trials:
            raise SystemExit(f"Not enough final 3B checkpoints for {method}: {len(group)}/{trials}")
        for trial in group:
            add_selected_item(
                selected,
                model="llama3b",
                method=method,
                trial=trial["trial"],
                run_dir=trial["run_dir"],
                source=str(LLAMA3B_FINAL_SUITE),
            )


def select_items(models, methods, trials, logit_bias_trials, llama3b_trials):
    selected = {}
    unsupported = sorted(set(methods) - set(DEFAULT_METHODS))
    if unsupported:
        raise SystemExit(
            "legacy selection only supports "
            f"{', '.join(DEFAULT_METHODS)}; use --selection-policy table1_ours for "
            f"{', '.join(unsupported)}"
        )
    all_methods = [method for method in methods if method == "tokmem_eoc"]
    if all_methods:
        select_status_items(
            selected,
            status_path=ALL_METHODS_STATUS,
            source="all_methods",
            models=models,
            methods=all_methods,
            trials_by_method={"tokmem_eoc": trials},
            skip_models=("llama3b",),
        )
    final_head = [method for method in methods if method == "tokmem_eoc_logit_bias"]
    if final_head:
        select_status_items(
            selected,
            status_path=PAPER_HEAD_STATUS,
            source="paper_compositional_head_8gpu",
            models=models,
            methods=final_head,
            trials_by_method={"tokmem_eoc_logit_bias": logit_bias_trials},
            skip_models=("llama3b",),
        )
    select_llama3b_items(selected, models, methods, llama3b_trials)
    return selected


def select_table1_ours_items(models, methods, logit_bias_trials):
    selected = {}
    supported = {"tokmem_eoc_logit_bias", "adap_tokmem_eoc_logit_bias"}
    unsupported = sorted(set(methods) - supported)
    if unsupported:
        raise SystemExit(
            "table1_ours only supports Table 1 Ours methods: "
            f"{', '.join(sorted(supported))}; got unsupported methods: {', '.join(unsupported)}"
        )
    trials_by_method = {method: logit_bias_trials for method in methods}
    select_status_items(
        selected,
        status_path=PAPER_HEAD_STATUS,
        source="paper_compositional_head_8gpu",
        models=models,
        methods=methods,
        trials_by_method=trials_by_method,
    )
    return selected


def output_path(output_dir, item):
    label = METHOD_LABELS.get(item["method"], item["method"])
    return (
        output_dir
        / "predictions"
        / item["model"]
        / label
        / f"{item['run_dir'].name}_eoc_boundaries.jsonl"
    )


def generate_raw_batch(model, tokenizer, batch, device, max_new_tokens):
    import torch

    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [build_user_text(item, tokenizer, model) for item in batch],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    user_tokens = encoded["input_ids"]
    user_mask = encoded["attention_mask"]

    if not model.use_logit_bias and not model.use_tool_head_replacement:
        generate_kwargs = {
            "input_ids": user_tokens,
            "attention_mask": user_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "pad_token_id": getattr(
                model,
                "_native_generation_pad_token_id",
                tokenizer.eos_token_id,
            ),
            "use_cache": True,
        }
        native_eos_token_id = getattr(
            model,
            "_native_generation_eos_token_id",
            None,
        )
        if native_eos_token_id is not None:
            generate_kwargs["eos_token_id"] = native_eos_token_id
        with torch.no_grad():
            generated = model.model.generate(**generate_kwargs)
    else:
        generated = generate_raw_with_custom_decoding(model, user_tokens, user_mask, tokenizer, max_new_tokens)

    return extract_raw_sequences(generated, user_tokens.shape[1], tokenizer)


def generate_raw_with_custom_decoding(model, user_tokens, user_mask, tokenizer, max_new_tokens):
    import torch

    if model.use_logit_bias and model.use_tool_head_replacement:
        raise ValueError("use_logit_bias and use_tool_head_replacement are decode-time alternatives")

    batch_size = user_tokens.shape[0]
    device = user_tokens.device
    input_ids = user_tokens.clone()
    attention_mask = user_mask.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    step_input_ids = user_tokens
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            decision_context = model._build_decision_context(
                input_ids=input_ids,
                batch_size=batch_size,
                device=device,
                step=step,
            )
            active_decision_rows = decision_context & ~finished
            need_last_hidden_state = bool(
                (model.use_logit_bias or model.use_tool_head_replacement) and active_decision_rows.any()
            )
            next_logits, last_hidden_states, past_key_values = model._generation_forward_step(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                return_last_hidden_state=need_last_hidden_state,
            )
            next_logits = next_logits.clone()
            selection_logits = next_logits
            if model.use_logit_bias and active_decision_rows.any():
                selection_logits = model._apply_logit_bias_to_logits(
                    logits=next_logits,
                    hidden_states=last_hidden_states,
                    active_decision_rows=active_decision_rows,
                )
            next_tokens = model._sample_next_tokens(
                selection_logits,
                temperature=0.6,
                top_p=0.9,
                do_sample=False,
            )
            if model.use_tool_head_replacement and active_decision_rows.any():
                next_tokens = model._replace_tool_triggers_with_head_predictions(
                    next_tokens=next_tokens,
                    hidden_states=last_hidden_states,
                    active_decision_rows=active_decision_rows,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                )
            next_tokens = next_tokens.masked_fill(finished, tokenizer.eos_token_id)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device)], dim=-1)
            finished = finished | (next_tokens == tokenizer.eos_token_id)
            if finished.all():
                break
            step_input_ids = next_tokens.unsqueeze(-1)
    return input_ids


def extract_raw_sequences(generated, input_length, tokenizer):
    sequences = []
    for row in generated:
        tokens = []
        for token in row[input_length:].tolist():
            if token == tokenizer.eos_token_id:
                break
            tokens.append(int(token))
        sequences.append(tokens)
    return sequences


def boundary_counts(raw_token_ids, model, expected_count):
    eoc_id = model.eoc_token_id
    tool_positions = []
    eoc_positions = []
    malformed_boundary_positions = []
    closed_segments = 0
    open_segment = False
    seen_eoc_in_segment = False

    for position, token_id in enumerate(raw_token_ids):
        if model.is_tool_token_id(token_id):
            tool_positions.append(position)
            if open_segment and not seen_eoc_in_segment:
                malformed_boundary_positions.append(position)
            open_segment = True
            seen_eoc_in_segment = False
        elif token_id == eoc_id:
            eoc_positions.append(position)
            if open_segment and not seen_eoc_in_segment:
                closed_segments += 1
                seen_eoc_in_segment = True
            else:
                malformed_boundary_positions.append(position)

    predicted_count = len(eoc_positions)
    tp = min(closed_segments, expected_count)
    fp = max(predicted_count - tp, 0)
    fn = max(expected_count - tp, 0)
    precision = tp / (tp + fp) if tp + fp > 0 else (1.0 if expected_count == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "expected_eoc_count": expected_count,
        "predicted_eoc_count": predicted_count,
        "tool_token_count": len(tool_positions),
        "valid_eoc_closures": closed_segments,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "eoc_precision": precision,
        "eoc_recall": recall,
        "eoc_f1": f1,
        "eoc_count_exact": predicted_count == expected_count,
        "missing_eoc_count": max(expected_count - predicted_count, 0),
        "extra_eoc_count": max(predicted_count - expected_count, 0),
        "malformed_boundary_count": len(malformed_boundary_positions),
        "malformed_boundary": bool(malformed_boundary_positions),
        "tool_positions": tool_positions,
        "eoc_positions": eoc_positions,
        "malformed_boundary_positions": malformed_boundary_positions,
    }


def run_prediction(args, item, out_path):
    import torch
    from transformers import AutoTokenizer

    run_config = load_json(item["run_config_path"])
    checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
    dtype = torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(run_config["args"]["model_name"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(run_config, checkpoint, tokenizer, args.device, dtype)
    if not model.use_eoc or model.eoc_token_id is None:
        raise SystemExit(f"Selected run does not expose an EOC token: {item['run_dir']}")
    del checkpoint
    gc.collect()

    data = load_json(item["data_path"])
    if args.limit is not None:
        data = data[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    method_label = METHOD_LABELS.get(item["method"], item["method"])
    with open(out_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(data), args.eval_batch_size):
            batch = data[start : start + args.eval_batch_size]
            raw_sequences = generate_raw_batch(model, tokenizer, batch, args.device, args.max_new_tokens)
            for offset, (sample, raw_token_ids) in enumerate(zip(batch, raw_sequences)):
                index = start + offset
                expected_tools = sample.get("tools", [sample.get("tool_name", "unknown")])
                metrics = boundary_counts(raw_token_ids, model, len(expected_tools))
                record = {
                    "index": index,
                    "model": item["model"],
                    "method": method_label,
                    "trial": item["trial"],
                    "run_name": item["run_dir"].name,
                    "user_input": sample["user_input"],
                    "expected_tools": expected_tools,
                    "raw_generated_token_ids": raw_token_ids,
                    "raw_generated_tokens": [tokenizer.decode([token_id]) for token_id in raw_token_ids],
                    "eoc_token_id": model.eoc_token_id,
                    "eoc_token": tokenizer.decode([model.eoc_token_id]),
                    **metrics,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = start + len(batch)
            if args.progress_every > 0 and completed % args.progress_every < args.eval_batch_size:
                print(f"Generated {completed}/{len(data)} EOC records for {item['run_dir'].name}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_file(path):
    records = load_jsonl(path)
    if not records:
        return {"samples": 0}
    total_tp = sum(int(record["tp"]) for record in records)
    total_fp = sum(int(record["fp"]) for record in records)
    total_fn = sum(int(record["fn"]) for record in records)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "samples": len(records),
        "eoc_count_exact": mean([1.0 if record["eoc_count_exact"] else 0.0 for record in records]),
        "eoc_precision": precision,
        "eoc_recall": recall,
        "eoc_f1": f1,
        "avg_expected_eoc": mean([record["expected_eoc_count"] for record in records]),
        "avg_predicted_eoc": mean([record["predicted_eoc_count"] for record in records]),
        "avg_missing_eoc": mean([record["missing_eoc_count"] for record in records]),
        "avg_extra_eoc": mean([record["extra_eoc_count"] for record in records]),
        "malformed_boundary_rate": mean([1.0 if record["malformed_boundary"] else 0.0 for record in records]),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def global_prf(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def summarize_all(selected, output_paths):
    summaries = {}
    for key, items in selected.items():
        trials = []
        for item in items:
            path = output_paths[(item["model"], item["method"], item["trial"])]
            if not path.exists():
                raise SystemExit(f"Prediction file missing for summary: {path}")
            trials.append(
                {
                    "trial": item["trial"],
                    "run_name": item["run_dir"].name,
                    "source": item["source"],
                    "run_dir": str(item["run_dir"]),
                    "checkpoint_path": str(item["checkpoint_path"]),
                    "data_path": str(item["data_path"]),
                    "prediction_path": str(path),
                    "metrics": summarize_file(path),
                }
            )

        aggregate = {"trials": len(trials)}
        for metric_key, _label in METRIC_FIELDS:
            values = [trial["metrics"].get(metric_key) for trial in trials]
            values = [value for value in values if value is not None]
            aggregate[metric_key] = mean(values)
            aggregate[f"{metric_key}_std"] = stdev(values)
        aggregate["samples"] = sum(trial["metrics"].get("samples", 0) for trial in trials)
        aggregate["total_tp"] = sum(trial["metrics"].get("total_tp", 0) for trial in trials)
        aggregate["total_fp"] = sum(trial["metrics"].get("total_fp", 0) for trial in trials)
        aggregate["total_fn"] = sum(trial["metrics"].get("total_fn", 0) for trial in trials)
        precision, recall, f1 = global_prf(aggregate["total_tp"], aggregate["total_fp"], aggregate["total_fn"])
        aggregate["eoc_precision"] = precision
        aggregate["eoc_recall"] = recall
        aggregate["eoc_f1"] = f1

        model, method = key
        summaries.setdefault(model, {})[method] = {
            "trials": trials,
            "aggregate": aggregate,
        }
    return summaries


def write_summary_json(output_dir, args, summaries):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_policy": args.selection_policy,
        "all_methods_status": str(ALL_METHODS_STATUS),
        "paper_head_status": str(PAPER_HEAD_STATUS),
        "llama3b_final_suite": str(LLAMA3B_FINAL_SUITE),
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
        "# Generated EOC Boundary Accuracy",
        "",
        f"- all_methods status: `{ALL_METHODS_STATUS}`",
        f"- paper head status: `{PAPER_HEAD_STATUS}`",
        f"- llama3b final checkpoint suite: `{LLAMA3B_FINAL_SUITE}`",
        f"- selection policy: `{args.selection_policy}`",
        f"- max new tokens: `{args.max_new_tokens}`",
        f"- eval batch size: `{args.eval_batch_size}`",
    ]
    if args.limit is not None:
        lines.append(f"- limit: `{args.limit}`")

    lines.extend(["", "## Aggregate", ""])
    header = ["Model", "Method", "Trials", "Samples"] + [label for _key, label in METRIC_FIELDS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in summaries:
        for method, payload in summaries[model].items():
            aggregate = payload["aggregate"]
            row = [
                model,
                METHOD_LABELS.get(method, method),
                str(aggregate["trials"]),
                str(aggregate["samples"]),
            ]
            row.extend(fmt(aggregate[key]) for key, _label in METRIC_FIELDS)
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Per Trial", ""])
    for model in summaries:
        for method, payload in summaries[model].items():
            lines.extend(
                [
                    f"### {model} / {METHOD_LABELS.get(method, method)}",
                    "",
                    "| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for trial in payload["trials"]:
                metrics = trial["metrics"]
                row = [
                    str(trial["trial"]),
                    str(metrics.get("samples", "")),
                    fmt(metrics.get("eoc_count_exact")),
                    fmt(metrics.get("eoc_precision")),
                    fmt(metrics.get("eoc_recall")),
                    fmt(metrics.get("eoc_f1")),
                    fmt(metrics.get("avg_predicted_eoc")),
                    fmt(metrics.get("avg_missing_eoc")),
                    fmt(metrics.get("avg_extra_eoc")),
                    f"`{trial['run_name']}`",
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    path = output_dir / "summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_per_trial(output_dir, summaries):
    path = output_dir / "per_trial.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for model, model_payload in summaries.items():
            for method, method_payload in model_payload.items():
                for trial in method_payload["trials"]:
                    record = {
                        "model": model,
                        "method": METHOD_LABELS.get(method, method),
                        "trial": trial["trial"],
                        "run_name": trial["run_name"],
                        "source": trial["source"],
                        "checkpoint_path": trial["checkpoint_path"],
                        "data_path": trial["data_path"],
                        "prediction_path": trial["prediction_path"],
                        "metrics": trial["metrics"],
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main():
    args = parse_args()
    models = split_csv(args.models)
    methods = split_csv(args.methods)
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise SystemExit(f"Unsupported method: {method}")

    output_dir = Path(args.output_dir)
    if args.selection_policy == "table1_ours":
        selected = select_table1_ours_items(models, methods, args.logit_bias_trials)
    else:
        selected = select_items(models, methods, args.trials, args.logit_bias_trials, args.llama3b_trials)
    output_paths = {}
    for items in selected.values():
        for item in items:
            output_paths[(item["model"], item["method"], item["trial"])] = output_path(output_dir, item)

    print(f"Output directory: {output_dir}")
    for model in models:
        for method in methods:
            print(f"\n{model} / {METHOD_LABELS.get(method, method)}:")
            for item in selected.get((model, method), []):
                path = output_paths[(item["model"], item["method"], item["trial"])]
                print(f"  trial {item['trial']}: {item['checkpoint_path']} -> {path}")

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for model in models:
            for method in methods:
                for item in selected.get((model, method), []):
                    path = output_paths[(item["model"], item["method"], item["trial"])]
                    if path.exists() and not args.force:
                        print(f"Skipping existing EOC records: {path}")
                        continue
                    print(
                        f"Running {model}/{METHOD_LABELS.get(method, method)} "
                        f"trial {item['trial']}: {item['run_dir'].name}"
                    )
                    run_prediction(args, item, path)

    summaries = summarize_all(selected, output_paths)
    summary_json = write_summary_json(output_dir, args, summaries)
    summary_md = write_summary_md(output_dir, args, summaries)
    per_trial = write_per_trial(output_dir, summaries)
    print(f"\nWrote summary JSON: {summary_json}")
    print(f"Wrote summary MD: {summary_md}")
    print(f"Wrote per-trial JSONL: {per_trial}")


if __name__ == "__main__":
    main()
