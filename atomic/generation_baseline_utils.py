#!/usr/bin/env python3
"""Shared helpers for atomic generation-only baselines."""

import copy
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from natural_instructions_eval import (
    evaluate_predictions,
    exact_match,
    metric_max_over_ground_truths,
    rouge_score,
)
from task_dataset import sample_natural_instructions_tasks


def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def write_json(path, payload):
    """Write JSON with stable formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    """Write JSONL rows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_tasks_dir(tasks_dir):
    """Resolve the tasks directory against the repo layout used in this workspace."""
    if os.path.exists(tasks_dir):
        return os.path.abspath(tasks_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, tasks_dir),
        os.path.join(script_dir, "..", "datasets", "natural-instructions-2.8", "tasks"),
        os.path.join(script_dir, "natural-instructions-2.8", "tasks"),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(tasks_dir)


def default_run_dir(experiment_tag, model_name, num_tasks):
    """Build a default run directory under atomic/runs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_label = os.path.basename(str(model_name).rstrip("/")) or str(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"atomic_{model_label}_{num_tasks}tasks_{experiment_tag}_{timestamp}"
    return os.path.join(script_dir, "runs", run_name)


def prepare_run_dir(run_dir, experiment_tag, model_name, num_tasks):
    """Create the target run directory and return its absolute path."""
    final_run_dir = os.path.abspath(
        run_dir if run_dir else default_run_dir(experiment_tag, model_name, num_tasks)
    )
    os.makedirs(final_run_dir, exist_ok=True)
    return final_run_dir


def ensure_tokenizer_padding(tokenizer):
    """Ensure the tokenizer is ready for left-padded generation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def is_qwen_tokenizer(tokenizer):
    """Return whether the tokenizer belongs to a Qwen model family."""
    return "qwen" in tokenizer.name_or_path.lower()


def build_generation_prompt(tokenizer, instruction, query, few_shot_examples=None):
    """Build a generation prompt compatible with the repo's training/eval format."""
    examples = few_shot_examples or []
    conversation_parts = []

    if is_qwen_tokenizer(tokenizer):
        if examples:
            for idx, example in enumerate(examples):
                example_input = example.get("input", "")
                example_output = normalize_example_output(example.get("output", ""))
                if idx == 0:
                    conversation_parts.append(
                        f"<|im_start|>user\n{instruction}\n\n{example_input}<|im_end|>\n"
                    )
                else:
                    conversation_parts.append(
                        f"<|im_start|>user\n{example_input}<|im_end|>\n"
                    )
                conversation_parts.append(
                    f"<|im_start|>assistant\n{example_output}<|im_end|>\n"
                )
            conversation_parts.append(f"<|im_start|>user\n{query}<|im_end|>\n")
            conversation_parts.append("<|im_start|>assistant\n")
        else:
            conversation_parts.append(
                f"<|im_start|>user\n{instruction}\n\n{query}<|im_end|>\n"
            )
            conversation_parts.append("<|im_start|>assistant\n")
    else:
        conversation_parts.append("<|begin_of_text|>")
        if examples:
            for idx, example in enumerate(examples):
                example_input = example.get("input", "")
                example_output = normalize_example_output(example.get("output", ""))
                if idx == 0:
                    conversation_parts.append(
                        f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n\n{example_input}<|eot_id|>"
                    )
                else:
                    conversation_parts.append(
                        f"<|start_header_id|>user<|end_header_id|>\n\n{example_input}<|eot_id|>"
                    )
                conversation_parts.append(
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n{example_output}<|eot_id|>"
                )
            conversation_parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
            )
            conversation_parts.append("<|start_header_id|>assistant<|end_header_id|>")
        else:
            conversation_parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n\n{query}<|eot_id|>"
            )
            conversation_parts.append("<|start_header_id|>assistant<|end_header_id|>")

    return "".join(conversation_parts)


def normalize_example_output(output_value):
    """Normalize Positive Examples output into a printable string."""
    if isinstance(output_value, list):
        return output_value[0] if output_value else ""
    return str(output_value)


def extract_response_text(tokenizer, generated_text, prompt_text=""):
    """Extract the assistant response from generated text."""
    response = generated_text

    if is_qwen_tokenizer(tokenizer):
        marker = "<|im_start|>assistant\n"
        stop_marker = "<|im_end|>"
    else:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        stop_marker = "<|eot_id|>"

    if marker in response:
        response = response.rsplit(marker, 1)[-1]

    if prompt_text and response.startswith(prompt_text):
        response = response[len(prompt_text):]

    if stop_marker in response:
        response = response.split(stop_marker, 1)[0]

    return response.strip()


def get_model_input_device(model):
    """Return the device expected by the input embeddings."""
    return model.get_input_embeddings().weight.device


def load_cached_split_relaxed(args):
    """Load a cached split while allowing few-shot augmentation at evaluation time."""
    cache_path = os.path.abspath(args.split_cache_path)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Split cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu")
    required_keys = ("train_data", "val_data", "test_data", "task_names")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise KeyError(f"Split cache missing keys: {missing_keys}")

    cached_metadata = payload.get("metadata", {})
    strict_keys = {
        "num_tasks": args.num_tasks,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "max_length": args.max_length,
    }
    mismatches = []
    for key, expected_value in strict_keys.items():
        cached_value = cached_metadata.get(key)
        if cached_value is not None and cached_value != expected_value:
            mismatches.append((key, expected_value, cached_value))

    if mismatches:
        mismatch_text = "\n".join(
            f"  {key}: expected={expected!r}, cached={cached!r}"
            for key, expected, cached in mismatches
        )
        raise ValueError(f"Cached split metadata mismatch for {cache_path}:\n{mismatch_text}")

    return (
        payload["train_data"],
        payload["val_data"],
        payload["test_data"],
        payload["task_names"],
        cache_path,
    )


def load_generation_split(args, few_shot):
    """Load a cached split or sample a fresh runtime split."""
    resolved_tasks_dir = resolve_tasks_dir(args.tasks_dir)
    if args.split_cache_path:
        train_data, val_data, test_data, task_names, split_source = load_cached_split_relaxed(args)
    else:
        train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(
            tasks_dir=resolved_tasks_dir,
            num_tasks=args.num_tasks,
            max_instruction_tokens=args.max_instruction_tokens,
            tokenizer=args.tokenizer_for_sampling,
            stable_test_split=True,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            few_shot=few_shot,
        )
        split_source = "runtime_sampling"
    return train_data, val_data, test_data, task_names, split_source, resolved_tasks_dir


def build_prediction_row(
    example,
    predicted_response,
    full_generated_sequence,
    prompt_text,
    mode,
    extra=None,
    include_verbose=False,
):
    """Build a per-example prediction row."""
    expected_tasks = example.get("tasks", ["unknown"])
    expected_responses = example.get("responses", [""])

    response_exact_match = bool(
        metric_max_over_ground_truths(
            lambda prediction, ground_truth, xlingual=False: (
                1.0 if exact_match(prediction, ground_truth, xlingual) else 0.0
            ),
            predicted_response,
            expected_responses,
            xlingual=False,
        )
    )
    response_rouge_l = metric_max_over_ground_truths(
        rouge_score,
        predicted_response,
        expected_responses,
        xlingual=False,
    )

    row = {
        "mode": mode,
        "query": example.get("query", ""),
        "expected_task": expected_tasks[0] if expected_tasks else None,
        "expected_tasks": expected_tasks,
        "expected_response": expected_responses[0] if expected_responses else "",
        "expected_responses": expected_responses,
        "predicted_response": predicted_response,
        "response_exact_match": response_exact_match,
        "response_rouge_l": round(response_rouge_l, 4),
    }
    if include_verbose:
        row.update(
            {
                "instruction": example.get("instruction", ""),
                "prompt_preview": prompt_text[:500],
                "full_generated_sequence": full_generated_sequence,
            }
        )
    if extra:
        row.update(extra)
    return row


def evaluate_generation_batches(
    model,
    tokenizer,
    test_examples,
    batch_builder,
    max_length,
    max_new_tokens,
    batch_size,
    mode,
    logger=None,
    include_verbose_predictions=False,
):
    """Run batch generation and evaluation with a callback that builds prompts."""
    model.eval()

    all_predictions = []
    all_references = []
    all_task_names = []
    prediction_rows = []

    input_device = get_model_input_device(model)
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.use_cache = True
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    total_examples = len(test_examples)
    for start_idx in range(0, total_examples, batch_size):
        batch_examples = test_examples[start_idx:start_idx + batch_size]
        batch_payloads = [batch_builder(example) for example in batch_examples]
        batch_texts = [payload["prompt_text"] for payload in batch_payloads]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        for offset, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=False)
            prompt_text = batch_payloads[offset]["prompt_text"]
            predicted_response = extract_response_text(
                tokenizer,
                generated_text,
                prompt_text=prompt_text,
            )

            example = batch_examples[offset]
            all_predictions.append(predicted_response)
            all_references.append(example["responses"])
            all_task_names.append(example["tasks"][0] if example.get("tasks") else "unknown")
            row = build_prediction_row(
                example=example,
                predicted_response=predicted_response,
                full_generated_sequence=generated_text,
                prompt_text=prompt_text,
                mode=mode,
                extra=batch_payloads[offset].get("row_extra"),
                include_verbose=include_verbose_predictions,
            )
            row["example_index"] = start_idx + offset
            prediction_rows.append(row)

        processed = min(start_idx + len(batch_examples), total_examples)
        progress_message = f"Progress: {processed}/{total_examples} ({processed / total_examples * 100:.1f}%)"
        print(f"   {progress_message}")
        if logger is not None:
            logger.info(progress_message)

    metrics = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names,
    )
    metrics.update(
        {
            "exact_accuracy": round(metrics["exact_match"] / 100.0, 4),
            "task_accuracy": None,
            "avg_response_score": round(metrics["rougeL"] / 100.0, 4),
            "total_examples": total_examples,
            "ni_exact_match": metrics["exact_match"],
            "ni_rouge_l": metrics["rougeL"],
            "ni_per_task": metrics.get("per_task", {}),
            "task_breakdown": {},
        }
    )
    return metrics, all_predictions, prediction_rows
