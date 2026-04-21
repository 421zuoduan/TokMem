#!/usr/bin/env python3
"""Evaluate the raw base model on an existing atomic fixed split."""

import argparse
import copy
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from main_in_domain_fixed_split import load_split_cache
from natural_instructions_eval import (
    evaluate_predictions,
    exact_match,
    metric_max_over_ground_truths,
    print_evaluation_results,
    rouge_score,
)
from run_layout import DEFAULT_RUNS_DIR, build_run_config, resolve_run_context, write_json
from task_dataset import DEFAULT_TASKS_DIR, NaturalInstructionsTaskDataset, collate_fn
from task_training import setup_logging, write_jsonl


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def build_prompt_text(example, model_name, include_instruction=True):
    """Build the evaluation prompt using the repo's chat formatting."""
    instruction = example.get("instruction", "")
    query = example.get("query", "")
    user_content = f"{instruction}\n\n{query}" if include_instruction else str(query)

    if "qwen" in str(model_name).lower():
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


def extract_response_text(generated_text, prompt_text=""):
    """Extract the assistant response from generated text."""
    response = generated_text

    for marker in ("<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>"):
        if marker in response:
            response = response.rsplit(marker, 1)[-1]

    if prompt_text and response.startswith(prompt_text):
        response = response[len(prompt_text):]

    for stop_marker in ("<|im_end|>", "<|eot_id|>"):
        if stop_marker in response:
            response = response.split(stop_marker, 1)[0]

    return response.strip()


def summarize_metrics(results):
    """Return a compact summary payload for downstream logging/tests."""
    return {
        "prompt_mode": results.get("prompt_mode"),
        "task_accuracy": results.get("task_accuracy"),
        "ni_exact_match": results.get("ni_exact_match"),
        "ni_rouge_l": results.get("ni_rouge_l"),
        "total_examples": results.get("total_examples"),
    }


def create_eval_dataloader(test_data, tokenizer, max_length, test_batch_size, include_instruction):
    """Create the eval dataloader without any task-token model dependency."""
    test_dataset = NaturalInstructionsTaskDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        model=None,
        mode="eval",
        include_instruction_in_prompt=include_instruction,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    print(f"Test dataset created ({'instruction_and_query' if include_instruction else 'query_only'}): {len(test_dataset)} samples")
    return test_dataloader


def get_model_input_device(model):
    """Return the device expected by the input embeddings."""
    return model.get_input_embeddings().weight.device


def decode_generated_tokens(tokenizer, token_ids):
    """Decode only the newly generated continuation up to EOS."""
    valid_tokens = []
    eos_token_id = tokenizer.eos_token_id
    for token_id in token_ids:
        value = int(token_id)
        if eos_token_id is not None and value == eos_token_id:
            break
        valid_tokens.append(value)
    return tokenizer.decode(valid_tokens, skip_special_tokens=False)


def build_prediction_row(example, predicted_response, full_generated_sequence, prompt_mode, error=None):
    """Build a JSONL row consistent with the repo's evaluation outputs."""
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
        "example_index": None,
        "mode": "base_model_generation",
        "prompt_mode": prompt_mode,
        "prediction_status": "response_only_base_model" if predicted_response else "empty_prediction",
        "instruction": example.get("instruction", ""),
        "query": example.get("query", ""),
        "expected_task": expected_tasks[0] if expected_tasks else None,
        "predicted_task": None,
        "expected_tasks": expected_tasks,
        "predicted_tasks": [],
        "has_task_prediction": False,
        "task_match": None,
        "expected_response": expected_responses[0] if expected_responses else "",
        "predicted_response": predicted_response,
        "expected_responses": expected_responses,
        "predicted_responses": [predicted_response] if predicted_response else [],
        "response_exact_match": response_exact_match,
        "response_rouge_l": round(response_rouge_l, 4),
        "full_generated_sequence": full_generated_sequence,
        "task_token_used": None,
        "source_instance_index": example.get("source_instance_index"),
    }
    if error is not None:
        row["error"] = error
    return row


def evaluate_base_model(
    model,
    tokenizer,
    test_dataloader,
    prompt_mode_label,
    predictions_output_path=None,
    max_new_tokens=256,
):
    """Run generation-only evaluation for the raw base model."""
    eval_logger = logging.getLogger("evaluation")
    model.eval()

    total_examples = len(test_dataloader.dataset)
    print(f"\n=== Base Model Evaluation (Prompt: {prompt_mode_label}) ===")
    print(f"Evaluating on {total_examples} test examples")
    eval_logger.info(f"BASE MODEL EVALUATION START - Prompt:{prompt_mode_label} Examples:{total_examples}")

    all_predictions = []
    all_references = []
    all_task_names = []
    prediction_rows = []

    start_time = time.time()
    input_device = get_model_input_device(model)
    prompt_width = None

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.use_cache = True
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    for batch_idx, batch in enumerate(test_dataloader):
        raw_examples = batch["raw_data"]
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        prompt_width = input_ids.shape[1]

        if batch_idx % 10 == 0 or (batch_idx + 1) * len(raw_examples) >= total_examples:
            processed = min((batch_idx + 1) * len(raw_examples), total_examples)
            print(f"   Progress: {processed}/{total_examples} ({processed / total_examples * 100:.1f}%)")

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

            continuations = outputs[:, prompt_width:]
            for example, continuation in zip(raw_examples, continuations):
                full_generated_sequence = decode_generated_tokens(tokenizer, continuation.tolist())
                predicted_response = extract_response_text(full_generated_sequence)

                all_predictions.append(predicted_response)
                expected_responses = example.get("responses", [""])
                all_references.append(expected_responses)
                all_task_names.append(example.get("tasks", ["unknown"])[0])

                row = build_prediction_row(
                    example=example,
                    predicted_response=predicted_response,
                    full_generated_sequence=full_generated_sequence,
                    prompt_mode=prompt_mode_label,
                )
                row["example_index"] = len(prediction_rows)
                prediction_rows.append(row)
        except Exception as exc:
            error_message = str(exc)
            print(f"   Error processing batch {batch_idx + 1}: {error_message}")
            eval_logger.info(f"BATCH ERROR - Batch:{batch_idx + 1} Error:{error_message}")
            for example in raw_examples:
                all_predictions.append("")
                all_references.append(example.get("responses", [""]))
                all_task_names.append(example.get("tasks", ["unknown"])[0])
                row = build_prediction_row(
                    example=example,
                    predicted_response="",
                    full_generated_sequence="",
                    prompt_mode=prompt_mode_label,
                    error=error_message,
                )
                row["example_index"] = len(prediction_rows)
                prediction_rows.append(row)

    print("\n🔍 Computing Natural Instructions metrics...")
    ni_results = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names,
        xlingual=False,
    )
    print_evaluation_results(ni_results, f"BASE MODEL EVALUATION ({prompt_mode_label})")

    eval_time = time.time() - start_time
    print(f"⏱️  Total evaluation time: {eval_time:.2f} seconds")
    eval_logger.info(
        "BASE MODEL EVALUATION COMPLETE - "
        f"ExactMatch:{ni_results['exact_match']:.1f}% RougeL:{ni_results['rougeL']:.1f}% "
        f"Time:{eval_time:.1f}s"
    )

    if predictions_output_path is not None:
        write_jsonl(predictions_output_path, prediction_rows)
        print(f"Saved per-example evaluation predictions to: {predictions_output_path}")

    return {
        "prompt_mode": prompt_mode_label,
        "exact_accuracy": ni_results["exact_match"] / 100.0,
        "task_accuracy": None,
        "avg_response_score": ni_results["rougeL"] / 100.0,
        "total_examples": total_examples,
        "ni_exact_match": ni_results["exact_match"],
        "ni_rouge_l": ni_results["rougeL"],
        "ni_per_task": ni_results.get("per_task", {}),
        "task_breakdown": {},
        "predictions_output_path": predictions_output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the raw base model on an atomic fixed split")
    parser.add_argument("--tasks_dir", type=str, default=DEFAULT_TASKS_DIR, help="Natural Instructions tasks directory")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--num_tasks", type=int, default=700, help="Number of tasks in the cached split")
    parser.add_argument("--train_size", type=int, default=500, help="Training samples per task in cached split metadata")
    parser.add_argument("--val_size", type=int, default=10, help="Validation samples per task in cached split metadata")
    parser.add_argument("--test_size", type=int, default=50, help="Test samples per task in cached split metadata")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated response length")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument(
        "--test_prompt_mode",
        type=str,
        default="instruction_and_query",
        choices=["instruction_and_query", "query_only"],
        help="Whether to include instructions in the evaluation prompt",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use when device_map is not set")
    parser.add_argument(
        "--device_map",
        type=str,
        default="balanced",
        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
        help="Optional Hugging Face device_map for sharding the model",
    )
    parser.add_argument("--few_shot", action="store_true", help="Expected few-shot flag stored in split metadata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_cache_path", type=str, required=True, help="Path to a saved split cache file")
    parser.add_argument("--run_root_dir", type=str, default=DEFAULT_RUNS_DIR, help="Directory where run folders will be created")
    parser.add_argument("--run_name", type=str, default=None, help="Optional explicit run folder name")
    parser.add_argument("--run_tag", type=str, default="fixed_split_base_model", help="Tag appended to auto-generated run names")
    args = parser.parse_args()

    run_context = resolve_run_context(
        experiment_name="atomic_base_model",
        model_name=args.model_name,
        num_tasks=args.num_tasks,
        run_root_dir=args.run_root_dir,
        run_name=args.run_name,
        run_tag=args.run_tag,
    )

    _, _, training_log, evaluation_log, stdout_log, timestamp = setup_logging(
        log_dir=run_context["run_dir"],
        model_name=args.model_name,
        num_tasks=args.num_tasks,
        stdout_prefix="evaluation",
        timestamp=run_context["timestamp"],
    )

    set_random_seed(args.seed)
    print()
    print("=" * 60)
    print("RAW BASE MODEL EVALUATION (FIXED SPLIT)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Device map: {args.device_map}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Test prompt mode: {args.test_prompt_mode}")
    print(f"Run directory: {run_context['run_dir']}")
    print(f"Split cache path: {os.path.abspath(args.split_cache_path)}")
    print("Setting up logging...")
    print(f"   Training log: {training_log}")
    print(f"   Evaluation log: {evaluation_log}")
    print(f"   Stdout log: {stdout_log}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print()

    print("Loading cached split...")
    train_data, val_data, test_data, task_names, split_cache_metadata = load_split_cache(args)
    print()

    include_instruction = args.test_prompt_mode == "instruction_and_query"
    example_prompt = build_prompt_text(test_data[0], args.model_name, include_instruction=include_instruction) if test_data else ""

    write_json(
        os.path.join(run_context["run_dir"], "run_config.json"),
        build_run_config(
            vars(args),
            run_context,
            extra={
                "split_cache_path": os.path.abspath(args.split_cache_path),
                "dataset_summary": {
                    "train_examples": len(train_data),
                    "val_examples": len(val_data),
                    "test_examples": len(test_data),
                    "task_count": len(task_names),
                },
                "base_model_only": True,
                "example_prompt_preview": example_prompt[:500],
            },
        ),
    )
    write_json(
        os.path.join(run_context["run_dir"], "split_cache_metadata.json"),
        split_cache_metadata,
    )

    print("Loading base model...")
    model_load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.device_map is not None:
        model_load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    print("Base model loaded")
    print()

    print("Creating eval dataloader...")
    test_dataloader = create_eval_dataloader(
        test_data=test_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        test_batch_size=args.test_batch_size,
        include_instruction=include_instruction,
    )
    print()

    print("Running comprehensive evaluation...")
    predictions_output_path = os.path.join(
        run_context["run_dir"],
        f"evaluation_predictions_{args.test_prompt_mode}.jsonl",
    )
    results = evaluate_base_model(
        model=model,
        tokenizer=tokenizer,
        test_dataloader=test_dataloader,
        prompt_mode_label=args.test_prompt_mode,
        predictions_output_path=predictions_output_path,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY:")
    print(f"   Prompt mode: {args.test_prompt_mode}")
    print("   Task Prediction Accuracy: N/A (raw base model, no task routing)")
    print(f"   Exact Match Accuracy: {results['exact_accuracy']:.3f}")
    print(f"   Average Response Score: {results['avg_response_score']:.3f}")
    print("=" * 50)

    evaluation_results_path = os.path.join(
        run_context["run_dir"],
        f"evaluation_results_{args.test_prompt_mode}.json",
    )
    write_json(evaluation_results_path, results)
    write_json(os.path.join(run_context["run_dir"], "evaluation_results.json"), results)
    write_json(
        os.path.join(run_context["run_dir"], "run_summary.json"),
        {
            "run_name": run_context["run_name"],
            "run_dir": run_context["run_dir"],
            "split_cache_path": os.path.abspath(args.split_cache_path),
            "model_name": args.model_name,
            "num_tasks": len(task_names),
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "test_examples": len(test_data),
            "base_model_only": True,
            "task_tokens_path": None,
            "best_task_tokens_path": None,
            "final_task_tokens_path": None,
            "evaluation_predictions_path": results.get("predictions_output_path"),
            "evaluation_results_path": evaluation_results_path,
            "metrics": results,
            "summary": summarize_metrics(results),
        },
    )

    print("\nRaw base-model evaluation completed!")


if __name__ == "__main__":
    main()
