#!/usr/bin/env python3
"""Evaluate the raw base model on atomic Natural Instructions tasks."""

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_baseline_utils import (
    build_generation_prompt,
    ensure_tokenizer_padding,
    evaluate_generation_batches,
    load_generation_split,
    prepare_run_dir,
    set_random_seed,
    write_json,
    write_jsonl,
)
from task_training import setup_logging


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Raw base-model evaluation for atomic Natural Instructions")
    parser.add_argument("--tasks_dir", type=str, default="natural-instructions-2.8/tasks",
                        help="Directory containing Natural Instructions task files")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--num_tasks", type=int, default=50, help="Number of tasks to sample")
    parser.add_argument("--train_size", type=int, default=500, help="Training samples per task")
    parser.add_argument("--val_size", type=int, default=10, help="Validation samples per task")
    parser.add_argument("--test_size", type=int, default=50, help="Test samples per task")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_instruction_tokens", type=int, default=1024,
                        help="Maximum token length for instructions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum generated response length")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use when device_map is not set")
    parser.add_argument("--device_map", type=str, default=None,
                        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
                        help="Optional Hugging Face device_map for sharding the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_cache_path", type=str, default=None,
                        help="Path to a cached train/val/test split")
    parser.add_argument("--save_verbose_predictions", action="store_true",
                        help="Save instruction, prompt preview, and full decoded sequence in evaluation_predictions.jsonl")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory where logs and results will be written")
    return parser.parse_args()


def main():
    """Run raw base-model evaluation."""
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = prepare_run_dir(
        run_dir=args.run_dir,
        experiment_tag="base_model",
        model_name=args.model_name,
        num_tasks=args.num_tasks,
    )
    _, eval_logger, training_log, evaluation_log, timestamp = setup_logging(log_dir=run_dir)

    print()
    print("=" * 60)
    print("RAW BASE MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Device map: {args.device_map}")
    print(f"Run directory: {run_dir}")
    print(f"Training log: {training_log}")
    print(f"Evaluation log: {evaluation_log}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = ensure_tokenizer_padding(tokenizer)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}, padding side: {tokenizer.padding_side}")
    print()

    args.tokenizer_for_sampling = tokenizer
    train_data, val_data, test_data, task_names, split_source, resolved_tasks_dir = load_generation_split(
        args,
        few_shot=False,
    )

    print(f"Tasks directory: {resolved_tasks_dir}")
    print(f"Split source: {split_source}")
    print(f"Dataset summary - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Tasks: {len(task_names)}")
    print()

    print("Loading base model...")
    model_load_kwargs = {"torch_dtype": "auto"}
    if args.device_map is not None:
        model_load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    print("Base model loaded")
    print()

    write_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "mode": "base_model",
            "model_name": args.model_name,
            "tasks_dir": resolved_tasks_dir,
            "num_tasks": args.num_tasks,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "max_length": args.max_length,
            "max_instruction_tokens": args.max_instruction_tokens,
            "max_new_tokens": args.max_new_tokens,
            "test_batch_size": args.test_batch_size,
            "device": args.device,
            "device_map": args.device_map,
            "seed": args.seed,
            "split_cache_path": os.path.abspath(args.split_cache_path) if args.split_cache_path else None,
            "save_verbose_predictions": args.save_verbose_predictions,
            "run_dir": run_dir,
            "timestamp": timestamp,
            "dataset_summary": {
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "task_count": len(task_names),
            },
        },
    )

    print("Running generation evaluation...")

    def build_batch_payload(example):
        prompt_text = build_generation_prompt(
            tokenizer=tokenizer,
            instruction=example.get("instruction", ""),
            query=example.get("query", ""),
            few_shot_examples=None,
        )
        return {
            "prompt_text": prompt_text,
            "row_extra": {
                "prompt_style": "instruction_and_query",
                "few_shot_count": 0,
            },
        }

    results, _, prediction_rows = evaluate_generation_batches(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_data,
        batch_builder=build_batch_payload,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.test_batch_size,
        mode="base_model_generation",
        logger=eval_logger,
        include_verbose_predictions=args.save_verbose_predictions,
    )

    predictions_path = os.path.join(run_dir, "evaluation_predictions.jsonl")
    write_jsonl(predictions_path, prediction_rows)
    results["predictions_output_path"] = predictions_path

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print(f"Exact Match Accuracy: {results['exact_accuracy']:.3f}")
    print(f"Average Response Score: {results['avg_response_score']:.3f}")
    print("=" * 50)

    evaluation_results_path = os.path.join(run_dir, "evaluation_results.json")
    write_json(evaluation_results_path, results)
    write_json(
        os.path.join(run_dir, "run_summary.json"),
        {
            "mode": "base_model",
            "run_dir": run_dir,
            "model_name": args.model_name,
            "split_source": split_source,
            "metrics": results,
            "dataset_summary": {
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "task_count": len(task_names),
            },
            "artifacts": {
                "training_log": training_log,
                "evaluation_log": evaluation_log,
                "evaluation_results": evaluation_results_path,
                "evaluation_predictions": predictions_path,
            },
        },
    )

    print("\nRaw base-model evaluation completed!")


if __name__ == "__main__":
    main()
