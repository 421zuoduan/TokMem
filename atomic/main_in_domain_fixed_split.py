#!/usr/bin/env python3
"""
Natural Instructions task learning with a precomputed cached train/val/test split.

This variant always loads an existing split cache file so repeated runs share
the exact same task/sample split produced ahead of time.
"""

import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from run_layout import DEFAULT_RUNS_DIR, build_run_config, resolve_run_context, write_json
from task_model import TaskCallingModel, print_model_info
from task_dataset import (
    DEFAULT_TASKS_DIR,
    create_natural_instructions_dataloader,
)
from task_training import (
    demo_task_calling,
    eval_task_calling,
    setup_logging,
    save_trained_model,
    train_task_calling_model,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def add_reserved_special_tokens(tokenizer, num_of_tasks, device="cuda"):
    """Add reserved special tokens to the tokenizer."""
    start_idx = len([t for t in tokenizer.get_vocab() if t.startswith("<|reserved_special_token_")])

    if num_of_tasks <= start_idx:
        return tokenizer, False

    num_additional_tokens = num_of_tasks - start_idx
    new_tokens = [
        f"<|reserved_special_token_{i}|>" for i in range(start_idx, start_idx + num_additional_tokens)
    ]
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    assert added == num_additional_tokens, (
        f"Expected to add {num_additional_tokens} tokens, but added {added}"
    )

    return tokenizer, True


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


def build_split_metadata(args):
    """Build metadata describing how the cached split was created."""
    return {
        "tasks_dir": os.path.abspath(args.tasks_dir),
        "model_name": args.model_name,
        "num_tasks": args.num_tasks,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "few_shot": args.few_shot,
        "seed": args.seed,
    }


def validate_cached_split_metadata(cache_path, expected_metadata, cached_metadata):
    """Ensure the cache matches the current split-defining arguments."""
    mismatches = []
    for key, expected_value in expected_metadata.items():
        cached_value = cached_metadata.get(key)
        if key == "model_name" and cached_value != expected_value:
            compatible_model_names = cached_metadata.get("compatible_model_names", [])
            if expected_value in compatible_model_names:
                continue
        if cached_value != expected_value:
            mismatches.append((key, expected_value, cached_value))

    if mismatches:
        lines = [f"Cached split metadata in {cache_path} does not match current arguments:"]
        for key, expected_value, cached_value in mismatches:
            lines.append(
                f"  {key}: expected={expected_value!r}, cached={cached_value!r}"
            )
        lines.append(
            "Change --split_cache_path or rebuild/resample it with the corresponding "
            "scripts/build_atomic_*_task_pool.sh and scripts/sample_atomic_*_fixed_split*.sh scripts."
        )
        raise ValueError("\n".join(lines))


def load_split_cache(args):
    """Load and validate a precomputed split cache."""
    cache_path = os.path.abspath(args.split_cache_path)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Split cache not found: {cache_path}\n"
            "Generate it first with the corresponding scripts/sample_atomic_*_fixed_split*.sh "
            "script or point --split_cache_path to an existing .pt file."
        )

    expected_metadata = build_split_metadata(args)

    print(f"Loading cached split from: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    cached_metadata = payload.get("metadata", {})
    validate_cached_split_metadata(cache_path, expected_metadata, cached_metadata)
    print(
        "Cached split loaded. "
        f"Train: {len(payload['train_data'])}, Val: {len(payload['val_data'])}, "
        f"Test: {len(payload['test_data'])}, Tasks: {len(payload['task_names'])}"
    )
    return (
        payload["train_data"],
        payload["val_data"],
        payload["test_data"],
        payload["task_names"],
        cached_metadata,
    )


def main():
    parser = argparse.ArgumentParser(description="Natural Instructions Task Learning (fixed split cache)")
    parser.add_argument(
        "--tasks_dir",
        type=str,
        default=DEFAULT_TASKS_DIR,
        help="Directory containing Natural Instructions task files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks to sample")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Test batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
        help="Optional Hugging Face device_map for sharding the frozen backbone across multiple GPUs",
    )
    parser.add_argument(
        "--decouple_embeddings",
        action="store_true",
        help="Use separate input/output embeddings for task tokens",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--demo", action="store_true", help="Only run demo on a few examples")
    parser.add_argument(
        "--load_task_tokens",
        type=str,
        default=None,
        help="Path to saved task tokens file (for evaluation/inference)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--train_size", type=int, default=None, help="Absolute number of training samples per task")
    parser.add_argument("--val_size", type=int, default=None, help="Absolute number of validation samples per task")
    parser.add_argument("--test_size", type=int, default=None, help="Absolute number of test samples per task")
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot instructions")
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=1000,
        help="Validate every n steps",
    )
    parser.add_argument(
        "--split_cache_path",
        type=str,
        required=True,
        help="Path to a saved split cache file",
    )
    parser.add_argument(
        "--run_root_dir",
        type=str,
        default=DEFAULT_RUNS_DIR,
        help="Directory where atomic run folders will be created",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run folder name",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="fixed_split",
        help="Short tag appended to auto-generated run names",
    )
    args = parser.parse_args()

    run_context = resolve_run_context(
        experiment_name="atomic_tokmem",
        model_name=args.model_name,
        num_tasks=args.num_tasks,
        run_root_dir=args.run_root_dir,
        run_name=args.run_name,
        run_tag=args.run_tag,
    )

    stdout_prefix = "evaluation" if args.skip_training else "training"
    training_logger, eval_logger, training_log, evaluation_log, stdout_log, timestamp = setup_logging(
        log_dir=run_context["run_dir"],
        model_name=args.model_name,
        num_tasks=args.num_tasks,
        stdout_prefix=stdout_prefix,
        timestamp=run_context["timestamp"],
    )

    set_random_seed(args.seed)
    print()

    print("=" * 60)
    print("NATURAL INSTRUCTIONS TASK LEARNING (FIXED SPLIT)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Device map: {args.device_map}")
    print(f"Number of tasks to sample: {args.num_tasks}")
    print(f"Decouple embeddings: {args.decouple_embeddings}")
    print(f"Validation batch size: {args.val_batch_size}")
    print(f"Test batch size: {args.test_batch_size}")
    print(f"Run directory: {run_context['run_dir']}")
    print(f"Split cache path: {os.path.abspath(args.split_cache_path)}")
    if any(x is not None for x in [args.train_size, args.val_size, args.test_size]):
        print(
            f"Sizes mode per task - Train: {args.train_size}, Val: {args.val_size}, "
            f"Test: {args.test_size} (test is selected first, stable)"
        )
    print(f"Random seed: {args.seed}")
    print()

    print("Setting up logging...")
    print(f"   Training log: {training_log}")
    print(f"   Evaluation log: {evaluation_log}")
    print(f"   Stdout log: {stdout_log}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print()

    print("Loading tokenizer with reserved task tokens...")
    tokenizer, is_extended = add_reserved_special_tokens(tokenizer, args.num_tasks)
    print(f"Tokenizer loaded with adjustments. Vocab size: {len(tokenizer)}")
    print()

    print("Loading cached split...")
    train_data, val_data, test_data, task_names, split_cache_metadata = load_split_cache(args)
    print()

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
            },
        ),
    )
    write_json(
        os.path.join(run_context["run_dir"], "split_cache_metadata.json"),
        split_cache_metadata,
    )

    print("Initializing Task Calling Model...")
    model = TaskCallingModel(
        model_name=args.model_name,
        num_tasks=len(task_names),
        task_names=task_names,
        tokenizer=tokenizer,
        device=args.device,
        decouple_embeddings=args.decouple_embeddings,
        is_extended=is_extended,
        device_map=args.device_map,
    )

    print("\nModel Information:")
    print_model_info(model.model, "Base Model (Frozen)")
    print_model_info(model, "Task Model (Trainable Task Tokens)")
    print()

    if args.load_task_tokens:
        if os.path.exists(args.load_task_tokens):
            print(f"Loading task tokens from: {args.load_task_tokens}")
            model.load_task_tokens(args.load_task_tokens)
            print("Task tokens loaded successfully!")
        else:
            print(f"Error: Task tokens file not found: {args.load_task_tokens}")
            return
        print()

    print("Creating data loaders...")
    train_dataloader, val_dataloader, test_dataloader, tokenizer, test_examples = (
        create_natural_instructions_dataloader(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            val_batch_size=args.val_batch_size,
            test_batch_size=args.test_batch_size,
        )
    )

    if not args.skip_training and train_dataloader:
        print("Starting Training...")
        train_results = train_task_calling_model(
            model=model,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=args.device,
            timestamp=timestamp,
            save_dir=run_context["run_dir"],
            validate_every_n_steps=args.validate_every_n_steps,
        )
        print(f"Training completed with average loss: {train_results['avg_total_loss']:.4f}")
        final_model_path = save_trained_model(
            model,
            save_dir=run_context["run_dir"],
            timestamp=timestamp,
            suffix="final",
        )
        write_json(
            os.path.join(run_context["run_dir"], "train_results.json"),
            {
                "avg_total_loss": train_results["avg_total_loss"],
                "best_val_loss": train_results["best_val_loss"],
                "best_model_path": train_results["best_model_path"],
                "final_model_path": final_model_path,
            },
        )
        print(f"Final model saved to: {final_model_path}")

        # Restore best validation checkpoint for downstream demo/evaluation consistency.
        if train_results["best_model_state"] is not None:
            print(f"Loading best model state (validation loss: {train_results['best_val_loss']:.4f})")
            best_state = train_results["best_model_state"]
            model.load_state_dict(best_state, strict=False)
        print()

    if args.demo:
        print("Running demo on sample examples...")
        demo_examples = random.sample(test_examples, min(5, len(test_examples)))
        demo_task_calling(model, tokenizer, demo_examples, device=args.device)
        print()

    if test_dataloader:
        print("Running comprehensive evaluation...")
        predictions_output_path = os.path.join(
            run_context["run_dir"], "evaluation_predictions.jsonl"
        )
        results = eval_task_calling(
            model=model,
            tokenizer=tokenizer,
            test_dataloader=test_dataloader,
            device=args.device,
            use_ground_truth_tasks=False,
            predictions_output_path=predictions_output_path,
        )

        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY:")
        print(f"   Task Prediction Accuracy: {results['task_accuracy']:.3f}")
        print(f"   Exact Match Accuracy: {results['exact_accuracy']:.3f}")
        print(f"   Average Response Score: {results['avg_response_score']:.3f}")
        print("=" * 50)
        write_json(
            os.path.join(run_context["run_dir"], "evaluation_results.json"),
            results,
        )
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
                "task_tokens_path": train_results["best_model_path"] if not args.skip_training and train_dataloader else None,
                "best_task_tokens_path": train_results["best_model_path"] if not args.skip_training and train_dataloader else None,
                "final_task_tokens_path": final_model_path if not args.skip_training and train_dataloader else None,
                "evaluation_predictions_path": results.get("predictions_output_path"),
                "metrics": results,
            },
        )

    print("\nTask learning pipeline completed!")


if __name__ == "__main__":
    main()
