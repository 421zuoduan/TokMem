#!/usr/bin/env python3
"""
Entry point for atomic TokMem runs with a runtime-sampled split.

This script samples English Natural Instructions tasks on the fly, saves the
resulting split into the run folder, trains task-token embeddings, and then
evaluates with instruction+query prompts.
"""

import torch
from transformers import AutoTokenizer
import argparse
import os
import random
import numpy as np

from run_layout import DEFAULT_RUNS_DIR, build_run_config, resolve_run_context, write_json
# Import our custom modules
from task_model import TaskCallingModel, print_model_info
from task_dataset import (
    DEFAULT_TASKS_DIR,
    create_natural_instructions_dataloader, 
    sample_natural_instructions_tasks
)
from task_training import (
    train_task_calling_model,
    demo_task_calling,
    eval_task_calling,
    setup_logging,
    save_trained_model,
)


def parse_bool_arg(value):
    """Parse a CLI boolean argument from explicit True/False style strings."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value: True or False')

def add_reserved_special_tokens(tokenizer, num_of_tasks, device="cuda"):
    """Add reserved special tokens to the tokenizer"""
    start_idx = len([t for t in tokenizer.get_vocab() if t.startswith("<|reserved_special_token_")])

    if num_of_tasks <= start_idx:
        return tokenizer, False
    else:
        num_additional_tokens = num_of_tasks - start_idx
        new_tokens = [f"<|reserved_special_token_{i}|>" for i in range(start_idx, start_idx + num_additional_tokens)]
        added = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        assert added == num_additional_tokens, f"Expected to add {num_additional_tokens} tokens, but added {added}"

        return tokenizer, True

def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Make deterministic operations more deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")


def build_runtime_split_payload(args, train_data, val_data, test_data, task_names):
    metadata = {
        "tasks_dir": os.path.abspath(args.tasks_dir),
        "model_name": args.model_name,
        "num_tasks": args.num_tasks,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "few_shot": args.few_shot,
        "seed": args.seed,
        "source": "runtime_split",
        "max_length": args.max_length,
    }
    return {
        "metadata": metadata,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "task_names": task_names,
    }

def main():
    parser = argparse.ArgumentParser(description='Natural Instructions Task Learning')
    parser.add_argument('--tasks_dir', type=str, default=DEFAULT_TASKS_DIR, 
                        help='Directory containing Natural Instructions task files')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help='HuggingFace model name')
    parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks to sample')
    # Remove ratio-based splitting in favor of absolute sizes
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument(
        '--generation_routing',
        type=str,
        default='first_step_routing',
        choices=['full_vocab_generation', 'first_step_routing'],
        help='How to handle the first generated token during inference'
    )
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--device_map', type=str, default=None,
                        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
                        help='Optional Hugging Face device_map for sharding the frozen backbone across multiple GPUs')
    parser.add_argument('--decouple_embeddings', action='store_true', 
                        help='Use separate input/output embeddings for task tokens')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only run evaluation')
    parser.add_argument('--demo', action='store_true', help='Only run demo on a few examples')
    parser.add_argument('--load_task_tokens', type=str, default=None, 
                        help='Path to saved task tokens file (for evaluation/inference)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--shuffle_train', action='store_true',
                        help='Shuffle the training dataloader')
    # Absolute per-task sizes (override ratios when provided)
    parser.add_argument('--train_size', type=int, default=None, help='Absolute number of training samples per task (overrides train_ratio)')
    parser.add_argument('--val_size', type=int, default=None, help='Absolute number of validation samples per task (overrides val_ratio)')
    parser.add_argument('--test_size', type=int, default=None, help='Absolute number of test samples per task (overrides test_ratio; test selected first deterministically)')
    parser.add_argument('--few_shot', action='store_true', help='Use few-shot instructions')
    parser.add_argument('--validate_every_n_steps', type=int, default=1000, 
                        help='Validate every n steps')
    parser.add_argument('--use_task_loss', type=parse_bool_arg, default=False, metavar='BOOL',
                        help='Whether to include task-token cross entropy in the optimization objective')
    parser.add_argument('--task_loss_weight', type=float, default=0.01,
                        help='Weight for the task-token routing cross entropy loss')
    parser.add_argument('--mean_loss_weight', type=float, default=0.01,
                        help='Weight for the mean-direction memory bank regularizer')
    parser.add_argument('--use_mean_loss', type=parse_bool_arg, default=False, metavar='BOOL',
                        help='Whether to include the mean-direction memory bank regularizer')
    parser.add_argument('--use_hard_negative_loss', type=parse_bool_arg, default=False, metavar='BOOL',
                        help='Whether to include hardest-negative routing margin loss inside the memory bank')
    parser.add_argument('--hard_negative_loss_weight', type=float, default=0.01,
                        help='Weight for the hard-negative routing loss')
    parser.add_argument('--hard_negative_margin', type=float, default=0.2,
                        help='Margin required between the positive and hardest-negative routing logits')
    parser.add_argument('--use_sep_loss', type=parse_bool_arg, default=False, metavar='BOOL',
                        help='Whether to include separation loss between task embeddings in the optimization objective')
    parser.add_argument('--sep_loss_weight', type=float, default=0.01,
                        help='Weight for the memory-token separation loss')
    parser.add_argument('--sep_loss_tau', type=float, default=0.3,
                        help='Cosine-similarity margin for the memory-token separation loss')
    parser.add_argument('--use_centered_sep', type=parse_bool_arg, default=False, metavar='BOOL',
                        help='Whether to compute separation loss after subtracting the mean direction')
    parser.add_argument('--compute_memory_bank_geometry_stats', type=parse_bool_arg, default=False, metavar='BOOL',
                        help='Whether to compute auxiliary memory-bank geometry stats such as effective rank')
    parser.add_argument('--run_root_dir', type=str, default=DEFAULT_RUNS_DIR,
                        help='Directory where atomic run folders will be created')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional explicit run folder name')
    parser.add_argument('--run_tag', type=str, default='runtime_split',
                        help='Short tag appended to auto-generated run names')
    args = parser.parse_args()

    # ---- Run context and logging ----
    run_context = resolve_run_context(
        experiment_name="atomic_tokmem",
        model_name=args.model_name,
        num_tasks=args.num_tasks,
        run_root_dir=args.run_root_dir,
        run_name=args.run_name,
        run_tag=args.run_tag,
    )
    
    stdout_prefix = "evaluation" if args.skip_training else "training"
    _, _, training_log, evaluation_log, stdout_log, timestamp = setup_logging(
        log_dir=run_context["run_dir"],
        model_name=args.model_name,
        num_tasks=args.num_tasks,
        stdout_prefix=stdout_prefix,
        timestamp=run_context["timestamp"],
    )

    # ---- Reproducibility and startup summary ----
    # Set random seed first for full reproducibility
    set_random_seed(args.seed)
    print()

    print("=" * 60)
    print("NATURAL INSTRUCTIONS TASK LEARNING")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Device map: {args.device_map}")
    print(f"Number of tasks to sample: {args.num_tasks}")
    # Ratios removed; using sizes mode only
    print(f"Decouple embeddings: {args.decouple_embeddings}")
    print(f"Validation batch size: {args.val_batch_size}")
    print(f"Test batch size: {args.test_batch_size}")
    print("Test prompt mode: instruction_and_query")
    print(f"Generation routing mode: {args.generation_routing}")
    print(f"Shuffle training dataloader: {args.shuffle_train}")
    print(f"Use task loss: {args.use_task_loss}")
    print(f"Task loss weight: {args.task_loss_weight}")
    print(f"Use mean loss: {args.use_mean_loss}")
    print(f"Mean loss weight: {args.mean_loss_weight}")
    print(f"Use hard-negative routing loss: {args.use_hard_negative_loss}")
    print(f"Hard-negative routing loss weight: {args.hard_negative_loss_weight}")
    print(f"Hard-negative routing margin: {args.hard_negative_margin}")
    print(f"Use separation loss: {args.use_sep_loss}")
    print(f"Separation loss weight: {args.sep_loss_weight}")
    print(f"Separation loss tau: {args.sep_loss_tau}")
    print(f"Use centered separation loss: {args.use_centered_sep}")
    print(f"Compute memory bank geometry stats: {args.compute_memory_bank_geometry_stats}")
    print(f"Run directory: {run_context['run_dir']}")
    if any(x is not None for x in [args.train_size, args.val_size, args.test_size]):
        print(f"Sizes mode per task - Train: {args.train_size}, Val: {args.val_size}, Test: {args.test_size} (test is selected first, stable)")
    print(f"Random seed: {args.seed}")
    print()

    print("Setting up logging...")
    print(f"   Training log: {training_log}")
    print(f"   Evaluation log: {evaluation_log}")
    print(f"   Stdout log: {stdout_log}")
    print()
    
    # ---- Tokenizer and task sampling ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print()

    # Add reserved special tokens to the tokenizer
    tokenizer, is_extended = add_reserved_special_tokens(tokenizer, args.num_tasks)
    print(f"Tokenizer loaded with adjustments. Vocab size: {len(tokenizer)}")
    print()
    
    # Sample tasks from Natural Instructions dataset
    print(f"Sampling {args.num_tasks} tasks from Natural Instructions dataset...")
    train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(
        tasks_dir=args.tasks_dir,
        num_tasks=args.num_tasks,
        max_length=args.max_length,
        tokenizer=tokenizer,
        stable_test_split=True,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        few_shot=args.few_shot,
    )
    split_payload = build_runtime_split_payload(
        args,
        train_data,
        val_data,
        test_data,
        task_names,
    )
    split_cache_path = os.path.join(run_context["run_dir"], "split_cache.pt")
    torch.save(split_payload, split_cache_path)
    write_json(
        os.path.join(run_context["run_dir"], "split_cache_metadata.json"),
        split_payload["metadata"],
    )
    write_json(
        os.path.join(run_context["run_dir"], "run_config.json"),
        build_run_config(
            vars(args),
            run_context,
            extra={
                "split_cache_path": split_cache_path,
                "dataset_summary": {
                    "train_examples": len(train_data),
                    "val_examples": len(val_data),
                    "test_examples": len(test_data),
                    "task_count": len(task_names),
                },
            },
        ),
    )
    
    # ---- Model and dataloaders ----
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
        generation_routing=args.generation_routing,
    )
    
    print("\nModel Information:")
    print_model_info(model.model, "Base Model (Frozen)")
    print_model_info(model, "Task Model (Trainable Task Tokens)")
    print()
    
    # Load task tokens if specified
    if args.load_task_tokens:
        if os.path.exists(args.load_task_tokens):
            print(f"Loading task tokens from: {args.load_task_tokens}")
            model.load_task_tokens(args.load_task_tokens)
            print("Task tokens loaded successfully!")
        else:
            print(f"❌ Error: Task tokens file not found: {args.load_task_tokens}")
            return
        print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, val_dataloader, test_dataloader, tokenizer, test_examples = create_natural_instructions_dataloader(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        shuffle_train=args.shuffle_train,
    )
    
    # ---- Training ----
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
            use_task_loss=args.use_task_loss,
            task_loss_weight=args.task_loss_weight,
            use_mean_loss=args.use_mean_loss,
            mean_loss_weight=args.mean_loss_weight,
            use_hard_negative_loss=args.use_hard_negative_loss,
            hard_negative_loss_weight=args.hard_negative_loss_weight,
            hard_negative_margin=args.hard_negative_margin,
            use_sep_loss=args.use_sep_loss,
            sep_loss_weight=args.sep_loss_weight,
            sep_loss_tau=args.sep_loss_tau,
            use_centered_sep=args.use_centered_sep,
            compute_memory_bank_geometry_stats=args.compute_memory_bank_geometry_stats,
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
                "avg_task_loss": train_results["avg_task_loss"],
                "avg_mean_loss": train_results["avg_mean_loss"],
                "avg_hard_negative_loss": train_results["avg_hard_negative_loss"],
                "avg_sep_loss": train_results["avg_sep_loss"],
                "avg_sep_loss_raw": train_results["avg_sep_loss_raw"],
                "avg_sep_loss_centered": train_results["avg_sep_loss_centered"],
                "routing_bank_acc": train_results["routing_bank_acc"],
                "routing_bank_margin_avg": train_results["routing_bank_margin_avg"],
                "best_val_loss": train_results["best_val_loss"],
                "best_model_path": train_results["best_model_path"],
                "final_model_path": final_model_path,
                "memory_bank_geometry_stats": train_results.get("memory_bank_geometry_stats"),
            },
        )
        print(f"Final model saved to: {final_model_path}")
        
        # Restore best validation checkpoint for downstream demo/evaluation consistency.
        if train_results['best_model_state'] is not None:
            print(f"Loading best model state (validation loss: {train_results['best_val_loss']:.4f})")
            best_state = train_results['best_model_state']
            model.load_state_dict(best_state, strict=False)
        print()
    
    # ---- Optional demo ----
    if args.demo:
        print("Running demo on sample examples...")
        # Show up to 5 examples
        demo_examples = random.sample(test_examples, 5)
        demo_task_calling(model, tokenizer, demo_examples, device=args.device)
        print()
    
    # ---- Evaluation ----
    if test_dataloader:
        print("Running comprehensive evaluation (instruction_and_query)...")
        predictions_output_path = os.path.join(
            run_context["run_dir"], "evaluation_predictions_instruction_and_query.jsonl"
        )
        results = eval_task_calling(
            model=model,
            tokenizer=tokenizer,
            test_dataloader=test_dataloader,
            device=args.device,
            use_ground_truth_tasks=False,
            predictions_output_path=predictions_output_path,
            prompt_mode_label="instruction_and_query",
        )

        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY:")
        print("   Prompt mode: instruction_and_query")
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
                "split_cache_path": split_cache_path,
                "model_name": args.model_name,
                "num_tasks": len(task_names),
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "task_tokens_path": train_results["best_model_path"] if not args.skip_training and train_dataloader else None,
                "best_task_tokens_path": train_results["best_model_path"] if not args.skip_training and train_dataloader else None,
                "final_task_tokens_path": final_model_path if not args.skip_training and train_dataloader else None,
                "evaluation_predictions_path": results.get("predictions_output_path"),
                "evaluation_results_path": os.path.join(run_context["run_dir"], "evaluation_results.json"),
                "metrics": results,
            },
        )
        
        # # Optional: Ground truth task evaluation for comparison
        # print("\nRunning ground truth task evaluation for comparison...")
        # gt_results = eval_task_calling(
        #     model=model,
        #     tokenizer=tokenizer,
        #     test_dataloader=test_dataloader,
        #     device=args.device,
        #     use_ground_truth_tasks=True
        # )
        
        # print("\n" + "=" * 50)
        # print("COMPARISON RESULTS:")
        # print(f"   Task Prediction Mode - Exact Match: {results['exact_accuracy']:.3f}")
        # print(f"   Ground Truth Mode - Exact Match: {gt_results['exact_accuracy']:.3f}")
        # print(f"   Performance Gap: {gt_results['exact_accuracy'] - results['exact_accuracy']:.3f}")
        # print("=" * 50)
    
    print("\nTask learning pipeline completed!")


if __name__ == "__main__":
    main()
