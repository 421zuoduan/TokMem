#!/usr/bin/env python3
"""
Natural Instructions task learning with reserved task tokens.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration, init_empty_weights
from accelerate.utils import (
    FullyShardedDataParallelPlugin,
    broadcast_object_list,
    disable_fsdp_ram_efficient_loading,
    enable_fsdp_ram_efficient_loading,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from task_dataset import create_natural_instructions_dataloader, sample_natural_instructions_tasks
from task_model import TaskCallingModel, print_model_info
from task_training import (
    create_optimizer,
    create_scheduler,
    demo_task_calling,
    eval_task_calling,
    save_trained_model,
    setup_logging,
    train_task_calling_model,
)


def add_reserved_special_tokens(tokenizer, num_of_tasks, device="cuda"):
    """Add reserved special tokens to the tokenizer."""
    start_idx = len(
        [t for t in tokenizer.get_vocab() if t.startswith("<|reserved_special_token_")]
    )

    if num_of_tasks <= start_idx:
        return tokenizer, False

    num_additional_tokens = num_of_tasks - start_idx
    new_tokens = [
        f"<|reserved_special_token_{i}|>"
        for i in range(start_idx, start_idx + num_additional_tokens)
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

    if os.environ.get("RANK", "0") == "0":
        print(f"Random seed set to: {seed}")


def write_json(output_path, payload):
    """Write a JSON payload with stable formatting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_expected_split_metadata(args):
    """Build the compact metadata contract for fixed cached splits."""
    return {
        "tasks_dir": os.path.abspath(args.tasks_dir),
        "model_name": args.model_name,
        "num_tasks": args.num_tasks,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "few_shot": args.few_shot,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_instruction_tokens": args.max_instruction_tokens,
    }


def validate_split_cache_metadata(cache_path, expected_metadata, cached_metadata):
    """Verify that a cached split matches the split-defining CLI arguments."""
    mismatches = []
    compatible_model_names = set(cached_metadata.get("compatible_model_names", []))
    strict_keys = {"num_tasks", "train_size", "val_size", "test_size", "few_shot", "seed", "max_length"}

    for key, expected_value in expected_metadata.items():
        cached_value = cached_metadata.get(key)
        if key in strict_keys:
            if cached_value != expected_value:
                mismatches.append((key, expected_value, cached_value))
            continue

        if key == "max_instruction_tokens":
            if cached_value is None:
                continue
            if cached_value != expected_value:
                mismatches.append((key, expected_value, cached_value))
            continue

        if key == "model_name":
            if cached_value == expected_value or expected_value in compatible_model_names:
                continue
            print(
                "Warning: cached split metadata model_name does not exactly match the current "
                f"model_name ({cached_value!r} vs {expected_value!r}), but loading will continue."
            )
            continue

        if key == "tasks_dir":
            if cached_value == expected_value:
                continue
            print(
                "Warning: cached split metadata tasks_dir does not exactly match the current "
                f"tasks_dir ({cached_value!r} vs {expected_value!r}), but loading will continue."
            )
            continue

    if mismatches:
        details = "\n".join(
            f"  {key}: expected={expected_value!r}, cached={cached_value!r}"
            for key, expected_value, cached_value in mismatches
        )
        raise ValueError(f"Cached split metadata mismatch for {cache_path}:\n{details}")


def load_split_cache(args):
    """Load a cached train/val/test split instead of sampling one at runtime."""
    cache_path = os.path.abspath(args.split_cache_path)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Split cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu")
    expected_metadata = build_expected_split_metadata(args)
    cached_metadata = payload.get("metadata", {})
    validate_split_cache_metadata(cache_path, expected_metadata, cached_metadata)

    required_keys = ("train_data", "val_data", "test_data", "task_names")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise KeyError(f"Split cache missing keys: {missing_keys}")

    return (
        payload["train_data"],
        payload["val_data"],
        payload["test_data"],
        payload["task_names"],
        cache_path,
    )


def resolve_tasks_dir(tasks_dir):
    """Resolve the tasks directory against the repo layout used in this workspace."""
    if os.path.exists(tasks_dir):
        return tasks_dir

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

    return tasks_dir


def infer_transformer_layer_cls_name(model_name):
    """Infer the decoder block class name for transformer-based FSDP auto-wrap."""
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        backbone = AutoModelForCausalLM.from_config(config)

    layer_candidates = [
        getattr(getattr(backbone, "model", None), "layers", None),
        getattr(backbone, "layers", None),
        getattr(getattr(backbone, "transformer", None), "h", None),
    ]
    for layers in layer_candidates:
        if layers is not None and len(layers) > 0:
            return layers[0].__class__.__name__
    return None


def build_accelerator(args, model_name):
    """Build Accelerator and optional FSDP plugin for the atomic training path."""
    dataloader_config = DataLoaderConfiguration(
        non_blocking=args.pin_memory,
        use_seedable_sampler=args.shuffle_train,
    )

    fsdp_plugin = None
    if args.use_fsdp:
        layer_cls_name = infer_transformer_layer_cls_name(model_name)
        if layer_cls_name is None:
            raise ValueError(
                "--use_fsdp requires a recognized transformer decoder block class. "
                f"Failed to infer one for model {model_name!r}."
            )
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=args.fsdp_sharding_strategy,
            backward_prefetch=(
                None if args.fsdp_backward_prefetch == "none" else args.fsdp_backward_prefetch
            ),
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=[layer_cls_name],
            state_dict_type="FULL_STATE_DICT",
            use_orig_params=True,
            limit_all_gathers=True,
            cpu_ram_efficient_loading=True,
        )

    return Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        fsdp_plugin=fsdp_plugin,
    )


def synchronize_checkpoint_path(checkpoint_path, accelerator):
    """Broadcast a checkpoint path from rank 0 so all ranks follow the same load path."""
    if accelerator is None or accelerator.num_processes <= 1:
        return checkpoint_path

    payload = [checkpoint_path]
    broadcast_object_list(payload, from_process=0)
    return payload[0]


def configure_fsdp_ignored_modules(model, accelerator):
    """Keep the tiny trainable task-token module replicated outside backbone FSDP sharding."""
    if accelerator is None or getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return

    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is None:
        return
    if not hasattr(model, "get_fsdp_trainable_modules"):
        raise AttributeError(
            "FSDP atomic path requires get_fsdp_trainable_modules() on the task model."
        )

    ignored_modules = list(model.get_fsdp_trainable_modules())
    if not ignored_modules:
        raise ValueError("FSDP atomic path expected at least one ignored trainable module.")
    for module in ignored_modules:
        module.to(accelerator.device)
    fsdp_plugin.ignored_modules = ignored_modules


def main():
    parser = argparse.ArgumentParser(description="Natural Instructions Task Learning")
    parser.add_argument("--tasks_dir", type=str, default="natural-instructions-2.8/tasks",
                        help="Directory containing Natural Instructions task files")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks to sample")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per process")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size per process")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Test batch size per process")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_fsdp", action="store_true",
                        help="Wrap the backbone with Accelerate FSDP for multi-GPU training")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Accelerate mixed precision mode")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "HYBRID_SHARD_ZERO2"],
                        help="FSDP sharding strategy")
    parser.add_argument("--fsdp_backward_prefetch", type=str, default="BACKWARD_PRE",
                        choices=["BACKWARD_PRE", "BACKWARD_POST", "none"],
                        help="FSDP backward prefetch mode")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable backbone gradient checkpointing for larger-model training")
    parser.add_argument("--decouple_embeddings", action="store_true",
                        help="Use separate input/output embeddings for task tokens")
    parser.add_argument("--max_instruction_tokens", type=int, default=1024,
                        help="Maximum token length for instructions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--demo", action="store_true", help="Only run demo on a few examples")
    parser.add_argument("--load_task_tokens", type=str, default=None,
                        help="Path to saved task tokens file")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--shuffle_train", action=argparse.BooleanOptionalAction, default=True,
                        help="Shuffle the training dataloader")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader worker processes")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Pin dataloader host memory")
    parser.add_argument("--train_size", type=int, default=None,
                        help="Absolute number of training samples per task")
    parser.add_argument("--val_size", type=int, default=None,
                        help="Absolute number of validation samples per task")
    parser.add_argument("--test_size", type=int, default=None,
                        help="Absolute number of test samples per task")
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot instructions")
    parser.add_argument("--validate_every_n_steps", type=int, default=1000,
                        help="Validate every n forward steps")
    parser.add_argument("--split_cache_path", type=str, default=None,
                        help="Path to a cached train/val/test split")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory for run-scoped logs and structured outputs")
    parser.add_argument("--use_logit_bias", action="store_true",
                        help="Train and apply a first-step task logit-bias head")
    parser.add_argument("--logit_bias_loss_weight", type=float, default=1.0,
                        help="Weight for the detached hidden-state bias-head loss")
    parser.add_argument("--logit_bias_network", type=str, default="linear", choices=["linear", "mlp"],
                        help="Architecture for the task logit-bias head")
    parser.add_argument("--logit_bias_scale", type=float, default=1.0,
                        help="Scale applied to first-step bias logits")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.tasks_dir = resolve_tasks_dir(args.tasks_dir)
    if args.run_dir:
        args.run_dir = os.path.abspath(args.run_dir)
    accelerator = build_accelerator(args, args.model_name)
    should_log = accelerator.is_main_process
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)

    set_random_seed(args.seed)
    if should_log:
        print()
        print("=" * 60)
        print("NATURAL INSTRUCTIONS TASK LEARNING")
        print("=" * 60)
        print(f"Model: {args.model_name}")
        print(f"Use FSDP: {args.use_fsdp}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"World size: {world_size}")
        print(f"Number of tasks to sample: {args.num_tasks}")
        print(f"Validation batch size: {args.val_batch_size}")
        print(f"Test batch size: {args.test_batch_size}")
        print(f"Decouple embeddings: {args.decouple_embeddings}")
        print(f"Shuffle training dataloader: {args.shuffle_train}")
        print(f"Use logit bias: {args.use_logit_bias}")
        print(f"Logit bias loss weight: {args.logit_bias_loss_weight}")
        print(f"Logit bias network: {args.logit_bias_network}")
        print(f"Logit bias scale: {args.logit_bias_scale}")
        print(f"Dataloader workers: {args.num_workers}")
        print(f"Pin memory: {args.pin_memory}")
        if any(x is not None for x in [args.train_size, args.val_size, args.test_size]):
            print(
                f"Sizes mode per task - Train: {args.train_size}, Val: {args.val_size}, "
                f"Test: {args.test_size} (test is selected first, stable)"
            )
        if args.split_cache_path:
            print(f"Split source: cached split at {os.path.abspath(args.split_cache_path)}")
        else:
            print("Split source: runtime sampling")
        print(f"Random seed: {args.seed}")
        print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    if os.environ.get("RANK", "0") == "0":
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
        print()

    if args.split_cache_path:
        if os.environ.get("RANK", "0") == "0":
            print(f"Loading cached split from: {os.path.abspath(args.split_cache_path)}")
        train_data, val_data, test_data, task_names, split_cache_path = load_split_cache(args)
        effective_num_tasks = len(task_names)
        if os.environ.get("RANK", "0") == "0":
            print(
                f"Cached split loaded. Train: {len(train_data)}, Val: {len(val_data)}, "
                f"Test: {len(test_data)}, Tasks: {effective_num_tasks}"
            )
    else:
        effective_num_tasks = args.num_tasks
        split_cache_path = None

    tokenizer, is_extended = add_reserved_special_tokens(tokenizer, effective_num_tasks)
    if os.environ.get("RANK", "0") == "0":
        print(f"Tokenizer loaded with adjustments. Vocab size: {len(tokenizer)}")
        print()

    if args.split_cache_path:
        if os.environ.get("RANK", "0") == "0":
            print(f"Using precomputed task split from cache: {split_cache_path}")
    else:
        if os.environ.get("RANK", "0") == "0":
            print(f"Sampling {args.num_tasks} tasks from Natural Instructions dataset...")
            print(f"   Max instruction length: {args.max_instruction_tokens} tokens")
        train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(
            tasks_dir=args.tasks_dir,
            num_tasks=args.num_tasks,
            max_instruction_tokens=args.max_instruction_tokens,
            tokenizer=tokenizer,
            stable_test_split=True,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            few_shot=args.few_shot,
        )

    try:
        if should_log:
            print("Initializing Task Calling Model...")
        if args.use_fsdp and fsdp_plugin is not None:
            enable_fsdp_ram_efficient_loading()
        try:
            model = TaskCallingModel(
                model_name=args.model_name,
                num_tasks=len(task_names),
                task_names=task_names,
                tokenizer=tokenizer,
                device="cpu",
                decouple_embeddings=args.decouple_embeddings,
                is_extended=is_extended,
                use_logit_bias=args.use_logit_bias,
                logit_bias_network=args.logit_bias_network,
                logit_bias_scale=args.logit_bias_scale,
            )
        finally:
            if args.use_fsdp and fsdp_plugin is not None:
                disable_fsdp_ram_efficient_loading()
        if args.gradient_checkpointing:
            model.model.gradient_checkpointing_enable()

        if should_log:
            print("\nModel Information:")
            print_model_info(model.model, "Base Model (Frozen)")
            print_model_info(model, "Task Model (Trainable Task Tokens)")
            print()

        if args.load_task_tokens:
            if not os.path.exists(args.load_task_tokens):
                raise FileNotFoundError(f"Task tokens file not found: {args.load_task_tokens}")
            if should_log:
                print(f"Loading task tokens from: {args.load_task_tokens}")
            model.load_task_tokens(args.load_task_tokens)
            if should_log:
                print("Task tokens loaded successfully!")
                print()

        configure_fsdp_ignored_modules(model, accelerator)
        if should_log and getattr(accelerator.distributed_type, "name", "") == "FSDP":
            print("Configured FSDP ignored module: task_token_module")
            print()

        training_logger, eval_logger, training_log, evaluation_log, timestamp = setup_logging(
            log_dir=args.run_dir if args.run_dir else "logs",
            is_main_process=should_log,
        )
        if should_log:
            print("Setting up logging...")
            print(f"   Training log: {training_log}")
            print(f"   Evaluation log: {evaluation_log}")
            print()
            if accelerator is not None:
                print(f"Accelerator device: {accelerator.device}")
                print(f"Accelerator distributed type: {accelerator.distributed_type}")
                print()

        if should_log and args.run_dir:
            write_json(
                os.path.join(args.run_dir, "run_config.json"),
                {
                    "args": vars(args),
                    "world_size": world_size,
                    "split_cache_path": split_cache_path,
                    "task_names": task_names,
                    "training_log": training_log,
                    "evaluation_log": evaluation_log,
                },
            )

        if should_log:
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
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        optimizer = None
        scheduler = None
        if not args.skip_training and train_dataloader is not None:
            optimizer = create_optimizer(model=model, lr=args.lr)

        prepare_args = [model]
        has_optimizer = optimizer is not None
        has_scheduler = scheduler is not None
        if has_optimizer:
            prepare_args.append(optimizer)
        prepare_args.append(train_dataloader)
        if val_dataloader is not None:
            prepare_args.append(val_dataloader)
        if test_dataloader is not None:
            prepare_args.append(test_dataloader)
        if has_scheduler:
            prepare_args.append(scheduler)

        prepared = list(accelerator.prepare(*prepare_args))
        idx = 0
        model = prepared[idx]
        idx += 1
        if has_optimizer:
            optimizer = prepared[idx]
            idx += 1
        train_dataloader = prepared[idx]
        idx += 1
        if val_dataloader is not None:
            val_dataloader = prepared[idx]
            idx += 1
        if test_dataloader is not None:
            test_dataloader = prepared[idx]
            idx += 1
        if has_scheduler:
            scheduler = prepared[idx]
            idx += 1

        if optimizer is not None:
            scheduler, _, _ = create_scheduler(
                optimizer=optimizer,
                dataloader=train_dataloader,
                num_epochs=args.num_epochs,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

        train_results = None
        task_token_checkpoint = args.load_task_tokens
        if not args.skip_training and train_dataloader is not None:
            if should_log:
                print("Starting Training...")
            train_results = train_task_calling_model(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                val_dataloader=val_dataloader,
                num_epochs=args.num_epochs,
                lr=args.lr,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                device=str(accelerator.device),
                timestamp=timestamp,
                validate_every_n_steps=args.validate_every_n_steps,
                use_logit_bias=args.use_logit_bias,
                logit_bias_loss_weight=args.logit_bias_loss_weight,
                accelerator=accelerator,
            )
            task_token_checkpoint = train_results["best_model_path"] or task_token_checkpoint
            if should_log:
                print(f"Training completed with average loss: {train_results['avg_total_loss']:.4f}")
                print()
                if args.run_dir:
                    write_json(
                        os.path.join(args.run_dir, "train_results.json"),
                        train_results,
                    )

        accelerator.wait_for_everyone()
        task_token_checkpoint = synchronize_checkpoint_path(task_token_checkpoint, accelerator)

        if task_token_checkpoint is None and not args.skip_training:
            task_token_checkpoint = save_trained_model(
                model=model,
                timestamp=timestamp,
                suffix="final",
                accelerator=accelerator,
            )
            accelerator.wait_for_everyone()
            task_token_checkpoint = synchronize_checkpoint_path(task_token_checkpoint, accelerator)

        if task_token_checkpoint:
            if should_log:
                print(f"Loading evaluation task tokens from: {task_token_checkpoint}")
            target_model = accelerator.unwrap_model(model)
            target_model.load_task_tokens(task_token_checkpoint)
            if should_log:
                print("Evaluation task tokens loaded successfully!")
                print()

        if args.demo and should_log:
            print("Running demo on sample examples...")
            demo_examples = random.sample(test_examples, min(5, len(test_examples)))
            demo_task_calling(
                model=model,
                tokenizer=tokenizer,
                test_examples=demo_examples,
                device=str(accelerator.device),
                accelerator=accelerator,
            )
            print()

        if test_dataloader is not None:
            if should_log:
                print("Running comprehensive evaluation...")
            results = eval_task_calling(
                model=model,
                tokenizer=tokenizer,
                test_dataloader=test_dataloader,
                device=str(accelerator.device),
                use_ground_truth_tasks=False,
                accelerator=accelerator,
            )

            if should_log and results is not None:
                if args.run_dir:
                    write_json(
                        os.path.join(args.run_dir, "evaluation_results.json"),
                        results,
                    )
                print("\n" + "=" * 50)
                print("FINAL RESULTS SUMMARY:")
                print(f"   Task Prediction Accuracy: {results['task_accuracy']:.3f}")
                print(f"   Exact Match Accuracy: {results['exact_accuracy']:.3f}")
                print(f"   Average Response Score: {results['avg_response_score']:.3f}")
                print("=" * 50)

        if should_log:
            print("\nTask learning pipeline completed!")
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
