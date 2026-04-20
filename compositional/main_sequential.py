#!/usr/bin/env python3
"""
Sequential training script for function calling with tool set progression.
Supports multiple rounds of training with different tool sets.
"""

import argparse
import inspect
import random
import os
import numpy as np
import torch
from transformers import AutoTokenizer
import json
import logging

from model import FunctionCallingModel, print_model_info
from dataset import create_native_dataloader, discover_available_tools
from training import (
    demo_native_function_calling,
    eval_native_function_calling,
    save_training_plot_images,
    train_native_function_calling_model,
)
from run_layout import (
    DEFAULT_RUNS_DIR,
    artifact_path,
    build_training_summary_payload,
    build_run_config,
    resolve_run_context,
    write_json,
)


def discover_all_tool_names_from_rounds(rounds, data_dir, train_max_calls_by_round, test_max_calls_by_round):
    """Pre-discover all tool names from all rounds' data files

    Args:
        rounds: List of round specifications
        data_dir: Base directory for data files
        train_max_calls_by_round: Per-round train max call limits
        test_max_calls_by_round: Per-round test max call limits

    Returns:
        List of all tool names in order (tools 1-10, 11-20, etc.)
    """
    print("Pre-discovering all tool names from all rounds...")
    all_tool_names = []
    seen_tool_names = set()
    
    for round_idx, round_spec in enumerate(rounds):
        round_num = round_idx + 1
        tools_range = round_spec['tools']
        
        # Construct data file names
        train_data_file = os.path.join(
            data_dir,
            f"training/function_calling_train_tools{tools_range}_{train_max_calls_by_round[round_idx]}calls.json"
        )
        test_data_file = os.path.join(
            data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_max_calls_by_round[round_idx]}calls.json"
        )
        
        # Check if files exist
        if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
            print(f"Warning: Data files not found for round {round_num} (tools {tools_range})")
            print(f"Expected: {train_data_file}")
            print(f"Expected: {test_data_file}")
            continue
        
        # Discover tools for this round
        round_tools = discover_available_tools(train_data_file, test_data_file)
        print(f"Round {round_num} (tools {tools_range}): {len(round_tools)} tools")
        
        # Add to complete list while preserving the first-seen order
        for tool_name in round_tools:
            if tool_name not in seen_tool_names:
                all_tool_names.append(tool_name)
                seen_tool_names.add(tool_name)
    
    print(f"Total discovered tools: {len(all_tool_names)}")
    print(f"First few tools: {all_tool_names[:5]}...")
    print(f"Last few tools: {all_tool_names[-5:]}")
    
    return all_tool_names


def freeze_lora_parameters(model):
    """Freeze all LoRA parameters in the model"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"Frozen {frozen_count} LoRA parameters")
    return frozen_count


def unfreeze_lora_parameters(model):
    """Unfreeze all LoRA parameters in the model"""
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            unfrozen_count += 1
    print(f"Unfrozen {unfrozen_count} LoRA parameters")
    return unfrozen_count


# Active tool mapping system removed - model now uses original mappings for all 60 tools


def apply_orthogonal_init_all_tools(model, total_tools):
    """Apply orthogonal initialization to ALL tool embeddings at once
    
    Args:
        model: The model with tool embeddings
        total_tools: Total number of tools to initialize
    """
    print(f"Applying orthogonal initialization to all {total_tools} tool embeddings")
    
    with torch.no_grad():
        if model.decouple_embeddings:
            # Apply orthogonal initialization to both input and output embeddings
            input_embeddings = model.trainable_tool_input_embeddings[:total_tools]
            output_embeddings = model.trainable_tool_output_embeddings[:total_tools]
            
            # Convert to float32 for orthogonal initialization (BFloat16 not supported)
            original_dtype = input_embeddings.dtype
            input_temp = input_embeddings.float()
            output_temp = output_embeddings.float()
            
            # Initialize with orthogonal matrices
            torch.nn.init.orthogonal_(input_temp)
            torch.nn.init.orthogonal_(output_temp)
            
            # Convert back to original dtype
            input_embeddings.copy_(input_temp.to(original_dtype))
            output_embeddings.copy_(output_temp.to(original_dtype))
            
        else:
            # Apply orthogonal initialization to shared embeddings
            shared_embeddings = model.trainable_tool_embeddings[:total_tools]
            
            # Convert to float32 for orthogonal initialization (BFloat16 not supported)
            original_dtype = shared_embeddings.dtype
            shared_temp = shared_embeddings.float()
            
            # Initialize with orthogonal matrices
            torch.nn.init.orthogonal_(shared_temp)
            
            # Convert back to original dtype
            shared_embeddings.copy_(shared_temp.to(original_dtype))
    
    print(f"Orthogonal initialization applied to all {total_tools} tool embeddings - each embedding is orthogonal to all others")


def reinitialize_tool_embeddings(model, new_tool_names):
    """DEPRECATED: This function should not be used with pre-allocated embeddings.
    Use update_tool_mappings_for_round instead."""
    # Suppress unused parameter warnings
    _ = model
    _ = new_tool_names
    raise NotImplementedError(
        "reinitialize_tool_embeddings should not be used with pre-allocated embeddings. "
        "Use update_tool_mappings_for_round instead."
    )


def parse_training_rounds(rounds_str):
    """
    Parse training rounds specification.
    Format: "tools:epochs,tools:epochs,..."
    Example: "1-10:3,11-20:3,21-30:2"
    """
    rounds = []
    for round_spec in rounds_str.split(','):
        parts = round_spec.strip().split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid round specification: {round_spec}. Use 'tools:epochs' format.")
        
        tools = parts[0].strip()
        epochs = int(parts[1].strip())
        rounds.append({'tools': tools, 'epochs': epochs})
    
    return rounds


def resolve_run_file_path(run_context, requested_path, default_name):
    if requested_path:
        if os.path.isabs(requested_path):
            return requested_path
        return artifact_path(run_context, os.path.basename(requested_path))
    return artifact_path(run_context, default_name)


def strip_call_count_breakdown(value):
    if isinstance(value, dict):
        return {
            key: strip_call_count_breakdown(subvalue)
            for key, subvalue in value.items()
            if key != "call_count_breakdown"
        }
    if isinstance(value, list):
        return [strip_call_count_breakdown(item) for item in value]
    return value


def filter_supported_kwargs(callable_obj, **kwargs):
    """Return only the kwargs accepted by the callable."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}

    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in parameters
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Sequential Function Calling Training with Tool Progression")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model name")
    
    # Sequential training arguments
    parser.add_argument("--training_rounds", type=str, required=True,
                        help="Training rounds specification (e.g., '1-10:3,11-20:3,21-30:2' for 3 rounds)")
    parser.add_argument("--freeze_lora_after_first", action="store_true",
                        help="Freeze LoRA parameters after first round")
    
    # Training arguments  
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the epoch count for a single no-adaptation training round")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate for tokenized memory embeddings")
    parser.add_argument("--lora_lr", type=float, default=5e-4,
                        help="Learning rate for LoRA parameters (default: 5e-4)")
    
    # Evaluation arguments
    parser.add_argument("--eval_after_each_round", action="store_true",
                        help="Run evaluation after each training round")
    parser.add_argument("--eval_all_previous", action="store_true",
                        help="Evaluate on all previous test sets after each round")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum generated tokens for training-time evaluation and demo decoding")
    parser.add_argument("--demo", type=int, default=None,
                        help="Number of demo examples to show after training")
    parser.add_argument("--use_ground_truth_tools", action="store_true",
                        help="Force gold tool ids at assistant-start and each explicit eoc boundary during inference")
    parser.add_argument("--use_eoc", action="store_true",
                        help="Insert an explicit end-of-control token after each tool-controlled span")
    parser.add_argument("--use_js_trunc", action="store_true",
                        help="Enable JS truncation decode-only routing at decision positions")
    parser.add_argument("--use_logit_bias", action="store_true",
                        help="Train an external detached tool prior head and use it as a soft decode-time logit bias")
    parser.add_argument("--logit_bias_loss_weight", type=float, default=0.1,
                        help="Weight for the detached tool-prior CE loss")
    parser.add_argument("--logit_bias_network", type=str, default="linear", choices=["mlp", "linear"],
                        help="Architecture used by the detached tool prior head")
    parser.add_argument("--logit_bias_scale", type=float, default=1.0,
                        help="Multiplier applied to the prior head log-softmax bias at decode time")
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--decouple_embeddings", action="store_true",
                        help="Use separate weights for input and output tool token embeddings")
    parser.add_argument("--renorm_active_tools", action="store_true",
                        help="Renormalize active tool output-embedding norms post-step")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA fine-tuning of base model")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate (default: 0.1)")
    parser.add_argument("--lora_layer_indices", type=str, default=None,
                        help="Apply LoRA only to specific layer indices")
    parser.add_argument("--lora_target_modules", type=str, default="o_proj",
                        help="Target modules for LoRA")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--train_max_function_calls", type=int, default=4,
                        help="Maximum function calls used in data generation")
    parser.add_argument("--train_max_function_calls_per_round", type=str, default=None,
                        help="Comma-separated max function call limits per round (overrides --train_max_function_calls)")
    parser.add_argument("--test_max_function_calls", type=int, default=4,
                        help="Maximum function calls used in data generation")
    parser.add_argument("--test_max_function_calls_per_round", type=str, default=None,
                        help="Comma-separated max function call limits per round for evaluation (overrides --test_max_function_calls)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save model checkpoints after each round")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file for evaluation results")
    parser.add_argument("--curriculum_learning", action="store_true",
                        help="Enable curriculum learning")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Explicit run name")
    parser.add_argument("--run_root_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Root directory for compositional runs")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Optional tag appended to the generated run name")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Save post-training loss and learning-rate plots under the current run directory")

    return parser


def validate_args(args, parser):
    if args.use_js_trunc and not args.use_eoc:
        parser.error("--use_js_trunc requires --use_eoc")
    if args.use_logit_bias and not args.use_eoc:
        parser.error("--use_logit_bias requires --use_eoc")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.logit_bias_loss_weight < 0:
        parser.error("--logit_bias_loss_weight must be non-negative")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup simple logging
    import sys
    from datetime import datetime
    
    run_context = resolve_run_context(
        experiment_name="tokmem",
        model_name=args.model_name,
        run_root_dir=args.run_root_dir,
        run_name=args.run_name,
        run_tag=args.run_tag,
    )

    evaluation_log_file = resolve_run_file_path(run_context, args.log_file, "evaluation.log")
    loss_plot_path = artifact_path(run_context, "loss_step.png") if args.tensorboard else None
    lr_plot_path = artifact_path(run_context, "lr_step.png") if args.tensorboard else None
    checkpoint_dir = run_context["run_dir"]
    if args.checkpoint_dir:
        checkpoint_dir = (
            args.checkpoint_dir
            if os.path.isabs(args.checkpoint_dir)
            else artifact_path(run_context, os.path.basename(args.checkpoint_dir.rstrip("/")))
        )
    open(evaluation_log_file, "a", encoding="utf-8").close()
    if args.tensorboard:
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            parser.error(
                "--tensorboard now saves loss/lr PNG plots and requires the 'matplotlib' package "
                "in the active environment. Run `pip install -r requirements.txt` inside the tokmem environment."
            )

    log_handlers = [logging.StreamHandler(sys.stdout)]
    log_handlers.append(logging.FileHandler(evaluation_log_file, mode='a'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=log_handlers
    )
    logger = logging.getLogger(__name__)
    
    # Log start time and configuration
    logger.info(f"=== Sequential Training Started at {datetime.now()} ===")
    logger.info(f"Configuration: model={args.model_name}, rounds={args.training_rounds}, batch_size={args.batch_size}")
    logger.info(f"Run directory: {run_context['run_dir']}")
    
    # Parse training rounds
    try:
        rounds = parse_training_rounds(args.training_rounds)
        if args.epochs is not None:
            if len(rounds) != 1:
                parser.error("--epochs only supports single-round no-adaptation runs")
            rounds[0]["epochs"] = args.epochs
        print(f"Parsed {len(rounds)} training rounds:")
        for i, round_spec in enumerate(rounds, 1):
            print(f"  Round {i}: Tools {round_spec['tools']}, {round_spec['epochs']} epochs")
    except ValueError as e:
        parser.error(str(e))

    num_rounds = len(rounds)

    def expand_per_round_values(values_str, fallback, arg_label):
        if values_str is None:
            return [fallback] * num_rounds

        raw_values = [part.strip() for part in values_str.split(',')]
        parsed_values = []
        for value in raw_values:
            if not value:
                continue
            try:
                parsed_values.append(int(value))
            except ValueError:
                parser.error(f"Invalid value '{value}' for {arg_label}; expected integers.")

        if not parsed_values:
            parser.error(f"No valid values provided for {arg_label}.")

        if len(parsed_values) < num_rounds:
            parsed_values.extend([parsed_values[-1]] * (num_rounds - len(parsed_values)))
        elif len(parsed_values) > num_rounds:
            print(f"Warning: {arg_label} specified {len(parsed_values)} values; using first {num_rounds}.")
            parsed_values = parsed_values[:num_rounds]

        return parsed_values

    train_max_calls_by_round = expand_per_round_values(
        args.train_max_function_calls_per_round,
        args.train_max_function_calls,
        "train_max_function_calls_per_round"
    )

    test_max_calls_by_round = expand_per_round_values(
        args.test_max_function_calls_per_round,
        args.test_max_function_calls,
        "test_max_function_calls_per_round"
    )
    
    # Set dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    print("\n=== Sequential Function Calling Training ===")
    print(f"Model: {args.model_name}")
    print(f"Training rounds: {len(rounds)}")
    print(f"LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    print(f"EOC: {'Enabled' if args.use_eoc else 'Disabled'}")
    print(f"JS truncation: {'Enabled' if args.use_js_trunc else 'Disabled'}")
    print(f"Logit bias: {'Enabled' if args.use_logit_bias else 'Disabled'}")
    if args.use_logit_bias:
        print(f"Logit bias network: {args.logit_bias_network}")
        print(f"Logit bias loss weight: {args.logit_bias_loss_weight}")
        print(f"Logit bias scale: {args.logit_bias_scale}")
    print(f"Max length: {args.max_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Run directory: {run_context['run_dir']}")
    if args.use_lora and args.freeze_lora_after_first:
        print("LoRA will be frozen after first round")
    print()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    
    # Create checkpoint directory if needed
    if args.save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model (will be created in first round)
    model = None
    lora_config = None
    
    # Build LoRA config if enabled
    if args.use_lora:
        target_modules = [mod.strip() for mod in args.lora_target_modules.split(',')]
        
        if args.lora_layer_indices is not None:
            layer_indices = [int(idx.strip()) for idx in args.lora_layer_indices.split(',')]
            lora_config = {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "layer_indices": layer_indices,
                "target_modules": target_modules
            }
        else:
            lora_config = {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_modules
            }
    
    # Pre-discover actual tool names from all rounds' data files
    all_tool_names = discover_all_tool_names_from_rounds(
        rounds,
        args.data_dir,
        train_max_calls_by_round,
        test_max_calls_by_round,
    )

    if not all_tool_names:
        parser.error("No tools discovered from the round datasets. Check --data_dir and the generated split files.")

    total_tools = len(all_tool_names)
    print(f"Total tools to be learned: {total_tools}")

    write_json(
        artifact_path(run_context, "run_config.json"),
        build_run_config(
            vars(args),
            run_context,
            extra={
                "experiment_type": "tokmem_sequential",
                "rounds": rounds,
                "total_tools": total_tools,
                "discovered_tool_count": len(all_tool_names),
                "data_dir": os.path.abspath(args.data_dir),
                "artifacts": {
                    "evaluation_log": evaluation_log_file,
                    "checkpoint_dir": checkpoint_dir,
                    "loss_plot": loss_plot_path,
                    "lr_plot": lr_plot_path,
                },
            },
        ),
    )
    
    # Store results for all rounds
    all_results = []
    # Store test dataloaders for cumulative evaluation
    all_test_dataloaders = []
    plot_history = {"loss_steps": [], "lr_steps": [], "round_boundaries": []} if args.tensorboard else None
    plot_step_offset = 0
    
    # Training loop for each round
    for round_idx, round_spec in enumerate(rounds):
        round_num = round_idx + 1
        tools_range = round_spec['tools']
        epochs = round_spec['epochs']
        
        print("\n" + "="*60)
        print(f"ROUND {round_num}/{len(rounds)}: Training on tools {tools_range}")
        print("="*60 + "\n")
        
        # Construct data file names based on tool range
        train_data_file = os.path.join(
            args.data_dir,
            f"training/function_calling_train_tools{tools_range}_{train_max_calls_by_round[round_idx]}calls.json"
        )
        test_data_file = os.path.join(
            args.data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_max_calls_by_round[round_idx]}calls.json"
        )
        
        # Check if data files exist
        if not os.path.exists(train_data_file):
            print(f"Warning: Training data file not found: {train_data_file}")
            print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        if not os.path.exists(test_data_file):
            print(f"Warning: Test data file not found: {test_data_file}")
            print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        
        # Discover tools for this round
        print(f"Discovering tools from round {round_num} dataset...")
        round_tools = discover_available_tools(train_data_file, test_data_file)
        print(f"Found {len(round_tools)} tools for round {round_num}: {round_tools[:5]}..." if len(round_tools) > 5 else f"Found {len(round_tools)} tools: {round_tools}")
        
        # Create or update model
        if model is None:
            # First round - create new model with ALL tool slots and actual tool names
            print(f"Initializing model with {total_tools} total tool slots...")
            model_kwargs = filter_supported_kwargs(
                FunctionCallingModel,
                model_name=args.model_name,
                num_tools=total_tools,  # Initialize with ALL tools
                tool_names=all_tool_names,  # Use actual discovered tool names
                tokenizer=tokenizer,
                device=args.device,
                dtype=dtype,
                decouple_embeddings=args.decouple_embeddings,
                lora_config=lora_config,
                use_eoc=args.use_eoc,
                use_js_trunc=args.use_js_trunc,
                use_logit_bias=args.use_logit_bias,
                logit_bias_network=args.logit_bias_network,
                logit_bias_scale=args.logit_bias_scale,
            )
            model = FunctionCallingModel(**model_kwargs)
            print_model_info(model, f"Model with {total_tools} tool slots")
            print(f"Model initialized with all tool names: {model.tool_names[:5]}...{model.tool_names[-5:]}")
            
            # Apply orthogonal initialization to ALL tool embeddings once at the beginning
            apply_orthogonal_init_all_tools(model, total_tools)
        else:
            # Subsequent rounds - just check LoRA freezing
            if args.use_lora and args.freeze_lora_after_first and round_num == 2:
                print("\nFreezing LoRA parameters after first round...")
                freeze_lora_parameters(model)
        
        print(f"Round {round_num}: Training will focus on tools {tools_range}")
        print(f"Dataset will provide labels only for: {round_tools[:3]}...{round_tools[-3:]}")
        
        # Create datasets for this round
        print(f"Creating round {round_num} dataset...")
        dataloader_kwargs = filter_supported_kwargs(
            create_native_dataloader,
            model=model,
            train_data_path=train_data_file,
            test_data_path=test_data_file,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            eval_batch_size=args.eval_batch_size,
            curriculum_learning=args.curriculum_learning,
            validation_split=0,
            random_seed=args.seed,  # Use consistent seed across all rounds
            use_eoc=args.use_eoc,
        )
        train_dataloader, _, test_dataloader, _, test_examples = create_native_dataloader(**dataloader_kwargs)
        
        # Store test dataloader for cumulative evaluation
        all_test_dataloaders.append((tools_range, test_dataloader))
        
        # Determine if LoRA should be trained this round
        train_lora = args.use_lora and (round_num == 1 or not args.freeze_lora_after_first)
        
        # Train this round
        print(f"\nTraining round {round_num} for {epochs} epochs...")
        if train_lora:
            print("Training: Embeddings + LoRA")
        else:
            print("Training: Embeddings only (LoRA frozen)" if args.use_lora else "Training: Embeddings only")
        
        # Determine active tool IDs (freeze others via grad mask in training)
        try:
            active_tool_ids = [model.tool_name_to_id[name] for name in round_tools if name in model.tool_name_to_id]
        except Exception:
            active_tool_ids = None

        training_kwargs = filter_supported_kwargs(
            train_native_function_calling_model,
            model=model,
            dataloader=train_dataloader,
            num_epochs=epochs,
            lr=args.lr,
            lora_lr=args.lora_lr if train_lora else None,
            device=args.device,
            active_tool_ids=active_tool_ids,
            renorm_active_rows=(args.renorm_active_tools and round_num > 2),
            use_eoc=args.use_eoc,
            use_js_trunc=args.use_js_trunc,
            use_logit_bias=args.use_logit_bias,
            logit_bias_loss_weight=args.logit_bias_loss_weight,
            plot_history=plot_history,
            plot_step_offset=plot_step_offset,
            plot_round=round_num,
        )
        round_results = train_native_function_calling_model(**training_kwargs)
        plot_step_offset = round_results.get("plot_next_step", plot_step_offset)
        if plot_history is not None:
            plot_history["round_boundaries"].append(
                {
                    "round": round_num,
                    "step": round_results.get("plot_end_step", plot_step_offset),
                    "tools": tools_range,
                }
            )
        
        print(f"Round {round_num} training completed! Average loss: {round_results['avg_total_loss']:.4f}")
        logger.info(f"\n[ROUND {round_num} RESULTS] Tools: {tools_range}, Epochs: {epochs}, Loss: {round_results['avg_total_loss']:.4f}")
        
        # Store results
        all_results.append({
            'round': round_num,
            'tools': tools_range,
            'epochs': epochs,
            'avg_loss': round_results['avg_total_loss'],
            'results': round_results
        })
        
        # Evaluate this round
        if args.eval_after_each_round:
            print(f"\nEvaluating round {round_num} model...")
            
            # Capture the formatted evaluation output
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capture stdout to get the formatted evaluation results
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                eval_results = eval_native_function_calling(
                    **filter_supported_kwargs(
                        eval_native_function_calling,
                        model=model,
                        tokenizer=tokenizer,
                        test_dataloader=test_dataloader,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                        use_ground_truth_tools=args.use_ground_truth_tools,
                        use_eoc=args.use_eoc,
                        use_js_trunc=args.use_js_trunc,
                        use_logit_bias=args.use_logit_bias,
                    )
                )
            
            # Get the captured formatted output
            formatted_eval_output = captured_output.getvalue()
            
            # Print the formatted output to console as well
            print(formatted_eval_output)
            
            all_results[-1]['eval_results'] = eval_results
            
            # Log the formatted evaluation results to file
            with open(evaluation_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"ROUND {round_num} EVALUATION - Tools: {tools_range}, Epochs: {epochs}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n")
                f.write(formatted_eval_output)
                f.write(f"\n{'='*60}\n\n")
            
            # Evaluate on all previous test sets if requested
            if args.eval_all_previous and round_num > 1:
                print(f"\nEvaluating on all previous test sets...")
                cumulative_results = {}
                # Evaluate on all previous rounds (current round is at the end, so [:-1] gives us all previous)
                for prev_tools, prev_test_dataloader in all_test_dataloaders[:-1]:
                    print(f"\nEvaluating on tools {prev_tools}...")
                    
                    # Capture the formatted evaluation output
                    captured_output = io.StringIO()
                    with redirect_stdout(captured_output):
                        prev_eval_results = eval_native_function_calling(
                            **filter_supported_kwargs(
                                eval_native_function_calling,
                                model=model,
                                tokenizer=tokenizer,
                                test_dataloader=prev_test_dataloader,
                                device=args.device,
                                max_new_tokens=args.max_new_tokens,
                                use_ground_truth_tools=args.use_ground_truth_tools,
                                use_eoc=args.use_eoc,
                                use_js_trunc=args.use_js_trunc,
                                use_logit_bias=args.use_logit_bias,
                            )
                        )
                    
                    # Get the captured formatted output
                    prev_formatted_eval_output = captured_output.getvalue()
                    
                    # Print the formatted output to console as well
                    print(prev_formatted_eval_output)
                    
                    cumulative_results[f"tools_{prev_tools}"] = prev_eval_results
                    
                    # Also log cumulative results to eval file
                    with open(evaluation_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"ROUND {round_num} CUMULATIVE EVALUATION - Previous Tools: {prev_tools}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*60}\n")
                        f.write(prev_formatted_eval_output)
                        f.write(f"\n{'='*60}\n\n")
                        
                all_results[-1]['cumulative_eval_results'] = cumulative_results
        
        # Save checkpoint if requested
        if args.save_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f"round_{round_num}_tools_{tools_range.replace('-', '_')}.pt")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'round': round_num,
                'tools': round_tools,
                'model_state_dict': model.state_dict(),
                'results': round_results,
            }, checkpoint_path)
            all_results[-1]["checkpoint_path"] = checkpoint_path
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"Round {result['round']} (tools {result['tools']}): "
              f"{result['epochs']} epochs, avg loss: {result['avg_loss']:.4f}")
        if 'eval_results' in result and result['eval_results']:
            print(f"  Evaluation accuracy: {result['eval_results'].get('exact_accuracy', 'N/A'):.3f}")
    
    # Run demo on last round if requested
    if args.demo is not None and test_examples is not None:
        print("\n" + "="*60)
        print(f"DEMO: Testing on final round examples")
        print("="*60 + "\n")
        demo_native_function_calling(
            **filter_supported_kwargs(
                demo_native_function_calling,
                model=model,
                tokenizer=tokenizer,
                test_examples=test_examples[:args.demo],
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                use_ground_truth_tools=args.use_ground_truth_tools,
                use_eoc=args.use_eoc,
                use_js_trunc=args.use_js_trunc,
                use_logit_bias=args.use_logit_bias,
            )
        )
    
    print("\n" + "="*60)
    print("Sequential training completed!")
    print(f"Trained {len(rounds)} rounds with different tool sets")
    if args.use_lora and args.freeze_lora_after_first:
        print("LoRA was frozen after first round")
    if args.tensorboard:
        saved_plot_paths = save_training_plot_images(
            plot_history,
            loss_plot_path,
            lr_plot_path,
            run_context["run_name"],
        )
        if saved_plot_paths:
            print("Saved training plots:")
            for saved_plot_path in saved_plot_paths:
                print(f"  {saved_plot_path}")
    print("="*60)
    
    evaluation_results_payload = {
        "experiment_type": "tokmem_sequential",
        "run_name": run_context["run_name"],
        "eval_after_each_round": args.eval_after_each_round,
        "rounds": [
            {
                "round": result["round"],
                "tools": result["tools"],
                "epochs": result["epochs"],
                "eval_results": result.get("eval_results"),
                "cumulative_eval_results": result.get("cumulative_eval_results"),
            }
            for result in all_results
            if result.get("eval_results") is not None or result.get("cumulative_eval_results") is not None
        ],
    }
    write_json(
        artifact_path(run_context, "evaluation_results.json"),
        strip_call_count_breakdown(evaluation_results_payload),
    )
    write_json(
        artifact_path(run_context, "training_summary.json"),
        build_training_summary_payload(
            run_name=run_context["run_name"],
            all_results=all_results,
            experiment_type="tokmem_sequential",
        ),
    )
    
    # Log completion
    logger.info(f"=== Sequential Training Completed at {datetime.now()} ===")


if __name__ == "__main__":
    main()
