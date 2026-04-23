#!/usr/bin/env python3
"""
LoRA Baseline - Sequential training script for function calling.
Uses standard LoRA fine-tuning instead of tool-specific embeddings.
"""

import argparse
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import disable_fsdp_ram_efficient_loading, enable_fsdp_ram_efficient_loading
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
import json
import os
import logging
from contextlib import nullcontext

from fsdp_utils import build_accelerator
from dataset import discover_available_tools
from replay_buffer import SimpleReplayBuffer
from run_layout import (
    DEFAULT_RUNS_DIR,
    artifact_path,
    build_training_summary_payload,
    build_run_config,
    resolve_run_context,
    write_json,
)

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:
    FSDP = None


def set_random_seed(seed):
    """Seed Python, NumPy, and Torch RNGs for reproducible LoRA runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _unwrap_model(model, accelerator=None):
    return accelerator.unwrap_model(model) if accelerator is not None else model


def _maybe_move_batch(batch, device, accelerator=None):
    if accelerator is not None:
        return batch["input_ids"], batch["attention_mask"], batch.get("labels")
    labels = batch.get("labels")
    if labels is None:
        return batch["input_ids"].to(device), batch["attention_mask"].to(device), None
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        labels.to(device),
    )


def _fsdp_generation_context(model, accelerator=None):
    if FSDP is None:
        return nullcontext()
    if not isinstance(model, FSDP):
        return nullcontext()
    if accelerator is not None and getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return nullcontext()
    fsdp_plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None) if accelerator is not None else None
    sharding_strategy = getattr(fsdp_plugin, "sharding_strategy", None)
    strategy_name = getattr(sharding_strategy, "name", str(sharding_strategy))
    if strategy_name == "NO_SHARD":
        return nullcontext()
    return FSDP.summon_full_params(model, recurse=True, writeback=False)


def _optimizer_step_count(num_batches, num_epochs, gradient_accumulation_steps):
    return max(1, math.ceil(num_batches / gradient_accumulation_steps) * num_epochs)


def _generated_continuation_tokens(generated_sequence, padded_input_ids):
    """Slice off the full padded prompt width before decoding generated tokens."""
    prompt_width = int(padded_input_ids.shape[-1])
    return generated_sequence[prompt_width:]


def _save_lora_checkpoint(model, checkpoint_path, accelerator=None):
    should_write = accelerator is None or accelerator.is_main_process
    base_model = _unwrap_model(model, accelerator=accelerator)
    with _fsdp_generation_context(model, accelerator=accelerator):
        if not should_write:
            return None
        os.makedirs(checkpoint_path, exist_ok=True)
        state_dict = get_peft_model_state_dict(base_model)
        base_model.save_pretrained(checkpoint_path, state_dict=state_dict)
    return checkpoint_path


def _cast_trainable_params_to_model_dtype(model):
    """Keep PEFT adapter params in the same dtype as the base model for FSDP wrapping."""
    target_dtype = next(model.parameters()).dtype
    for param in model.parameters():
        if param.requires_grad and param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)


def collate_fn(batch):
    """Custom collate function to handle variable length sequences with left padding"""
    # Extract components
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    raw_data = [item['raw_data'] for item in batch]
    
    # Stack tensors (they should all be the same length due to max_length padding)
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'raw_data': raw_data
    }


class FunctionCallingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, mode="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Create a consistent mapping of all tools to generic labels
        all_tools = set()
        for example in self.data:
            if 'tools' in example:
                all_tools.update(example['tools'])
        
        # Sort tools for consistent mapping across runs
        sorted_tools = sorted(list(all_tools))
        self.tool_mapping = {tool: f"tool_{i+1}" for i, tool in enumerate(sorted_tools)}
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        user_input = example['user_input']
        
        if self.mode == "train":
            # Training: include expected function calls in output
            tools = example.get('tools', [])
            function_calls = example.get('function_calls', [])
            
            conversation = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n"
            
            if tools and function_calls:
                # Use generic tool tokens for fair comparison
                for tool, func_call in zip(tools, function_calls):
                    generic_tool = self.tool_mapping.get(tool, "tool_unknown")
                    conversation += f"\n[{generic_tool}]{func_call}"  # Add generic tool marker
            
            conversation += "<|eot_id|>"
            
        else:
            # Evaluation: only user input
            conversation = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n"
        
        encoding = self.tokenizer(
            conversation,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        if self.mode == "train":
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Only train on assistant response
            assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n"
            if assistant_start in conversation:
                prefix = conversation.split(assistant_start)[0] + assistant_start
                prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                if len(prefix_tokens) < len(labels):
                    labels[:len(prefix_tokens)] = -100
        else:
            labels = torch.tensor(-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_data": example
        }


def create_lora_dataloader(
    train_data_path,
    test_data_path,
    tokenizer,
    batch_size=4,
    max_length=512,
    eval_batch_size=32,
):
    """Create dataloaders for LoRA training"""
    
    # Create datasets
    train_dataset = FunctionCallingDataset(train_data_path, tokenizer, max_length, "train")
    test_dataset = FunctionCallingDataset(test_data_path, tokenizer, max_length, "eval")
    
    # Create dataloaders with collate_fn
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader


def create_mixed_dataloader_with_replay(
    train_dataset,
    replay_buffer,
    batch_size,
    replay_ratio,
):
    """
    Create a dataloader that mixes new training samples with replay samples.
    
    Args:
        train_dataset: Dataset with new training samples
        replay_buffer: SimpleReplayBuffer with previous samples
        batch_size: Total batch size
        replay_ratio: Proportion of batch that should be replay samples (0.0-1.0)
    """
    
    class MixedDataset(Dataset):
        def __init__(self, train_dataset, replay_samples, replay_ratio):
            self.train_dataset = train_dataset
            self.replay_samples = replay_samples
            self.replay_ratio = replay_ratio
            
            # Calculate how many samples of each type per epoch
            self.replay_per_batch = int(batch_size * replay_ratio)
            self.new_per_batch = batch_size - self.replay_per_batch
            
            # Create indices for sampling
            self.train_indices = list(range(len(train_dataset)))
            
        def __len__(self):
            return len(self.train_dataset)
            
        def __getitem__(self, idx):
            # This gets called by DataLoader, but we'll handle mixing in the dataloader
            return self.train_dataset[idx]
    
    # Get replay samples
    replay_samples = replay_buffer.get_all() if replay_buffer else []
    
    if not replay_samples or replay_ratio == 0.0:
        # No replay samples, return regular dataloader
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    
    # Create mixed dataset
    mixed_dataset = MixedDataset(train_dataset, replay_samples, replay_ratio)
    
    def mixed_collate_fn(batch):
        """Custom collate function that mixes new and replay samples"""
        replay_per_batch = int(len(batch) * replay_ratio)
        new_per_batch = len(batch) - replay_per_batch
        
        # Keep only new_per_batch new samples
        new_batch = batch[:new_per_batch] if new_per_batch > 0 else []
        
        # Sample replay_per_batch replay samples
        replay_batch = []
        if replay_per_batch > 0 and replay_samples:
            sampled_replay = random.sample(replay_samples, min(replay_per_batch, len(replay_samples)))
            for replay_sample in sampled_replay:
                # Convert replay sample back to dataset format
                replay_batch.append(replay_sample)
        
        # Combine and shuffle
        combined_batch = new_batch + replay_batch
        random.shuffle(combined_batch)
        
        # Use original collate function
        return collate_fn(combined_batch)
    
    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mixed_collate_fn,
    )


def train_lora_model(
    model,
    train_dataloader,
    num_epochs=3,
    lr=5e-4,
    device="cuda",
    gradient_accumulation_steps=1,
    accelerator=None,
):
    """Train LoRA model using standard fine-tuning approach"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    model.train()
    should_log = accelerator is None or accelerator.is_main_process
    base_model = _unwrap_model(model, accelerator=accelerator)
    
    # Set up optimizer for LoRA parameters
    optimizer = AdamW((param for param in base_model.parameters() if param.requires_grad), lr=lr, weight_decay=0.01)
    
    total_optimizer_steps = _optimizer_step_count(
        len(train_dataloader),
        num_epochs,
        gradient_accumulation_steps,
    )
    
    # Create linear learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_optimizer_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_optimizer_steps
    )
    if accelerator is not None:
        optimizer = accelerator.prepare_optimizer(optimizer)
        scheduler = accelerator.prepare_scheduler(scheduler)

    if should_log:
        print(f"Training LoRA for {num_epochs} epochs, {len(train_dataloader)} batches per epoch")
        print(f"Total optimizer steps: {total_optimizer_steps}")
        print(f"Learning rate: {lr} (with linear schedule + warmup)")
        print(f"Warmup steps: {total_optimizer_steps // 10}")
    
    total_loss = 0
    step_count = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            accumulation_context = accelerator.accumulate(model) if accelerator is not None else nullcontext()
            with accumulation_context:
                input_ids, attention_mask, labels = _maybe_move_batch(batch, device, accelerator=accelerator)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass
                if accelerator is not None:
                    accelerator.backward(loss)
                    should_step = accelerator.sync_gradients
                else:
                    (loss / gradient_accumulation_steps).backward()
                    should_step = (
                        ((step_count + 1) % gradient_accumulation_steps == 0)
                        or (batch_idx + 1 == len(train_dataloader))
                    )

                if should_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            step_count += 1
            
            if should_log and (batch_idx + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if should_log:
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.4f}")
    
    if accelerator is not None:
        reduced = accelerator.reduce(torch.tensor([total_loss, float(step_count)], device=accelerator.device), reduction="sum")
        total_loss = reduced[0].item()
        step_count = int(reduced[1].item())
    avg_total_loss = total_loss / step_count
    if should_log:
        print(f"Training completed! Average total loss: {avg_total_loss:.4f}")
        print(f"Optimizer steps: {optimizer_steps}")
    
    return {
        'avg_loss': avg_total_loss,
        'total_steps': step_count,
        'optimizer_steps': optimizer_steps,
    }


def extract_function_calls_from_text(text):
    """Extract function calls and tool tokens from generated text"""
    import re
    function_calls = []
    tools = []
    
    # Split text into lines and process each line
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for lines with tool markers: [tool_X]{json}
        tool_pattern = r'^\s*\[(tool_\d+)\](.+)$'
        tool_match = re.match(tool_pattern, line)
        
        if tool_match:
            tool = tool_match.group(1)
            json_str = tool_match.group(2).strip()
            
            try:
                # Validate it's proper JSON
                json.loads(json_str)
                function_calls.append(json_str)
                tools.append(tool)
            except json.JSONDecodeError:
                # Try to extract JSON from the string
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, json_str)
                for json_match in json_matches:
                    try:
                        json.loads(json_match)
                        function_calls.append(json_match)
                        tools.append(tool)
                        break  # Only take first valid JSON
                    except json.JSONDecodeError:
                        pass
        else:
            # Fallback: look for JSON without tool marker
            json_pattern = r'^\s*(\{.*\})\s*$'
            match = re.match(json_pattern, line)
            
            if match:
                json_candidate = match.group(1)
                try:
                    json.loads(json_candidate)
                    function_calls.append(json_candidate)
                    tools.append("tool_unknown")
                except json.JSONDecodeError:
                    pass
    
    return function_calls, tools


def eval_lora_model(model, tokenizer, test_dataloader, device="cuda", accelerator=None):
    """Evaluate LoRA model using function calling metrics"""
    import time
    from collections import defaultdict
    from eval import compare_function_calls_advanced, calculate_tool_metrics
    
    model.eval()
    should_log = accelerator is None or accelerator.is_main_process
    
    # Calculate total examples from dataloader
    total_examples = len(test_dataloader.dataset)
    if should_log:
        print("🔄 Running LoRA model evaluation...")
    start_time = time.time()
    
    exact_matches = 0
    tool_tp = 0
    tool_tn = 0
    tool_fp = 0
    tool_fn = 0
    tool_exact_matches = 0
    processed_examples = 0
    parse_errors = 0
    f1_scores = []
    precision_scores = []
    recall_scores = []
    tool_f1_scores = []
    tool_precision_scores = []
    tool_recall_scores = []
    call_count_breakdown = defaultdict(lambda: {
        'total': 0, 
        'exact_matches': 0, 
        'tool_tp': 0,
        'tool_tn': 0,
        'tool_fp': 0,
        'tool_fn': 0,
        'tool_exact_matches': 0,
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'tool_f1_scores': [],
        'tool_precision_scores': [],
        'tool_recall_scores': [],
        'parse_errors': 0
    })

    dataset = test_dataloader.dataset
    candidate_tools = list(getattr(dataset, "tool_mapping", {}).values())
    if not candidate_tools:
        candidate_tools = [
            tool
            for example in getattr(dataset, "data", [])
            for tool in example.get("tools", [])
        ]
    candidate_tools = list(dict.fromkeys(candidate_tools))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            input_ids, attention_mask, _ = _maybe_move_batch(batch, device, accelerator=accelerator)
            batch_size = len(batch['raw_data'])
            processed_examples += batch_size
            
            if should_log and (batch_idx % 10 == 0 or processed_examples == total_examples):
                print(f"   Progress: {processed_examples}/{total_examples} ({100 * processed_examples / total_examples:.1f}%)")
            
            # Generate responses
            with _fsdp_generation_context(model, accelerator=accelerator):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            local_outputs = []
            for i, example in enumerate(batch["raw_data"]):
                generated_tokens = _generated_continuation_tokens(generated[i], input_ids[i])
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                local_outputs.append({"example": example, "generated_text": generated_text})

            if accelerator is not None:
                local_outputs = accelerator.gather_for_metrics(local_outputs, use_gather_object=True)
            if not should_log:
                continue
            
            # Process each example
            for output in local_outputs:
                example = output["example"]
                expected_calls = example.get('function_calls', [])
                expected_tools = example.get('tools', [])
                expected_call_count = len(expected_calls)
                
                # Get dataset from dataloader to access tool mapping
                dataset = test_dataloader.dataset
                expected_generic_tools = [dataset.tool_mapping.get(tool, "tool_unknown") for tool in expected_tools]
                
                generated_text = output["generated_text"]
                
                # Extract function calls and predicted tools
                predicted_calls, predicted_tools = extract_function_calls_from_text(generated_text)
                
                tool_metrics = calculate_tool_metrics(
                    predicted_tools=predicted_tools,
                    expected_tools=expected_generic_tools,
                    candidate_tools=candidate_tools,
                )
                tool_f1_scores.append(tool_metrics['tool_f1_score'])
                tool_precision_scores.append(tool_metrics['tool_precision'])
                tool_recall_scores.append(tool_metrics['tool_recall'])
                tool_tp += tool_metrics['tool_tp']
                tool_tn += tool_metrics['tool_tn']
                tool_fp += tool_metrics['tool_fp']
                tool_fn += tool_metrics['tool_fn']
                if tool_metrics['tool_exact_match_acc'] >= 1.0:
                    tool_exact_matches += 1
                    call_count_breakdown[expected_call_count]['tool_exact_matches'] += 1
                call_count_breakdown[expected_call_count]['tool_tp'] += tool_metrics['tool_tp']
                call_count_breakdown[expected_call_count]['tool_tn'] += tool_metrics['tool_tn']
                call_count_breakdown[expected_call_count]['tool_fp'] += tool_metrics['tool_fp']
                call_count_breakdown[expected_call_count]['tool_fn'] += tool_metrics['tool_fn']
                
                # Use the same evaluation function as native function calling for fair comparison
                eval_result = compare_function_calls_advanced(
                    predicted_calls,
                    expected_calls,
                    ignore_order=True
                )
                
                if eval_result.exact_match:
                    exact_matches += 1
                
                f1_scores.append(eval_result.f1_score)
                precision_scores.append(eval_result.precision)
                recall_scores.append(eval_result.recall)
                
                # Track parse errors from eval_result
                if 'parse_errors' in eval_result.details:
                    current_parse_errors = eval_result.details['parse_errors']['outputs']
                    parse_errors += current_parse_errors
                else:
                    current_parse_errors = 0
                
                # Track by call count
                call_count_breakdown[expected_call_count]['total'] += 1
                if eval_result.exact_match:
                    call_count_breakdown[expected_call_count]['exact_matches'] += 1
                call_count_breakdown[expected_call_count]['f1_scores'].append(eval_result.f1_score)
                call_count_breakdown[expected_call_count]['precision_scores'].append(eval_result.precision)
                call_count_breakdown[expected_call_count]['recall_scores'].append(eval_result.recall)
                call_count_breakdown[expected_call_count]['tool_f1_scores'].append(tool_metrics['tool_f1_score'])
                call_count_breakdown[expected_call_count]['tool_precision_scores'].append(tool_metrics['tool_precision'])
                call_count_breakdown[expected_call_count]['tool_recall_scores'].append(tool_metrics['tool_recall'])
                call_count_breakdown[expected_call_count]['parse_errors'] += current_parse_errors
    
    end_time = time.time()
    eval_time = end_time - start_time
    if not should_log:
        return None
    
    # Calculate final metrics
    exact_accuracy = exact_matches / total_examples
    tool_judgments = tool_tp + tool_tn + tool_fp + tool_fn
    tool_accuracy = (tool_tp + tool_tn) / tool_judgments if tool_judgments > 0 else 1.0
    tool_exact_match_acc = tool_exact_matches / total_examples if total_examples > 0 else 0.0
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_tool_f1_score = sum(tool_f1_scores) / len(tool_f1_scores) if tool_f1_scores else 0.0
    avg_tool_precision = sum(tool_precision_scores) / len(tool_precision_scores) if tool_precision_scores else 0.0
    avg_tool_recall = sum(tool_recall_scores) / len(tool_recall_scores) if tool_recall_scores else 0.0
    parse_error_rate = parse_errors / total_examples
    
    # Print formatted results matching the style from training.py
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"📋 Dataset: {total_examples} examples")
    print(f"⏱️  Evaluation time: {eval_time:.2f} seconds")
    print(f"🔧 Mode: LoRA Baseline")
    print()
    
    print("🎯 RESULTS:")
    print(f"   Exact Match Accuracy:     {exact_accuracy:.3f} ({exact_matches}/{total_examples})")
    print(
        "   Tool Acc (tool_accuracy): "
        f"{tool_accuracy:.3f} (TP={tool_tp}, TN={tool_tn}, FP={tool_fp}, FN={tool_fn})"
    )
    print(
        "   Tool Exact Match Acc (tool_exact_match_acc): "
        f"{tool_exact_match_acc:.3f} ({tool_exact_matches}/{total_examples})"
    )
    print(f"   Average F1 Score:         {avg_f1_score:.3f}")
    print(f"   Average Precision:        {avg_precision:.3f}")
    print(f"   Average Recall:           {avg_recall:.3f}")
    print(f"   Average Tool F1 Score:    {avg_tool_f1_score:.3f}")
    print(f"   Average Tool Precision:   {avg_tool_precision:.3f}")
    print(f"   Average Tool Recall:      {avg_tool_recall:.3f}")
    print(f"   Parse Error Rate:         {parse_error_rate:.3f}")
    print("=" * 50)
    
    # Breakdown by function call count
    print("\n📊 EXACT MATCH ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        accuracy = stats['exact_matches'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {accuracy:.3f} ({stats['exact_matches']}/{stats['total']})")
    print("=" * 50)
    
    print("\n📊 TOOL ACCURACY (tool_accuracy):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        tool_total = stats['tool_tp'] + stats['tool_tn'] + stats['tool_fp'] + stats['tool_fn']
        per_call_tool_accuracy = (stats['tool_tp'] + stats['tool_tn']) / tool_total if tool_total > 0 else 1.0
        print(
            f"   {call_count} call(s): {per_call_tool_accuracy:.3f} "
            f"(TP={stats['tool_tp']}, TN={stats['tool_tn']}, FP={stats['tool_fp']}, FN={stats['tool_fn']})"
        )
    print("=" * 50)

    print("\n📊 TOOL EXACT MATCH ACCURACY (tool_exact_match_acc):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        per_call_tool_exact_match_acc = stats['tool_exact_matches'] / stats['total'] if stats['total'] > 0 else 0.0
        print(
            f"   {call_count} call(s): {per_call_tool_exact_match_acc:.3f} "
            f"({stats['tool_exact_matches']}/{stats['total']})"
        )
    print("=" * 50)
    
    print("\n📊 AVERAGE F1 SCORE (Function Calls):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_f1 = sum(stats['f1_scores']) / len(stats['f1_scores']) if stats['f1_scores'] else 0.0
        avg_prec = sum(stats['precision_scores']) / len(stats['precision_scores']) if stats['precision_scores'] else 0.0
        avg_rec = sum(stats['recall_scores']) / len(stats['recall_scores']) if stats['recall_scores'] else 0.0
        print(f"   {call_count} call(s): F1={avg_f1:.3f}, P={avg_prec:.3f}, R={avg_rec:.3f}")
    print("=" * 50)
    
    print("\n📊 AVERAGE TOOL F1 SCORE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_tool_f1 = sum(stats['tool_f1_scores']) / len(stats['tool_f1_scores']) if stats['tool_f1_scores'] else 0.0
        avg_tool_prec = sum(stats['tool_precision_scores']) / len(stats['tool_precision_scores']) if stats['tool_precision_scores'] else 0.0
        avg_tool_rec = sum(stats['tool_recall_scores']) / len(stats['tool_recall_scores']) if stats['tool_recall_scores'] else 0.0
        print(f"   {call_count} call(s): Tool F1={avg_tool_f1:.3f}, P={avg_tool_prec:.3f}, R={avg_tool_rec:.3f}")
    print("=" * 50)
    
    print("\n📊 PARSE ERROR RATE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        error_rate = stats['parse_errors'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {error_rate:.3f} ({stats['parse_errors']}/{stats['total']})")
    print("=" * 50)
    
    return {
        'exact_accuracy': exact_accuracy,
        'tool_accuracy': tool_accuracy, 
        'tool_exact_match_acc': tool_exact_match_acc,
        'avg_f1_score': avg_f1_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_tool_f1_score': avg_tool_f1_score,
        'avg_tool_precision': avg_tool_precision,
        'avg_tool_recall': avg_tool_recall,
        'parse_error_rate': parse_error_rate,
        'total_examples': total_examples,
        'call_count_breakdown': dict(call_count_breakdown)
    }


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


def main():
    parser = argparse.ArgumentParser(description="LoRA Baseline - Sequential Function Calling Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model name")
    
    # Sequential training arguments
    parser.add_argument("--training_rounds", type=str, required=True,
                        help="Training rounds specification (e.g., '1-10:3,11-20:3,21-30:2' for 3 rounds)")
    
    # Training arguments  
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    
    # Evaluation arguments
    parser.add_argument("--eval_after_each_round", action="store_true",
                        help="Run evaluation after each training round")
    parser.add_argument("--eval_all_previous", action="store_true",
                        help="Evaluate on all previous test sets after each round")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    
    # Replay buffer arguments
    parser.add_argument("--use_replay_buffer", action="store_true",
                        help="Use replay buffer to mitigate catastrophic forgetting")
    parser.add_argument("--replay_buffer_size", type=int, default=1000,
                        help="Size of replay buffer")
    parser.add_argument("--replay_ratio", type=float, default=0.2,
                        help="Ratio of replay samples in each batch (0.0-1.0)")
    parser.add_argument("--demo", type=int, default=None,
                        help="Number of demo examples to show after training")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype")
    parser.add_argument("--use_fsdp", action="store_true",
                        help="Wrap the backbone with Accelerate FSDP for multi-GPU training")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Accelerate mixed precision mode")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="NO_SHARD",
                       choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "HYBRID_SHARD_ZERO2"],
                       help="FSDP sharding strategy")
    parser.add_argument("--fsdp_backward_prefetch", type=str, default="BACKWARD_PRE",
                        choices=["BACKWARD_PRE", "BACKWARD_POST", "none"],
                        help="FSDP backward prefetch mode")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate (default: 0.1)")
    parser.add_argument("--lora_target_modules", type=str, 
                        default="q_proj,v_proj",
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
    parser.add_argument("--reinit_lora_after_each_round", action="store_true",
                        help="Reinitialize LoRA parameters after each training round")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Explicit run name")
    parser.add_argument("--run_root_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Root directory for compositional runs")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Optional tag appended to the generated run name")
    
    args = parser.parse_args()
    accelerator = build_accelerator(args, args.model_name, use_seedable_sampler=True)
    should_log = accelerator.is_main_process
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    set_random_seed(args.seed)
    
    # Setup simple logging
    import sys
    from datetime import datetime
    
    run_context = resolve_run_context(
        experiment_name="compositional_lora",
        model_name=args.model_name,
        run_root_dir=args.run_root_dir,
        run_name=args.run_name,
        run_tag=args.run_tag,
    )

    evaluation_log_file = resolve_run_file_path(run_context, args.log_file, "evaluation.log")
    checkpoint_dir = run_context["run_dir"]
    if args.checkpoint_dir:
        checkpoint_dir = (
            args.checkpoint_dir
            if os.path.isabs(args.checkpoint_dir)
            else artifact_path(run_context, os.path.basename(args.checkpoint_dir.rstrip("/")))
        )
    if should_log:
        open(evaluation_log_file, "a", encoding="utf-8").close()

    log_handlers = []
    if should_log:
        log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(evaluation_log_file, mode='a')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=log_handlers)
    logger = logging.getLogger(__name__)
    
    # Log start time and configuration
    if should_log:
        logger.info(f"=== LoRA Sequential Training Started at {datetime.now()} ===")
        logger.info(f"Configuration: model={args.model_name}, rounds={args.training_rounds}, batch_size={args.batch_size}")
        logger.info(f"Run directory: {run_context['run_dir']}")
        if args.reinit_lora_after_each_round:
            logger.info("LoRA reinitialization enabled: Parameters will be reset after each round")
    
    # Parse training rounds
    try:
        rounds = parse_training_rounds(args.training_rounds)
        if should_log:
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
            if should_log:
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
    
    if should_log:
        print("\n=== LoRA Baseline - Sequential Function Calling Training ===")
        print(f"Model: {args.model_name}")
        print(f"Training rounds: {len(rounds)}")
        print(f"World size: {world_size}")
        print(f"Use FSDP: {args.use_fsdp}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Reinit LoRA after each round: {args.reinit_lora_after_each_round}")
        print(f"Method: Standard LoRA fine-tuning{' with reinitialization' if args.reinit_lora_after_each_round else ''}")
        print(f"Run directory: {run_context['run_dir']}")
        print()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"  # Use left padding for decoder-only models
    
    # Create checkpoint directory if needed
    if args.save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def load_base_model():
        if should_log:
            print("Loading base model...")
        model_device = "cpu" if args.use_fsdp else (str(accelerator.device) if args.device == "cuda" else args.device)
        if args.use_fsdp:
            enable_fsdp_ram_efficient_loading()
        try:
            return AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=dtype,
            ).to(model_device)
        finally:
            if args.use_fsdp:
                disable_fsdp_ram_efficient_loading()
    
    # Parse target modules once
    target_modules = [mod.strip() for mod in args.lora_target_modules.split(',')]
    
    # Store results for all rounds
    all_results = []
    # Store test dataloaders for cumulative evaluation
    all_test_dataloaders = []
    
    # Initialize model variable
    model = None
    model_prepared = False
    
    # Initialize replay buffer if enabled
    replay_buffer = None
    if args.use_replay_buffer:
        replay_buffer = SimpleReplayBuffer(max_size=args.replay_buffer_size)
        if should_log:
            print(f"Initialized replay buffer with size {args.replay_buffer_size}, replay ratio {args.replay_ratio}")

    if should_log:
        write_json(
            artifact_path(run_context, "run_config.json"),
            build_run_config(
                vars(args),
                run_context,
                extra={
                    "experiment_type": "lora_sequential",
                    "rounds": rounds,
                    "world_size": world_size,
                    "data_dir": os.path.abspath(args.data_dir),
                    "artifacts": {
                        "evaluation_log": evaluation_log_file,
                        "checkpoint_dir": checkpoint_dir,
                    },
                },
            ),
        )
    
    # Training loop for each round
    for round_idx, round_spec in enumerate(rounds):
        round_num = round_idx + 1
        tools_range = round_spec['tools']
        epochs = round_spec['epochs']
        
        if should_log:
            print("\n" + "="*60)
            print(f"ROUND {round_num}/{len(rounds)}: Training on tools {tools_range}")
            print("="*60 + "\n")
        
        # Initialize or reinitialize LoRA for this round
        if round_idx == 0 or args.reinit_lora_after_each_round:
            if round_idx > 0 and should_log:
                print("Reinitializing LoRA parameters for new round...")
            base_model = load_base_model()
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )
            
            # Apply LoRA to base model
            model = get_peft_model(base_model, lora_config)
            _cast_trainable_params_to_model_dtype(model)
            model_prepared = False
            if should_log:
                print("LoRA configuration applied!")
                model.print_trainable_parameters()
        
        # Construct data file names based on tool range
        train_max_calls = train_max_calls_by_round[round_idx]
        test_max_calls = test_max_calls_by_round[round_idx]
        train_data_file = os.path.join(
            args.data_dir, 
            f"training/function_calling_train_tools{tools_range}_{train_max_calls}calls.json"
        )
        test_data_file = os.path.join(
            args.data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_max_calls}calls.json"
        )
        
        # Check if data files exist
        if not os.path.exists(train_data_file):
            if should_log:
                print(f"Warning: Training data file not found: {train_data_file}")
                print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        if not os.path.exists(test_data_file):
            if should_log:
                print(f"Warning: Test data file not found: {test_data_file}")
                print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        
        # Discover tools for this round
        if should_log:
            print(f"Discovering tools from round {round_num} dataset...")
        round_tools = discover_available_tools(train_data_file, test_data_file)
        if should_log:
            print(
                f"Found {len(round_tools)} tools for round {round_num}: {round_tools[:5]}..."
                if len(round_tools) > 5
                else f"Found {len(round_tools)} tools: {round_tools}"
            )
        
        # Create datasets for this round
        if should_log:
            print(f"Creating round {round_num} dataset...")
        train_dataloader, test_dataloader = create_lora_dataloader(
            train_data_path=train_data_file,
            test_data_path=test_data_file,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            eval_batch_size=args.eval_batch_size,
        )
        
        # Create mixed dataloader with replay buffer if enabled
        if args.use_replay_buffer and replay_buffer is not None:
            if should_log:
                print(f"Creating mixed dataloader with {replay_buffer.size()} replay samples (ratio: {args.replay_ratio})")
            # Get the train dataset from the dataloader
            train_dataset = train_dataloader.dataset
            train_dataloader = create_mixed_dataloader_with_replay(
                train_dataset,
                replay_buffer,
                args.batch_size,
                args.replay_ratio,
            )

        if not model_prepared:
            model, train_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, test_dataloader)
            model_prepared = True
        else:
            train_dataloader, test_dataloader = accelerator.prepare(train_dataloader, test_dataloader)
        
        # Store test dataloader for cumulative evaluation
        all_test_dataloaders.append((tools_range, test_dataloader))
        
        # Train this round
        if should_log:
            print(f"\nTraining round {round_num} for {epochs} epochs...")
            print("Training: LoRA fine-tuning")
        
        round_results = train_lora_model(
            model=model,
            train_dataloader=train_dataloader,
            num_epochs=epochs,
            lr=args.lr,
            device=str(accelerator.device),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            accelerator=accelerator,
        )
        
        if should_log:
            print(f"Round {round_num} training completed! Average loss: {round_results['avg_loss']:.4f}")
            logger.info(f"\n[ROUND {round_num} RESULTS] Tools: {tools_range}, Epochs: {epochs}, Loss: {round_results['avg_loss']:.4f}")
        
        # Add samples to replay buffer after training
        if args.use_replay_buffer and replay_buffer is not None:
            # Load training data to add to replay buffer
            with open(train_data_file, 'r') as f:
                training_data = json.load(f)
            
            # Sample a subset for the replay buffer (don't add all samples)
            max_samples_to_add = min(len(training_data) // 2, 500)  # Add at most 500 samples per round
            samples_to_add = random.sample(training_data, max_samples_to_add)
            
            replay_buffer.add(samples_to_add)
            if should_log:
                print(f"Added {len(samples_to_add)} samples to replay buffer. Buffer size: {replay_buffer.size()}")
        
        # Store results
        all_results.append({
            'round': round_num,
            'tools': tools_range,
            'epochs': epochs,
            'avg_loss': round_results['avg_loss'],
            'results': round_results
        })
        
        # Import necessary modules for output capture
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Evaluate this round
        if args.eval_after_each_round:
            if should_log:
                print(f"\nEvaluating round {round_num} model...")
                captured_output = io.StringIO()
                with redirect_stdout(captured_output):
                    eval_results = eval_lora_model(
                        model=model,
                        tokenizer=tokenizer,
                        test_dataloader=test_dataloader,
                        device=str(accelerator.device),
                        accelerator=accelerator,
                    )
                formatted_eval_output = captured_output.getvalue()
                print(formatted_eval_output)
                all_results[-1]['eval_results'] = eval_results
                with open(evaluation_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"ROUND {round_num} EVALUATION - Tools: {tools_range}, Epochs: {epochs}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*60}\n")
                    f.write(formatted_eval_output)
                    f.write(f"\n{'='*60}\n\n")
            else:
                eval_lora_model(
                    model=model,
                    tokenizer=tokenizer,
                    test_dataloader=test_dataloader,
                    device=str(accelerator.device),
                    accelerator=accelerator,
                )
            
            # Evaluate on all previous test sets if requested
            if args.eval_all_previous and round_num > 1:
                if should_log:
                    print(f"\nEvaluating on all previous test sets...")
                cumulative_results = {}
                # Evaluate on all previous rounds (current round is at the end, so [:-1] gives us all previous)
                for prev_tools, prev_test_dataloader in all_test_dataloaders[:-1]:
                    if should_log:
                        print(f"  Evaluating on tools {prev_tools}...")
                        prev_captured_output = io.StringIO()
                        with redirect_stdout(prev_captured_output):
                            prev_eval_results = eval_lora_model(
                                model=model,
                                tokenizer=tokenizer,
                                test_dataloader=prev_test_dataloader,
                                device=str(accelerator.device),
                                accelerator=accelerator,
                            )
                        prev_formatted_eval_output = prev_captured_output.getvalue()
                        print(prev_formatted_eval_output)
                        cumulative_results[f"tools_{prev_tools}"] = prev_eval_results
                        with open(evaluation_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"ROUND {round_num} EVAL on tools {prev_tools}\n")
                            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                            f.write(f"{'='*60}\n")
                            f.write(prev_formatted_eval_output)
                            f.write(f"\n{'='*60}\n\n")
                    else:
                        eval_lora_model(
                            model=model,
                            tokenizer=tokenizer,
                            test_dataloader=prev_test_dataloader,
                            device=str(accelerator.device),
                            accelerator=accelerator,
                        )
                if should_log:
                    all_results[-1]['cumulative_eval_results'] = cumulative_results
        
        # Save checkpoint if requested
        if args.save_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f"round_{round_num}_tools_{tools_range.replace('-', '_')}")
            if should_log:
                print(f"Saving checkpoint to {checkpoint_path}")
            _save_lora_checkpoint(model, checkpoint_path, accelerator=accelerator)
            if accelerator is not None:
                accelerator.wait_for_everyone()
            if should_log:
                all_results[-1]["checkpoint_path"] = checkpoint_path
    
    # Final summary
    if should_log:
        print("\n" + "="*60)
        print("LORA TRAINING SUMMARY")
        print("="*60)
        for result in all_results:
            print(f"Round {result['round']} (tools {result['tools']}): "
                  f"{result['epochs']} epochs, avg loss: {result['avg_loss']:.4f}")
            if 'eval_results' in result and result['eval_results']:
                print(f"  Evaluation accuracy: {result['eval_results'].get('exact_accuracy', 'N/A'):.3f}")

        print("\n" + "="*60)
        print("LoRA sequential training completed!")
        print(f"Trained {len(rounds)} rounds with different tool sets")
        if args.reinit_lora_after_each_round:
            print("Method: Standard LoRA fine-tuning with reinitialization after each round")
        else:
            print("Method: Standard LoRA fine-tuning")
        print("="*60)
    
    evaluation_results_payload = {
        "experiment_type": "lora_sequential",
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
    if should_log:
        write_json(
            artifact_path(run_context, "evaluation_results.json"),
            strip_call_count_breakdown(evaluation_results_payload),
        )
        write_json(
            artifact_path(run_context, "training_summary.json"),
            build_training_summary_payload(
                run_name=run_context["run_name"],
                all_results=all_results,
                experiment_type="lora_sequential",
            ),
        )
    
    # Log completion
    if should_log:
        logger.info(f"=== LoRA Sequential Training Completed at {datetime.now()} ===")


if __name__ == "__main__":
    main()
