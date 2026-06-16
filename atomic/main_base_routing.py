#!/usr/bin/env python3
"""Evaluate raw base-model task routing by task-name likelihood scoring."""

import argparse
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_baseline_utils import (
    ensure_tokenizer_padding,
    is_qwen_tokenizer,
    load_generation_split,
    prepare_run_dir,
    set_random_seed,
    write_json,
    write_jsonl,
)
from task_training import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt-based base-model routing evaluation for atomic Natural Instructions"
    )
    parser.add_argument("--tasks_dir", type=str, default="natural-instructions-2.8/tasks")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, default=700)
    parser.add_argument("--train_size", type=int, default=80)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_instruction_tokens", type=int, default=1024)
    parser.add_argument("--candidate_batch_size", type=int, default=128)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_cache_path", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    return parser.parse_args()


def build_routing_prompt(tokenizer, instruction, query):
    user_text = (
        "Route this Natural Instructions example to its exact task identifier.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{query}\n\n"
        "Return only the task identifier."
    )
    if is_qwen_tokenizer(tokenizer):
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\nTask:"
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\nTask:"
    )


def encode_candidates(tokenizer, task_names):
    candidates = []
    for task_name in task_names:
        token_ids = tokenizer.encode(f" {task_name}", add_special_tokens=False)
        if token_ids:
            candidates.append({"task_name": task_name, "token_ids": token_ids})
    if len(candidates) != len(task_names):
        raise ValueError("Some task names produced empty tokenizations.")
    return candidates


def get_model_input_device(model):
    return model.get_input_embeddings().weight.device


def trim_prompt_ids(prompt_ids, max_length, max_candidate_tokens):
    max_prompt_tokens = max(1, max_length - max_candidate_tokens)
    if len(prompt_ids) > max_prompt_tokens:
        return prompt_ids[-max_prompt_tokens:]
    return prompt_ids


def score_candidate_batch(model, input_ids, attention_mask, prompt_lengths, candidate_lengths, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    scores = []
    for row_index, (prompt_len, candidate_len) in enumerate(zip(prompt_lengths, candidate_lengths)):
        token_positions = torch.arange(
            prompt_len,
            prompt_len + candidate_len,
            device=logits.device,
        )
        prediction_positions = token_positions - 1
        target_ids = input_ids[row_index, token_positions]
        token_log_probs = torch.log_softmax(
            logits[row_index, prediction_positions, :].float(),
            dim=-1,
        )
        gathered = token_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        scores.append(float(gathered.mean().item()))
    return scores


def predict_task_for_example(model, tokenizer, example, candidates, max_length, candidate_batch_size):
    prompt = build_routing_prompt(
        tokenizer=tokenizer,
        instruction=example.get("instruction", ""),
        query=example.get("query", ""),
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_candidate_tokens = max(len(candidate["token_ids"]) for candidate in candidates)
    prompt_ids = trim_prompt_ids(prompt_ids, max_length, max_candidate_tokens)
    device = get_model_input_device(model)
    pad_token_id = tokenizer.pad_token_id

    best_task = None
    best_score = None
    best_token_count = None

    for start in range(0, len(candidates), candidate_batch_size):
        batch_candidates = candidates[start:start + candidate_batch_size]
        sequences = [prompt_ids + candidate["token_ids"] for candidate in batch_candidates]
        max_seq_len = max(len(sequence) for sequence in sequences)
        input_rows = []
        attention_rows = []
        prompt_lengths = []
        candidate_lengths = []
        for sequence, candidate in zip(sequences, batch_candidates):
            pad_len = max_seq_len - len(sequence)
            input_rows.append(sequence + [pad_token_id] * pad_len)
            attention_rows.append([1] * len(sequence) + [0] * pad_len)
            prompt_lengths.append(len(prompt_ids))
            candidate_lengths.append(len(candidate["token_ids"]))

        input_ids = torch.tensor(input_rows, dtype=torch.long)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long)
        scores = score_candidate_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            candidate_lengths=candidate_lengths,
            device=device,
        )

        for candidate, score in zip(batch_candidates, scores):
            if best_score is None or score > best_score:
                best_score = score
                best_task = candidate["task_name"]
                best_token_count = len(candidate["token_ids"])

    return {
        "predicted_task": best_task,
        "routing_score": best_score,
        "candidate_token_count": best_token_count,
    }


def compute_routing_metrics(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["predicted_task"] == row["expected_task"])
    per_task_counts = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in rows:
        expected_task = row["expected_task"]
        per_task_counts[expected_task]["total"] += 1
        if row["predicted_task"] == expected_task:
            per_task_counts[expected_task]["correct"] += 1

    per_task = {}
    for task_name, counts in sorted(per_task_counts.items()):
        task_total = counts["total"]
        task_correct = counts["correct"]
        per_task[task_name] = {
            "total": task_total,
            "correct": task_correct,
            "routing_acc": task_correct / task_total if task_total else 0.0,
        }

    return {
        "task_accuracy": correct / total if total else 0.0,
        "task_correct": correct,
        "total_examples": total,
        "per_task": per_task,
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = prepare_run_dir(
        run_dir=args.run_dir,
        experiment_tag="base_routing",
        model_name=args.model_name,
        num_tasks=args.num_tasks,
    )
    _, eval_logger, training_log, evaluation_log, timestamp = setup_logging(log_dir=run_dir)

    print()
    print("=" * 60)
    print("RAW BASE MODEL ROUTING EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Device map: {args.device_map}")
    print(f"Run directory: {run_dir}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = ensure_tokenizer_padding(tokenizer)
    args.tokenizer_for_sampling = tokenizer

    train_data, val_data, test_data, task_names, split_source, resolved_tasks_dir = load_generation_split(
        args,
        few_shot=False,
    )
    if args.max_examples is not None:
        test_data = test_data[:args.max_examples]

    print(f"Tasks directory: {resolved_tasks_dir}")
    print(f"Split source: {split_source}")
    print(f"Dataset summary - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Tasks: {len(task_names)}")
    print()

    model_load_kwargs = {"torch_dtype": "auto"}
    if args.device_map is not None:
        model_load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    model.eval()

    candidates = encode_candidates(tokenizer, task_names)
    prediction_rows = []
    for index, example in enumerate(tqdm(test_data, desc="Routing examples")):
        prediction = predict_task_for_example(
            model=model,
            tokenizer=tokenizer,
            example=example,
            candidates=candidates,
            max_length=args.max_length,
            candidate_batch_size=args.candidate_batch_size,
        )
        expected_task = example["tasks"][0] if example.get("tasks") else "unknown"
        row = {
            "mode": "base_task_name_likelihood_routing",
            "example_index": index,
            "query": example.get("query", ""),
            "expected_task": expected_task,
            "expected_tasks": example.get("tasks", []),
            "predicted_task": prediction["predicted_task"],
            "routing_correct": prediction["predicted_task"] == expected_task,
            "routing_score": prediction["routing_score"],
            "candidate_token_count": prediction["candidate_token_count"],
        }
        prediction_rows.append(row)

        if (index + 1) % 100 == 0:
            partial = compute_routing_metrics(prediction_rows)
            message = (
                f"Progress: {index + 1}/{len(test_data)} "
                f"routing_acc={partial['task_accuracy']:.4f}"
            )
            print(message)
            eval_logger.info(message)

    metrics = compute_routing_metrics(prediction_rows)
    metrics.update(
        {
            "mode": "base_task_name_likelihood_routing",
            "candidate_count": len(candidates),
            "split_source": split_source,
            "scoring": "mean log probability over candidate task-name tokens",
        }
    )

    predictions_path = os.path.join(run_dir, "base_routing_predictions.jsonl")
    results_path = os.path.join(run_dir, "base_routing_results.json")
    write_jsonl(predictions_path, prediction_rows)
    metrics["predictions_output_path"] = predictions_path
    write_json(results_path, metrics)

    write_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "mode": "base_task_name_likelihood_routing",
            "model_name": args.model_name,
            "tasks_dir": resolved_tasks_dir,
            "num_tasks": args.num_tasks,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "max_length": args.max_length,
            "max_instruction_tokens": args.max_instruction_tokens,
            "candidate_batch_size": args.candidate_batch_size,
            "max_examples": args.max_examples,
            "device": args.device,
            "device_map": args.device_map,
            "seed": args.seed,
            "split_cache_path": os.path.abspath(args.split_cache_path) if args.split_cache_path else None,
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
    write_json(
        os.path.join(run_dir, "run_summary.json"),
        {
            "mode": "base_task_name_likelihood_routing",
            "run_dir": run_dir,
            "model_name": args.model_name,
            "split_source": split_source,
            "metrics": metrics,
            "dataset_summary": {
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "task_count": len(task_names),
            },
            "artifacts": {
                "training_log": training_log,
                "evaluation_log": evaluation_log,
                "base_routing_results": results_path,
                "base_routing_predictions": predictions_path,
            },
        },
    )

    print("\n" + "=" * 50)
    print("FINAL BASE ROUTING RESULTS")
    print(f"Task Prediction Accuracy: {metrics['task_accuracy']:.4f} ({metrics['task_correct']}/{metrics['total_examples']})")
    print(f"Results: {results_path}")
    print(f"Predictions: {predictions_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
