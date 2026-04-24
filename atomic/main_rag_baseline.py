#!/usr/bin/env python3
"""Evaluate the raw base model with SBERT retrieval-augmented few-shot prompts."""

import argparse
import gc
import json
import os
from collections import defaultdict

import torch
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
from test_sbert_retriever import SBERTRetriever


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="SBERT RAG baseline for atomic Natural Instructions")
    parser.add_argument("--tasks_dir", type=str, default="natural-instructions-2.8/tasks",
                        help="Directory containing Natural Instructions task files")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--retriever_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model used for retrieval")
    parser.add_argument("--num_tasks", type=int, default=50, help="Number of tasks to sample")
    parser.add_argument("--train_size", type=int, default=500, help="Training samples per task")
    parser.add_argument("--val_size", type=int, default=10, help="Validation samples per task")
    parser.add_argument("--test_size", type=int, default=50, help="Test samples per task")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_instruction_tokens", type=int, default=1024,
                        help="Maximum token length for instructions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum generated response length")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--retrieval_top_k", type=int, default=3,
                        help="How many retrieved demonstrations to place into the prompt")
    parser.add_argument("--retrieval_batch_size", type=int, default=512,
                        help="Batch size for precomputing SBERT retrievals")
    parser.add_argument("--retriever_device", type=str, default=None,
                        help="Device for SentenceTransformer retrieval; default uses cuda when available, otherwise cpu")
    parser.add_argument("--include_instruction_in_demos", action="store_true",
                        help="Include task instruction when encoding corpus items")
    parser.add_argument("--max_examples_per_task", type=int, default=None,
                        help="Optional cap on how many few-shot examples to keep per task in the retrieval corpus")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use when device_map is not set")
    parser.add_argument("--device_map", type=str, default=None,
                        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
                        help="Optional Hugging Face device_map for sharding the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_cache_path", type=str, default=None,
                        help="Path to a cached train/val/test split")
    parser.add_argument("--corpus_cache_path", type=str, default=None,
                        help="Optional path to save/load encoded SBERT corpus")
    parser.add_argument("--save_verbose_predictions", action="store_true",
                        help="Save instruction, prompt preview, and full decoded sequence in evaluation_predictions.jsonl")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory where logs and results will be written")
    return parser.parse_args()


def load_task_positive_examples(tasks_dir, task_name):
    """Load Positive Examples for one task from its source json."""
    task_path = os.path.join(tasks_dir, f"{task_name}.json")
    if not os.path.exists(task_path):
        return []
    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    return task_data.get("Positive Examples", [])


def collect_task_few_shot_examples(train_data, task_names, tasks_dir, max_examples_per_task=None):
    """Collect one few-shot example pool per task."""
    tasks_few_shot = defaultdict(list)
    task_instructions = {}

    for item in train_data:
        task_name = item["tasks"][0]
        if task_name not in task_instructions:
            task_instructions[task_name] = item.get("instruction", "")
        if not tasks_few_shot[task_name] and item.get("few_shot_examples"):
            tasks_few_shot[task_name] = item["few_shot_examples"]

    for task_name in task_names:
        if not tasks_few_shot[task_name]:
            tasks_few_shot[task_name] = load_task_positive_examples(tasks_dir, task_name)
        if max_examples_per_task is not None:
            tasks_few_shot[task_name] = tasks_few_shot[task_name][:max_examples_per_task]

    return tasks_few_shot, task_instructions


def build_or_load_corpus(
    retriever,
    tasks_few_shot,
    task_instructions,
    include_instruction_in_demos,
    corpus_cache_path,
    retriever_model_name,
):
    """Build the SBERT corpus or load it from cache."""
    if corpus_cache_path and os.path.exists(corpus_cache_path):
        payload = torch.load(corpus_cache_path, map_location="cpu")
        retriever.corpus_embeddings = payload["corpus_embeddings"]
        retriever.corpus_metadata = payload["corpus_metadata"]
        retriever._normalized_corpus_matrix = None
        return payload.get("corpus_size", len(retriever.corpus_metadata)), True

    corpus_size = retriever.build_corpus(
        tasks_few_shot,
        task_instructions if include_instruction_in_demos else None,
    )

    if corpus_cache_path:
        os.makedirs(os.path.dirname(corpus_cache_path), exist_ok=True)
        torch.save(
            {
                "corpus_embeddings": retriever.corpus_embeddings.detach().cpu(),
                "corpus_metadata": retriever.corpus_metadata,
                "corpus_size": corpus_size,
                "include_instruction_in_demos": include_instruction_in_demos,
                "retriever_model": retriever_model_name,
            },
            corpus_cache_path,
        )

    return corpus_size, False


def resolve_retriever_device(requested_device):
    """Resolve the SentenceTransformer device when the CLI leaves it automatic."""
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """Run SBERT RAG evaluation."""
    args = parse_args()
    set_random_seed(args.seed)
    retriever_device = resolve_retriever_device(args.retriever_device)

    run_dir = prepare_run_dir(
        run_dir=args.run_dir,
        experiment_tag="rag_baseline",
        model_name=args.model_name,
        num_tasks=args.num_tasks,
    )
    _, eval_logger, training_log, evaluation_log, timestamp = setup_logging(log_dir=run_dir)

    print()
    print("=" * 60)
    print("SBERT RAG BASELINE EVALUATION")
    print("=" * 60)
    print(f"Generator model: {args.model_name}")
    print(f"Retriever model: {args.retriever_model}")
    print(f"Retriever device: {retriever_device}")
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
        few_shot=True,
    )

    print(f"Tasks directory: {resolved_tasks_dir}")
    print(f"Split source: {split_source}")
    print(f"Dataset summary - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Tasks: {len(task_names)}")
    print()

    print("Collecting few-shot demonstrations...")
    tasks_few_shot, task_instructions = collect_task_few_shot_examples(
        train_data=train_data,
        task_names=task_names,
        tasks_dir=resolved_tasks_dir,
        max_examples_per_task=args.max_examples_per_task,
    )
    total_demo_count = sum(len(examples) for examples in tasks_few_shot.values())
    print(f"Collected {total_demo_count} few-shot examples across {len(task_names)} tasks")
    print()

    print("Loading retriever...")
    retriever = SBERTRetriever(model_name=args.retriever_model, device=retriever_device)
    corpus_size, loaded_from_cache = build_or_load_corpus(
        retriever=retriever,
        tasks_few_shot=tasks_few_shot,
        task_instructions=task_instructions,
        include_instruction_in_demos=args.include_instruction_in_demos,
        corpus_cache_path=args.corpus_cache_path,
        retriever_model_name=args.retriever_model,
    )
    print(
        f"Retriever corpus ready. Size: {corpus_size}. "
        f"Source: {'cache' if loaded_from_cache else 'fresh_encode'}"
    )
    print()

    retrieval_stats = {
        "top1_correct": 0,
        "topk_correct": 0,
        "total": 0,
    }

    print("Precomputing retrievals...")
    retrieval_queries = [
        f"{example.get('instruction', '')} {example.get('query', '')}"
        for example in test_data
    ]
    batched_retrievals = retriever.retrieve_top_k_batch(
        retrieval_queries,
        k=args.retrieval_top_k,
        batch_size=args.retrieval_batch_size,
    )
    for example, retrieved in zip(test_data, batched_retrievals):
        retrieved_examples = [
            {
                "input": item["input"],
                "output": item["output"],
            }
            for item in retrieved
        ]
        retrieved_tasks = [item["task"] for item in retrieved]
        retrieved_scores = [round(float(item["score"]), 6) for item in retrieved]
        true_task = example["tasks"][0] if example.get("tasks") else None

        retrieval_stats["total"] += 1
        if retrieved_tasks and retrieved_tasks[0] == true_task:
            retrieval_stats["top1_correct"] += 1
        if true_task in retrieved_tasks:
            retrieval_stats["topk_correct"] += 1

        example["_rag_retrieval_payload"] = {
            "retrieved_examples": retrieved_examples,
            "retrieved_tasks": retrieved_tasks,
            "retrieved_scores": retrieved_scores,
        }
    print(f"Precomputed retrievals for {len(test_data)} examples")
    print()

    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading base model...")
    model_load_kwargs = {"torch_dtype": "auto"}
    if args.device_map is not None:
        model_load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    print("Base model loaded")
    print()

    def build_batch_payload(example):
        retrieval_payload = example["_rag_retrieval_payload"]
        prompt_text = build_generation_prompt(
            tokenizer=tokenizer,
            instruction=example.get("instruction", ""),
            query=example.get("query", ""),
            few_shot_examples=retrieval_payload["retrieved_examples"],
        )
        return {
            "prompt_text": prompt_text,
            "row_extra": {
                "prompt_style": "retrieved_few_shot",
                "few_shot_count": len(retrieval_payload["retrieved_examples"]),
                "retrieved_tasks": retrieval_payload["retrieved_tasks"],
                "retrieved_scores": retrieval_payload["retrieved_scores"],
            },
        }

    print("Running retrieval-augmented generation evaluation...")
    results, _, prediction_rows = evaluate_generation_batches(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_data,
        batch_builder=build_batch_payload,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.test_batch_size,
        mode="rag_generation",
        logger=eval_logger,
        include_verbose_predictions=args.save_verbose_predictions,
    )

    retrieval_results = {
        "retrieval_top1_accuracy": (
            retrieval_stats["top1_correct"] / retrieval_stats["total"]
            if retrieval_stats["total"] > 0 else 0.0
        ),
        "retrieval_topk_accuracy": (
            retrieval_stats["topk_correct"] / retrieval_stats["total"]
            if retrieval_stats["total"] > 0 else 0.0
        ),
        "retrieval_total_examples": retrieval_stats["total"],
        "corpus_size": corpus_size,
        "corpus_loaded_from_cache": loaded_from_cache,
    }
    results.update(retrieval_results)

    predictions_path = os.path.join(run_dir, "evaluation_predictions.jsonl")
    write_jsonl(predictions_path, prediction_rows)
    results["predictions_output_path"] = predictions_path

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print(f"Exact Match Accuracy: {results['exact_accuracy']:.3f}")
    print(f"Average Response Score: {results['avg_response_score']:.3f}")
    print(f"Retrieval Top-1 Accuracy: {results['retrieval_top1_accuracy']:.3f}")
    print(f"Retrieval Top-{args.retrieval_top_k} Accuracy: {results['retrieval_topk_accuracy']:.3f}")
    print("=" * 50)

    write_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "mode": "rag_baseline",
            "model_name": args.model_name,
            "retriever_model": args.retriever_model,
            "tasks_dir": resolved_tasks_dir,
            "num_tasks": args.num_tasks,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "max_length": args.max_length,
            "max_instruction_tokens": args.max_instruction_tokens,
            "max_new_tokens": args.max_new_tokens,
            "test_batch_size": args.test_batch_size,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_batch_size": args.retrieval_batch_size,
            "retriever_device": retriever_device,
            "requested_retriever_device": args.retriever_device,
            "include_instruction_in_demos": args.include_instruction_in_demos,
            "max_examples_per_task": args.max_examples_per_task,
            "device": args.device,
            "device_map": args.device_map,
            "seed": args.seed,
            "split_cache_path": os.path.abspath(args.split_cache_path) if args.split_cache_path else None,
            "corpus_cache_path": os.path.abspath(args.corpus_cache_path) if args.corpus_cache_path else None,
            "save_verbose_predictions": args.save_verbose_predictions,
            "run_dir": run_dir,
            "timestamp": timestamp,
            "dataset_summary": {
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "task_count": len(task_names),
                "few_shot_example_count": total_demo_count,
            },
        },
    )

    evaluation_results_path = os.path.join(run_dir, "evaluation_results.json")
    write_json(evaluation_results_path, results)
    write_json(
        os.path.join(run_dir, "run_summary.json"),
        {
            "mode": "rag_baseline",
            "run_dir": run_dir,
            "model_name": args.model_name,
            "retriever_model": args.retriever_model,
            "split_source": split_source,
            "metrics": results,
            "dataset_summary": {
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "task_count": len(task_names),
                "few_shot_example_count": total_demo_count,
            },
            "artifacts": {
                "training_log": training_log,
                "evaluation_log": evaluation_log,
                "evaluation_results": evaluation_results_path,
                "evaluation_predictions": predictions_path,
            },
        },
    )

    print("\nSBERT RAG baseline evaluation completed!")


if __name__ == "__main__":
    main()
