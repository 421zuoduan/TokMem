#!/usr/bin/env python3
"""SBERT RAG generation entrypoint for atomic Natural Instructions runs."""

import argparse
import os

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from generation_utils import (
    create_prompt_dataloader,
    evaluate_generation,
    load_atomic_split,
    load_model,
    load_tokenizer,
    set_random_seed,
    setup_generation_run,
    write_run_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Atomic SBERT-based RAG generation")
    parser.add_argument("--tasks_dir", type=str, default="natural-instructions-2.8/tasks")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--retriever_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_instruction_tokens", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument(
        "--retrieval_text_mode",
        type=str,
        default="instruction_and_query",
        choices=["instruction_and_query", "query_only"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--retriever_device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_cache_path", type=str, default=None)
    parser.add_argument("--few_shot", action="store_true", help="Expected few-shot flag for split metadata")
    parser.add_argument(
        "--test_prompt_mode",
        type=str,
        default="instruction_and_query",
        choices=["instruction_and_query", "query_only"],
    )
    parser.add_argument(
        "--no_retrieved_instruction_in_demos",
        action="store_true",
        help="Use retrieved query text only inside demonstrations",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def retrieval_text(example, mode):
    if mode == "query_only":
        return str(example.get("query", ""))
    return f"{example.get('instruction', '')}\n\n{example.get('query', '')}"


def attach_retrieved_examples(train_data, test_data, args):
    if not train_data:
        raise ValueError("RAG generation requires non-empty train_data as the retrieval corpus")

    retriever_device = args.retriever_device
    if retriever_device is None:
        retriever_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Sentence-BERT retriever: {args.retriever_model_name}")
    print(f"Retriever device: {retriever_device}")
    retriever = SentenceTransformer(args.retriever_model_name, device=retriever_device)

    corpus_texts = [retrieval_text(item, args.retrieval_text_mode) for item in train_data]
    print(f"Encoding retrieval corpus: {len(corpus_texts)} examples")
    corpus_embeddings = retriever.encode(
        corpus_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    enriched_test_data = []
    correct_top1 = 0
    for item in tqdm(test_data, desc="Retrieving demos"):
        query_text = retrieval_text(item, args.retrieval_text_mode)
        query_embedding = retriever.encode(
            query_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k = min(args.retrieval_k, len(train_data))
        top_scores, top_indices = torch.topk(scores, k=top_k)

        retrieved_examples = []
        retrieval_metadata = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            retrieved = train_data[int(index)]
            retrieved_examples.append(retrieved)
            retrieval_metadata.append(
                {
                    "score": float(score),
                    "task": retrieved.get("tasks", ["unknown"])[0],
                    "query": retrieved.get("query", ""),
                    "response": retrieved.get("responses", [""])[0],
                }
            )

        if retrieval_metadata and retrieval_metadata[0]["task"] == item.get("tasks", ["unknown"])[0]:
            correct_top1 += 1

        enriched = dict(item)
        enriched["retrieved_examples"] = retrieved_examples
        enriched["retrieval_metadata"] = retrieval_metadata
        enriched_test_data.append(enriched)

    retrieval_top1_acc = correct_top1 / len(test_data) if test_data else 0.0
    print(f"Retriever top-1 task accuracy: {retrieval_top1_acc:.3f} ({correct_top1}/{len(test_data)})")
    print()
    return enriched_test_data, retrieval_top1_acc


def main():
    args = parse_args()
    set_random_seed(args.seed)

    artifacts = setup_generation_run(args.output_dir)
    include_instruction = args.test_prompt_mode == "instruction_and_query"
    include_demo_instruction = not args.no_retrieved_instruction_in_demos

    print("=" * 60)
    print("ATOMIC SBERT RAG GENERATION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Retriever: {args.retriever_model_name}")
    print(f"Retrieval k: {args.retrieval_k}")
    print(f"Retrieval text mode: {args.retrieval_text_mode}")
    print(f"Test prompt mode: {args.test_prompt_mode}")
    print(f"Output directory: {os.path.abspath(artifacts.output_dir)}")
    print(f"Training log: {artifacts.training_log}")
    print(f"Evaluation log: {artifacts.evaluation_log}")
    print()

    tokenizer = load_tokenizer(args.model_name)
    train_data, val_data, test_data, task_names, split_source = load_atomic_split(args, tokenizer)
    test_data, retrieval_top1_acc = attach_retrieved_examples(train_data, test_data, args)
    model = load_model(args.model_name, args.device, args.device_map, args.torch_dtype)

    print("Creating RAG evaluation dataloader...")
    test_dataloader = create_prompt_dataloader(
        examples=test_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.test_batch_size,
        include_instruction=include_instruction,
        include_demo_instruction=include_demo_instruction,
    )
    print(f"Test dataset created: {len(test_dataloader.dataset)} samples")
    print()

    predictions_path = os.path.join(artifacts.output_dir, "evaluation_predictions.jsonl")
    results = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_dataloader,
        mode="sbert_rag_generation",
        prompt_mode=args.test_prompt_mode,
        predictions_output_path=predictions_path,
        max_new_tokens=args.max_new_tokens,
    )
    results["retrieval_k"] = args.retrieval_k
    results["retrieval_text_mode"] = args.retrieval_text_mode
    results["retriever_top1_task_accuracy"] = retrieval_top1_acc

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY:")
    print(f"   Retriever Top-1 Task Accuracy: {retrieval_top1_acc:.3f}")
    print("   Task Prediction Accuracy: N/A (RAG generation)")
    print(f"   Exact Match Accuracy: {results['exact_accuracy']:.3f}")
    print(f"   Average Response Score: {results['avg_response_score']:.3f}")
    print("=" * 50)

    write_run_outputs(
        artifacts=artifacts,
        args=args,
        results=results,
        split_source=split_source,
        task_names=task_names,
        train_count=len(train_data),
        val_count=len(val_data),
        test_count=len(test_data),
    )
    print("\nAtomic SBERT RAG generation completed!")


if __name__ == "__main__":
    main()
