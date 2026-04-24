#!/usr/bin/env python3
"""Raw base-model generation entrypoint for atomic Natural Instructions runs."""

import argparse
import os

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
    parser = argparse.ArgumentParser(description="Atomic raw base generation")
    parser.add_argument("--tasks_dir", type=str, default="natural-instructions-2.8/tasks")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_instruction_tokens", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    artifacts = setup_generation_run(args.output_dir)
    include_instruction = args.test_prompt_mode == "instruction_and_query"

    print("=" * 60)
    print("ATOMIC RAW BASE GENERATION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Device map: {args.device_map}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Test prompt mode: {args.test_prompt_mode}")
    print(f"Output directory: {os.path.abspath(artifacts.output_dir)}")
    print(f"Training log: {artifacts.training_log}")
    print(f"Evaluation log: {artifacts.evaluation_log}")
    print()

    tokenizer = load_tokenizer(args.model_name)
    train_data, val_data, test_data, task_names, split_source = load_atomic_split(args, tokenizer)
    model = load_model(args.model_name, args.device, args.device_map, args.torch_dtype)

    print("Creating evaluation dataloader...")
    test_dataloader = create_prompt_dataloader(
        examples=test_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.test_batch_size,
        include_instruction=include_instruction,
        include_demo_instruction=True,
    )
    print(f"Test dataset created: {len(test_dataloader.dataset)} samples")
    print()

    predictions_path = os.path.join(artifacts.output_dir, "evaluation_predictions.jsonl")
    results = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_dataloader,
        mode="base_generation",
        prompt_mode=args.test_prompt_mode,
        predictions_output_path=predictions_path,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY:")
    print("   Task Prediction Accuracy: N/A (raw base generation)")
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
    print("\nAtomic raw base generation completed!")


if __name__ == "__main__":
    main()
