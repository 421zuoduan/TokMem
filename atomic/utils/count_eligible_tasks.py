#!/usr/bin/env python3
"""
Count how many Natural Instructions tasks are eligible under the current
atomic TokMem filtering rules.

This script mirrors the key filtering logic in `task_dataset.py`:
1. Only keep task JSON files whose input/output languages are English.
2. For each instance, keep it only if
   `definition + "\\n\\n" + input` token length <= max_instruction_tokens.
3. Report how many tasks remain under different downstream thresholds.

Examples:
    python atomic/utils/count_eligible_tasks.py
    python atomic/utils/count_eligible_tasks.py --max_instruction_tokens 512
    python atomic/utils/count_eligible_tasks.py --show-samples 5
"""

import argparse
import json
import os
from dataclasses import dataclass

from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ATOMIC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ATOMIC_DIR)
DEFAULT_TASKS_DIR = os.path.join(REPO_ROOT, "datasets", "natural-instructions-2.8", "tasks")
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "Llama-3.2-3B-Instruct")


@dataclass
class TaskStats:
    task_name: str
    raw_instances: int
    valid_instances: int


def is_english_only_task(task_data: dict) -> bool:
    """Match the English-only filter used in task_dataset.py."""
    input_lang = task_data.get("Input_language", [])
    output_lang = task_data.get("Output_language", [])
    return (
        (("English" in input_lang) or input_lang == ["English"])
        and (("English" in output_lang) or output_lang == ["English"])
    )


def count_valid_instances(task_data: dict, tokenizer, max_instruction_tokens: int) -> int:
    """Count instances whose instruction length passes the project filter."""
    definition = " ".join(task_data.get("Definition", []))
    valid_instances = 0

    for instance in task_data.get("Instances", []):
        instruction = f"{definition}\n\n{instance['input']}"
        token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
        if token_count <= max_instruction_tokens:
            valid_instances += 1

    return valid_instances


def summarize_tasks(stats: list[TaskStats], train_size: int, val_size: int, test_size: int) -> dict:
    """
    Summarize task counts under several thresholds.

    A task can contribute to training under the current splitting logic if:
    - valid_instances > test_size
      because test examples are selected first, and training uses the remainder.

    A task can fully satisfy the configured split if:
    - valid_instances >= train_size + val_size + test_size
    """
    at_least_one_valid = [s for s in stats if s.valid_instances >= 1]
    trainable_tasks = [s for s in stats if s.valid_instances > test_size]
    with_any_val = [s for s in stats if s.valid_instances >= test_size + train_size + 1]
    full_split_tasks = [s for s in stats if s.valid_instances >= test_size + train_size + val_size]

    return {
        "english_only_tasks": len(stats),
        "tasks_with_at_least_one_valid_instance": len(at_least_one_valid),
        "tasks_trainable_under_current_split": len(trainable_tasks),
        "tasks_with_at_least_one_val_sample_after_current_split": len(with_any_val),
        "tasks_that_fully_satisfy_current_train_val_test_sizes": len(full_split_tasks),
        "at_least_one_valid": at_least_one_valid,
        "trainable_tasks": trainable_tasks,
        "with_any_val": with_any_val,
        "full_split_tasks": full_split_tasks,
    }


def main():
    parser = argparse.ArgumentParser(description="Count eligible Natural Instructions tasks for atomic TokMem")
    parser.add_argument("--tasks_dir", type=str, default=DEFAULT_TASKS_DIR, help="Directory containing task JSON files")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_PATH, help="Tokenizer path or model name")
    parser.add_argument("--max_instruction_tokens", type=int, default=1024, help="Instruction length filter")
    parser.add_argument("--train_size", type=int, default=500, help="Per-task train size")
    parser.add_argument("--val_size", type=int, default=10, help="Per-task val size")
    parser.add_argument("--test_size", type=int, default=50, help="Per-task test size")
    parser.add_argument("--show_samples", type=int, default=0, help="Show up to N sample task names per bucket")
    args = parser.parse_args()

    if not os.path.exists(args.tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {args.tasks_dir}")

    print(f"Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=os.path.isdir(args.model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    all_task_files = sorted(
        f for f in os.listdir(args.tasks_dir) if f.startswith("task") and f.endswith(".json")
    )

    english_stats: list[TaskStats] = []

    for task_file in all_task_files:
        task_path = os.path.join(args.tasks_dir, task_file)
        with open(task_path, "r") as f:
            task_data = json.load(f)

        if not is_english_only_task(task_data):
            continue

        raw_instances = len(task_data.get("Instances", []))
        valid_instances = count_valid_instances(task_data, tokenizer, args.max_instruction_tokens)
        english_stats.append(
            TaskStats(
                task_name=task_file.replace(".json", ""),
                raw_instances=raw_instances,
                valid_instances=valid_instances,
            )
        )

    summary = summarize_tasks(
        english_stats,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    print()
    print("=== Eligibility Summary ===")
    print(f"Total task files: {len(all_task_files)}")
    print(f"English-only tasks: {summary['english_only_tasks']}")
    print(f"Tasks with >= 1 valid instance: {summary['tasks_with_at_least_one_valid_instance']}")
    print(f"Tasks trainable under current split (> test_size={args.test_size} valid instances): {summary['tasks_trainable_under_current_split']}")
    print(
        "Tasks with >= 1 validation sample after current split "
        f"(>= test_size + train_size + 1 = {args.test_size + args.train_size + 1}): "
        f"{summary['tasks_with_at_least_one_val_sample_after_current_split']}"
    )
    print(
        "Tasks that fully satisfy current train/val/test sizes "
        f"(>= {args.train_size + args.val_size + args.test_size} valid instances): "
        f"{summary['tasks_that_fully_satisfy_current_train_val_test_sizes']}"
    )

    if args.show_samples > 0:
        def show_bucket(name: str, items: list[TaskStats]):
            print()
            print(f"[{name}] showing up to {args.show_samples} tasks")
            for task in items[:args.show_samples]:
                print(
                    f"  {task.task_name}: raw_instances={task.raw_instances}, "
                    f"valid_instances={task.valid_instances}"
                )

        show_bucket("trainable_tasks", summary["trainable_tasks"])
        show_bucket("full_split_tasks", summary["full_split_tasks"])


if __name__ == "__main__":
    main()
