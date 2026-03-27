#!/usr/bin/env python3
"""
Filter Natural Instructions tasks to a common pool that satisfies all paper-model
tokenizer requirements for atomic TokMem.

The script:
1. keeps English-only tasks,
2. keeps only instances that are valid for every target tokenizer,
3. keeps only tasks whose common instance pool can satisfy the requested
   train/eval/test sizes,
4. optionally samples a fixed number of eligible tasks with a seed,
5. saves a fixed-split cache with the same payload structure used by
   `main_in_domain_fixed_split.py`,
6. saves extra files for manual long-context inspection.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
ATOMIC_DIR = SCRIPT_DIR.parent
REPO_ROOT = ATOMIC_DIR.parent
DEFAULT_TASKS_DIR = REPO_ROOT / "datasets" / "natural-instructions-2.8" / "tasks"
DEFAULT_OUTPUT_DIR = ATOMIC_DIR / "cached_splits" / "paper_model_common_pool"


def default_tokenizers():
    candidates = [
        (REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
        (REPO_ROOT / "models" / "Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
        (REPO_ROOT / "models" / "Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
        (REPO_ROOT / "models" / "Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ]
    return [str(local if local.exists() else remote) for local, remote in candidates]


def is_english_only(task_data):
    input_lang = task_data.get("Input_language", [])
    output_lang = task_data.get("Output_language", [])
    return (
        (("English" in input_lang) or input_lang == ["English"])
        and (("English" in output_lang) or output_lang == ["English"])
    )


def format_prompt(tokenizer_name, definition, query):
    if "qwen" in tokenizer_name.lower():
        return f"<|im_start|>user\n{definition}\n\n{query}<|im_end|>\n<|im_start|>assistant\n"
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n{definition}\n\n{query}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


def instance_rank(instance, index):
    output_text = instance["output"][0] if isinstance(instance.get("output", ""), list) else instance.get("output", "")
    key = f"{instance.get('id', '')}||{instance.get('input', '')}||{output_text}||{index}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=os.path.isdir(model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    return tokenizer


def build_sample(task_name, definition, instance, instance_index):
    output_text = instance["output"][0] if isinstance(instance.get("output", ""), list) else instance.get("output", "")
    return {
        "instruction": definition,
        "query": instance.get("input", ""),
        "tasks": [task_name],
        "responses": [output_text],
        "few_shot_examples": [],
        "source_instance_index": instance_index,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter atomic tasks to a common valid pool across all paper models")
    parser.add_argument("--tasks_dir", type=str, default=str(DEFAULT_TASKS_DIR))
    parser.add_argument("--tokenizer", action="append", default=None, help="Tokenizer path or model name; repeatable")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--eval_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    tokenizer_sources = args.tokenizer or default_tokenizers()
    tokenizers = [load_tokenizer(name) for name in tokenizer_sources]
    tokenizer_names = [tok.name_or_path for tok in tokenizers]
    compatible_model_names = list(dict.fromkeys(tokenizer_sources + tokenizer_names))
    required_pool = args.train_size + args.eval_size + args.test_size
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    english_task_files = []
    for task_file in sorted(tasks_dir.glob("task*.json")):
        with open(task_file, "r") as f:
            task_data = json.load(f)
        if is_english_only(task_data):
            english_task_files.append(task_file)

    task_rows = []
    for task_file in english_task_files:
        with open(task_file, "r") as f:
            task_data = json.load(f)

        task_name = task_file.stem
        definition = " ".join(task_data.get("Definition", []))
        instances = task_data.get("Instances", [])
        valid_by_model = {name: 0 for name in tokenizer_names}
        common_instances = []

        for index, instance in enumerate(instances):
            query = instance.get("input", "")
            response = instance["output"][0] if isinstance(instance.get("output", ""), list) else instance.get("output", "")
            coarse_text = f"{definition}\n\n{query}"
            coarse_lengths_by_model = {}
            full_lengths_by_model = {}
            is_common_valid = True

            for tokenizer, tokenizer_name in zip(tokenizers, tokenizer_names):
                coarse_len = len(tokenizer.encode(coarse_text, add_special_tokens=False))
                prompt_len = len(tokenizer.encode(format_prompt(tokenizer_name, definition, query), add_special_tokens=False))
                response_len = len(tokenizer.encode(str(response), add_special_tokens=False))
                eos_len = len(tokenizer.encode(tokenizer.eos_token or "", add_special_tokens=False))
                full_len = prompt_len + 1 + response_len + eos_len
                coarse_lengths_by_model[tokenizer_name] = coarse_len
                full_lengths_by_model[tokenizer_name] = full_len

                if full_len <= args.max_length:
                    valid_by_model[tokenizer_name] += 1
                else:
                    is_common_valid = False

            if is_common_valid:
                common_instances.append(
                    {
                        "index": index,
                        "rank": instance_rank(instance, index),
                        "query_preview": query[:200],
                        "response_preview": str(response)[:200],
                        "coarse_lengths_by_model": coarse_lengths_by_model,
                        "full_lengths_by_model": full_lengths_by_model,
                    }
                )

        common_instances.sort(key=lambda row: row["rank"])
        eligible = len(common_instances) >= required_pool
        split = {"train_indices": [], "eval_indices": [], "test_indices": []}
        if eligible:
            test_rows = common_instances[:args.test_size]
            remaining = common_instances[args.test_size:]
            train_rows = remaining[:args.train_size]
            eval_rows = remaining[args.train_size:args.train_size + args.eval_size]
            split = {
                "train_indices": [row["index"] for row in train_rows],
                "eval_indices": [row["index"] for row in eval_rows],
                "test_indices": [row["index"] for row in test_rows],
            }

        task_rows.append(
            {
                "task_name": task_name,
                "task_file": str(task_file),
                "raw_instances": len(instances),
                "common_valid_instances": len(common_instances),
                "valid_instances_by_model": valid_by_model,
                "eligible": eligible,
                "split": split,
                "definition_preview": definition[:300],
                "longest_common_full_length": max(
                    [max(row["full_lengths_by_model"].values()) for row in common_instances],
                    default=0,
                ),
                "common_instances": common_instances,
            }
        )

    eligible_rows = [row for row in task_rows if row["eligible"]]
    eligible_rows.sort(key=lambda row: row["task_name"])
    eligible_before_sampling = len(eligible_rows)

    with open(output_dir / "task_stats.json", "w") as f:
        json.dump(task_rows, f, indent=2)

    if args.num_tasks is not None and eligible_before_sampling < args.num_tasks:
        with open(output_dir / "eligible_tasks.txt", "w") as f:
            for row in eligible_rows:
                f.write(f"{row['task_name']}\n")
        print(
            f"无法满足要求：筛选后只有 {eligible_before_sampling} 个任务满足共同样本池要求，"
            f"少于所需的 {args.num_tasks} 个任务。",
            file=sys.stderr,
        )
        print(f"Saved: {output_dir / 'eligible_tasks.txt'}")
        print(f"Saved: {output_dir / 'task_stats.json'}")
        raise SystemExit(1)

    random.seed(args.seed)
    random.shuffle(eligible_rows)
    if args.num_tasks is not None:
        eligible_rows = eligible_rows[:args.num_tasks]

    selected_task_names = [row["task_name"] for row in eligible_rows]
    row_by_task = {row["task_name"]: row for row in eligible_rows}
    train_data = []
    val_data = []
    test_data = []
    audit_rows = []

    for task_name in selected_task_names:
        row = row_by_task[task_name]
        with open(row["task_file"], "r") as f:
            task_data = json.load(f)
        definition = " ".join(task_data.get("Definition", []))
        instances = task_data.get("Instances", [])
        common_lookup = {item["index"]: item for item in row["common_instances"]}

        for split_name, split_indices, target in [
            ("train", row["split"]["train_indices"], train_data),
            ("eval", row["split"]["eval_indices"], val_data),
            ("test", row["split"]["test_indices"], test_data),
        ]:
            for instance_index in split_indices:
                target.append(build_sample(task_name, definition, instances[instance_index], instance_index))
                audit_row = common_lookup[instance_index]
                audit_rows.append(
                    {
                        "task_name": task_name,
                        "task_file": row["task_file"],
                        "split": split_name,
                        "source_instance_index": instance_index,
                        "rank": audit_row["rank"],
                        "max_full_length": max(audit_row["full_lengths_by_model"].values()),
                        "coarse_lengths_by_model": audit_row["coarse_lengths_by_model"],
                        "full_lengths_by_model": audit_row["full_lengths_by_model"],
                        "query_preview": audit_row["query_preview"],
                        "response_preview": audit_row["response_preview"],
                    }
                )

    payload = {
        "metadata": {
            "tasks_dir": str(tasks_dir.resolve()),
            "model_name": "__COMMON_PAPER_MODELS__",
            "compatible_model_names": compatible_model_names,
            "num_tasks": len(selected_task_names) if args.num_tasks is None else args.num_tasks,
            "train_size": args.train_size,
            "val_size": args.eval_size,
            "test_size": args.test_size,
            "few_shot": False,
            "seed": args.seed,
            "max_length": args.max_length,
            "tokenizers": tokenizer_names,
        },
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "task_names": selected_task_names,
    }

    manifest = {
        "metadata": {
            "tasks_dir": str(tasks_dir),
            "tokenizers": tokenizer_names,
            "compatible_model_names": compatible_model_names,
            "train_size": args.train_size,
            "val_size": args.eval_size,
            "test_size": args.test_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "num_tasks": args.num_tasks,
            "english_tasks": len(english_task_files),
            "eligible_tasks_before_sampling": eligible_before_sampling,
            "selected_tasks": len(eligible_rows),
        },
        "selected_task_names": selected_task_names,
        "selected_tasks": eligible_rows,
        "all_task_stats": task_rows,
    }

    with open(output_dir / "eligible_tasks.txt", "w") as f:
        for task_name in selected_task_names:
            f.write(f"{task_name}\n")

    with open(output_dir / "common_pool_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    audit_rows.sort(key=lambda row: row["max_full_length"], reverse=True)
    with open(output_dir / "long_context_samples.jsonl", "w") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(payload, output_dir / "tokmem_atomic_fixed_split_common_all_models.pt")

    print(f"Tokenizers: {tokenizer_names}")
    print(f"English tasks: {len(english_task_files)}")
    print(f"Eligible tasks before sampling: {eligible_before_sampling}")
    print(f"Selected tasks: {len(eligible_rows)}")
    print(f"Required common pool per task: {required_pool}")
    print(f"Saved: {output_dir / 'eligible_tasks.txt'}")
    print(f"Saved: {output_dir / 'task_stats.json'}")
    print(f"Saved: {output_dir / 'common_pool_manifest.json'}")
    print(f"Saved: {output_dir / 'long_context_samples.jsonl'}")
    print(f"Saved: {output_dir / 'tokmem_atomic_fixed_split_common_all_models.pt'}")


if __name__ == "__main__":
    main()
