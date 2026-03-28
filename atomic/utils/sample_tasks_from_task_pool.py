#!/usr/bin/env python3
"""
Sample a fixed-size task split from a previously built atomic task pool.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from filter_tasks_for_all_models import DEFAULT_OUTPUT_ROOT, build_sample, normalize_model_label


def build_default_output_dir(model_name, num_tasks, train_size, val_size, test_size, seed):
    """Build a concise sampled-split directory under atomic/cached_splits."""
    dir_name = (
        f"{normalize_model_label(model_name)}-"
        f"task{num_tasks}-{train_size}-{val_size}-{test_size}-seed{seed}"
    )
    return DEFAULT_OUTPUT_ROOT / dir_name


def load_pool_manifest(pool_dir):
    """Load the task-pool manifest from the given directory."""
    manifest_path = Path(pool_dir) / "task_pool_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Task pool manifest not found: {manifest_path}\n"
            "Build the pool first with a build_atomic_*_task_pool.sh script."
        )
    with open(manifest_path, "r") as f:
        return json.load(f), manifest_path


def validate_pool(manifest, args, manifest_path):
    """Ensure the task pool matches the requested split configuration."""
    metadata = manifest.get("metadata", {})
    mismatches = []
    for key, expected_value in [
        ("train_size", args.train_size),
        ("val_size", args.val_size),
        ("test_size", args.test_size),
        ("max_length", args.max_length),
    ]:
        cached_value = metadata.get(key)
        if cached_value != expected_value:
            mismatches.append((key, expected_value, cached_value))

    if mismatches:
        lines = [f"Task pool metadata in {manifest_path} does not match current arguments:"]
        for key, expected_value, cached_value in mismatches:
            lines.append(f"  {key}: expected={expected_value!r}, cached={cached_value!r}")
        raise ValueError("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Sample an atomic fixed split from a task pool")
    parser.add_argument("--pool_dir", type=str, required=True, help="Directory containing task_pool_manifest.json")
    parser.add_argument("--model_name", type=str, required=True, help="Target model name/path for naming and cache metadata")
    parser.add_argument("--num_tasks", type=int, required=True, help="Number of tasks to sample from the pool")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to atomic/cached_splits/<model>-taskN-train-val-test-seedS",
    )
    args = parser.parse_args()

    manifest, manifest_path = load_pool_manifest(args.pool_dir)
    validate_pool(manifest, args, manifest_path)

    eligible_rows = list(manifest.get("eligible_tasks", []))
    eligible_count = len(eligible_rows)
    if eligible_count < args.num_tasks:
        print(
            f"Cannot sample {args.num_tasks} tasks from pool {manifest_path}: only {eligible_count} eligible tasks available.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else build_default_output_dir(
            args.model_name,
            args.num_tasks,
            args.train_size,
            args.val_size,
            args.test_size,
            args.seed,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    random.shuffle(eligible_rows)
    selected_rows = eligible_rows[:args.num_tasks]
    selected_task_names = [row["task_name"] for row in selected_rows]

    train_data = []
    val_data = []
    test_data = []

    for row in selected_rows:
        with open(row["task_file"], "r") as f:
            task_data = json.load(f)
        definition = " ".join(task_data.get("Definition", []))
        instances = task_data.get("Instances", [])

        for split_indices, target in [
            (row["split"]["train_indices"], train_data),
            (row["split"]["eval_indices"], val_data),
            (row["split"]["test_indices"], test_data),
        ]:
            for instance_index in split_indices:
                target.append(build_sample(row["task_name"], definition, instances[instance_index], instance_index))

    compatible_model_names = manifest.get("metadata", {}).get("compatible_model_names", [])
    tokenizers = manifest.get("metadata", {}).get("tokenizers", [])
    payload = {
        "metadata": {
            "tasks_dir": manifest.get("metadata", {}).get("tasks_dir"),
            "model_name": args.model_name,
            "compatible_model_names": compatible_model_names,
            "num_tasks": args.num_tasks,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "few_shot": False,
            "seed": args.seed,
            "max_length": args.max_length,
            "tokenizers": tokenizers,
            "source_pool": str(Path(args.pool_dir).resolve() / "task_pool_manifest.json"),
        },
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "task_names": selected_task_names,
    }

    sample_manifest = {
        "source_pool": str(Path(args.pool_dir).resolve() / "task_pool_manifest.json"),
        "eligible_task_count": eligible_count,
        "selected_task_count": len(selected_rows),
        "seed": args.seed,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "selected_tasks": selected_task_names,
    }

    with open(output_dir / "selected_tasks.txt", "w") as f:
        for task_name in selected_task_names:
            f.write(f"{task_name}\n")

    with open(output_dir / "sample_manifest.json", "w") as f:
        json.dump(sample_manifest, f, indent=2)

    split_cache_path = output_dir / f"tokmem_atomic_fixed_split_maxlen{args.max_length}.pt"
    torch.save(payload, split_cache_path)

    print(f"Loaded pool: {manifest_path}")
    print(f"Eligible tasks in pool: {eligible_count}")
    print(f"Selected tasks: {len(selected_rows)}")
    print(f"Saved: {output_dir / 'selected_tasks.txt'}")
    print(f"Saved: {output_dir / 'sample_manifest.json'}")
    print(f"Saved: {split_cache_path}")


if __name__ == "__main__":
    main()
