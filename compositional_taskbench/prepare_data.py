#!/usr/bin/env python3
import argparse
from pathlib import Path

from taskbench_data import load_matching_taskbench_split, prepare_taskbench_splits


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare TaskBench DailyLife data for TokMem/TapMem runs")
    parser.add_argument(
        "--source_path",
        type=str,
        default="datasets/taskbench/data_dailylifeapis/data.json",
        help="Path to TaskBench DailyLife data.json JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="compositional_taskbench/data",
        help="Directory where converted train/test JSON files are written",
    )
    parser.add_argument(
        "--sample_types",
        type=str,
        default="node_chain",
        help="Sample type selector: chain, node, node_chain, or comma-separated values",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    summary = load_matching_taskbench_split(
        source_path=Path(args.source_path),
        output_dir=Path(args.output_dir),
        sample_types=args.sample_types,
        train_ratio=args.train_ratio,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    if summary is None:
        summary = prepare_taskbench_splits(
            source_path=Path(args.source_path),
            output_dir=Path(args.output_dir),
            sample_types=args.sample_types,
            train_ratio=args.train_ratio,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
        )

    if summary.get("reused_existing_split"):
        print("Reused existing TaskBench DailyLife splits:")
    else:
        print("Prepared TaskBench DailyLife splits:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
