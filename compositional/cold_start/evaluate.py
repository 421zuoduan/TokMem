#!/usr/bin/env python3
"""Evaluate a base checkpoint plus a cold-start delta on mixed tool calls."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from backbone_prompting import format_user_assistant_prompt  # noqa: E402
from cold_start.common import (  # noqa: E402
    load_checkpoint_model,
    load_json,
    torch_dtype,
)
from cold_start.runtime import apply_cold_start_delta  # noqa: E402
from eval import calculate_tool_metrics, compare_function_calls_advanced  # noqa: E402


DEFAULT_DATA = (
    COMPOSITIONAL_DIR
    / "data"
    / "test"
    / "function_calling_test_tools51-100_plus_cold20_4calls.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run mixed old/new tool evaluation with a cold-start delta."
    )
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--output", required=True, help="Per-sample JSONL output")
    parser.add_argument("--summary", default=None, help="Summary JSON output")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _generate_one(model, tokenizer, item, device, max_new_tokens):
    prompt = format_user_assistant_prompt(
        tokenizer,
        item["user_input"],
        model=model,
    )
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        return model.generate_with_tool_prediction(
            encoded["input_ids"],
            encoded["attention_mask"],
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=False,
        )[0]


def _prediction_record(index, item, result, candidate_tools, new_tool_set):
    expected_tools = item.get("tools", [item.get("tool_name", "unknown")])
    expected_calls = item.get(
        "function_calls",
        [item.get("function_call", "{}")],
    )
    predicted_tools = [
        tool["tool_name"] for tool in result.get("predicted_tools", [])
    ]
    predicted_calls = result.get("function_calls", [])
    call_evaluation = compare_function_calls_advanced(
        predicted_calls,
        expected_calls,
        ignore_order=True,
    )
    tool_metrics = calculate_tool_metrics(
        predicted_tools=predicted_tools,
        expected_tools=expected_tools,
        candidate_tools=candidate_tools,
    )
    expected_new_calls = [
        call
        for tool, call in zip(expected_tools, expected_calls)
        if tool in new_tool_set
    ]
    predicted_new_calls = [
        call
        for tool, call in zip(predicted_tools, predicted_calls)
        if tool in new_tool_set
    ]
    expected_old_calls = [
        call
        for tool, call in zip(expected_tools, expected_calls)
        if tool not in new_tool_set
    ]
    predicted_old_calls = [
        call
        for tool, call in zip(predicted_tools, predicted_calls)
        if tool not in new_tool_set
    ]
    new_call_evaluation = compare_function_calls_advanced(
        predicted_new_calls,
        expected_new_calls,
        ignore_order=True,
    )
    old_call_evaluation = compare_function_calls_advanced(
        predicted_old_calls,
        expected_old_calls,
        ignore_order=True,
    )
    new_count = sum(tool in new_tool_set for tool in expected_tools)
    old_count = len(expected_tools) - new_count
    return {
        "index": index,
        "user_input": item["user_input"],
        "expected_tools": expected_tools,
        "expected_calls": expected_calls,
        "predicted_tools": predicted_tools,
        "predicted_calls": predicted_calls,
        "full_generated_sequence": result.get("full_generated_sequence", ""),
        "composition": f"{new_count}:{old_count}",
        "tool_sequence_exact": predicted_tools == expected_tools,
        "tool_multiset_exact": bool(
            tool_metrics["tool_exact_match_acc"] >= 1.0
        ),
        "call_exact": bool(call_evaluation.exact_match),
        "tool_f1": float(tool_metrics["tool_f1_score"]),
        "call_f1": float(call_evaluation.f1_score),
        "new_call_f1": float(new_call_evaluation.f1_score),
        "old_call_f1": float(old_call_evaluation.f1_score),
        "parse_errors": call_evaluation.details.get("parse_errors", {}),
    }


def _update_partition_counts(counts, expected, predicted, membership):
    expected_counter = Counter(tool for tool in expected if membership(tool))
    predicted_counter = Counter(tool for tool in predicted if membership(tool))
    counts["true_positive"] += sum(
        min(expected_counter[tool], predicted_counter[tool])
        for tool in expected_counter.keys() | predicted_counter.keys()
    )
    counts["expected"] += sum(expected_counter.values())
    counts["predicted"] += sum(predicted_counter.values())


def _partition_metrics(counts):
    precision = (
        counts["true_positive"] / counts["predicted"]
        if counts["predicted"]
        else 0.0
    )
    recall = (
        counts["true_positive"] / counts["expected"]
        if counts["expected"]
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        **dict(counts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean(records, field):
    return sum(record[field] for record in records) / max(1, len(records))


def main():
    args = parse_args()
    dtype = torch_dtype(args.dtype)
    run_config, tokenizer, model = load_checkpoint_model(
        args.run_config,
        args.checkpoint,
        args.device,
        dtype,
    )
    delta = torch.load(
        args.delta,
        map_location="cpu",
        weights_only=False,
    )
    apply_cold_start_delta(model, delta)

    data = load_json(args.data_path)
    if args.limit is not None:
        data = data[: args.limit]
    max_new_tokens = (
        int(run_config["args"].get("max_new_tokens", 256))
        if args.max_new_tokens is None
        else int(args.max_new_tokens)
    )
    new_tool_set = set(delta["new_tool_names"])
    old_tool_set = set(delta["base_tool_names"])
    candidate_tools = list(model.tool_names)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    partition_counts = {
        "new": defaultdict(int),
        "old": defaultdict(int),
    }
    composition_counts = Counter()
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, item in enumerate(data):
            result = _generate_one(
                model,
                tokenizer,
                item,
                args.device,
                max_new_tokens,
            )
            record = _prediction_record(
                index,
                item,
                result,
                candidate_tools,
                new_tool_set,
            )
            records.append(record)
            composition_counts[record["composition"]] += 1
            _update_partition_counts(
                partition_counts["new"],
                record["expected_tools"],
                record["predicted_tools"],
                lambda tool: tool in new_tool_set,
            )
            _update_partition_counts(
                partition_counts["old"],
                record["expected_tools"],
                record["predicted_tools"],
                lambda tool: tool in old_tool_set,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (
                args.progress_every > 0
                and (index + 1) % args.progress_every == 0
            ):
                print(f"Generated {index + 1}/{len(data)} predictions")

    avg_tool_f1_score = _mean(records, "tool_f1")
    avg_f1_score = _mean(records, "call_f1")
    summary = {
        "examples": len(records),
        "base_tools": len(old_tool_set),
        "new_tools": len(new_tool_set),
        "avg_tool_f1_score": avg_tool_f1_score,
        "avg_f1_score": avg_f1_score,
        "mean_tool_f1": avg_tool_f1_score,
        "mean_call_f1": avg_f1_score,
        "mean_new_call_f1": _mean(records, "new_call_f1"),
        "mean_old_call_f1": _mean(records, "old_call_f1"),
        "tool_sequence_exact": _mean(records, "tool_sequence_exact"),
        "tool_multiset_exact": _mean(records, "tool_multiset_exact"),
        "call_exact": _mean(records, "call_exact"),
        "new_tool_metrics": _partition_metrics(partition_counts["new"]),
        "old_tool_metrics": _partition_metrics(partition_counts["old"]),
        "composition_counts": dict(sorted(composition_counts.items())),
        "predictions": str(output_path.resolve()),
    }
    summary_path = (
        Path(args.summary)
        if args.summary is not None
        else output_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote predictions to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
