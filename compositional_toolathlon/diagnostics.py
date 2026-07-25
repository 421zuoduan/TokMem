from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compositional.eval import (
    calculate_tool_metrics,
    compare_function_calls_advanced,
)
from .manifest import load_manifest


def _normalize_call(call: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(call, dict):
        raise ValueError("each function call must be an object")
    if "tool_id" in call:
        tool_id = call.get("tool_id")
        arguments = call.get("arguments")
    elif len(call) == 1:
        tool_id, arguments = next(iter(call.items()))
    else:
        raise ValueError("call requires tool_id/arguments or a one-key call object")
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("call tool_id must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ValueError("call arguments must be an object")
    return tool_id, arguments


def _legacy_call(call: Any) -> dict[str, dict[str, Any]]:
    tool_id, arguments = _normalize_call(call)
    return {tool_id: arguments}


def score_call_record(record: dict[str, Any]) -> dict[str, Any]:
    predicted_raw = record.get("predicted_calls")
    target_raw = record.get("target_calls")
    if not isinstance(predicted_raw, list) or not isinstance(target_raw, list):
        raise ValueError("record requires predicted_calls and target_calls lists")
    predicted = [_legacy_call(call) for call in predicted_raw]
    target = [_legacy_call(call) for call in target_raw]
    predicted_tools = [next(iter(call)) for call in predicted]
    target_tools = [next(iter(call)) for call in target]
    available_tools = record.get("available_tool_ids", [])
    if not isinstance(available_tools, list) or any(
        not isinstance(tool_id, str) for tool_id in available_tools
    ):
        raise ValueError("available_tool_ids must be a list of strings")

    tool_metrics = calculate_tool_metrics(
        predicted_tools=predicted_tools,
        expected_tools=target_tools,
        candidate_tools=available_tools,
    )
    argument_metrics = compare_function_calls_advanced(
        predicted,
        target,
        ignore_order=True,
    )
    return {
        "sample_id": record.get("sample_id"),
        "next_tool_accuracy": float(
            bool(predicted_tools)
            and bool(target_tools)
            and predicted_tools[0] == target_tools[0]
        ),
        "tool_precision": tool_metrics["tool_precision"],
        "tool_recall": tool_metrics["tool_recall"],
        "tool_f1": tool_metrics["tool_f1_score"],
        "tool_exact_match": tool_metrics["tool_exact_match_acc"],
        "arguments_precision": argument_metrics.precision,
        "arguments_recall": argument_metrics.recall,
        "arguments_f1": argument_metrics.f1_score,
        "function_call_exact_match": float(argument_metrics.exact_match),
    }


def aggregate_call_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot score an empty diagnostics set")
    per_sample = [score_call_record(record) for record in records]
    metric_names = [
        key
        for key in per_sample[0]
        if key != "sample_id" and isinstance(per_sample[0][key], (int, float))
    ]
    macro = {
        metric_name: sum(float(sample[metric_name]) for sample in per_sample)
        / len(per_sample)
        for metric_name in metric_names
    }
    return {
        "sample_count": len(per_sample),
        "aggregation": "macro_over_samples",
        "arguments_f1_definition": (
            "normalized complete function-call set F1; not argument-field F1"
        ),
        "macro": macro,
        "per_sample": per_sample,
    }


def _filter_record_tools(
    record: dict[str, Any],
    included_tool_ids: set[str],
) -> dict[str, Any]:
    filtered = dict(record)
    filtered["predicted_calls"] = [
        call
        for call in record["predicted_calls"]
        if _normalize_call(call)[0] in included_tool_ids
    ]
    filtered["target_calls"] = [
        call
        for call in record["target_calls"]
        if _normalize_call(call)[0] in included_tool_ids
    ]
    filtered["available_tool_ids"] = [
        tool_id
        for tool_id in record.get("available_tool_ids", [])
        if tool_id in included_tool_ids
    ]
    return filtered


def aggregate_scoped_call_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    task_action_ids = {
        record["stable_id"]
        for record in manifest["tools"]
        if record["dispatch_kind"] not in {"terminal", "runner_state"}
    }
    return {
        "metric_policy": {
            "primary_scope": "environment_and_effectful_task_tools",
            "terminal_and_runner_state_excluded_from_primary": True,
            "all_model_selected_actions_reported_separately": True,
        },
        "task_actions": aggregate_call_records(
            [_filter_record_tools(record, task_action_ids) for record in records]
        ),
        "all_model_selected_actions": aggregate_call_records(records),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score synthetic Toolathlon function-call diagnostics"
    )
    parser.add_argument("--input", required=True, help="JSONL prediction records")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = aggregate_scoped_call_records(
        read_jsonl(args.input),
        load_manifest(args.manifest),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["task_actions"]["macro"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
