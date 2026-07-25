from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    value = payload
    for encoded in pointer[1:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError(f"JSON pointer traversed a scalar at {key!r}")
    return value


def attach_official_evaluator(
    agent_result: dict[str, Any],
    evaluator_result: dict[str, Any],
    *,
    pass_pointer: str,
) -> dict[str, Any]:
    passed = json_pointer(evaluator_result, pass_pointer)
    if not isinstance(passed, bool):
        raise ValueError("official evaluator pass field must be boolean")
    merged = dict(agent_result)
    merged["official_evaluator"] = {
        "passed": passed,
        "raw": evaluator_result,
        "pass_pointer": pass_pointer,
        "original_evaluator_unmodified": True,
    }
    merged["official_evaluator_pending"] = False
    return merged


def summarize_official_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("official result set is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for field in ("method", "task_id", "seed", "trial"):
            if field not in record:
                raise ValueError(f"official record is missing {field}")
        evaluator = record.get("official_evaluator")
        if not isinstance(evaluator, dict) or not isinstance(
            evaluator.get("passed"),
            bool,
        ):
            raise ValueError("official record has no attached boolean evaluator result")
        grouped[str(record["method"])].append(record)

    methods = {}
    for method, method_records in sorted(grouped.items()):
        task_seed_runs: dict[tuple[str, int], list[bool]] = defaultdict(list)
        per_task_runs: dict[str, list[bool]] = defaultdict(list)
        for record in method_records:
            passed = bool(record["official_evaluator"]["passed"])
            task_id = str(record["task_id"])
            seed = int(record["seed"])
            task_seed_runs[(task_id, seed)].append(passed)
            per_task_runs[task_id].append(passed)
        trial_counts = sorted({len(values) for values in task_seed_runs.values()})
        if len(trial_counts) != 1:
            raise ValueError(
                f"{method} has unequal trial counts across task/seed groups: {trial_counts}"
            )
        k = trial_counts[0]
        passes = [
            bool(record["official_evaluator"]["passed"])
            for record in method_records
        ]
        methods[method] = {
            "run_count": len(method_records),
            "task_count": len(per_task_runs),
            "trials_per_task_seed": k,
            "pass_at_1": sum(passes) / len(passes),
            f"pass_at_{k}": (
                sum(any(values) for values in task_seed_runs.values())
                / len(task_seed_runs)
            ),
            f"pass_power_{k}": (
                sum(all(values) for values in task_seed_runs.values())
                / len(task_seed_runs)
            ),
            "mean_tool_calls": (
                sum(int(record.get("tool_call_count", 0)) for record in method_records)
                / len(method_records)
            ),
            "timeout_rate": (
                sum(record.get("termination_reason") == "timeout" for record in method_records)
                / len(method_records)
            ),
            "environment_failure_count": sum(
                record.get("failure_source") == "environment"
                for record in method_records
            ),
            "per_task_pass_rate": {
                task_id: sum(values) / len(values)
                for task_id, values in sorted(per_task_runs.items())
            },
        }
    return {
        "schema_version": 1,
        "statistical_unit": "task (paired); runs are repeated measurements",
        "methods": methods,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("official JSONL records must be objects")
                records.append(value)
    return records


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach unchanged Toolathlon evaluator output and summarize formal runs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    attach = subparsers.add_parser("attach")
    attach.add_argument("--agent-result", required=True)
    attach.add_argument("--evaluator-result", required=True)
    attach.add_argument("--pass-pointer", required=True)
    attach.add_argument("--output", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "attach":
        agent = json.loads(Path(args.agent_result).read_text(encoding="utf-8"))
        evaluator = json.loads(
            Path(args.evaluator_result).read_text(encoding="utf-8")
        )
        result = attach_official_evaluator(
            agent,
            evaluator,
            pass_pointer=args.pass_pointer,
        )
    else:
        result = summarize_official_records(_read_jsonl(args.input))
    _write(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
