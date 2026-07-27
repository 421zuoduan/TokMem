from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


READ_MARKERS = (
    "read",
    "list",
    "get",
    "search",
    "inspect",
    "extract",
    "metadata",
    "info",
)
WRITE_MARKERS = (
    "write",
    "create",
    "add",
    "insert",
    "update",
    "delete",
    "remove",
    "move",
    "rename",
    "copy",
    "format",
    "merge",
    "apply",
    "replace",
)


def action_kind(wire_name: str | None) -> str:
    if wire_name is None:
        return "finish"
    lowered = wire_name.casefold()
    if any(marker in lowered for marker in WRITE_MARKERS):
        return "write"
    if any(marker in lowered for marker in READ_MARKERS):
        return "read"
    return "transform"


def analyze_rollout(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollout = payload.get("rollout")
    if not isinstance(rollout, dict):
        failure = payload.get("failure") or {}
        return {
            "path": str(path),
            "usable": False,
            "failure_type": failure.get("error_type"),
            "failure": failure.get("error"),
        }
    events = rollout["events"]
    kinds = [action_kind(event.get("wire_name")) for event in events]
    first_write = next(
        (index for index, kind in enumerate(kinds) if kind == "write"),
        None,
    )
    failures = [
        index for index, event in enumerate(events) if not event.get("success")
    ]
    changed_retries = 0
    unchanged_retries = 0
    for index in failures:
        if index + 1 >= len(events):
            continue
        current = events[index]
        following = events[index + 1]
        changed = (
            current.get("wire_name") != following.get("wire_name")
            or current.get("arguments") != following.get("arguments")
        )
        if changed:
            changed_retries += 1
        else:
            unchanged_retries += 1
    tool_counts = Counter(
        str(event["wire_name"])
        for event in events
        if event.get("wire_name") is not None
    )
    return {
        "path": str(path),
        "usable": True,
        "task_dir": rollout.get("task_dir"),
        "prompt_variant": rollout.get("prompt_variant"),
        "termination_reason": rollout.get("termination_reason"),
        "tool_call_count": len(events),
        "execution_error_count": len(failures),
        "phase_sequence": kinds,
        "pre_write_read_count": (
            sum(kind == "read" for kind in kinds)
            if first_write is None
            else sum(kind == "read" for kind in kinds[:first_write])
        ),
        "post_write_read_count": (
            0
            if first_write is None
            else sum(kind == "read" for kind in kinds[first_write + 1 :])
        ),
        "changed_retry_count": changed_retries,
        "unchanged_retry_count": unchanged_retries,
        "claimed_done": bool(kinds and kinds[-1] == "finish"),
        "tool_counts": dict(sorted(tool_counts.items())),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [record for record in records if record["usable"]]
    totals = Counter()
    for record in usable:
        totals.update(record["tool_counts"])
    count = len(usable)
    return {
        "trajectory_files": len(records),
        "usable_trajectories": count,
        "failed_trajectories": len(records) - count,
        "claim_done_rate": (
            sum(record["claimed_done"] for record in usable) / count
            if count
            else 0.0
        ),
        "mean_tool_calls": (
            sum(record["tool_call_count"] for record in usable) / count
            if count
            else 0.0
        ),
        "mean_pre_write_reads": (
            sum(record["pre_write_read_count"] for record in usable) / count
            if count
            else 0.0
        ),
        "mean_post_write_reads": (
            sum(record["post_write_read_count"] for record in usable) / count
            if count
            else 0.0
        ),
        "execution_errors": sum(
            record["execution_error_count"] for record in usable
        ),
        "changed_retries": sum(
            record["changed_retry_count"] for record in usable
        ),
        "unchanged_retries": sum(
            record["unchanged_retry_count"] for record in usable
        ),
        "tool_counts": dict(sorted(totals.items())),
    }


def render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# GPT-5.6 非测试题参考轨迹统计",
        "",
        f"- 可用轨迹：{summary['usable_trajectories']}/{summary['trajectory_files']}",
        f"- 平均工具调用数：{summary['mean_tool_calls']:.2f}",
        f"- 写入前平均读取数：{summary['mean_pre_write_reads']:.2f}",
        f"- 写入后平均读取数：{summary['mean_post_write_reads']:.2f}",
        f"- 改变调用后的失败恢复数：{summary['changed_retries']}",
        f"- 原样重试数：{summary['unchanged_retries']}",
        f"- claim_done 比例：{summary['claim_done_rate']:.2%}",
        "",
        "## 分轨迹",
        "",
    ]
    for record in records:
        if not record["usable"]:
            lines.append(
                f"- `{record['path']}`：不可用（{record.get('failure_type')}）"
            )
            continue
        phases = " → ".join(record["phase_sequence"])
        lines.append(
            f"- `{record.get('task_dir')}` / `{record.get('prompt_variant')}`："
            f"{record['tool_call_count']} 次，`{phases}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize GPT-5.6 reference trajectories without task content"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.root).glob("**/gpt5.6_rollout.json"))
    records = [analyze_rollout(path) for path in paths]
    summary = aggregate(records)
    output = {"summary": summary, "trajectories": records}
    Path(args.json_output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown_output).write_text(
        render_markdown(summary, records),
        encoding="utf-8",
    )
    print(
        f"reference={len(records)} usable={summary['usable_trajectories']} "
        f"failed={summary['failed_trajectories']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
