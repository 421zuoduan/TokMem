from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .generate_tasks import write_jsonl


def merge_jsonl(
    paths: list[str | Path],
    unique_key: str,
    excluded_values: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    excluded = excluded_values or set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                value = record.get(unique_key)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{path}:{line_number} requires non-empty {unique_key}"
                    )
                if value in excluded:
                    continue
                if value in seen:
                    raise ValueError(f"duplicate {unique_key}: {value}")
                seen.add(value)
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge JSONL files in argument order")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unique-key", default="task_id")
    parser.add_argument("--exclude-value", action="append", default=[])
    args = parser.parse_args()

    records = merge_jsonl(
        args.input,
        args.unique_key,
        set(args.exclude_value),
    )
    write_jsonl(Path(args.output), records)
    print(f"Wrote {len(records)} records to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
