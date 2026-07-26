#!/usr/bin/env python3
"""Select the largest safe prefix using only old-tool replay records."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tcra-gain", type=float, default=1.0)
    parser.add_argument("--new-logit-penalty", type=float, default=0.0)
    parser.add_argument("--max-old-to-new-rate", type=float, default=0.05)
    parser.add_argument(
        "--min-baseline-correct-retention",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-old-accuracy-drop",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-strength",
        type=float,
        default=0.5,
        help="Conservative cap on donor-coherence norm restoration.",
    )
    return parser.parse_args()


def _record_key(summary):
    return (
        f"strength={summary['strength']},"
        f"gain={summary.get('new_tcra_gain', 1.0)},"
        f"penalty={summary.get('new_logit_penalty', 0.0)}"
    )


def select_largest_safe_prefix(
    sweep,
    tcra_gain=1.0,
    new_logit_penalty=0.0,
    max_old_to_new_rate=0.05,
    min_baseline_correct_retention=1.0,
    max_old_accuracy_drop=0.0,
    max_strength=0.5,
):
    records_by_candidate = sweep.get("records")
    if not records_by_candidate:
        raise ValueError("The sweep must be produced with --save-records")

    candidates = sorted(
        (
            summary
            for summary in sweep["strengths"]
            if summary.get("new_tcra_gain", 1.0) == tcra_gain
            and summary.get("new_logit_penalty", 0.0)
            == new_logit_penalty
            and summary["strength"] <= max_strength
        ),
        key=lambda summary: summary["strength"],
    )
    if not candidates or candidates[0]["strength"] != 0.0:
        raise ValueError("Safe-prefix selection requires a zero-strength row")

    baseline = candidates[0]
    baseline_records = records_by_candidate[_record_key(baseline)]
    baseline_correct = {
        record["index"]
        for record in baseline_records
        if record["correct"]
    }
    baseline_accuracy = baseline["old_tools"]["accuracy"]

    audit = []
    selected = baseline
    for summary in candidates:
        records = records_by_candidate[_record_key(summary)]
        retained = sum(
            record["index"] in baseline_correct and record["correct"]
            for record in records
        )
        retention_rate = retained / max(1, len(baseline_correct))
        old_accuracy_drop = (
            baseline_accuracy - summary["old_tools"]["accuracy"]
        )
        safe = (
            summary["old_to_new_rate"] <= max_old_to_new_rate
            and retention_rate >= min_baseline_correct_retention
            and old_accuracy_drop <= max_old_accuracy_drop
        )
        audit.append(
            {
                "strength": summary["strength"],
                "old_accuracy": summary["old_tools"]["accuracy"],
                "old_accuracy_drop": old_accuracy_drop,
                "old_to_new_rate": summary["old_to_new_rate"],
                "baseline_correct_retention": retention_rate,
                "safe": safe,
            }
        )
        if not safe:
            break
        selected = summary

    return {
        "selection_rule": "largest_safe_prefix",
        "selected_strength": selected["strength"],
        "tcra_gain": tcra_gain,
        "new_logit_penalty": new_logit_penalty,
        "constraints": {
            "max_old_to_new_rate": max_old_to_new_rate,
            "min_baseline_correct_retention": (
                min_baseline_correct_retention
            ),
            "max_old_accuracy_drop": max_old_accuracy_drop,
            "max_strength": max_strength,
        },
        "baseline": baseline,
        "selected": selected,
        "audit": audit,
    }


def main():
    args = parse_args()
    with open(args.sweep, "r", encoding="utf-8") as handle:
        sweep = json.load(handle)
    selection = select_largest_safe_prefix(
        sweep,
        tcra_gain=args.tcra_gain,
        new_logit_penalty=args.new_logit_penalty,
        max_old_to_new_rate=args.max_old_to_new_rate,
        min_baseline_correct_retention=(
            args.min_baseline_correct_retention
        ),
        max_old_accuracy_drop=args.max_old_accuracy_drop,
        max_strength=args.max_strength,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
    print(json.dumps(selection["selected"], ensure_ascii=False, indent=2))
    print(f"Wrote safe-prefix selection to {output_path}")


if __name__ == "__main__":
    main()
