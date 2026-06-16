#!/usr/bin/env python3
"""Compare TokMem and EOC+logit-bias per-sample prediction JSONL files."""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find samples where EOC+logit-bias is correct and TokMem has later tool-selection errors."
    )
    parser.add_argument("--tokmem", required=True, help="TokMem prediction JSONL")
    parser.add_argument("--eoc-logit-bias", required=True, help="EOC+logit-bias prediction JSONL")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--require-dissimilar",
        action="store_true",
        help="Keep later TokMem tool errors only when expected and predicted tools look semantically different.",
    )
    return parser.parse_args()


def load_jsonl(path):
    records = {}
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[int(record["index"])] = record
    return records


def tool_terms(name):
    stop = {
        "find",
        "get",
        "calculate",
        "is",
        "valid",
        "all",
        "between",
        "from",
        "to",
        "of",
        "in",
        "and",
        "the",
    }
    return {
        normalize_term(term)
        for term in re.split(r"[_\W]+", (name or "").lower())
        if term and term not in stop
    }


def normalize_term(term):
    for suffix in ["ization", "ation", "tion", "ing", "ate", "ed", "s"]:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def tool_category(name):
    name = (name or "").lower()
    if any(term in name for term in ["ip", "palindrome", "rotation", "leap", "power", "sudoku", "attend"]):
        return "predicate"
    if any(
        term in name
        for term in [
            "area",
            "velocity",
            "energy",
            "integrat",
            "mortgage",
            "profit",
            "cagr",
            "factorial",
            "fibonacci",
            "divisor",
            "quadratic",
            "wire",
            "median",
            "grade",
            "bits",
            "subarray",
            "equilibrium",
            "minimum",
            "duplicate",
            "binary",
        ]
    ):
        return "math_algo"
    if any(
        term in name
        for term in [
            "product",
            "order",
            "safeway",
            "whole_foods",
            "pokemon",
            "city",
            "zipcode",
            "whois",
            "directions",
            "auto_complete",
        ]
    ):
        return "external_lookup"
    if any(term in name for term in ["string", "word", "password", "date", "histogram", "list"]):
        return "text_data"
    return "other"


def dissimilar_tools(predicted, expected):
    if predicted is None or expected is None:
        return False
    if predicted == expected:
        return False
    if "integrat" in predicted and "integrat" in expected:
        return False
    if tool_category(predicted) != tool_category(expected):
        return True
    return len(tool_terms(predicted) & tool_terms(expected)) == 0


def later_tool_mismatches(tokmem_record):
    predicted = tokmem_record["predicted_tools"]
    expected = tokmem_record["expected_tools"]
    mismatches = []
    for position in range(1, max(len(predicted), len(expected))):
        predicted_tool = predicted[position] if position < len(predicted) else None
        expected_tool = expected[position] if position < len(expected) else None
        if predicted_tool != expected_tool:
            expected_present = expected_tool is not None
            predicted_present = predicted_tool is not None
            mismatches.append(
                {
                    "position": position,
                    "expected": expected_tool,
                    "tokmem_predicted": predicted_tool,
                    "expected_present": expected_present,
                    "predicted_present": predicted_present,
                    "expected_category": tool_category(expected_tool),
                    "predicted_category": tool_category(predicted_tool),
                    "dissimilar": dissimilar_tools(predicted_tool, expected_tool),
                }
            )
    return mismatches


def eoc_is_correct(record):
    return bool(record.get("tool_sequence_exact") and record.get("call_exact"))


def build_match(tokmem_record, eoc_record, mismatches):
    return {
        "index": tokmem_record["index"],
        "user_input": tokmem_record["user_input"],
        "expected_tools": tokmem_record["expected_tools"],
        "expected_calls": tokmem_record["expected_calls"],
        "tokmem": {
            "predicted_tools": tokmem_record["predicted_tools"],
            "predicted_calls": tokmem_record["predicted_calls"],
            "tool_tokens": tokmem_record.get("tool_tokens", []),
            "tool_sequence_exact": tokmem_record.get("tool_sequence_exact"),
            "call_exact": tokmem_record.get("call_exact"),
        },
        "tokmem_eoc_logit_bias": {
            "predicted_tools": eoc_record["predicted_tools"],
            "predicted_calls": eoc_record["predicted_calls"],
            "tool_tokens": eoc_record.get("tool_tokens", []),
            "tool_sequence_exact": eoc_record.get("tool_sequence_exact"),
            "call_exact": eoc_record.get("call_exact"),
        },
        "later_tool_mismatches": mismatches,
    }


def write_markdown(path, matches):
    with open(path, "w") as handle:
        handle.write("# TokMem vs EOC+Logit-Bias Matching Samples\n\n")
        handle.write(f"Matched samples: {len(matches)}\n\n")
        for match in matches:
            handle.write(f"## Index {match['index']}\n\n")
            handle.write(f"User input: {match['user_input']}\n\n")
            handle.write(f"Expected tools: `{match['expected_tools']}`\n\n")
            handle.write(f"TokMem tools: `{match['tokmem']['predicted_tools']}`\n\n")
            handle.write(
                "EOC+logit-bias tools: "
                f"`{match['tokmem_eoc_logit_bias']['predicted_tools']}`\n\n"
            )
            handle.write(f"Later mismatches: `{match['later_tool_mismatches']}`\n\n")


def main():
    args = parse_args()
    tokmem_records = load_jsonl(args.tokmem)
    eoc_records = load_jsonl(args.eoc_logit_bias)

    matches = []
    for index in sorted(set(tokmem_records) & set(eoc_records)):
        tokmem_record = tokmem_records[index]
        eoc_record = eoc_records[index]
        if not eoc_is_correct(eoc_record):
            continue
        mismatches = later_tool_mismatches(tokmem_record)
        if args.require_dissimilar:
            mismatches = [
                mismatch
                for mismatch in mismatches
                if mismatch["expected_present"]
                and mismatch["predicted_present"]
                and mismatch["dissimilar"]
            ]
        if mismatches:
            matches.append(build_match(tokmem_record, eoc_record, mismatches))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as handle:
        json.dump(
            {
                "tokmem_predictions": args.tokmem,
                "eoc_logit_bias_predictions": args.eoc_logit_bias,
                "require_dissimilar": args.require_dissimilar,
                "matched_count": len(matches),
                "matches": matches,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_markdown(output_md, matches)

    print(f"Matched samples: {len(matches)}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
