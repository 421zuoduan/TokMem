import json
from collections import Counter, defaultdict
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from eval import compare_function_calls_advanced, calculate_tool_metrics  # noqa: E402


def lcs_length(left, right):
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for j, right_item in enumerate(right, 1):
            if left_item == right_item:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(prediction, reference):
    pred_tokens = str(prediction or "").split()
    ref_tokens = str(reference or "").split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def serialize_target_sequence(example, use_eoc=False):
    parts = []
    for tool_name, function_call in zip(example.get("tools", []), example.get("function_calls", [])):
        parts.append(tool_name)
        parts.append(function_call)
        if use_eoc:
            parts.append("<EOC>")
    return " ".join(parts)


def serialize_prediction_sequence(predicted_tools, predicted_calls, use_eoc=False):
    parts = []
    for idx, tool_name in enumerate(predicted_tools):
        parts.append(tool_name)
        if idx < len(predicted_calls):
            parts.append(predicted_calls[idx])
        if use_eoc:
            parts.append("<EOC>")
    return " ".join(parts)


def extract_predicted_tools_and_calls(result):
    predicted_tools = []
    for tool_info in result.get("predicted_tools", []) or []:
        if isinstance(tool_info, dict) and tool_info.get("tool_name"):
            predicted_tools.append(tool_info["tool_name"])
    predicted_calls = list(result.get("function_calls", []) or [])

    if not predicted_tools and result.get("predicted_tool_name") not in (None, "none"):
        predicted_tools = [result["predicted_tool_name"]]
    if not predicted_calls and result.get("function_call"):
        predicted_calls = [result["function_call"]]

    return predicted_tools, predicted_calls


def calculate_transition_error(predicted_tools, expected_tools):
    if len(expected_tools) <= 1:
        return {
            "transition_total": 0,
            "transition_errors": 0,
            "transition_error_rate": 0.0,
        }

    errors = 0
    total = len(expected_tools) - 1
    for idx in range(total):
        expected_pair = (expected_tools[idx], expected_tools[idx + 1])
        predicted_pair = (
            predicted_tools[idx] if idx < len(predicted_tools) else None,
            predicted_tools[idx + 1] if idx + 1 < len(predicted_tools) else None,
        )
        if predicted_pair != expected_pair:
            errors += 1

    return {
        "transition_total": total,
        "transition_errors": errors,
        "transition_error_rate": errors / total if total else 0.0,
    }


def calculate_argument_f1(predicted_calls, expected_calls):
    result = compare_function_calls_advanced(
        predicted_calls,
        expected_calls,
        ignore_order=False,
    )
    return {
        "argument_f1": result.f1_score,
        "argument_precision": result.precision,
        "argument_recall": result.recall,
        "argument_exact_match": result.exact_match,
        "parse_errors": result.details.get("parse_errors", {}).get("outputs", 0),
    }


def evaluate_taskbench_predictions(prediction_records, candidate_tools=None, use_eoc=False):
    if candidate_tools is None:
        candidate_tools = sorted({
            tool
            for record in prediction_records
            for tool in record["example"].get("tools", [])
        })

    totals = {
        "total_examples": len(prediction_records),
        "routing_exact_matches": 0,
        "transition_errors": 0,
        "transition_total": 0,
        "argument_exact_matches": 0,
        "parse_errors": 0,
        "parse_error_examples": 0,
    }
    score_lists = defaultdict(list)
    tool_counts = Counter()
    breakdown = defaultdict(lambda: defaultdict(float))

    for record in prediction_records:
        example = record["example"]
        result = record["prediction"]
        expected_tools = list(example.get("tools", []))
        expected_calls = list(example.get("function_calls", []))
        predicted_tools, predicted_calls = extract_predicted_tools_and_calls(result)
        call_count = len(expected_tools)
        tool_counts[call_count] += 1

        sequence_match = predicted_tools == expected_tools
        if sequence_match:
            totals["routing_exact_matches"] += 1
            breakdown[call_count]["routing_exact_matches"] += 1

        transition = calculate_transition_error(predicted_tools, expected_tools)
        totals["transition_errors"] += transition["transition_errors"]
        totals["transition_total"] += transition["transition_total"]
        breakdown[call_count]["transition_errors"] += transition["transition_errors"]
        breakdown[call_count]["transition_total"] += transition["transition_total"]

        tool_metrics = calculate_tool_metrics(
            predicted_tools=predicted_tools,
            expected_tools=expected_tools,
            candidate_tools=candidate_tools,
        )
        score_lists["tool_f1"].append(tool_metrics["tool_f1_score"])
        score_lists["tool_precision"].append(tool_metrics["tool_precision"])
        score_lists["tool_recall"].append(tool_metrics["tool_recall"])
        breakdown[call_count]["tool_f1_sum"] += tool_metrics["tool_f1_score"]

        argument_metrics = calculate_argument_f1(predicted_calls, expected_calls)
        score_lists["argument_f1"].append(argument_metrics["argument_f1"])
        score_lists["argument_precision"].append(argument_metrics["argument_precision"])
        score_lists["argument_recall"].append(argument_metrics["argument_recall"])
        if argument_metrics["argument_exact_match"]:
            totals["argument_exact_matches"] += 1
            breakdown[call_count]["argument_exact_matches"] += 1
        totals["parse_errors"] += argument_metrics["parse_errors"]
        if argument_metrics["parse_errors"] > 0:
            totals["parse_error_examples"] += 1
            breakdown[call_count]["parse_error_examples"] += 1
        breakdown[call_count]["argument_f1_sum"] += argument_metrics["argument_f1"]

        target_sequence = serialize_target_sequence(example, use_eoc=use_eoc)
        predicted_sequence = serialize_prediction_sequence(predicted_tools, predicted_calls, use_eoc=use_eoc)
        rouge_l = rouge_l_score(predicted_sequence, target_sequence)
        score_lists["rouge_l"].append(rouge_l)
        breakdown[call_count]["rouge_l_sum"] += rouge_l

    total_examples = totals["total_examples"]
    metrics = {
        "total_examples": total_examples,
        "routing_acc": totals["routing_exact_matches"] / total_examples if total_examples else 0.0,
        "task_prediction_accuracy": totals["routing_exact_matches"] / total_examples if total_examples else 0.0,
        "transition_error": totals["transition_errors"] / totals["transition_total"] if totals["transition_total"] else 0.0,
        "avg_tool_f1_score": average(score_lists["tool_f1"]),
        "avg_tool_precision": average(score_lists["tool_precision"]),
        "avg_tool_recall": average(score_lists["tool_recall"]),
        "avg_argument_f1": average(score_lists["argument_f1"]),
        "avg_argument_precision": average(score_lists["argument_precision"]),
        "avg_argument_recall": average(score_lists["argument_recall"]),
        "argument_exact_match_acc": totals["argument_exact_matches"] / total_examples if total_examples else 0.0,
        "avg_rouge_l": average(score_lists["rouge_l"]),
        "parse_error_rate": totals["parse_error_examples"] / total_examples if total_examples else 0.0,
        "parse_error_call_count": totals["parse_errors"],
        "call_count_breakdown": {},
    }

    for call_count in sorted(tool_counts):
        count = tool_counts[call_count]
        stats = breakdown[call_count]
        transition_total = stats.get("transition_total", 0)
        metrics["call_count_breakdown"][str(call_count)] = {
            "total": count,
            "routing_acc": stats.get("routing_exact_matches", 0) / count if count else 0.0,
            "task_prediction_accuracy": stats.get("routing_exact_matches", 0) / count if count else 0.0,
            "transition_error": stats.get("transition_errors", 0) / transition_total if transition_total else 0.0,
            "avg_tool_f1_score": stats.get("tool_f1_sum", 0.0) / count if count else 0.0,
            "avg_argument_f1": stats.get("argument_f1_sum", 0.0) / count if count else 0.0,
            "avg_rouge_l": stats.get("rouge_l_sum", 0.0) / count if count else 0.0,
            "argument_exact_match_acc": stats.get("argument_exact_matches", 0) / count if count else 0.0,
            "parse_error_rate": stats.get("parse_error_examples", 0) / count if count else 0.0,
        }

    return metrics


def average(values):
    return sum(values) / len(values) if values else 0.0


def write_predictions(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for record in records:
        serializable.append(
            {
                "id": record["example"].get("id"),
                "expected_tools": record["example"].get("tools", []),
                "expected_function_calls": record["example"].get("function_calls", []),
                "prediction": record["prediction"],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
