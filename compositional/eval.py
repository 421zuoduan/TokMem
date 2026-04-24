import ast
import json
import re
from collections import Counter
from typing import List, Union, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EvalResult:
    """Detailed evaluation result with breakdown of matches and mismatches"""
    exact_match: bool
    f1_score: float  # 0.0 to 1.0
    precision: float  # 0.0 to 1.0
    recall: float  # 0.0 to 1.0
    details: Dict[str, Any]

def normalize_json_string(text: str) -> str:
    """Normalize JSON string by removing extra whitespace and standardizing format"""
    try:
        # Try to parse and re-serialize to normalize format
        parsed = json.loads(text)
        return json.dumps(parsed, sort_keys=True, separators=(',', ':'))
    except (json.JSONDecodeError, TypeError):
        # Fallback: basic normalization
        return re.sub(r'\s+', ' ', text.strip())

def extract_json_from_text(text: str) -> Any:
    """Extract a JSON-like value from text, handling objects, arrays, scalars, and call syntax."""
    if not text or not isinstance(text, str):
        return {'raw_text': str(text), 'parse_error': True}
    
    text = text.strip()
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    if '{' in text and '}' in text:
        start = text.find('{')
        end = text.rfind('}') + 1
        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to handle function call format: func_name(arg1=val1, arg2=val2)
    func_call_match = re.match(r'(\w+)\((.*)\)', text)
    if func_call_match:
        func_name, args_str = func_call_match.groups()
        try:
            # Parse arguments
            args_dict = {}
            if args_str.strip():
                # Simple parsing for key=value pairs
                for arg in args_str.split(','):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to evaluate as Python literal
                        try:
                            args_dict[key] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            args_dict[key] = value
            return {func_name: args_dict}
        except Exception:
            pass
    
    # Fallback to raw text
    return {'raw_text': text, 'parse_error': True}

def parse_function_call(call_input: Union[str, Dict, List, int, float, bool, None]) -> Any:
    """Parse function call from various input formats."""
    if isinstance(call_input, dict):
        return call_input
    elif isinstance(call_input, str):
        return extract_json_from_text(call_input)
    else:
        return {'raw_text': str(call_input), 'parse_error': True}

def _is_parse_error_payload(call: Any) -> bool:
    return isinstance(call, dict) and ('parse_error' in call or 'raw_text' in call)

def normalize_function_call(call: Any) -> str:
    """Convert a parsed function call into a stable comparable string."""
    if _is_parse_error_payload(call):
        # Handle unparseable calls by normalizing the raw text
        raw_text = call.get('raw_text', str(call))
        return normalize_json_string(raw_text)

    if not isinstance(call, dict):
        try:
            return json.dumps(call, sort_keys=True, separators=(',', ':'))
        except (TypeError, ValueError):
            return str(call)
    
    # Handle structured calls - convert to AST representation
    if len(call) == 1:
        # Standard format: {"func_name": {"arg1": val1, "arg2": val2}}
        func_name, params = next(iter(call.items()))
        if isinstance(params, dict):
            try:
                args_src = ", ".join(f"{k}={repr(v)}" for k, v in sorted(params.items()))
                expr = f"{func_name}({args_src})"
                node = ast.parse(expr).body[0].value
                return ast.dump(node, annotate_fields=False, include_attributes=False)
            except Exception:
                pass
    
    # Fallback to normalized JSON
    try:
        return json.dumps(call, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(call)

def compare_function_calls_advanced(
    output_calls: List[Union[str, Dict]], 
    target_calls: List[Union[str, Dict]], 
    ignore_order: bool = True
) -> EvalResult:
    """
    Advanced comparison of function call lists with detailed evaluation.
    
    Args:
        output_calls: Model-generated calls (strings or dicts)
        target_calls: Ground-truth calls (strings or dicts)  
        ignore_order: If True, compare as multisets; else compare sequence-equal
        
    Returns:
        EvalResult with detailed breakdown
    """
    # Parse all calls
    parsed_outputs = [parse_function_call(call) for call in output_calls]
    parsed_targets = [parse_function_call(call) for call in target_calls]
    
    # Normalize to AST representations
    norm_outputs = [normalize_function_call(call) for call in parsed_outputs]
    norm_targets = [normalize_function_call(call) for call in parsed_targets]
    
    # Calculate exact match
    if ignore_order:
        exact_match = sorted(norm_outputs) == sorted(norm_targets)
    else:
        exact_match = norm_outputs == norm_targets
    
    details = {
        'output_count': len(output_calls),
        'target_count': len(target_calls),
        'parsed_outputs': parsed_outputs,
        'parsed_targets': parsed_targets,
        'normalized_outputs': norm_outputs,
        'normalized_targets': norm_targets,
        'parse_errors': {
            'outputs': sum(1 for call in parsed_outputs if _is_parse_error_payload(call)),
            'targets': sum(1 for call in parsed_targets if _is_parse_error_payload(call))
        }
    }
    
    # Calculate F1 metrics
    f1_metrics = calculate_f1_score(norm_outputs, norm_targets)
    
    # Add F1 breakdown to details
    details['f1_breakdown'] = f1_metrics
    
    return EvalResult(
        exact_match=exact_match,
        f1_score=f1_metrics["f1_score"],
        precision=f1_metrics["precision"],
        recall=f1_metrics["recall"],
        details=details
    )


def calculate_f1_score(outputs: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate F1 score following NESTful benchmark methodology"""
    if not outputs and not targets:
        return {"f1_score": 1.0, "precision": 1.0, "recall": 1.0}
    
    if not outputs:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
    
    if not targets:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Convert to sets for intersection calculations
    output_set = set(outputs)
    target_set = set(targets)
    
    # Calculate precision and recall
    true_positives = len(output_set & target_set)
    precision = true_positives / len(output_set)
    recall = true_positives / len(target_set)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": len(output_set) - true_positives,
        "false_negatives": len(target_set) - true_positives
    }


def extract_tool_names(calls: List[Union[str, Dict]]) -> List[str]:
    """Extract tool/function names from function calls"""
    tool_names = []
    
    for call in calls:
        parsed_call = parse_function_call(call)

        if _is_parse_error_payload(parsed_call):
            # Try to extract function name from raw text
            raw_text = parsed_call.get('raw_text', '')
            match = re.match(r'^(\w+)\s*\(', raw_text)
            if match:
                tool_names.append(match.group(1))
            continue

        if not isinstance(parsed_call, dict):
            continue

        # Extract function name from structured call
        if len(parsed_call) == 1:
            func_name = next(iter(parsed_call.keys()))
            tool_names.append(func_name)
        elif 'function' in parsed_call:
            tool_names.append(parsed_call['function'])
        elif 'name' in parsed_call:
            tool_names.append(parsed_call['name'])
    
    return tool_names


def extract_tool_argument_pairs(calls: List[Union[str, Dict]]) -> List[Tuple[str, str]]:
    """Extract comparable (tool_name, normalized_args) pairs from function calls."""
    pairs = []

    for call in calls:
        parsed_call = parse_function_call(call)
        if _is_parse_error_payload(parsed_call):
            continue
        if not isinstance(parsed_call, dict) or len(parsed_call) != 1:
            continue

        func_name, params = next(iter(parsed_call.items()))
        try:
            normalized_args = json.dumps(params, sort_keys=True, separators=(',', ':'))
        except (TypeError, ValueError):
            normalized_args = str(params)
        pairs.append((func_name, normalized_args))

    return pairs


def calculate_argument_accuracy(output_calls: List[Union[str, Dict]],
                                target_calls: List[Union[str, Dict]]) -> Dict[str, float]:
    """Calculate how often gold tool calls also get the full argument payload correct."""
    output_counter = Counter(extract_tool_argument_pairs(output_calls))
    target_counter = Counter(extract_tool_argument_pairs(target_calls))

    matched_arguments = 0
    for pair, target_count in target_counter.items():
        matched_arguments += min(target_count, output_counter.get(pair, 0))

    total_target_arguments = sum(target_counter.values())
    arguments_accuracy = (
        matched_arguments / total_target_arguments
        if total_target_arguments > 0
        else 1.0
    )

    return {
        "arguments_accuracy": arguments_accuracy,
        "matched_arguments": matched_arguments,
        "total_target_arguments": total_target_arguments,
    }


def calculate_tool_metrics(
    predicted_tools: Optional[List[str]] = None,
    expected_tools: Optional[List[str]] = None,
    output_calls: Optional[List[Union[str, Dict]]] = None,
    target_calls: Optional[List[Union[str, Dict]]] = None,
    candidate_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate tool-level accuracy, exact match, and F1-style metrics.

    The helper accepts pre-extracted tool lists or raw function-call lists.
    ``candidate_tools`` should contain the full tool universe for the run when
    tool_accuracy needs binary TP/TN/FP/FN over every sample-tool pair.
    """

    if predicted_tools is None:
        predicted_tools = extract_tool_names(output_calls or [])
    if expected_tools is None:
        expected_tools = extract_tool_names(target_calls or [])

    predicted_counter = Counter(predicted_tools)
    expected_counter = Counter(expected_tools)

    candidate_tool_set = set(candidate_tools or [])
    candidate_tool_set.update(predicted_counter.keys())
    candidate_tool_set.update(expected_counter.keys())

    tool_tp = 0
    tool_tn = 0
    tool_fp = 0
    tool_fn = 0

    for tool_name in candidate_tool_set:
        predicted_present = predicted_counter.get(tool_name, 0) > 0
        expected_present = expected_counter.get(tool_name, 0) > 0

        if predicted_present and expected_present:
            tool_tp += 1
        elif predicted_present and not expected_present:
            tool_fp += 1
        elif not predicted_present and expected_present:
            tool_fn += 1
        else:
            tool_tn += 1

    total_judgments = tool_tp + tool_tn + tool_fp + tool_fn
    tool_accuracy = (tool_tp + tool_tn) / total_judgments if total_judgments > 0 else 1.0
    tool_exact_match_acc = 1.0 if predicted_counter == expected_counter else 0.0
    tool_f1_metrics = calculate_f1_score(predicted_tools, expected_tools)

    return {
        "tool_tp": tool_tp,
        "tool_tn": tool_tn,
        "tool_fp": tool_fp,
        "tool_fn": tool_fn,
        "tool_total_judgments": total_judgments,
        "tool_accuracy": tool_accuracy,
        "tool_exact_match_acc": tool_exact_match_acc,
        "tool_f1_score": tool_f1_metrics["f1_score"],
        "tool_precision": tool_f1_metrics["precision"],
        "tool_recall": tool_f1_metrics["recall"],
        "predicted_tools": list(predicted_tools),
        "expected_tools": list(expected_tools),
        "candidate_tools": sorted(candidate_tool_set),
    }

def calculate_tool_selection_accuracy(
    output_calls: Optional[List[Union[str, Dict]]] = None,
    target_calls: Optional[List[Union[str, Dict]]] = None,
    predicted_tools: Optional[List[str]] = None,
    expected_tools: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate nuanced tool selection accuracy metrics."""

    if predicted_tools is None:
        predicted_tools = extract_tool_names(output_calls or [])
    if expected_tools is None:
        expected_tools = extract_tool_names(target_calls or [])

    # Calculate basic F1 for tool selection
    tool_f1_results = calculate_f1_score(predicted_tools, expected_tools)

    # Calculate additional metrics
    output_tool_set = set(predicted_tools)
    target_tool_set = set(expected_tools)
    
    # Tool coverage: what percentage of required tools were selected
    tool_coverage = len(output_tool_set & target_tool_set) / len(target_tool_set) if expected_tools else 1.0

    # Tool precision: what percentage of selected tools were correct
    tool_precision = len(output_tool_set & target_tool_set) / len(output_tool_set) if predicted_tools else 0.0

    # Over-selection penalty: penalize selecting too many tools
    over_selection_ratio = (
        len(output_tool_set) / len(target_tool_set)
        if expected_tools
        else (1.0 if not predicted_tools else float('inf'))
    )
    over_selection_penalty = max(0.0, 1.0 - (over_selection_ratio - 1.0)) if over_selection_ratio > 1.0 else 1.0

    # Under-selection penalty: penalize selecting too few tools  
    under_selection_ratio = len(target_tool_set) / len(output_tool_set) if predicted_tools else float('inf')
    under_selection_penalty = max(0.0, 1.0 - (under_selection_ratio - 1.0)) if under_selection_ratio > 1.0 else 1.0
    
    # Combined selection accuracy with penalties
    selection_accuracy = tool_f1_results['f1_score'] * over_selection_penalty * under_selection_penalty
    
    return {
        'tool_f1_score': tool_f1_results['f1_score'],
        'tool_precision': tool_precision,
        'tool_recall': tool_coverage,
        'tool_coverage': tool_coverage,
        'over_selection_penalty': over_selection_penalty,
        'under_selection_penalty': under_selection_penalty,
        'nuanced_tool_accuracy': selection_accuracy,
        'selected_tools': list(output_tool_set),
        'expected_tools': list(target_tool_set),
        'correct_tools': list(output_tool_set & target_tool_set),
        'missing_tools': list(target_tool_set - output_tool_set),
        'extra_tools': list(output_tool_set - target_tool_set)
    }
