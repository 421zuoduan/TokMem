#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compositional.dataset import discover_available_tools
from compositional.eval import compare_function_calls_advanced, parse_function_call


CATEGORY_ORDER = ["last_input", "tool", "eoc", "args"]
CATEGORY_COLORS = {
    "last_input": "#D55E00",
    "tool": "#0072B2",
    "eoc": "#009E73",
    "args": "#666666",
}
SPECIAL_LABELS = {
    "last_input": "<last_input>",
    "tool": "<tool>",
    "eoc": "<eoc>",
}


@dataclass
class GenerationCapture:
    generated_token_ids: List[int]
    token_records: List[Dict[str, Any]]
    full_generated_tensor: torch.Tensor
    parsed_result: Dict[str, Any]


def parse_training_rounds(rounds_str: str) -> List[Dict[str, Any]]:
    rounds = []
    for round_spec in rounds_str.split(","):
        tools_part, epochs_part = round_spec.strip().split(":")
        rounds.append({"tools": tools_part.strip(), "epochs": int(epochs_part.strip())})
    return rounds


def expand_per_round_values(values_str: Optional[str], fallback: int, num_rounds: int) -> List[int]:
    if values_str is None:
        return [fallback] * num_rounds

    values = [int(part.strip()) for part in values_str.split(",") if part.strip()]
    if not values:
        return [fallback] * num_rounds
    if len(values) < num_rounds:
        values.extend([values[-1]] * (num_rounds - len(values)))
    return values[:num_rounds]


def resolve_dtype(dtype_name: Optional[str]) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("js_explore")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "analysis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id"])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def find_run_config_path(checkpoint_path: Path, explicit_run_config_path: Optional[Path]) -> Optional[Path]:
    if explicit_run_config_path is not None:
        return explicit_run_config_path
    candidate = checkpoint_path.parent / "run_config.json"
    return candidate if candidate.exists() else None


def derive_output_dir(checkpoint_path: Path, requested_output_dir: Optional[str]) -> Path:
    if requested_output_dir:
        output_dir = Path(requested_output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_stem = checkpoint_path.stem
    return REPO_ROOT / "compositional" / "runs" / "js_explore" / f"{checkpoint_stem}_{timestamp}"


def flatten_dict(prefix: str, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    for key, value in payload.items():
        joined = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flatten_dict(joined, value, result)
        else:
            result[joined] = value


def classify_generated_token(
    token_id: int,
    position: int,
    last_input_position: int,
    tool_token_ids: Sequence[int],
    eoc_token_id: Optional[int],
    eot_token_ids: Sequence[int],
) -> Optional[str]:
    # Token classification rule:
    # 1. The prompt-side last input token is its own category.
    # 2. Generated tool tokens and <|eoc_id|> are split explicitly.
    # 3. <|eot_id|> is excluded from the four-way analysis.
    # 4. All remaining generated tokens are treated as args / JSON tokens.
    if position == last_input_position:
        return "last_input"
    if token_id in set(eot_token_ids):
        return None
    if token_id in set(tool_token_ids):
        return "tool"
    if eoc_token_id is not None and token_id == eoc_token_id:
        return "eoc"
    return "args"


def compute_js_divergence_against_final(logits_by_layer: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    logits_by_layer = logits_by_layer.float()
    log_probs = torch.log_softmax(logits_by_layer, dim=-1)
    probs = log_probs.exp()

    final_probs = probs[-1:].expand_as(probs)
    final_log_probs = log_probs[-1:].expand_as(log_probs)

    mixture = 0.5 * (probs + final_probs)
    log_mixture = torch.log(mixture.clamp_min(eps))

    kl_layer = (probs * (log_probs - log_mixture)).sum(dim=-1)
    kl_final = (final_probs * (final_log_probs - log_mixture)).sum(dim=-1)
    return 0.5 * (kl_layer + kl_final)


def compute_final_layer_js_metrics(js_curve: Sequence[float]) -> Dict[str, float]:
    curve = np.asarray(list(js_curve), dtype=np.float64)
    if curve.size == 0:
        return {
            "js_mean": 0.0,
            "js_var": 0.0,
            "js_std": 0.0,
            "first_half_mean": 0.0,
            "second_half_mean": 0.0,
            "first_minus_second": 0.0,
            "first_half_ratio": 0.0,
            "second_half_ratio": 0.0,
        }

    split_index = max(1, curve.size // 2) if curve.size > 1 else 1
    first_half = curve[:split_index]
    second_half = curve[split_index:] if split_index < curve.size else np.asarray([], dtype=np.float64)

    first_half_mean = float(first_half.mean()) if first_half.size else 0.0
    second_half_mean = float(second_half.mean()) if second_half.size else 0.0
    total_sum = float(curve.sum())
    first_sum = float(first_half.sum()) if first_half.size else 0.0
    second_sum = float(second_half.sum()) if second_half.size else 0.0

    if total_sum <= 0.0:
        first_ratio = 0.0
        second_ratio = 0.0
    else:
        first_ratio = first_sum / total_sum
        second_ratio = second_sum / total_sum

    return {
        "js_mean": float(curve.mean()),
        "js_var": float(curve.var()),
        "js_std": float(curve.std()),
        "first_half_mean": first_half_mean,
        "second_half_mean": second_half_mean,
        "first_minus_second": first_half_mean - second_half_mean,
        "first_half_ratio": first_ratio,
        "second_half_ratio": second_ratio,
    }


def summarize_category_metrics(token_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"count": len(token_metrics)}
    metric_names = [
        "js_mean",
        "js_var",
        "js_std",
        "first_half_mean",
        "second_half_mean",
        "first_minus_second",
        "first_half_ratio",
        "second_half_ratio",
    ]

    for metric_name in metric_names:
        values = [float(item[metric_name]) for item in token_metrics if metric_name in item]
        if values:
            array = np.asarray(values, dtype=np.float64)
            summary[f"{metric_name}_mean"] = float(array.mean())
            summary[f"{metric_name}_var"] = float(array.var())
        else:
            summary[f"{metric_name}_mean"] = None
            summary[f"{metric_name}_var"] = None

    return summary


def build_structured_calls(tool_names: Sequence[str], function_calls: Sequence[str]) -> List[Dict[str, Any]]:
    structured_calls: List[Dict[str, Any]] = []
    for tool_name, function_call in zip(tool_names, function_calls):
        parsed = parse_function_call(function_call)
        if isinstance(parsed, dict):
            structured_calls.append({tool_name: parsed})
        else:
            structured_calls.append({tool_name: {"value": parsed}})
    return structured_calls


def evaluate_complete_correctness(
    predicted_tools: Sequence[str],
    predicted_calls: Sequence[str],
    expected_tools: Sequence[str],
    expected_calls: Sequence[str],
) -> Dict[str, Any]:
    structured_predicted = build_structured_calls(predicted_tools, predicted_calls)
    structured_expected = build_structured_calls(expected_tools, expected_calls)
    eval_result = compare_function_calls_advanced(
        structured_predicted,
        structured_expected,
        ignore_order=True,
    )
    return {
        "correct": bool(eval_result.exact_match),
        "structured_predicted": structured_predicted,
        "structured_expected": structured_expected,
        "f1_score": eval_result.f1_score,
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "details": eval_result.details,
    }


def encode_eval_prompt(tokenizer: AutoTokenizer, user_input: str) -> torch.Tensor:
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return torch.tensor(token_ids, dtype=torch.long)


def compute_expected_tool_markers(
    sample: Dict[str, Any],
    tokenizer: Any,
    model: Any,
) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    cursor = 0

    for tool_name, function_call in zip(sample.get("tools", []), sample.get("function_calls", [])):
        tool_token_id = model.get_tool_token_id(tool_name)
        if tool_token_id is None:
            continue
        markers.append(
            {
                "position": cursor,
                "tool_name": tool_name,
                "token_id": int(tool_token_id),
            }
        )
        cursor += 1
        cursor += len(tokenizer(function_call, add_special_tokens=False)["input_ids"])
        if getattr(model, "use_eoc", False) and getattr(model, "eoc_token_id", None) is not None:
            cursor += 1

    return markers


def discover_tool_names_from_test_file(test_data_path: Path) -> List[str]:
    with test_data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tools = sorted({tool for item in data for tool in item.get("tools", [])})
    if not tools:
        raise ValueError(f"No tools found in test data: {test_data_path}")
    return tools


def discover_all_tool_names(run_args: Dict[str, Any]) -> List[str]:
    rounds = parse_training_rounds(run_args["training_rounds"])
    data_dir = run_args["data_dir"]
    train_calls = expand_per_round_values(
        run_args.get("train_max_function_calls_per_round"),
        int(run_args["train_max_function_calls"]),
        len(rounds),
    )
    test_calls = expand_per_round_values(
        run_args.get("test_max_function_calls_per_round"),
        int(run_args["test_max_function_calls"]),
        len(rounds),
    )

    all_tool_names: List[str] = []
    seen = set()
    for round_idx, round_spec in enumerate(rounds):
        tools_range = round_spec["tools"]
        train_data_file = os.path.join(
            data_dir,
            f"training/function_calling_train_tools{tools_range}_{train_calls[round_idx]}calls.json",
        )
        test_data_file = os.path.join(
            data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_calls[round_idx]}calls.json",
        )
        for tool_name in discover_available_tools(train_data_file, test_data_file):
            if tool_name not in seen:
                seen.add(tool_name)
                all_tool_names.append(tool_name)
    return all_tool_names


def resolve_test_data_path(
    checkpoint_path: Path,
    requested_test_data_path: Optional[str],
    run_args: Optional[Dict[str, Any]],
) -> Path:
    if requested_test_data_path:
        test_data_path = Path(requested_test_data_path)
        if not test_data_path.is_absolute():
            test_data_path = REPO_ROOT / test_data_path
        return test_data_path

    if run_args is None:
        raise ValueError("--test_data_path is required when run_config.json is unavailable")

    checkpoint_name = checkpoint_path.name
    match = re.search(r"round_(\d+)_tools_([\d_]+)\.pt$", checkpoint_name)
    if not match:
        raise ValueError(f"Unable to infer test data path from checkpoint name: {checkpoint_name}")

    round_index = int(match.group(1)) - 1
    rounds = parse_training_rounds(run_args["training_rounds"])
    if round_index < 0 or round_index >= len(rounds):
        raise ValueError(f"Checkpoint round index {round_index + 1} is out of range for run_config rounds")

    test_calls = expand_per_round_values(
        run_args.get("test_max_function_calls_per_round"),
        int(run_args["test_max_function_calls"]),
        len(rounds),
    )
    tools_range = rounds[round_index]["tools"]
    data_dir = Path(run_args["data_dir"])
    return data_dir / "test" / f"function_calling_test_tools{tools_range}_{test_calls[round_index]}calls.json"


def resolve_lora_config(run_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not run_args.get("use_lora"):
        return None
    config = {
        "r": run_args.get("lora_r", 8),
        "alpha": run_args.get("lora_alpha", 32),
        "dropout": run_args.get("lora_dropout", 0.1),
        "target_modules": [part.strip() for part in run_args.get("lora_target_modules", "o_proj").split(",") if part.strip()],
    }
    if run_args.get("lora_layer_indices"):
        config["layer_indices"] = [int(part.strip()) for part in run_args["lora_layer_indices"].split(",") if part.strip()]
    return config


def load_checkpoint_bundle(
    checkpoint_path: Path,
    run_config_path: Optional[Path],
    test_data_path: Path,
    device: str,
    dtype_name: Optional[str],
    model_name_override: Optional[str],
    logger: logging.Logger,
) -> Tuple[Any, AutoTokenizer, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    run_config = {}
    run_args: Dict[str, Any] = {}
    if run_config_path is not None and run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as f:
            run_config = json.load(f)
        run_args = deepcopy(run_config.get("args", {}))
    logger.info("Loading checkpoint from %s", checkpoint_path)

    model_name = model_name_override or run_args.get("model_name")
    if model_name is None:
        raise ValueError("Base model path is required. Provide --model_name or keep run_config.json next to the checkpoint.")

    dtype = resolve_dtype(dtype_name or run_args.get("dtype"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    if run_args:
        tool_names = discover_all_tool_names(run_args)
    else:
        tool_names = discover_tool_names_from_test_file(test_data_path)

    if not tool_names:
        raise ValueError("Failed to recover tool names for the checkpoint")

    from compositional.model import FunctionCallingModel

    state_dict = checkpoint_payload["model_state_dict"]
    has_routing_probe = any(
        key.startswith(("routing_probe.", "gate_mlp.", "toolmix_head."))
        for key in state_dict
    )
    use_eoc = bool(run_args.get("use_eoc", False))
    use_gate = bool(run_args.get("use_gate", False))
    use_toolmix = bool(run_args.get("use_toolmix", False))
    enable_routing_probe = has_routing_probe or use_gate or use_toolmix
    if not use_eoc and enable_routing_probe:
        use_eoc = True
    if not use_gate and not use_toolmix and has_routing_probe:
        if any(key.startswith("toolmix_head.") for key in state_dict):
            use_toolmix = True
        elif any(key.startswith("gate_mlp.") for key in state_dict):
            use_gate = True
    gate_network = run_args.get("gate_network")
    if gate_network is None and has_routing_probe:
        if any(
            key.startswith(("routing_probe.0.", "gate_mlp.0.", "toolmix_head.0."))
            for key in state_dict
        ):
            gate_network = "mlp"
        else:
            gate_network = "linear"
    if gate_network is None:
        gate_network = "linear"

    model = FunctionCallingModel(
        model_name=model_name,
        num_tools=len(tool_names),
        tool_names=tool_names,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        decouple_embeddings=bool(run_args.get("decouple_embeddings", False)),
        lora_config=resolve_lora_config(run_args),
        enable_routing_probe=enable_routing_probe,
        use_toolmix=use_toolmix,
        use_eoc=use_eoc,
        use_gate=use_gate,
        gate_network=gate_network,
        gate_threshold=float(run_args.get("gate_threshold", 0.5)),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    with test_data_path.open("r", encoding="utf-8") as f:
        test_samples = json.load(f)

    return model, tokenizer, checkpoint_payload, test_samples, run_args


def get_generation_core_model(model: Any) -> Any:
    return model.model


def get_final_norm_module(core_model: Any) -> Optional[torch.nn.Module]:
    if hasattr(core_model, "get_base_model"):
        base_model = core_model.get_base_model()
    else:
        base_model = core_model

    if hasattr(base_model, "model") and hasattr(base_model.model, "norm"):
        return base_model.model.norm
    return None


def get_lm_head_module(core_model: Any) -> torch.nn.Module:
    if hasattr(core_model, "lm_head"):
        return core_model.lm_head
    if hasattr(core_model, "get_output_embeddings"):
        output_embeddings = core_model.get_output_embeddings()
        if output_embeddings is not None:
            return output_embeddings
    raise ValueError("Unable to locate lm_head / output embeddings on the model")


def prepare_layer_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    final_norm_module: Optional[torch.nn.Module],
) -> torch.Tensor:
    if len(hidden_states) < 2:
        raise ValueError("Expected hidden_states to include embeddings and layer outputs")

    layer_outputs = list(hidden_states[1:])
    if final_norm_module is None or len(layer_outputs) <= 1:
        return torch.stack([tensor[0, -1, :] for tensor in layer_outputs], dim=0)

    intermediate_layers = layer_outputs[:-1]
    final_layer = layer_outputs[-1][0, -1, :].unsqueeze(0)
    if intermediate_layers:
        intermediate_tensor = torch.stack([tensor[0, -1, :] for tensor in intermediate_layers], dim=0)
        intermediate_tensor = final_norm_module(intermediate_tensor)
        return torch.cat([intermediate_tensor, final_layer], dim=0)
    return final_layer


def analyze_hidden_states_with_logit_lens(
    model: Any,
    hidden_states: Sequence[torch.Tensor],
    final_norm_module: Optional[torch.nn.Module],
    lm_head_module: torch.nn.Module,
) -> Tuple[List[float], Dict[str, float]]:
    # Final-JS rule:
    # project every layer's hidden state with the model's own unembedding / lm_head,
    # then compare each layer distribution only against the final layer distribution.
    layer_hidden_states = prepare_layer_hidden_states(hidden_states, final_norm_module)
    logits_by_layer = lm_head_module(layer_hidden_states)
    js_curve = compute_js_divergence_against_final(logits_by_layer).detach().cpu().tolist()
    return js_curve, compute_final_layer_js_metrics(js_curve)


def sample_next_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    if do_sample:
        if temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True")
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = 0
            logits = logits.clone()
            for batch_index in range(logits.size(0)):
                indices_to_remove = sorted_indices[batch_index][remove_mask[batch_index]]
                logits[batch_index, indices_to_remove] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.argmax(logits, dim=-1)


def build_token_record(
    tokenizer: AutoTokenizer,
    token_id: int,
    token_position: int,
    category: str,
    js_curve: Sequence[float],
    js_metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "position": token_position,
        "token_id": int(token_id),
        "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
        "category": category,
        "js_curve": [float(value) for value in js_curve],
        **{key: float(value) for key, value in js_metrics.items()},
    }


def run_online_generation_capture(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> GenerationCapture:
    core_model = get_generation_core_model(model)
    lm_head_module = get_lm_head_module(core_model)
    final_norm_module = get_final_norm_module(core_model)
    eot_token_ids = tokenizer("<|eot_id|>", add_special_tokens=False)["input_ids"]
    tool_token_ids = set(model.tool_reserved_token_ids)

    input_ids = prompt_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    token_records: List[Dict[str, Any]] = []
    generated_token_ids: List[int] = []

    with torch.no_grad():
        outputs = core_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        next_logits = outputs.logits[:, -1, :].clone()
        past_key_values = outputs.past_key_values

        prompt_js_curve, prompt_js_metrics = analyze_hidden_states_with_logit_lens(
            model,
            outputs.hidden_states,
            final_norm_module,
            lm_head_module,
        )
        last_input_record = build_token_record(
            tokenizer=tokenizer,
            token_id=int(input_ids[0, -1].item()),
            token_position=-1,
            category="last_input",
            js_curve=prompt_js_curve,
            js_metrics=prompt_js_metrics,
        )
        token_records.append(last_input_record)

        finished = False
        step = 0
        previous_generated_token_id: Optional[int] = None

        while step < max_new_tokens and not finished:
            logits_to_decode = next_logits.clone()

            if model.use_gate:
                gate_context = previous_generated_token_id == model.eoc_token_id if step > 0 else True
                if gate_context and previous_generated_token_id != tokenizer.eos_token_id:
                    last_hidden_states = outputs.hidden_states[-1][:, -1, :]
                    gate_scores = model._get_gate_scores(last_hidden_states)
                    gate_positive = torch.sigmoid(gate_scores) >= model.gate_threshold
                    if gate_positive.item():
                        logits_to_decode = model.mask_logits_to_tool_tokens(logits_to_decode)

            sampled_token = sample_next_tokens(
                logits=logits_to_decode,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            sampled_token_id = int(sampled_token.item())

            if sampled_token_id == tokenizer.eos_token_id:
                generated_token_ids.append(sampled_token_id)
                finished = True
                break

            generated_token_ids.append(sampled_token_id)

            step_input_ids = sampled_token.unsqueeze(0)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            outputs = core_model(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values

            category = classify_generated_token(
                token_id=sampled_token_id,
                position=step,
                last_input_position=-1,
                tool_token_ids=tool_token_ids,
                eoc_token_id=model.eoc_token_id,
                eot_token_ids=eot_token_ids,
            )
            if category is not None:
                js_curve, js_metrics = analyze_hidden_states_with_logit_lens(
                    model,
                    outputs.hidden_states,
                    final_norm_module,
                    lm_head_module,
                )
                token_records.append(
                    build_token_record(
                        tokenizer=tokenizer,
                        token_id=sampled_token_id,
                        token_position=step,
                        category=category,
                        js_curve=js_curve,
                        js_metrics=js_metrics,
                    )
                )

            previous_generated_token_id = sampled_token_id
            step += 1

    full_sequence = torch.tensor(
        [prompt_ids.tolist() + generated_token_ids],
        dtype=torch.long,
        device=device,
    )
    prompt_batch = prompt_ids.unsqueeze(0).to(device)
    parsed = model._parse_generated_sequences(full_sequence, prompt_batch, tokenizer)[0]

    return GenerationCapture(
        generated_token_ids=generated_token_ids,
        token_records=token_records,
        full_generated_tensor=full_sequence,
        parsed_result=parsed,
    )


def plot_single_sample(
    sample_id: str,
    correctness_label: str,
    token_records: Sequence[Dict[str, Any]],
    category_summaries: Dict[str, Dict[str, Any]],
    expected_tool_markers: Optional[Sequence[Dict[str, Any]]],
    output_path: Path,
) -> None:
    if not token_records:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(14, len(token_records) * 0.22 + 6), 5.5),
        gridspec_kw={"width_ratios": [2.7, 1.3]},
    )
    left_ax, right_ax = axes

    for category in CATEGORY_ORDER:
        category_records = [record for record in token_records if record["category"] == category]
        if not category_records:
            continue
        x_values = [record["position"] for record in category_records]
        y_values = [record["js_mean"] for record in category_records]
        y_errors = [record["js_std"] for record in category_records]
        left_ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            fmt="o",
            capsize=2,
            markersize=4.5,
            color=CATEGORY_COLORS[category],
            ecolor=CATEGORY_COLORS[category],
            alpha=0.9,
            label=category,
        )
        if category in SPECIAL_LABELS:
            for x_value, y_value in zip(x_values, y_values):
                left_ax.annotate(
                    SPECIAL_LABELS[category],
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=CATEGORY_COLORS[category],
                )

    if correctness_label == "incorrect" and expected_tool_markers:
        ymax = max(record["js_mean"] + record["js_std"] for record in token_records)
        for marker in expected_tool_markers:
            x_value = marker["position"]
            left_ax.axvline(x=x_value, color="#AA3377", linestyle="--", linewidth=0.9, alpha=0.45)
            left_ax.annotate(
                f"<gold:{marker['tool_name']}>",
                (x_value, ymax),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=7,
                color="#AA3377",
            )

    left_ax.set_title("Token JS Mean +/- Std")
    left_ax.set_xlabel("token index (-1 = last input)")
    left_ax.set_ylabel("js_mean")
    left_ax.grid(alpha=0.25, linestyle="--")
    left_ax.legend(frameon=False, fontsize=8)

    category_positions = np.arange(len(CATEGORY_ORDER))
    bar_heights = []
    bar_errors = []
    for category in CATEGORY_ORDER:
        summary = category_summaries.get(category, {})
        mean_value = summary.get("js_mean_mean")
        variance_value = summary.get("js_mean_var")
        bar_heights.append(0.0 if mean_value is None else mean_value)
        bar_errors.append(0.0 if variance_value is None else math.sqrt(max(variance_value, 0.0)))

    right_ax.bar(
        category_positions,
        bar_heights,
        yerr=bar_errors,
        color=[CATEGORY_COLORS[category] for category in CATEGORY_ORDER],
        alpha=0.85,
        capsize=4,
    )
    right_ax.set_xticks(category_positions)
    right_ax.set_xticklabels(CATEGORY_ORDER, rotation=20)
    right_ax.set_title("Category Mean JS")
    right_ax.set_ylabel("mean(js_mean)")
    right_ax.grid(alpha=0.2, linestyle="--", axis="y")

    summary_lines = []
    for category in CATEGORY_ORDER:
        summary = category_summaries.get(category, {})
        summary_lines.append(
            f"{category}: n={summary.get('count', 0)} "
            f"f-s={_format_metric(summary.get('first_minus_second_mean'))} "
            f"s2={_format_metric(summary.get('second_half_ratio_mean'))}"
        )
    right_ax.text(
        1.03,
        0.98,
        "\n".join(summary_lines),
        transform=right_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#dddddd"},
    )

    # Plot semantics:
    # left: per-token js_mean with js_std error bars, colored by token category.
    # right: per-category aggregate js_mean comparison with dispersion as error bars.
    fig.suptitle(f"{sample_id} | {correctness_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_metric(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.3f}"


def flatten_sample_summary(sample_record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "sample_id": sample_record["sample_id"],
        "correctness": sample_record["correctness"],
        "correct": int(sample_record["correctness"] == "correct"),
        "tool_count_predicted": len(sample_record["predicted_tools"]),
        "tool_count_expected": len(sample_record["expected_tools"]),
    }
    for category in CATEGORY_ORDER:
        summary = sample_record["category_summaries"].get(category, {"count": 0})
        flatten_dict(category, summary, flat)
    return flat


def aggregate_global_summary(sample_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_tokens: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "all": {category: [] for category in CATEGORY_ORDER},
        "correct": {category: [] for category in CATEGORY_ORDER},
        "incorrect": {category: [] for category in CATEGORY_ORDER},
    }

    for sample_record in sample_records:
        correctness_key = "correct" if sample_record["correctness"] == "correct" else "incorrect"
        for token_record in sample_record["token_records"]:
            category = token_record["category"]
            if category not in CATEGORY_ORDER:
                continue
            grouped_tokens["all"][category].append(token_record)
            grouped_tokens[correctness_key][category].append(token_record)

    summary: Dict[str, Any] = {}
    for subset_name, category_payload in grouped_tokens.items():
        summary[subset_name] = {}
        for category, token_records in category_payload.items():
            summary[subset_name][category] = summarize_category_metrics(token_records)
    return summary


def plot_global_comparison(global_summary: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(CATEGORY_ORDER))
    width = 0.34

    for offset, subset_name, color in [(-width / 2, "correct", "#2E8B57"), (width / 2, "incorrect", "#B22222")]:
        means = []
        errors = []
        for category in CATEGORY_ORDER:
            category_summary = global_summary.get(subset_name, {}).get(category, {})
            mean_value = category_summary.get("js_mean_mean")
            variance_value = category_summary.get("js_mean_var")
            means.append(0.0 if mean_value is None else mean_value)
            errors.append(0.0 if variance_value is None else math.sqrt(max(variance_value, 0.0)))
        ax.bar(positions + offset, means, width=width, label=subset_name, color=color, alpha=0.82, yerr=errors, capsize=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(CATEGORY_ORDER, rotation=15)
    ax.set_ylabel("mean(js_mean)")
    ax.set_title("Correct vs Incorrect")
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_sample_record(
    sample_index: int,
    sample: Dict[str, Any],
    capture: GenerationCapture,
    correctness_payload: Dict[str, Any],
    expected_tool_markers: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    predicted_tools = [item["tool_name"] for item in capture.parsed_result.get("predicted_tools", [])]
    predicted_calls = capture.parsed_result.get("function_calls", [])
    expected_tools = sample.get("tools", [])
    expected_calls = sample.get("function_calls", [])

    category_groups = {category: [] for category in CATEGORY_ORDER}
    for token_record in capture.token_records:
        category_groups[token_record["category"]].append(token_record)

    category_summaries = {
        category: summarize_category_metrics(category_groups[category]) if category_groups[category] else {"count": 0}
        for category in CATEGORY_ORDER
    }

    return {
        "sample_id": f"sample_{sample_index:05d}",
        "sample_index": sample_index,
        "user_input": sample["user_input"],
        "correctness": "correct" if correctness_payload["correct"] else "incorrect",
        "predicted_tools": predicted_tools,
        "predicted_function_calls": predicted_calls,
        "expected_tools": expected_tools,
        "expected_function_calls": expected_calls,
        "full_generated_sequence": capture.parsed_result.get("full_generated_sequence", ""),
        "correctness_details": correctness_payload,
        "expected_tool_markers": list(expected_tool_markers),
        "token_records": capture.token_records,
        "token_counts": {category: len(category_groups[category]) for category in CATEGORY_ORDER},
        "category_summaries": category_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a trained compositional checkpoint with online JS analysis")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--model_name", type=str, default=None, help="Optional base model/tokenizer path override")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to the test JSON data")
    parser.add_argument("--run_config_path", type=str, default=None, help="Optional explicit run_config.json path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: compositional/runs/js_explore/<timestamp>)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to analyze")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to process per outer loop chunk")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--dtype", type=str, default=None, help="Override model dtype")
    parser.add_argument("--max_plot_samples_per_group", type=int, default=20, help="Maximum number of plots for correct and incorrect groups")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    run_config_path = find_run_config_path(
        checkpoint_path=checkpoint_path,
        explicit_run_config_path=Path(args.run_config_path).resolve() if args.run_config_path else None,
    )
    run_args = None
    if run_config_path is not None and run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as f:
            run_args = json.load(f).get("args", {})
    test_data_path = resolve_test_data_path(checkpoint_path, args.test_data_path, run_args)
    output_dir = derive_output_dir(checkpoint_path, args.output_dir)
    logger = setup_logger(output_dir)

    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Run config: %s", run_config_path)
    logger.info("Test data: %s", test_data_path)
    logger.info("Output dir: %s", output_dir)

    model, tokenizer, checkpoint_payload, test_samples, run_args = load_checkpoint_bundle(
        checkpoint_path=checkpoint_path,
        run_config_path=run_config_path,
        test_data_path=test_data_path,
        device=args.device,
        dtype_name=args.dtype,
        model_name_override=args.model_name,
        logger=logger,
    )

    if args.max_samples is not None:
        test_samples = test_samples[: args.max_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "analysis_config.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "model_name": args.model_name or run_args.get("model_name"),
            "run_config_path": str(run_config_path) if run_config_path else None,
            "test_data_path": str(test_data_path),
            "output_dir": str(output_dir),
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "device": args.device,
            "dtype": args.dtype or run_args.get("dtype"),
            "seed": args.seed,
            "checkpoint_round": checkpoint_payload.get("round"),
            "checkpoint_tools": checkpoint_payload.get("tools"),
        },
    )

    sample_records: List[Dict[str, Any]] = []
    plotted_counts = {"correct": 0, "incorrect": 0}
    total_samples = len(test_samples)
    batch_size = max(1, int(args.batch_size))

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(total_samples, batch_start + batch_size)
        logger.info("Processing samples %d-%d / %d", batch_start, batch_end - 1, total_samples)

        for sample_index in range(batch_start, batch_end):
            sample = test_samples[sample_index]
            prompt_ids = encode_eval_prompt(tokenizer, sample["user_input"])

            capture = run_online_generation_capture(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
            predicted_tools = [item["tool_name"] for item in capture.parsed_result.get("predicted_tools", [])]
            predicted_calls = capture.parsed_result.get("function_calls", [])
            correctness_payload = evaluate_complete_correctness(
                predicted_tools=predicted_tools,
                predicted_calls=predicted_calls,
                expected_tools=sample.get("tools", []),
                expected_calls=sample.get("function_calls", []),
            )
            expected_tool_markers = compute_expected_tool_markers(
                sample=sample,
                tokenizer=tokenizer,
                model=model,
            )
            sample_record = build_sample_record(
                sample_index=sample_index,
                sample=sample,
                capture=capture,
                correctness_payload=correctness_payload,
                expected_tool_markers=expected_tool_markers,
            )
            sample_records.append(sample_record)

            correctness_group = sample_record["correctness"]
            if plotted_counts[correctness_group] < args.max_plot_samples_per_group:
                plot_single_sample(
                    sample_id=sample_record["sample_id"],
                    correctness_label=correctness_group,
                    token_records=sample_record["token_records"],
                    category_summaries=sample_record["category_summaries"],
                    expected_tool_markers=sample_record["expected_tool_markers"],
                    output_path=output_dir / "plots" / correctness_group / f"{sample_record['sample_id']}.png",
                )
                plotted_counts[correctness_group] += 1

    global_summary = aggregate_global_summary(sample_records)
    plot_global_comparison(global_summary, output_dir / "plots" / "global_correct_vs_incorrect_js_mean.png")

    flattened_rows = [flatten_sample_summary(sample_record) for sample_record in sample_records]
    correct_records = [record for record in sample_records if record["correctness"] == "correct"]
    incorrect_records = [record for record in sample_records if record["correctness"] == "incorrect"]

    write_jsonl(output_dir / "samples" / "all_samples.jsonl", sample_records)
    write_jsonl(output_dir / "samples" / "correct" / "samples.jsonl", correct_records)
    write_jsonl(output_dir / "samples" / "incorrect" / "samples.jsonl", incorrect_records)
    write_csv(output_dir / "samples" / "all_samples.csv", flattened_rows)
    write_csv(output_dir / "samples" / "correct" / "samples.csv", [flatten_sample_summary(record) for record in correct_records])
    write_csv(output_dir / "samples" / "incorrect" / "samples.csv", [flatten_sample_summary(record) for record in incorrect_records])
    write_json(output_dir / "summaries" / "global_summary.json", global_summary)

    logger.info("Finished %d samples", len(sample_records))
    logger.info("Correct: %d | Incorrect: %d", len(correct_records), len(incorrect_records))
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
