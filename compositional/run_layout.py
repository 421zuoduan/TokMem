import json
import os
import re
import shlex
import sys
from datetime import datetime


COMPOSITIONAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(COMPOSITIONAL_DIR, "runs")


def normalize_label(value):
    if value is None:
        return ""
    label = os.path.basename(str(value).rstrip("/")) or str(value)
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
    return label or "value"


def resolve_timestamp(provided=None):
    if provided:
        return provided
    env_timestamp = os.environ.get("COMPOSITIONAL_RUN_TIMESTAMP")
    if env_timestamp:
        return env_timestamp
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_run_context(
    experiment_name,
    model_name=None,
    run_root_dir=None,
    run_name=None,
    run_tag=None,
    timestamp=None,
):
    timestamp = resolve_timestamp(timestamp)
    run_root_dir = os.path.abspath(
        run_root_dir or os.environ.get("COMPOSITIONAL_RUNS_DIR") or DEFAULT_RUNS_DIR
    )

    if run_name is None:
        run_name = os.environ.get("COMPOSITIONAL_RUN_NAME")

    if not run_name:
        parts = [normalize_label(experiment_name)]
        if model_name:
            parts.append(normalize_label(model_name))
        if run_tag:
            parts.append(normalize_label(run_tag))
        parts.append(timestamp)
        run_name = "_".join(parts)

    run_dir = os.path.join(run_root_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return {
        "timestamp": timestamp,
        "run_name": run_name,
        "run_root_dir": run_root_dir,
        "run_dir": run_dir,
    }


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _summary_loss_value(results, key, enabled):
    if not enabled:
        return None
    return results.get(key)


def build_training_summary_payload(run_name, all_results, experiment_type="tokmem_sequential"):
    rounds = []
    for result in all_results:
        metrics = result.get("results", {})
        rounds.append(
            {
                "round": result["round"],
                "tools": result["tools"],
                "epochs": result["epochs"],
                "avg_total_loss": metrics.get("avg_total_loss", result.get("avg_loss")),
                "avg_ar_loss": metrics.get("avg_ar_loss"),
                "avg_eoc_loss": _summary_loss_value(
                    metrics,
                    "avg_eoc_loss",
                    metrics.get("use_eoc_loss", False),
                ),
                "avg_tool_loss": _summary_loss_value(
                    metrics,
                    "avg_tool_loss",
                    metrics.get("use_tool_loss", False),
                ),
                "avg_gate_loss": _summary_loss_value(
                    metrics,
                    "avg_gate_loss",
                    metrics.get("use_gate", False),
                ),
                "avg_gate_prob": (
                    metrics.get("avg_gate_prob")
                    if metrics.get("use_gate", False)
                    else None
                ),
                "avg_toolmix_aux_loss": _summary_loss_value(
                    metrics,
                    "avg_toolmix_aux_loss",
                    metrics.get("use_toolmix", False),
                ),
                "avg_logit_bias_loss": _summary_loss_value(
                    metrics,
                    "avg_logit_bias_loss",
                    metrics.get("use_logit_bias", False),
                ),
                "avg_toolmix_prob": (
                    metrics.get("avg_toolmix_prob")
                    if metrics.get("use_toolmix", False)
                    else None
                ),
                "toolmix_alpha": (
                    metrics.get("toolmix_alpha")
                    if metrics.get("use_toolmix", False)
                    else None
                ),
            }
        )

    return {
        "experiment_type": experiment_type,
        "run_name": run_name,
        "rounds": rounds,
    }


def build_command_string():
    return shlex.join(sys.argv)


def build_run_config(args_dict, run_context, extra=None):
    tracked_env = {
        key: os.environ.get(key)
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_LAUNCH_BLOCKING",
            "TORCH_SHOW_CPP_STACKTRACES",
            "NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_DEBUG",
            "COMPOSITIONAL_RUN_NAME",
            "COMPOSITIONAL_RUNS_DIR",
            "COMPOSITIONAL_RUN_TIMESTAMP",
            "TOKMEM_COMPOSITIONAL_MODEL",
        ]
        if os.environ.get(key) is not None
    }

    payload = {
        "run_name": run_context["run_name"],
        "run_dir": run_context["run_dir"],
        "timestamp": run_context["timestamp"],
        "command": build_command_string(),
        "args": args_dict,
        "environment": tracked_env,
    }
    if extra:
        payload.update(extra)
    return payload


def artifact_path(run_context, filename):
    return os.path.join(run_context["run_dir"], filename)
