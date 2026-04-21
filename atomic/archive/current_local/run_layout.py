import json
import os
import re
import shlex
import sys
from datetime import datetime


ATOMIC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(ATOMIC_DIR, "runs")


def normalize_label(value):
    if value is None:
        return ""
    label = os.path.basename(str(value).rstrip("/")) or str(value)
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
    return label or "value"


def resolve_timestamp(provided=None):
    if provided:
        return provided
    env_timestamp = os.environ.get("ATOMIC_RUN_TIMESTAMP")
    if env_timestamp:
        return env_timestamp
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_run_context(
    experiment_name,
    model_name=None,
    num_tasks=None,
    run_root_dir=None,
    run_name=None,
    run_tag=None,
    timestamp=None,
):
    timestamp = resolve_timestamp(timestamp)
    run_root_dir = os.path.abspath(
        run_root_dir or os.environ.get("ATOMIC_RUNS_DIR") or DEFAULT_RUNS_DIR
    )

    if run_name is None:
        run_name = os.environ.get("ATOMIC_RUN_NAME")

    if not run_name:
        parts = [normalize_label(experiment_name)]
        if model_name:
            parts.append(normalize_label(model_name))
        if num_tasks is not None:
            parts.append(f"{num_tasks}tasks")
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
            "ATOMIC_RUN_NAME",
            "ATOMIC_RUNS_DIR",
            "ATOMIC_RUN_TIMESTAMP",
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
