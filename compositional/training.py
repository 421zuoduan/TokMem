import inspect
import math
import os
import tempfile
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:
    FSDP = None


LOSS_METRIC_ORDER = (
    "total_loss",
    "ar_loss",
    "logit_bias_loss",
)


def _resolve_mode_flags(model, use_eoc=None, use_js_trunc=None):
    resolved_use_eoc = bool(getattr(model, "use_eoc", False) if use_eoc is None else use_eoc)
    resolved_use_js_trunc = bool(
        getattr(model, "use_js_trunc", False) if use_js_trunc is None else use_js_trunc
    )

    if resolved_use_js_trunc and not resolved_use_eoc:
        raise ValueError("use_js_trunc=True requires use_eoc=True")

    return resolved_use_eoc, resolved_use_js_trunc


def _call_method_with_supported_kwargs(method, **kwargs):
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)

    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return method(**supported_kwargs)
def _metric_value_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _unwrap_model(model, accelerator=None):
    return accelerator.unwrap_model(model) if accelerator is not None else model


def _maybe_move_batch(batch, device, accelerator=None):
    if accelerator is not None:
        return batch["input_ids"], batch["attention_mask"], batch.get("labels")
    labels = batch.get("labels")
    if labels is None:
        return batch["input_ids"].to(device), batch["attention_mask"].to(device), None
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        labels.to(device),
    )


def _fsdp_generation_context(model, accelerator=None):
    if FSDP is None:
        return nullcontext()
    if not isinstance(model, FSDP):
        return nullcontext()
    if accelerator is not None and getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return nullcontext()
    fsdp_plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None) if accelerator is not None else None
    sharding_strategy = getattr(fsdp_plugin, "sharding_strategy", None)
    strategy_name = getattr(sharding_strategy, "name", str(sharding_strategy))
    if strategy_name == "NO_SHARD":
        return nullcontext()
    return FSDP.summon_full_params(model, recurse=True, writeback=False)


def _is_sharded_fsdp_model(model, accelerator=None):
    if FSDP is None:
        return False
    if not isinstance(model, FSDP):
        return False
    if accelerator is not None and getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return False

    fsdp_plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None) if accelerator is not None else None
    sharding_strategy = getattr(fsdp_plugin, "sharding_strategy", None)
    if sharding_strategy is None:
        sharding_strategy = getattr(model, "sharding_strategy", None)

    strategy_name = getattr(sharding_strategy, "name", str(sharding_strategy))
    return strategy_name not in {None, "None", "NO_SHARD"}


def _distributed_any(local_flag, accelerator=None, device=None):
    if not local_flag and not (dist.is_available() and dist.is_initialized()):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_flag)

    if device is None:
        device = accelerator.device if accelerator is not None else None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flag_tensor = torch.tensor([1 if local_flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
    return bool(flag_tensor.item())


def _sync_fsdp_ignored_module_gradients(model, accelerator=None):
    if accelerator is None:
        return
    if getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return
    if not dist.is_available() or not dist.is_initialized():
        return

    base_model = _unwrap_model(model, accelerator=accelerator)
    if not hasattr(base_model, "get_fsdp_trainable_modules"):
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    for module in base_model.get_fsdp_trainable_modules():
        for param in module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def _sync_skip_flags(local_flags, accelerator=None, device=None):
    resolved_flags = {name: bool(value) for name, value in local_flags.items()}
    if accelerator is None or not resolved_flags:
        return resolved_flags

    if device is None:
        device = accelerator.device

    flag_names = list(resolved_flags.keys())
    flag_tensor = torch.tensor(
        [1.0 if resolved_flags[name] else 0.0 for name in flag_names],
        device=device,
    )
    reduced_flags = accelerator.reduce(flag_tensor, reduction="sum")
    return {
        name: bool(reduced_flags[idx].item() > 0)
        for idx, name in enumerate(flag_names)
    }



def _build_loss_metrics(total_loss, ar_loss, extra_loss_metrics=None):
    metrics = {
        "total_loss": total_loss,
        "ar_loss": ar_loss,
    }
    if extra_loss_metrics:
        for metric_name, metric_value in extra_loss_metrics.items():
            if metric_value is not None:
                metrics[metric_name] = metric_value
    return metrics


def _ordered_metric_items(metrics):
    emitted = set()
    for metric_name in LOSS_METRIC_ORDER:
        if metric_name in metrics:
            emitted.add(metric_name)
            yield metric_name, metrics[metric_name]
    for metric_name, metric_value in metrics.items():
        if metric_name not in emitted:
            yield metric_name, metric_value


def _average_metrics(metric_totals, count):
    if count <= 0:
        return {}
    return {
        metric_name: metric_total / count
        for metric_name, metric_total in metric_totals.items()
    }


def _append_loss_plot_record(plot_history, step, round_num, metrics):
    if plot_history is None:
        return

    record = {
        "step": int(step),
        "round": int(round_num) if round_num is not None else None,
    }
    for metric_name, metric_value in _ordered_metric_items(metrics):
        record[metric_name] = _metric_value_to_float(metric_value)
    plot_history.setdefault("loss_steps", []).append(record)


def _append_lr_plot_records(plot_history, step, round_num, optimizer, lr_values):
    if plot_history is None:
        return

    records = plot_history.setdefault("lr_steps", [])
    for lr_idx, lr_value in enumerate(lr_values):
        group_name = optimizer.param_groups[lr_idx].get("name", f"group_{lr_idx}")
        records.append(
            {
                "step": int(step),
                "round": int(round_num) if round_num is not None else None,
                "group": group_name,
                "lr": float(lr_value),
            }
        )


def _draw_round_boundaries(ax, round_boundaries):
    if not round_boundaries:
        return

    for boundary in round_boundaries[:-1]:
        boundary_step = boundary.get("step")
        if boundary_step is None:
            continue
        ax.axvline(boundary_step, color="gray", linestyle="--", linewidth=1.0, alpha=0.35)


def _moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values

    window_size = max(1, min(int(window_size), values.size))
    if window_size == 1:
        return values.copy()

    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    prefix = np.array(
        [values[: idx + 1].mean() for idx in range(window_size - 1)],
        dtype=np.float64,
    )
    return np.concatenate([prefix, smoothed])


def _trend_window(num_points):
    if num_points <= 1:
        return 1
    return min(num_points, max(20, min(100, num_points // 40 or 1)))


def _build_loss_plot_series(loss_steps):
    series = {}
    for metric_name in LOSS_METRIC_ORDER:
        points = [(record["step"], record[metric_name]) for record in loss_steps if metric_name in record]
        if not points:
            continue
        x_values = np.asarray([point[0] for point in points], dtype=np.int64)
        y_values = np.asarray([point[1] for point in points], dtype=np.float64)
        finite_mask = np.isfinite(y_values)
        if not finite_mask.any():
            continue
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]
        smooth_values = _moving_average(y_values, _trend_window(len(y_values)))
        series[metric_name] = {
            "x": x_values,
            "y": y_values,
            "smooth": smooth_values,
        }
    return series


def _loss_ylim(loss_series):
    preferred_metrics = [
        metric_name for metric_name in ("total_loss", "ar_loss") if metric_name in loss_series
    ]
    metrics_for_scale = preferred_metrics or list(loss_series.keys())
    if not metrics_for_scale:
        return None

    scale_values = np.concatenate([loss_series[metric_name]["smooth"] for metric_name in metrics_for_scale])
    scale_values = scale_values[np.isfinite(scale_values)]
    if scale_values.size == 0:
        return None

    upper_bound = float(np.percentile(scale_values, 99.5))
    initial_window = min(50, scale_values.size)
    initial_peak = float(np.max(scale_values[:initial_window]))
    overall_peak = float(np.max(scale_values))
    upper_bound = max(upper_bound, initial_peak * 1.05)
    if upper_bound <= 0:
        upper_bound = overall_peak if overall_peak > 0 else 1.0

    if upper_bound >= overall_peak * 0.98:
        return None
    return 0.0, upper_bound


def _plot_loss_trends(ax, loss_steps):
    loss_series = _build_loss_plot_series(loss_steps)
    if not loss_series:
        return

    line_styles = {
        "total_loss": {"color": "#1f77b4", "linewidth": 2.4, "alpha": 0.98, "linestyle": "-"},
        "ar_loss": {"color": "#ff7f0e", "linewidth": 2.0, "alpha": 0.92, "linestyle": "-"},
        "logit_bias_loss": {"color": "#2ca02c", "linewidth": 1.5, "alpha": 0.8, "linestyle": "--"},
    }

    y_limits = _loss_ylim(loss_series)
    clip_level = y_limits[1] if y_limits is not None else None

    for metric_name in LOSS_METRIC_ORDER:
        metric_series = loss_series.get(metric_name)
        if metric_series is None:
            continue
        style = line_styles.get(metric_name, {"linewidth": 1.5, "alpha": 0.8, "linestyle": "-"})
        smooth_values = metric_series["smooth"]
        display_values = np.minimum(smooth_values, clip_level) if clip_level is not None else smooth_values
        label = metric_name.replace("_", " ")
        ax.plot(metric_series["x"], display_values, label=label, **style)

        if clip_level is not None:
            spike_mask = smooth_values > clip_level
            if spike_mask.any():
                ax.scatter(
                    metric_series["x"][spike_mask],
                    np.full(int(spike_mask.sum()), clip_level),
                    color=style.get("color"),
                    alpha=0.35,
                    marker="o",
                    s=10,
                    linewidths=0,
                )

    if y_limits is not None:
        ax.set_ylim(*y_limits)


def save_training_plot_images(plot_history, loss_plot_path, lr_plot_path, run_name):
    loss_steps = plot_history.get("loss_steps", []) if plot_history else []
    lr_steps = plot_history.get("lr_steps", []) if plot_history else []
    round_boundaries = plot_history.get("round_boundaries", []) if plot_history else []

    if not loss_steps and not lr_steps:
        return []

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "tokmem-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Saving training plots requires matplotlib in the active environment."
        ) from exc

    saved_paths = []

    if loss_steps:
        loss_plot_path = Path(loss_plot_path)
        loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_loss_trends(ax, loss_steps)
        _draw_round_boundaries(ax, round_boundaries)
        ax.set_title(f"{run_name} Loss Trend")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_plot_path, dpi=200)
        plt.close(fig)
        saved_paths.append(str(loss_plot_path))

    if lr_steps:
        lr_plot_path = Path(lr_plot_path)
        lr_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        group_names = []
        for record in lr_steps:
            group_name = record["group"]
            if group_name not in group_names:
                group_names.append(group_name)

        for group_name in group_names:
            points = [(record["step"], record["lr"]) for record in lr_steps if record["group"] == group_name]
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.plot(x_values, y_values, label=group_name, linewidth=1.5)

        _draw_round_boundaries(ax, round_boundaries)
        ax.set_title(f"{run_name} Learning Rate Schedule")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(lr_plot_path, dpi=200)
        plt.close(fig)
        saved_paths.append(str(lr_plot_path))

    return saved_paths


def compute_tool_subset_targets(shift_labels, model):
    """Map shifted labels to tool indices for tool-only CE on the reserved-token subset."""
    tool_targets = torch.full_like(shift_labels, -100)
    token_id_to_tool_id = getattr(model, "token_id_to_tool_id", {})

    for token_id, tool_id in token_id_to_tool_id.items():
        tool_targets[shift_labels == token_id] = tool_id

    tool_mask = tool_targets != -100
    return tool_targets, tool_mask


def build_shift_supervision_masks(shift_labels, model, use_eoc=False):
    """Build post-truncation supervision masks from shifted labels."""
    valid_mask = shift_labels != -100
    tool_targets, tool_mask = compute_tool_subset_targets(shift_labels, model)

    eoc_token_id = getattr(model, "eoc_token_id", None) if use_eoc else None
    if eoc_token_id is not None:
        eoc_mask = valid_mask & (shift_labels == eoc_token_id)
    else:
        eoc_mask = torch.zeros_like(valid_mask)

    return {
        "valid_mask": valid_mask,
        "tool_mask": tool_mask,
        "tool_targets": tool_targets,
        "eoc_mask": eoc_mask,
    }


def gather_logit_bias_examples(hidden_states, labels, model, return_indices=False):
    """Collect boundary hidden states with gold tool labels for the detached tool-prior head."""
    eoc_token_id = getattr(model, "eoc_token_id", None)
    if eoc_token_id is None:
        empty_hidden = hidden_states.new_zeros((0, hidden_states.size(-1)))
        empty_targets = torch.zeros((0,), dtype=torch.long, device=hidden_states.device)
        empty_indices = torch.zeros((0,), dtype=torch.long, device=hidden_states.device)
        if return_indices:
            return empty_hidden, empty_targets, empty_indices, empty_indices, 0, 0
        return empty_hidden, empty_targets, 0, 0

    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    token_id_to_tool_id = getattr(model, "token_id_to_tool_id", {})

    prior_hidden_states = []
    prior_targets = []
    batch_indices = []
    time_indices = []
    initial_sites = 0
    eoc_sites = 0

    for batch_idx in range(labels.size(0)):
        valid_positions = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        first_valid_pos = int(valid_positions[0].item())
        if (
            first_valid_pos < shift_labels.size(1)
            and labels[batch_idx, first_valid_pos].item() == -100
        ):
            next_token_id = int(shift_labels[batch_idx, first_valid_pos].item())
            tool_id = token_id_to_tool_id.get(next_token_id)
            if tool_id is not None and first_valid_pos < hidden_states.size(1):
                prior_hidden_states.append(hidden_states[batch_idx, first_valid_pos])
                prior_targets.append(tool_id)
                batch_indices.append(batch_idx)
                time_indices.append(first_valid_pos)
                initial_sites += 1

        eoc_positions = torch.nonzero(
            (labels[batch_idx, :-1] == eoc_token_id) & valid_mask[batch_idx],
            as_tuple=False,
        ).flatten()
        for position in eoc_positions.tolist():
            next_token_id = int(shift_labels[batch_idx, position].item())
            tool_id = token_id_to_tool_id.get(next_token_id)
            if tool_id is None or position >= hidden_states.size(1):
                continue
            prior_hidden_states.append(hidden_states[batch_idx, position])
            prior_targets.append(tool_id)
            batch_indices.append(batch_idx)
            time_indices.append(position)
            eoc_sites += 1

    if not prior_hidden_states:
        empty_hidden = hidden_states.new_zeros((0, hidden_states.size(-1)))
        empty_targets = torch.zeros((0,), dtype=torch.long, device=hidden_states.device)
        empty_indices = torch.zeros((0,), dtype=torch.long, device=hidden_states.device)
        if return_indices:
            return empty_hidden, empty_targets, empty_indices, empty_indices, initial_sites, eoc_sites
        return empty_hidden, empty_targets, initial_sites, eoc_sites

    stacked_hidden_states = torch.stack(prior_hidden_states, dim=0)
    stacked_targets = torch.tensor(prior_targets, dtype=torch.long, device=hidden_states.device)
    stacked_batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=hidden_states.device)
    stacked_time_indices = torch.tensor(time_indices, dtype=torch.long, device=hidden_states.device)
    if return_indices:
        return (
            stacked_hidden_states,
            stacked_targets,
            stacked_batch_indices,
            stacked_time_indices,
            initial_sites,
            eoc_sites,
        )
    return stacked_hidden_states, stacked_targets, initial_sites, eoc_sites


def compute_logit_bias_loss(model, boundary_hidden_states, tool_targets):
    """Run the detached tool prior head and return logits with CE loss on gold tool ids."""
    tool_logits = model._get_logit_bias_scores(boundary_hidden_states.detach())
    tool_loss = F.cross_entropy(tool_logits, tool_targets)
    return tool_logits, tool_loss


def _forward_with_optional_hidden_states(
    model,
    input_ids,
    attention_mask,
    output_hidden_states=False,
    final_hidden_state_only=False,
    accelerator=None,
):
    base_model = _unwrap_model(model, accelerator=accelerator)
    if (
        output_hidden_states
        and final_hidden_state_only
        and hasattr(base_model, "forward_with_final_hidden_states")
    ):
        with _fsdp_generation_context(model, accelerator=accelerator):
            logits, final_hidden_states = base_model.forward_with_final_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return logits, final_hidden_states

    if hasattr(base_model, "forward_with_final_hidden_states"):
        with _fsdp_generation_context(model, accelerator=accelerator):
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=output_hidden_states,
            )
        if output_hidden_states:
            logits, hidden_states = outputs
            return logits, hidden_states
        return outputs, None

    forward_model = getattr(model, "model", model)
    with _fsdp_generation_context(model, accelerator=accelerator):
        outputs = forward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    hidden_states = None
    if output_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        hidden_states = outputs.hidden_states[-1]
    return logits, hidden_states


def _generate_results(
    model,
    tokenizer,
    user_tokens,
    user_mask,
    use_js_trunc=False,
    use_logit_bias=False,
    use_eoc=False,
    use_ground_truth_tools=False,
    ground_truth_tools=None,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.9,
    do_sample=False,
    accelerator=None,
):
    generation_model = _unwrap_model(model, accelerator=accelerator)
    candidate_methods = []
    if use_ground_truth_tools:
        candidate_methods.append("generate_with_ground_truth_tools")
    else:
        candidate_methods.append("generate_with_tool_prediction")

    generation_kwargs = {
        "user_tokens": user_tokens,
        "user_mask": user_mask,
        "tokenizer": tokenizer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "use_js_trunc": use_js_trunc,
        "use_logit_bias": use_logit_bias,
        "use_eoc": use_eoc,
    }
    if ground_truth_tools is not None:
        generation_kwargs["ground_truth_tools"] = ground_truth_tools

    for method_name in candidate_methods:
        method = getattr(generation_model, method_name, None)
        if callable(method):
            with _fsdp_generation_context(model, accelerator=accelerator):
                return _call_method_with_supported_kwargs(method, **generation_kwargs)

    raise AttributeError(
        f"Model {type(model).__name__} does not expose a compatible generation method "
        f"for use_js_trunc={use_js_trunc}, "
        f"use_logit_bias={use_logit_bias}, use_ground_truth_tools={use_ground_truth_tools}."
    )


def _empty_generation_result():
    """Return a safe empty prediction payload for a failed example."""
    return {
        "predicted_tools": [],
        "function_calls": [],
        "full_generated_sequence": "",
        "predicted_tool_id": None,
        "predicted_tool_name": "none",
        "function_call": "",
        "tool_token_used": None,
    }


def _generate_results_with_example_fallback(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    raw_examples,
    batch_idx,
    use_js_trunc,
    use_logit_bias,
    use_eoc,
    use_ground_truth_tools,
    max_new_tokens,
    accelerator=None,
):
    """Generate batch results, falling back to per-example decoding on batch failures."""
    batch_size = len(raw_examples)
    sharded_fsdp = _is_sharded_fsdp_model(model, accelerator=accelerator)

    if use_ground_truth_tools:
        batch_results = []
        for i in range(batch_size):
            example = raw_examples[i]
            expected_tools = example.get("tools", [example.get("tool_name", "unknown")])
            single_input = input_ids[i : i + 1]
            single_mask = attention_mask[i : i + 1]
            try:
                single_result = _generate_results(
                    model,
                    tokenizer,
                    single_input,
                    single_mask,
                    use_js_trunc=use_js_trunc,
                    use_logit_bias=use_logit_bias,
                    use_eoc=use_eoc,
                    use_ground_truth_tools=True,
                    ground_truth_tools=expected_tools,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                    accelerator=accelerator,
                )
                batch_results.extend(single_result)
            except Exception as exc:
                print(f"   Error generating example {i + 1} in batch {batch_idx + 1}: {str(exc)}")
                batch_results.append(_empty_generation_result())
        return batch_results

    batch_failure = None
    try:
        batch_results = _generate_results(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            use_js_trunc=use_js_trunc,
            use_logit_bias=use_logit_bias,
            use_eoc=use_eoc,
            use_ground_truth_tools=False,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=False,
            accelerator=accelerator,
        )
    except Exception as exc:
        batch_failure = exc
        print(f"   Error processing batch {batch_idx + 1}: {str(exc)}")

    if sharded_fsdp:
        any_rank_failed = _distributed_any(
            batch_failure is not None,
            accelerator=accelerator,
        )
        if any_rank_failed:
            failure_detail = str(batch_failure) if batch_failure is not None else "another rank failed batch generation"
            raise RuntimeError(
                f"Batch {batch_idx + 1} generation failed under sharded FSDP ({failure_detail}). "
                "Aborting evaluation to keep summon_full_params collectives aligned across ranks."
            ) from batch_failure

    if batch_failure is not None:
        print(f"   Falling back to per-example generation for batch {batch_idx + 1}")
        batch_results = []
        for i in range(batch_size):
            single_input = input_ids[i : i + 1]
            single_mask = attention_mask[i : i + 1]
            try:
                single_result = _generate_results(
                    model,
                    tokenizer,
                    single_input,
                    single_mask,
                    use_js_trunc=use_js_trunc,
                    use_logit_bias=use_logit_bias,
                    use_eoc=use_eoc,
                    use_ground_truth_tools=False,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                    accelerator=accelerator,
                )
                batch_results.extend(single_result)
            except Exception as single_exc:
                print(f"   Error generating example {i + 1} in batch {batch_idx + 1}: {str(single_exc)}")
                batch_results.append(_empty_generation_result())

    if len(batch_results) < batch_size:
        print(
            f"   Warning: batch {batch_idx + 1} produced {len(batch_results)} results for "
            f"{batch_size} examples; padding the remainder with empty predictions"
        )
        batch_results.extend(_empty_generation_result() for _ in range(batch_size - len(batch_results)))
    elif len(batch_results) > batch_size:
        print(
            f"   Warning: batch {batch_idx + 1} produced {len(batch_results)} results for "
            f"{batch_size} examples; truncating extras"
        )
        batch_results = batch_results[:batch_size]

    return batch_results


def create_native_optimizer(model, lr=0.01, lora_lr=None):
    """Create the optimizer for native token-memory training."""
    from torch.optim import AdamW

    if model.lora_config and lora_lr is not None:
        embedding_params, lora_params = model.get_trainable_parameters(separate_lora=True)
        known_param_ids = {id(param) for param in embedding_params + lora_params}
        extra_trainable_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in known_param_ids
        ]

        param_groups = [
            {"params": embedding_params, "lr": lr, "name": "embeddings", "weight_decay": 0.0},
            {"params": lora_params, "lr": lora_lr, "name": "lora", "weight_decay": 0.01},
        ]
        if extra_trainable_params:
            param_groups.append(
                {"params": extra_trainable_params, "lr": lr, "name": "auxiliary", "weight_decay": 0.0}
            )
        return AdamW(param_groups)

    embedding_params = model.get_trainable_parameters()
    known_param_ids = {id(param) for param in embedding_params}
    extra_trainable_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in known_param_ids
    ]
    param_groups = [{"params": embedding_params, "lr": lr, "weight_decay": 0.0, "name": "embeddings"}]
    if extra_trainable_params:
        param_groups.append(
            {"params": extra_trainable_params, "lr": lr, "weight_decay": 0.0, "name": "auxiliary"}
        )
    return AdamW(param_groups)


def save_native_checkpoint(model, checkpoint_path, payload, accelerator=None):
    """Save a round checkpoint while keeping FSDP full-param collectives matched across ranks."""
    should_write = accelerator is None or accelerator.is_main_process
    base_model = _unwrap_model(model, accelerator=accelerator)
    with _fsdp_generation_context(model, accelerator=accelerator):
        if not should_write:
            return None
        checkpoint_payload = dict(payload)
        checkpoint_payload["trainable_state"] = base_model.build_trainable_state_payload()
        torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def train_native_function_calling_model(
    model,
    dataloader,
    num_epochs=3,
    lr=0.01,
    gradient_accumulation_steps=1,
    device="cuda",
    lora_lr=None,
    optimizer=None,
    scheduler=None,
    active_tool_ids=None,
    renorm_active_rows=False,
    use_eoc=None,
    use_js_trunc=None,
    use_logit_bias=None,
    logit_bias_loss_weight=0.1,
    plot_history=None,
    plot_step_offset=0,
    plot_round=None,
    accelerator=None,
):
    """Train the native function calling model using reserved tokens."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    model.train()
    base_model = _unwrap_model(model, accelerator=accelerator)
    resolved_use_eoc, resolved_use_js_trunc = _resolve_mode_flags(
        base_model,
        use_eoc,
        use_js_trunc,
    )
    resolved_use_logit_bias = bool(
        getattr(base_model, "use_logit_bias", False) if use_logit_bias is None else use_logit_bias
    )
    if resolved_use_logit_bias and not resolved_use_eoc:
        raise ValueError("use_logit_bias=True requires use_eoc=True")

    if optimizer is None:
        if base_model.lora_config and lora_lr is not None:
            embedding_params, lora_params = base_model.get_trainable_parameters(separate_lora=True)
            known_param_ids = {id(param) for param in embedding_params + lora_params}
            extra_trainable_params = [
                param
                for param in base_model.parameters()
                if param.requires_grad and id(param) not in known_param_ids
            ]
            if extra_trainable_params:
                print(f"Found {len(extra_trainable_params)} additional trainable parameters")

            param_groups = [
                {"params": embedding_params, "lr": lr, "name": "embeddings", "weight_decay": 0.0},
                {"params": lora_params, "lr": lora_lr, "name": "lora", "weight_decay": 0.01},
            ]
            if extra_trainable_params:
                param_groups.append(
                    {"params": extra_trainable_params, "lr": lr, "name": "auxiliary", "weight_decay": 0.0}
                )
            optimizer = AdamW(param_groups)
            print(
                f"Using separate learning rates: embeddings={lr}, LoRA={lora_lr} "
                "(wd: emb=0.0, lora=0.01)"
            )
        else:
            embedding_params = base_model.get_trainable_parameters()
            known_param_ids = {id(param) for param in embedding_params}
            extra_trainable_params = [
                param
                for param in base_model.parameters()
                if param.requires_grad and id(param) not in known_param_ids
            ]
            param_groups = [{"params": embedding_params, "lr": lr, "weight_decay": 0.0, "name": "embeddings"}]
            if extra_trainable_params:
                print(f"Found {len(extra_trainable_params)} additional trainable parameters")
                param_groups.append(
                    {"params": extra_trainable_params, "lr": lr, "weight_decay": 0.0, "name": "auxiliary"}
                )
            optimizer = AdamW(param_groups)
            print(f"Using single learning rate: {lr} (wd: emb=0.0)")

    total_optimizer_steps = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps) * num_epochs)
    if scheduler is None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_optimizer_steps // 10,
            num_training_steps=total_optimizer_steps,
        )
    if accelerator is not None:
        optimizer = accelerator.prepare_optimizer(optimizer)
        scheduler = accelerator.prepare_scheduler(scheduler)

    should_log = accelerator is None or accelerator.is_main_process
    if should_log:
        print(f"Training for {num_epochs} epochs, {len(dataloader)} batches per epoch")
        print(f"Total optimizer steps: {total_optimizer_steps}")
        if base_model.lora_config and lora_lr is not None:
            print(f"Learning rates: embeddings={lr}, LoRA={lora_lr} (with linear schedule + warmup)")
        else:
            print(f"Learning rate: {lr} (with linear schedule + warmup)")
        print(f"Warmup steps: {total_optimizer_steps // 10}")
        print(
            "Mode: "
            f"use_eoc={resolved_use_eoc}, use_js_trunc={resolved_use_js_trunc}, "
            f"use_logit_bias={resolved_use_logit_bias}, "
            f"logit_bias_loss_weight={logit_bias_loss_weight}"
        )

    all_trainable_params = base_model.get_trainable_parameters()
    total_trainable = sum(param.numel() for param in all_trainable_params)

    if should_log:
        if base_model.decouple_embeddings:
            print("Training mode: Decoupled embeddings")
            print(
                "Trainable parameters: "
                f"{total_trainable:,} "
                f"(input: {base_model.trainable_tool_input_embeddings.numel():,}, "
                f"output: {base_model.trainable_tool_output_embeddings.numel():,})"
            )
        else:
            print("Training mode: Coupled embeddings")
            print(
                f"Trainable parameters: {total_trainable:,} "
                f"(shared: {base_model.trainable_tool_embeddings.numel():,})"
            )
        print(f"Tool token IDs to monitor: {base_model.reserved_token_ids}")
        print()

    total_loss_metrics = defaultdict(float)
    total_valid_positions = 0
    total_eoc_positions = 0
    total_tool_positions = 0
    total_logit_bias_positions = 0
    total_logit_bias_initial_positions = 0
    total_logit_bias_eoc_positions = 0
    successful_steps = 0
    optimizer_steps = 0
    step = 0

    window_batches = 0
    window_loss_metrics = defaultdict(float)
    window_valid_positions = 0
    window_eoc_positions = 0
    window_tool_positions = 0
    window_logit_bias_positions = 0

    work_device = accelerator.device if accelerator is not None else torch.device(device)
    zero = torch.tensor(0.0, device=work_device)
    has_logit_bias_head = hasattr(base_model, "logit_bias_head") and base_model.logit_bias_head is not None
    if should_log and resolved_use_logit_bias and not has_logit_bias_head:
        print("Warning: logit bias requested but model has no logit_bias_head; detached prior loss will stay zero.")

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            accumulation_context = accelerator.accumulate(model) if accelerator is not None else nullcontext()
            with accumulation_context:
                input_ids, attention_mask, labels = _maybe_move_batch(batch, device, accelerator=accelerator)

                logits, hidden_states = _forward_with_optional_hidden_states(
                    model,
                    input_ids,
                    attention_mask,
                    output_hidden_states=resolved_use_logit_bias,
                    final_hidden_state_only=resolved_use_logit_bias,
                    accelerator=accelerator,
                )

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                valid_mask = shift_labels != -100
                skip_flags = _sync_skip_flags(
                    {
                        "non_finite_logits": not torch.isfinite(logits).all().item(),
                        "no_valid_targets": valid_mask.sum().item() == 0,
                    },
                    accelerator=accelerator,
                    device=work_device,
                )

                if skip_flags["non_finite_logits"]:
                    if should_log:
                        print(
                        f"Warning: skipping non-finite logits at epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}/{len(dataloader)}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue

                if skip_flags["no_valid_targets"]:
                    if should_log:
                        print(
                        f"Warning: skipping batch with no valid targets at epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}/{len(dataloader)}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue

                ce_loss_per_position = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                ce_loss_per_position = ce_loss_per_position.view_as(shift_labels)
                ar_loss = ce_loss_per_position[valid_mask].mean()

                masks = build_shift_supervision_masks(
                    shift_labels,
                    base_model,
                    use_eoc=resolved_use_eoc,
                )
                eoc_count = int(masks["eoc_mask"].sum().item())
                tool_count = int(masks["tool_mask"].sum().item())
                logit_bias_loss = zero
                logit_bias_targets = None
                logit_bias_initial_count = 0
                logit_bias_eoc_count = 0

                if resolved_use_logit_bias and hidden_states is not None:
                    (
                        logit_bias_hidden_states,
                        logit_bias_targets,
                        _,
                        _,
                        logit_bias_initial_count,
                        logit_bias_eoc_count,
                    ) = gather_logit_bias_examples(
                        hidden_states,
                        labels,
                        base_model,
                        return_indices=True,
                    )
                    if logit_bias_targets.numel() > 0 and has_logit_bias_head:
                        _, logit_bias_loss = compute_logit_bias_loss(
                            base_model,
                            logit_bias_hidden_states,
                            logit_bias_targets,
                        )

                loss = ar_loss
                if resolved_use_logit_bias:
                    loss = loss + logit_bias_loss_weight * logit_bias_loss

                step_loss_metrics = _build_loss_metrics(
                    total_loss=loss,
                    ar_loss=ar_loss,
                    extra_loss_metrics={
                        "logit_bias_loss": logit_bias_loss if resolved_use_logit_bias else None,
                    },
                )

                skip_flags = _sync_skip_flags(
                    {
                        "non_finite_loss": not torch.isfinite(loss).item(),
                    },
                    accelerator=accelerator,
                    device=work_device,
                )
                if skip_flags["non_finite_loss"]:
                    if should_log:
                        print(
                        f"Warning: skipping non-finite loss at epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}/{len(dataloader)}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue

                backward_loss = loss
                if accelerator is not None:
                    accelerator.backward(backward_loss)
                else:
                    backward_loss = loss / gradient_accumulation_steps
                    backward_loss.backward()

                trainable_params = base_model.get_trainable_parameters()
                skip_flags = _sync_skip_flags(
                    {
                        "non_finite_gradients": any(
                            param.grad is not None and not torch.isfinite(param.grad).all().item()
                            for param in trainable_params
                        ),
                    },
                    accelerator=accelerator,
                    device=work_device,
                )
                if skip_flags["non_finite_gradients"]:
                    if should_log:
                        print(
                        f"Warning: skipping optimizer step with non-finite gradients at epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}/{len(dataloader)}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue

                if active_tool_ids is not None:
                    try:
                        total_tools = len(base_model.reserved_token_ids)
                        total_rows = base_model.trainable_tool_input_embeddings.size(0)
                        device_for_mask = base_model.trainable_tool_input_embeddings.device
                        active_mask = torch.zeros(total_rows, dtype=torch.bool, device=device_for_mask)
                        active_mask[active_tool_ids] = True
                        if total_rows > total_tools:
                            active_mask[total_tools:] = True

                        params_to_mask = []
                        if (
                            hasattr(base_model, "trainable_tool_input_embeddings")
                            and base_model.trainable_tool_input_embeddings is not None
                        ):
                            params_to_mask.append(base_model.trainable_tool_input_embeddings)
                        if (
                            hasattr(base_model, "trainable_tool_output_embeddings")
                            and base_model.trainable_tool_output_embeddings is not None
                        ):
                            if base_model.trainable_tool_output_embeddings is not base_model.trainable_tool_input_embeddings:
                                params_to_mask.append(base_model.trainable_tool_output_embeddings)

                        for param in params_to_mask:
                            if param.grad is not None:
                                inactive_rows = ~active_mask
                                param.grad[inactive_rows] = 0
                    except Exception as exc:
                        if should_log:
                            print(f"Warning: failed to apply active tool grad mask: {exc}")

                current_lr_values = [float(param_group["lr"]) for param_group in optimizer.param_groups]
                should_step = (
                    accelerator.sync_gradients
                    if accelerator is not None
                    else ((step + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1)
                )
                if should_step:
                    _sync_fsdp_ignored_module_gradients(model, accelerator=accelerator)
                    optimizer.step()
                    optimizer_steps += 1

                    if renorm_active_rows and active_tool_ids is not None:
                        try:
                            total_tools = len(base_model.reserved_token_ids)
                            device_for_mask = base_model.trainable_tool_input_embeddings.device
                            active_mask = torch.zeros(total_tools, dtype=torch.bool, device=device_for_mask)
                            active_mask[active_tool_ids] = True
                            inactive_mask = ~active_mask

                            params_to_norm = []
                            if (
                                hasattr(base_model, "trainable_tool_output_embeddings")
                                and base_model.trainable_tool_output_embeddings is not None
                            ):
                                params_to_norm.append(base_model.trainable_tool_output_embeddings)
                            elif (
                                hasattr(base_model, "trainable_tool_embeddings")
                                and base_model.trainable_tool_embeddings is not None
                            ):
                                params_to_norm.append(base_model.trainable_tool_embeddings)

                            with torch.no_grad():
                                for param in params_to_norm:
                                    tool_rows = param.data[:total_tools]
                                    if inactive_mask.any():
                                        target_norm = tool_rows[inactive_mask].norm(dim=1).mean().clamp(min=1e-6)
                                    else:
                                        target_norm = tool_rows.norm(dim=1).mean().clamp(min=1e-6)

                                    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
                                    if active_idx.numel() > 0:
                                        active_norms = tool_rows[active_idx].norm(dim=1, keepdim=True).clamp(min=1e-6)
                                        tool_rows[active_idx] = tool_rows[active_idx] * (target_norm / active_norms)
                        except Exception as exc:
                            if should_log:
                                print(f"Warning: failed to apply post-step renorm: {exc}")

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                total_valid_positions += int(valid_mask.sum().item())
                total_eoc_positions += eoc_count
                total_tool_positions += tool_count
                if resolved_use_logit_bias and logit_bias_targets is not None:
                    total_logit_bias_positions += int(logit_bias_targets.numel())
                    total_logit_bias_initial_positions += logit_bias_initial_count
                    total_logit_bias_eoc_positions += logit_bias_eoc_count
                successful_steps += 1
                for metric_name, metric_value in step_loss_metrics.items():
                    metric_float = _metric_value_to_float(metric_value)
                    total_loss_metrics[metric_name] += metric_float
                    window_loss_metrics[metric_name] += metric_float

                window_batches += 1
                window_valid_positions += int(valid_mask.sum().item())
                window_eoc_positions += eoc_count
                window_tool_positions += tool_count
                if resolved_use_logit_bias and logit_bias_targets is not None:
                    window_logit_bias_positions += int(logit_bias_targets.numel())
                if plot_history is not None and should_log:
                    plot_step = plot_step_offset + successful_steps
                    _append_loss_plot_record(plot_history, plot_step, plot_round, step_loss_metrics)
                    _append_lr_plot_records(
                        plot_history,
                        plot_step,
                        plot_round,
                        optimizer,
                        current_lr_values,
                    )

                if should_log and (batch_idx + 1) % 10 == 0:
                    lr_values = current_lr_values
                    if base_model.lora_config and lora_lr is not None:
                        lr_info = f"LR(emb/lora): {lr_values[0]:.6f}/{lr_values[1]:.6f}"
                    else:
                        lr_info = f"LR: {lr_values[0]:.6f}"

                    window_denom = max(1, window_batches)
                    window_avg_metrics = _average_metrics(window_loss_metrics, window_denom)
                    logit_bias_fragment = ""
                    if resolved_use_logit_bias:
                        logit_bias_fragment = (
                            f"LogitBias: {window_avg_metrics.get('logit_bias_loss', 0.0):.4f}, "
                        )
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                        f"Loss: {window_avg_metrics.get('total_loss', 0.0):.4f}, "
                        f"AR: {window_avg_metrics.get('ar_loss', 0.0):.4f}, "
                        f"{logit_bias_fragment}"
                        f"Sites(valid/eoc/tool/logit_bias): {window_valid_positions}/{window_eoc_positions}/"
                        f"{window_tool_positions}/{window_logit_bias_positions}, "
                        f"{lr_info}"
                    )

                    window_batches = 0
                    window_loss_metrics = defaultdict(float)
                    window_valid_positions = 0
                    window_eoc_positions = 0
                    window_tool_positions = 0
                    window_logit_bias_positions = 0

                step += 1

    local_successful_steps = successful_steps
    local_optimizer_steps = optimizer_steps
    aggregated_successful_steps = successful_steps
    if accelerator is not None:
        reduced_metrics = torch.tensor(
            [
                total_loss_metrics.get("total_loss", 0.0),
                total_loss_metrics.get("ar_loss", 0.0),
                total_loss_metrics.get("logit_bias_loss", 0.0),
                float(total_valid_positions),
                float(total_eoc_positions),
                float(total_tool_positions),
                float(total_logit_bias_positions),
                float(total_logit_bias_initial_positions),
                float(total_logit_bias_eoc_positions),
                float(successful_steps),
                float(optimizer_steps),
            ],
            device=accelerator.device,
        )
        reduced_metrics = accelerator.reduce(reduced_metrics, reduction="sum")
        total_loss_metrics = {
            "total_loss": reduced_metrics[0].item(),
            "ar_loss": reduced_metrics[1].item(),
            "logit_bias_loss": reduced_metrics[2].item(),
        }
        total_valid_positions = int(reduced_metrics[3].item())
        total_eoc_positions = int(reduced_metrics[4].item())
        total_tool_positions = int(reduced_metrics[5].item())
        total_logit_bias_positions = int(reduced_metrics[6].item())
        total_logit_bias_initial_positions = int(reduced_metrics[7].item())
        total_logit_bias_eoc_positions = int(reduced_metrics[8].item())
        aggregated_successful_steps = int(reduced_metrics[9].item())

    avg_loss_metrics = _average_metrics(total_loss_metrics, aggregated_successful_steps)
    default_inactive_avg = 0.0 if aggregated_successful_steps > 0 else float("nan")
    avg_total_loss = avg_loss_metrics.get("total_loss", float("nan"))
    avg_ar_loss = avg_loss_metrics.get("ar_loss", float("nan"))
    avg_logit_bias_loss = avg_loss_metrics.get("logit_bias_loss", default_inactive_avg)

    if should_log:
        print("\nTraining completed!")
        print(f"Average total loss: {avg_total_loss:.4f}")
        print(f"Average AR loss:    {avg_ar_loss:.4f}")
        if resolved_use_logit_bias:
            print(f"Average Logit bias loss: {avg_logit_bias_loss:.4f}")
        print(f"Total valid supervised positions: {total_valid_positions}")
        if resolved_use_eoc:
            print(f"Total EOC positions: {total_eoc_positions}")
            print(f"Total tool positions: {total_tool_positions}")
        if resolved_use_logit_bias:
            print(f"Total logit-bias positions: {total_logit_bias_positions}")
            print(f"Logit-bias tool sites from assistant-start positions: {total_logit_bias_initial_positions}")
            print(f"Logit-bias tool sites from EOC positions: {total_logit_bias_eoc_positions}")
        print(f"Successful batches: {local_successful_steps}")
        print(f"Optimizer steps: {local_optimizer_steps}")

    return {
        "avg_total_loss": avg_total_loss,
        "avg_ar_loss": avg_ar_loss,
        "avg_logit_bias_loss": avg_logit_bias_loss,
        "total_valid_positions": total_valid_positions,
        "total_eoc_positions": total_eoc_positions,
        "total_tool_positions": total_tool_positions,
        "total_logit_bias_positions": total_logit_bias_positions,
        "total_logit_bias_initial_positions": total_logit_bias_initial_positions,
        "total_logit_bias_eoc_positions": total_logit_bias_eoc_positions,
        "successful_steps": local_successful_steps,
        "aggregated_successful_steps": aggregated_successful_steps,
        "optimizer_steps": local_optimizer_steps,
        "use_eoc": resolved_use_eoc,
        "use_js_trunc": resolved_use_js_trunc,
        "use_logit_bias": resolved_use_logit_bias,
        "avg_loss_metrics": avg_loss_metrics,
        "plot_next_step": plot_step_offset + local_successful_steps,
        "plot_end_step": plot_step_offset + local_successful_steps,
    }


def demo_native_function_calling(
    model,
    tokenizer,
    test_examples,
    device="cuda",
    max_new_tokens=512,
    use_ground_truth_tools=False,
    use_eoc=None,
    use_js_trunc=None,
    use_logit_bias=None,
    accelerator=None,
):
    """Demo of native function calling using held-out test examples."""
    model.eval()
    base_model = _unwrap_model(model, accelerator=accelerator)
    should_log = accelerator is None or accelerator.is_main_process
    if not should_log:
        return
    resolved_use_eoc, resolved_use_js_trunc = _resolve_mode_flags(
        base_model,
        use_eoc,
        use_js_trunc,
    )
    resolved_use_logit_bias = bool(
        getattr(base_model, "use_logit_bias", False) if use_logit_bias is None else use_logit_bias
    )
    mode_desc = "Ground Truth Tool Inference" if use_ground_truth_tools else "Normal Tool Prediction"
    if resolved_use_js_trunc:
        mode_desc += " + JS trunc"
    if resolved_use_logit_bias:
        mode_desc += " + logit bias"

    print(f"\n=== Native Function Calling Demo ({mode_desc}) ===")
    print(f"Testing on {len(test_examples)} held-out examples")
    print(f"Available tools: {base_model.tool_names}")
    print()

    for i, example in enumerate(test_examples):
        user_input = example["user_input"]
        expected_tools = example.get("tools", [example.get("tool_name", "unknown")])
        expected_calls = example.get("function_calls", [example.get("function_call", "{}")])

        print(f"=== Test Example {i} ===")
        print(f"User Query: {user_input}")
        print(f"Expected Tool(s): {expected_tools}")
        print(f"Expected Call(s): {expected_calls}")
        print()

        user_text = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        user_tokens = tokenizer(user_text, return_tensors="pt").to(
            accelerator.device if accelerator is not None else device
        )

        results = _generate_results(
            model,
            tokenizer,
            user_tokens["input_ids"],
            user_tokens["attention_mask"],
            use_js_trunc=resolved_use_js_trunc,
            use_logit_bias=resolved_use_logit_bias,
            use_eoc=resolved_use_eoc,
            use_ground_truth_tools=use_ground_truth_tools,
            ground_truth_tools=expected_tools if use_ground_truth_tools else None,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            accelerator=accelerator,
        )
        mode_line = "Ground truth tools used" if use_ground_truth_tools else "Model predicts tools"
        if resolved_use_js_trunc:
            mode_line += " + JS trunc"
        if resolved_use_logit_bias:
            mode_line += " + logit bias"
        print(f"Mode: {mode_line}")

        result = results[0]

        if "predicted_tools" in result and result["predicted_tools"]:
            print(f"Generated {len(result['predicted_tools'])} tool(s):")
            for j, (tool_info, func_call) in enumerate(zip(result["predicted_tools"], result["function_calls"])):
                print(f"  Tool {j + 1}: {tool_info['tool_name']}")
                print(f"  Function Call {j + 1}: {func_call}")

                parsed = base_model.parse_function_call(func_call)
                print(f"  Parsed {j + 1}: {parsed}")
                print()
        else:
            print(f"Predicted Tool: {result['predicted_tool_name']}")
            print(f"Tool Token Used: {result.get('tool_token_used', 'N/A')}")
            print(f"Function Call: {result['function_call']}")

            parsed = base_model.parse_function_call(result["function_call"])
            print(f"Parsed: {parsed}")

        print(f"Full Generated: {result['full_generated_sequence']}")
        print("-" * 50)
        print()


def eval_native_function_calling(
    model,
    tokenizer,
    test_dataloader,
    device="cuda",
    max_new_tokens=512,
    use_ground_truth_tools=False,
    use_eoc=None,
    use_js_trunc=None,
    use_logit_bias=None,
    accelerator=None,
):
    """Comprehensive evaluation of native function calling model using batch processing."""
    from eval import compare_function_calls_advanced, calculate_argument_accuracy, calculate_tool_metrics
    import time

    model.eval()
    base_model = _unwrap_model(model, accelerator=accelerator)
    should_log = accelerator is None or accelerator.is_main_process
    resolved_use_eoc, resolved_use_js_trunc = _resolve_mode_flags(
        base_model,
        use_eoc,
        use_js_trunc,
    )
    resolved_use_logit_bias = bool(
        getattr(base_model, "use_logit_bias", False) if use_logit_bias is None else use_logit_bias
    )

    total_examples = len(test_dataloader.dataset)
    mode_desc = "Ground Truth Tool Inference" if use_ground_truth_tools else "Normal Tool Prediction"
    if resolved_use_js_trunc:
        mode_desc += " + JS trunc"
    if resolved_use_logit_bias:
        mode_desc += " + logit bias"

    if should_log:
        print(f"\n=== Native Function Calling Evaluation ({mode_desc}) ===")
        print(f"Evaluating on {total_examples} test examples")
        print()

    exact_matches = 0
    tool_tp = 0
    tool_tn = 0
    tool_fp = 0
    tool_fn = 0
    tool_exact_matches = 0
    matched_arguments = 0
    total_target_arguments = 0
    f1_scores = []
    precision_scores = []
    recall_scores = []
    tool_f1_scores = []
    tool_precision_scores = []
    tool_recall_scores = []
    parse_errors = 0

    call_count_breakdown = defaultdict(
        lambda: {
            "total": 0,
            "exact_matches": 0,
            "tool_tp": 0,
            "tool_tn": 0,
            "tool_fp": 0,
            "tool_fn": 0,
            "tool_exact_matches": 0,
            "matched_arguments": 0,
            "target_arguments": 0,
            "f1_scores": [],
            "precision_scores": [],
            "recall_scores": [],
            "tool_f1_scores": [],
            "tool_precision_scores": [],
            "tool_recall_scores": [],
            "parse_errors": 0,
        }
    )

    candidate_tools = list(getattr(base_model, "tool_names", []))
    if not candidate_tools:
        candidate_tools = [
            tool
            for example in getattr(test_dataloader.dataset, "data", [])
            for tool in example.get("tools", [])
        ]
    candidate_tools = list(dict.fromkeys(candidate_tools))

    start_time = time.time()
    if should_log:
        print("🔄 Running batch evaluation...")

    processed_examples = 0
    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch["raw_data"])
        processed_examples += batch_size

        if should_log and (batch_idx % 10 == 0 or processed_examples == total_examples):
            print(f"   Progress: {processed_examples}/{total_examples} ({100 * processed_examples / total_examples:.1f}%)")

        input_ids, attention_mask, _ = _maybe_move_batch(batch, device, accelerator=accelerator)
        batch_results = _generate_results_with_example_fallback(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            raw_examples=batch["raw_data"],
            batch_idx=batch_idx,
            use_js_trunc=resolved_use_js_trunc,
            use_logit_bias=resolved_use_logit_bias,
            use_eoc=resolved_use_eoc,
            use_ground_truth_tools=use_ground_truth_tools,
            max_new_tokens=max_new_tokens,
            accelerator=accelerator,
        )
        raw_examples = batch["raw_data"]
        if accelerator is not None:
            batch_results = accelerator.gather_for_metrics(batch_results, use_gather_object=True)
            raw_examples = accelerator.gather_for_metrics(raw_examples, use_gather_object=True)

        if not should_log:
            continue

        for i, example in enumerate(raw_examples):
            result = batch_results[i]
            expected_tools = example.get("tools", [example.get("tool_name", "unknown")])
            expected_calls = example.get("function_calls", [example.get("function_call", "{}")])
            expected_call_count = len(expected_calls)
            breakdown = call_count_breakdown[expected_call_count]
            breakdown["total"] += 1

            if "predicted_tools" in result and result["predicted_tools"]:
                predicted_tools = [tool_info["tool_name"] for tool_info in result["predicted_tools"]]
                predicted_calls = result["function_calls"]
            else:
                predicted_tools = [result["predicted_tool_name"]] if result["predicted_tool_name"] != "none" else []
                predicted_calls = [result["function_call"]] if result["function_call"] else []

            tool_metrics = calculate_tool_metrics(
                predicted_tools=predicted_tools,
                expected_tools=expected_tools,
                candidate_tools=candidate_tools,
            )
            tool_f1_scores.append(tool_metrics["tool_f1_score"])
            tool_precision_scores.append(tool_metrics["tool_precision"])
            tool_recall_scores.append(tool_metrics["tool_recall"])
            tool_tp += tool_metrics["tool_tp"]
            tool_tn += tool_metrics["tool_tn"]
            tool_fp += tool_metrics["tool_fp"]
            tool_fn += tool_metrics["tool_fn"]
            if tool_metrics["tool_exact_match_acc"] >= 1.0:
                tool_exact_matches += 1
                breakdown["tool_exact_matches"] += 1
            breakdown["tool_tp"] += tool_metrics["tool_tp"]
            breakdown["tool_tn"] += tool_metrics["tool_tn"]
            breakdown["tool_fp"] += tool_metrics["tool_fp"]
            breakdown["tool_fn"] += tool_metrics["tool_fn"]

            try:
                eval_result = compare_function_calls_advanced(
                    predicted_calls,
                    expected_calls,
                    ignore_order=True,
                )
                argument_result = calculate_argument_accuracy(predicted_calls, expected_calls)
                matched_arguments += argument_result["matched_arguments"]
                total_target_arguments += argument_result["total_target_arguments"]
                breakdown["matched_arguments"] += argument_result["matched_arguments"]
                breakdown["target_arguments"] += argument_result["total_target_arguments"]

                if eval_result.exact_match:
                    exact_matches += 1
                    breakdown["exact_matches"] += 1

                f1_scores.append(eval_result.f1_score)
                precision_scores.append(eval_result.precision)
                recall_scores.append(eval_result.recall)
                breakdown["f1_scores"].append(eval_result.f1_score)
                breakdown["precision_scores"].append(eval_result.precision)
                breakdown["recall_scores"].append(eval_result.recall)
                breakdown["tool_f1_scores"].append(tool_metrics["tool_f1_score"])
                breakdown["tool_precision_scores"].append(tool_metrics["tool_precision"])
                breakdown["tool_recall_scores"].append(tool_metrics["tool_recall"])

                if "parse_errors" in eval_result.details:
                    current_parse_errors = eval_result.details["parse_errors"]["outputs"]
                    parse_errors += current_parse_errors
                    breakdown["parse_errors"] += current_parse_errors
            except Exception as exc:
                print(
                    f"   Error evaluating example {i + 1} in batch {batch_idx + 1}: {str(exc)}"
                )
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                breakdown["f1_scores"].append(0.0)
                breakdown["precision_scores"].append(0.0)
                breakdown["recall_scores"].append(0.0)
                breakdown["tool_f1_scores"].append(tool_metrics["tool_f1_score"])
                breakdown["tool_precision_scores"].append(tool_metrics["tool_precision"])
                breakdown["tool_recall_scores"].append(tool_metrics["tool_recall"])

    eval_time = time.time() - start_time
    if not should_log:
        return None

    exact_accuracy = exact_matches / total_examples
    tool_judgments = tool_tp + tool_tn + tool_fp + tool_fn
    tool_accuracy = (tool_tp + tool_tn) / tool_judgments if tool_judgments > 0 else 1.0
    tool_exact_match_acc = tool_exact_matches / total_examples if total_examples > 0 else 0.0
    arguments_accuracy = matched_arguments / total_target_arguments if total_target_arguments > 0 else 1.0
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_tool_f1_score = sum(tool_f1_scores) / len(tool_f1_scores) if tool_f1_scores else 0.0
    avg_tool_precision = sum(tool_precision_scores) / len(tool_precision_scores) if tool_precision_scores else 0.0
    avg_tool_recall = sum(tool_recall_scores) / len(tool_recall_scores) if tool_recall_scores else 0.0
    parse_error_rate = parse_errors / total_examples

    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)

    print(f"📋 Dataset: {total_examples} examples")
    print(f"⏱️  Evaluation time: {eval_time:.2f} seconds")
    print(f"🔧 Mode: {mode_desc}")
    print()

    print("🎯 RESULTS:")
    print(f"   Exact Match Accuracy:     {exact_accuracy:.3f} ({exact_matches}/{total_examples})")
    print(f"   Full Correctness:         {exact_accuracy:.3f} ({exact_matches}/{total_examples})")
    print(
        "   Tool Acc (tool_accuracy): "
        f"{tool_accuracy:.3f} (TP={tool_tp}, TN={tool_tn}, FP={tool_fp}, FN={tool_fn})"
    )
    print(
        "   Tool Exact Match Acc (tool_exact_match_acc): "
        f"{tool_exact_match_acc:.3f} ({tool_exact_matches}/{total_examples})"
    )
    print(f"   Arguments Accuracy:       {arguments_accuracy:.3f} ({matched_arguments}/{total_target_arguments})")
    print(f"   Average F1 Score:         {avg_f1_score:.3f}")
    print(f"   Average Precision:        {avg_precision:.3f}")
    print(f"   Average Recall:           {avg_recall:.3f}")
    print(f"   Average Tool F1 Score:    {avg_tool_f1_score:.3f}")
    print(f"   Average Tool Precision:   {avg_tool_precision:.3f}")
    print(f"   Average Tool Recall:      {avg_tool_recall:.3f}")
    print(f"   Parse Error Rate:         {parse_error_rate:.3f}")
    print("=" * 50)

    print("\n📊 EXACT MATCH ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        accuracy = stats["exact_matches"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"   {call_count} call(s): {accuracy:.3f} ({stats['exact_matches']}/{stats['total']})")
    print("=" * 50)

    print("\n📊 TOOL ACCURACY (tool_accuracy):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        tool_total = stats["tool_tp"] + stats["tool_tn"] + stats["tool_fp"] + stats["tool_fn"]
        per_call_tool_accuracy = (stats["tool_tp"] + stats["tool_tn"]) / tool_total if tool_total > 0 else 1.0
        print(
            f"   {call_count} call(s): {per_call_tool_accuracy:.3f} "
            f"(TP={stats['tool_tp']}, TN={stats['tool_tn']}, FP={stats['tool_fp']}, FN={stats['tool_fn']})"
        )
    print("=" * 50)

    print("\n📊 TOOL EXACT MATCH ACCURACY (tool_exact_match_acc):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        per_call_tool_exact_match_acc = stats["tool_exact_matches"] / stats["total"] if stats["total"] > 0 else 0.0
        print(
            f"   {call_count} call(s): {per_call_tool_exact_match_acc:.3f} "
            f"({stats['tool_exact_matches']}/{stats['total']})"
        )
    print("=" * 50)

    print("\n📊 ARGUMENTS ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        per_call_arguments_accuracy = (
            stats["matched_arguments"] / stats["target_arguments"]
            if stats["target_arguments"] > 0
            else 1.0
        )
        print(
            f"   {call_count} call(s): {per_call_arguments_accuracy:.3f} "
            f"({stats['matched_arguments']}/{stats['target_arguments']})"
        )
    print("=" * 50)

    print("\n📊 AVERAGE F1 SCORE (Function Calls):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_f1 = sum(stats["f1_scores"]) / len(stats["f1_scores"]) if stats["f1_scores"] else 0.0
        avg_prec = sum(stats["precision_scores"]) / len(stats["precision_scores"]) if stats["precision_scores"] else 0.0
        avg_rec = sum(stats["recall_scores"]) / len(stats["recall_scores"]) if stats["recall_scores"] else 0.0
        print(f"   {call_count} call(s): F1={avg_f1:.3f}, P={avg_prec:.3f}, R={avg_rec:.3f}")
    print("=" * 50)

    print("\n📊 AVERAGE TOOL F1 SCORE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_tool_f1 = sum(stats["tool_f1_scores"]) / len(stats["tool_f1_scores"]) if stats["tool_f1_scores"] else 0.0
        avg_tool_prec = sum(stats["tool_precision_scores"]) / len(stats["tool_precision_scores"]) if stats["tool_precision_scores"] else 0.0
        avg_tool_rec = sum(stats["tool_recall_scores"]) / len(stats["tool_recall_scores"]) if stats["tool_recall_scores"] else 0.0
        print(f"   {call_count} call(s): Tool F1={avg_tool_f1:.3f}, P={avg_tool_prec:.3f}, R={avg_tool_rec:.3f}")
    print("=" * 50)

    print("\n📊 PARSE ERROR RATE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        error_rate = stats["parse_errors"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"   {call_count} call(s): {error_rate:.3f} ({stats['parse_errors']}/{stats['total']})")
    print("=" * 50)

    return {
        "exact_accuracy": exact_accuracy,
        "full_correctness": exact_accuracy,
        "tool_accuracy": tool_accuracy,
        "tool_exact_match_acc": tool_exact_match_acc,
        "arguments_accuracy": arguments_accuracy,
        "avg_f1_score": avg_f1_score,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_tool_f1_score": avg_tool_f1_score,
        "avg_tool_precision": avg_tool_precision,
        "avg_tool_recall": avg_tool_recall,
        "parse_error_rate": parse_error_rate,
        "total_examples": total_examples,
        "call_count_breakdown": dict(call_count_breakdown),
    }
