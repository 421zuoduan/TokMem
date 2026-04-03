import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import torch


class TeeStream:
    """Mirror writes to the original stream and a log file."""

    def __init__(self, primary_stream, log_stream):
        self.primary_stream = primary_stream
        self.log_stream = log_stream
        self.encoding = getattr(primary_stream, "encoding", None)

    def write(self, data):
        self.primary_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.primary_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return self.primary_stream.isatty()

    def writable(self):
        return True


def _normalize_model_label(model_name):
    """Convert a model name or path into a stable filename-safe label."""
    if not model_name:
        return "model"

    model_label = os.path.basename(str(model_name).rstrip("/")) or str(model_name)
    model_label = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-._")
    return model_label or "model"


def _format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _build_log_suffix(timestamp, model_name=None, num_tasks=None):
    """Build a shared timestamp/model/task suffix for log filenames."""
    model_label = _normalize_model_label(model_name)
    task_label = f"{num_tasks}tasks" if num_tasks is not None else "unknowntasks"
    return f"_{timestamp}_{model_label}_{task_label}.log"


def should_compute_bank_routing_metrics(use_hard_negative_loss):
    """Decide whether bank-only routing stats are needed for this run."""
    return bool(use_hard_negative_loss)


def format_training_progress_message(
    epoch,
    num_epochs,
    batch_idx,
    num_batches,
    avg_loss,
    avg_task_loss,
    avg_mean_loss,
    avg_hard_negative_loss,
    avg_sep_loss,
    avg_task_count,
    current_lr,
    time_summary,
    mean_norm,
    show_hard_negative_loss,
    include_bank_routing_metrics,
    avg_routing_bank_acc=0.0,
    avg_routing_bank_margin=0.0,
):
    """Build the human-readable training progress line."""
    message = (
        f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{num_batches}, "
        f"Avg Loss: {avg_loss:.4f}, Avg Task Loss: {avg_task_loss:.4f}, "
        f"Avg Mean Loss: {avg_mean_loss:.4f}, "
    )
    if show_hard_negative_loss:
        message += f"Avg HN Loss: {avg_hard_negative_loss:.4f}, "
    message += f"Avg Sep Loss: {avg_sep_loss:.4f}, "
    if include_bank_routing_metrics:
        message += (
            f"Routing Bank Acc: {avg_routing_bank_acc:.4f}, "
            f"Routing Bank Margin: {avg_routing_bank_margin:.4f}, "
        )
    message += (
        f"Mean Norm: {mean_norm:.4f}, Avg Task Tokens: {avg_task_count:.1f}, "
        f"LR: {current_lr:.6f}, {time_summary}"
    )
    return message


def format_training_logger_message(
    epoch,
    num_epochs,
    batch_idx,
    num_batches,
    avg_loss,
    avg_task_loss,
    avg_mean_loss,
    avg_hard_negative_loss,
    avg_sep_loss,
    avg_task_count,
    current_lr,
    time_summary,
    mean_norm,
    show_hard_negative_loss,
    include_bank_routing_metrics,
    avg_routing_bank_acc=0.0,
    avg_routing_bank_margin=0.0,
):
    """Build the machine-readable training log line."""
    message = (
        f"E{epoch}/{num_epochs} B{batch_idx}/{num_batches} "
        f"AvgLoss:{avg_loss:.4f} AvgTaskLoss:{avg_task_loss:.4f} "
        f"AvgMeanLoss:{avg_mean_loss:.4f} "
    )
    if show_hard_negative_loss:
        message += f"AvgHNLoss:{avg_hard_negative_loss:.4f} "
    message += f"AvgSepLoss:{avg_sep_loss:.4f} "
    if include_bank_routing_metrics:
        message += (
            f"RoutingBankAcc:{avg_routing_bank_acc:.4f} "
            f"RoutingBankMargin:{avg_routing_bank_margin:.4f} "
        )
    message += (
        f"MeanNorm:{mean_norm:.4f} AvgTokens:{avg_task_count:.1f} "
        f"LR:{current_lr:.6f} {time_summary}"
    )
    return message


def format_validation_message(
    prefix,
    avg_val_loss,
    avg_val_mean_loss,
    avg_val_hard_negative_loss,
    avg_val_sep_loss,
    show_hard_negative_loss,
    include_bank_routing_metrics,
    avg_val_routing_bank_acc=0.0,
    avg_val_routing_bank_margin=0.0,
):
    """Build the human-readable validation summary."""
    message = (
        f"{prefix} - Validation Loss: {avg_val_loss:.4f}, "
        f"Mean Loss: {avg_val_mean_loss:.4f}, "
    )
    if show_hard_negative_loss:
        message += f"HN Loss: {avg_val_hard_negative_loss:.4f}, "
    message += f"Sep Loss: {avg_val_sep_loss:.4f}"
    if include_bank_routing_metrics:
        message += (
            f", Routing Bank Acc: {avg_val_routing_bank_acc:.4f}, "
            f"Routing Bank Margin: {avg_val_routing_bank_margin:.4f}"
        )
    return message


def format_validation_logger_message(
    prefix,
    avg_val_loss,
    avg_val_mean_loss,
    avg_val_hard_negative_loss,
    avg_val_sep_loss,
    show_hard_negative_loss,
    include_bank_routing_metrics,
    avg_val_routing_bank_acc=0.0,
    avg_val_routing_bank_margin=0.0,
):
    """Build the machine-readable validation summary."""
    message = (
        f"{prefix} Loss:{avg_val_loss:.4f} "
        f"MeanLoss:{avg_val_mean_loss:.4f} "
    )
    if show_hard_negative_loss:
        message += f"HNLoss:{avg_val_hard_negative_loss:.4f} "
    message += f"SepLoss:{avg_val_sep_loss:.4f}"
    if include_bank_routing_metrics:
        message += (
            f" RoutingBankAcc:{avg_val_routing_bank_acc:.4f} "
            f"RoutingBankMargin:{avg_val_routing_bank_margin:.4f}"
        )
    return message


def format_training_completion_logger_message(
    avg_total_loss,
    avg_task_loss,
    avg_mean_loss,
    avg_hard_negative_loss,
    avg_sep_loss,
    show_hard_negative_loss,
    include_bank_routing_metrics,
    avg_routing_bank_acc=0.0,
    avg_routing_bank_margin=0.0,
    task_token_count=None,
    task_loss_batches=None,
    step=None,
    warning_no_task_tokens=False,
):
    """Build the machine-readable end-of-training summary."""
    message = (
        f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} "
        f"TaskLoss:{avg_task_loss:.4f} MeanLoss:{avg_mean_loss:.4f} "
    )
    if show_hard_negative_loss:
        message += f"HNLoss:{avg_hard_negative_loss:.4f} "
    message += f"SepLoss:{avg_sep_loss:.4f} "
    if include_bank_routing_metrics:
        message += (
            f"RoutingBankAcc:{avg_routing_bank_acc:.4f} "
            f"RoutingBankMargin:{avg_routing_bank_margin:.4f} "
        )
    if warning_no_task_tokens:
        message += "WARNING:NoTaskTokens"
    elif task_token_count is not None and task_loss_batches is not None and step is not None:
        message += f"TaskTokens:{task_token_count} TaskBatches:{task_loss_batches}/{step}"
    return message


def _capture_stdout(log_path):
    """Mirror stdout/stderr into a dedicated file while keeping terminal output."""
    stdout_stream = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, stdout_stream)
    sys.stderr = TeeStream(sys.stderr, stdout_stream)


def setup_logging(log_dir="logs", model_name=None, num_tasks=None, stdout_prefix=None, timestamp=None):
    """Set up logging configuration for training and evaluation."""
    os.makedirs(log_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _build_log_suffix(timestamp, model_name=model_name, num_tasks=num_tasks)

    training_log = os.path.join(log_dir, f"training{suffix}")
    evaluation_log = os.path.join(log_dir, f"evaluation{suffix}")
    stdout_log = None

    if stdout_prefix is not None:
        stdout_log = os.path.join(log_dir, f"{stdout_prefix}_stdout{suffix}")
        _capture_stdout(stdout_log)

    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.INFO)
    training_logger.propagate = False
    for handler in list(training_logger.handlers):
        handler.close()
        training_logger.removeHandler(handler)
    training_handler = logging.FileHandler(training_log)
    training_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    training_logger.addHandler(training_handler)

    eval_logger = logging.getLogger('evaluation')
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    for handler in list(eval_logger.handlers):
        handler.close()
        eval_logger.removeHandler(handler)
    eval_handler = logging.FileHandler(evaluation_log)
    eval_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    eval_logger.addHandler(eval_handler)

    return training_logger, eval_logger, training_log, evaluation_log, stdout_log, timestamp


def extract_trained_token_state(model):
    """Return a minimal state_dict containing only the trainable task token parameters.
    The tensors are cloned and moved to CPU to minimize GPU memory usage.
    """
    state = {}
    if getattr(model, 'decouple_embeddings', False):
        state['trainable_task_input_embeddings'] = (
            model.trainable_task_input_embeddings.detach().cpu().clone()
        )
        state['trainable_task_output_embeddings'] = (
            model.trainable_task_output_embeddings.detach().cpu().clone()
        )
    else:
        state['trainable_task_embeddings'] = (
            model.trainable_task_embeddings.detach().cpu().clone()
        )
    return state


def clear_cuda_cache():
    """Release Python garbage and any currently unused CUDA cache blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_task_loss(shift_logits, shift_labels, reserved_token_ids):
    """Compute task_loss as full-vocab cross entropy on task-token positions."""
    import torch.nn.functional as F

    valid_mask = shift_labels != -100
    task_token_mask = torch.isin(shift_labels, reserved_token_ids)
    task_loss_mask = task_token_mask & valid_mask
    routing_count = int(task_loss_mask.sum().item())

    if routing_count == 0:
        return torch.tensor(0.0, device=shift_logits.device), routing_count

    task_loss = F.cross_entropy(shift_logits[task_loss_mask], shift_labels[task_loss_mask])
    return task_loss, routing_count


def compute_bank_only_routing_outputs(
    shift_hidden_states,
    shift_labels,
    reserved_token_ids,
    task_embeddings,
    eps=1e-12,
):
    """Build bank-only routing logits and summary stats on task-token positions."""
    import torch.nn.functional as F

    zero = torch.zeros((), device=shift_hidden_states.device, dtype=torch.float32)
    empty_logits = torch.empty(
        (0, 0),
        device=shift_hidden_states.device,
        dtype=torch.float32,
    )
    empty_targets = torch.empty((0,), device=shift_labels.device, dtype=torch.long)

    if task_embeddings is None:
        raise ValueError("task_embeddings must be provided for routing losses")

    valid_mask = shift_labels != -100
    routing_mask = valid_mask & torch.isin(shift_labels, reserved_token_ids)
    routing_count = int(routing_mask.sum().item())
    if routing_count == 0:
        return {
            "bank_logits": empty_logits,
            "bank_targets": empty_targets,
            "routing_count": 0,
            "routing_bank_acc": zero,
            "routing_bank_margin_avg": zero,
            "routing_correct_count": 0,
            "routing_margin_sum": zero,
        }

    routing_hidden = shift_hidden_states[routing_mask].to(dtype=torch.float32)
    routing_labels = shift_labels[routing_mask]
    bank_targets = torch.searchsorted(reserved_token_ids, routing_labels)

    bank = task_embeddings.to(device=routing_hidden.device, dtype=torch.float32)
    normalized_hidden = F.normalize(routing_hidden, p=2, dim=-1, eps=eps)
    normalized_bank = F.normalize(bank, p=2, dim=-1, eps=eps)
    bank_logits = torch.matmul(normalized_hidden, normalized_bank.transpose(0, 1))

    predicted_targets = torch.argmax(bank_logits, dim=-1)
    routing_correct_count = int((predicted_targets == bank_targets).sum().item())
    routing_bank_acc = torch.tensor(
        routing_correct_count / routing_count,
        device=bank_logits.device,
        dtype=bank_logits.dtype,
    )

    if bank_logits.shape[-1] <= 1:
        routing_margin_sum = zero
        routing_bank_margin_avg = zero
    else:
        positive_logits = bank_logits.gather(1, bank_targets.unsqueeze(1)).squeeze(1)
        negative_mask = F.one_hot(bank_targets, num_classes=bank_logits.shape[-1]).bool()
        hardest_negative_logits = bank_logits.masked_fill(negative_mask, float("-inf")).max(dim=-1).values
        routing_margins = positive_logits - hardest_negative_logits
        routing_margin_sum = routing_margins.sum()
        routing_bank_margin_avg = routing_margins.mean()

    return {
        "bank_logits": bank_logits,
        "bank_targets": bank_targets,
        "routing_count": routing_count,
        "routing_bank_acc": routing_bank_acc,
        "routing_bank_margin_avg": routing_bank_margin_avg,
        "routing_correct_count": routing_correct_count,
        "routing_margin_sum": routing_margin_sum,
    }


def compute_hard_negative_loss(bank_logits, bank_targets, hard_negative_margin=0.2):
    """Push the positive routing logit above the hardest negative by a margin."""
    import torch.nn.functional as F

    if bank_logits.numel() == 0 or bank_logits.shape[-1] <= 1:
        return torch.zeros((), device=bank_logits.device, dtype=bank_logits.dtype)

    positive_logits = bank_logits.gather(1, bank_targets.unsqueeze(1)).squeeze(1)
    negative_mask = F.one_hot(bank_targets, num_classes=bank_logits.shape[-1]).bool()
    hardest_negative_logits = bank_logits.masked_fill(negative_mask, float("-inf")).max(dim=-1).values
    return torch.relu(hard_negative_margin - positive_logits + hardest_negative_logits).mean()


def compute_memory_bank_mean_stats(task_embeddings, eps=1e-12):
    """Return the mean-direction loss and mean-direction norm for a memory bank."""
    import torch.nn.functional as F

    if task_embeddings is None:
        raise ValueError("task_embeddings must be provided for mean loss")

    num_embeddings = task_embeddings.shape[0]
    zero = torch.zeros((), device=task_embeddings.device, dtype=task_embeddings.dtype)
    if num_embeddings <= 1:
        return zero, zero

    normalized_embeddings = F.normalize(task_embeddings, p=2, dim=-1, eps=eps)
    mean_vector = normalized_embeddings.mean(dim=0)
    mean_norm = mean_vector.norm(p=2)
    mean_loss = mean_norm.pow(2)
    return mean_loss, mean_norm


def _compute_pairwise_separation_penalty(normalized_embeddings, tau):
    """Compute the raw pairwise cosine margin penalty on normalized embeddings."""
    num_embeddings = normalized_embeddings.shape[0]
    cosine_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(0, 1))
    penalty_matrix = torch.relu(cosine_matrix - tau).pow(2)
    penalty_matrix.fill_diagonal_(0)
    normalizer = num_embeddings * (num_embeddings - 1)
    return penalty_matrix.sum() / normalizer


def compute_separation_loss_components(task_embeddings, tau=0.2, use_centered_sep=False, eps=1e-12):
    """Compute raw and optional centered separation losses for a memory bank."""
    import torch.nn.functional as F

    if task_embeddings is None:
        raise ValueError("task_embeddings must be provided for separation loss")

    num_embeddings = task_embeddings.shape[0]
    if num_embeddings <= 1:
        zero = torch.zeros((), device=task_embeddings.device, dtype=task_embeddings.dtype)
        return {
            "sep_loss": zero,
            "sep_loss_raw": zero,
            "sep_loss_centered": zero,
        }

    normalized_embeddings = F.normalize(task_embeddings, p=2, dim=-1, eps=eps)
    raw_sep_loss = _compute_pairwise_separation_penalty(normalized_embeddings, tau)

    centered_sep_loss = torch.zeros((), device=task_embeddings.device, dtype=task_embeddings.dtype)
    selected_sep_loss = raw_sep_loss
    if use_centered_sep:
        mean_direction = normalized_embeddings.mean(dim=0, keepdim=True)
        centered_embeddings = normalized_embeddings - mean_direction
        centered_embeddings = F.normalize(centered_embeddings, p=2, dim=-1, eps=eps)
        centered_sep_loss = _compute_pairwise_separation_penalty(centered_embeddings, tau)
        selected_sep_loss = centered_sep_loss

    return {
        "sep_loss": selected_sep_loss,
        "sep_loss_raw": raw_sep_loss,
        "sep_loss_centered": centered_sep_loss,
    }


def compute_separation_loss(task_embeddings, tau=0.2, use_centered_sep=False, eps=1e-12):
    """Penalize task embeddings whose cosine similarity exceeds the margin tau."""
    return compute_separation_loss_components(
        task_embeddings,
        tau=tau,
        use_centered_sep=use_centered_sep,
        eps=eps,
    )["sep_loss"]


def get_separation_loss_embeddings(model):
    """Return the task embeddings to regularize for separation loss."""
    return get_memory_bank_embeddings(model)


def get_memory_bank_embeddings(model):
    """Return the trainable embeddings that define the current memory bank."""
    if getattr(model, "decouple_embeddings", False):
        return model.trainable_task_output_embeddings
    return model.trainable_task_embeddings


def compute_memory_bank_geometry_stats(task_embeddings, eps=1e-12):
    """Compute geometry stats for a memory bank after row normalization.

    The eigenspectrum is derived from the centered normalized matrix via SVD for
    numerical stability.
    """
    import torch

    if task_embeddings is None:
        return {
            "memory_bank_mean_norm": 0.0,
            "memory_bank_pc1_ratio": 0.0,
            "memory_bank_top5_ratio": 0.0,
            "memory_bank_top10_ratio": 0.0,
            "memory_bank_effective_rank": 0.0,
        }

    with torch.no_grad():
        bank = task_embeddings.detach()
        if bank.numel() == 0 or bank.dim() != 2:
            return {
                "memory_bank_mean_norm": 0.0,
                "memory_bank_pc1_ratio": 0.0,
                "memory_bank_top5_ratio": 0.0,
                "memory_bank_top10_ratio": 0.0,
                "memory_bank_effective_rank": 0.0,
            }

        bank = bank.to(dtype=torch.float32)
        if bank.device.type != "cpu":
            bank = bank.cpu()

        row_norms = bank.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        normalized_bank = bank / row_norms
        mean_vector = normalized_bank.mean(dim=0)
        mean_norm = mean_vector.norm(p=2).item()

        centered = normalized_bank - mean_vector
        # SVD of the centered normalized bank gives the covariance spectrum
        # without forming the large feature covariance explicitly.
        singular_values = torch.linalg.svdvals(centered)
        eigvals = singular_values.pow(2) / max(int(normalized_bank.shape[0]), 1)
        eigvals = torch.clamp(eigvals, min=0.0)
        eigvals, _ = torch.sort(eigvals, descending=True)

        total_energy = eigvals.sum()
        total_energy_safe = total_energy.clamp_min(eps)

        pc1_ratio = (eigvals[:1].sum() / total_energy_safe).item() if eigvals.numel() > 0 else 0.0
        top5_ratio = (eigvals[: min(5, eigvals.numel())].sum() / total_energy_safe).item() if eigvals.numel() > 0 else 0.0
        top10_ratio = (eigvals[: min(10, eigvals.numel())].sum() / total_energy_safe).item() if eigvals.numel() > 0 else 0.0

        if eigvals.numel() == 0:
            effective_rank = 0.0
        else:
            probs = eigvals / total_energy_safe
            entropy = -(probs * torch.log(probs + eps)).sum()
            effective_rank = torch.exp(entropy).item()

    return {
        "memory_bank_mean_norm": mean_norm,
        "memory_bank_pc1_ratio": pc1_ratio,
        "memory_bank_top5_ratio": top5_ratio,
        "memory_bank_top10_ratio": top10_ratio,
        "memory_bank_effective_rank": effective_rank,
    }


def format_memory_bank_geometry_stats(stats):
    """Format memory bank geometry stats for human-readable logs."""
    return (
        "MemoryBank "
        f"mean_norm:{stats['memory_bank_mean_norm']:.4f} "
        f"pc1:{stats['memory_bank_pc1_ratio']:.4f} "
        f"top5:{stats['memory_bank_top5_ratio']:.4f} "
        f"top10:{stats['memory_bank_top10_ratio']:.4f} "
        f"erank:{stats['memory_bank_effective_rank']:.2f}"
    )


def maybe_get_memory_bank_geometry_stats(task_embeddings, compute_geometry_stats):
    """Return geometry stats only when the feature is enabled."""
    if not compute_geometry_stats:
        return None
    return compute_memory_bank_geometry_stats(task_embeddings)


def maybe_get_memory_bank_geometry_summary(task_embeddings, compute_geometry_stats):
    """Return formatted geometry stats only when the feature is enabled."""
    stats = maybe_get_memory_bank_geometry_stats(
        task_embeddings,
        compute_geometry_stats=compute_geometry_stats,
    )
    if stats is None:
        return None
    return format_memory_bank_geometry_stats(stats)


def run_validation(
    model,
    val_dataloader,
    device="cuda",
    ignore_index=-100,
    use_task_loss=False,
    task_loss_weight=0.0,
    use_mean_loss=True,
    mean_loss_weight=0.01,
    use_hard_negative_loss=True,
    hard_negative_loss_weight=0.1,
    hard_negative_margin=0.2,
    use_sep_loss=True,
    sep_loss_weight=0.0,
    sep_loss_tau=0.2,
    use_centered_sep=False,
    return_metrics=False,
):
    """Run validation pass and return average loss.
    Switches model to eval() and restores previous training state when finished.
    """
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    compute_bank_routing_metrics = should_compute_bank_routing_metrics(
        use_hard_negative_loss=use_hard_negative_loss,
    )

    val_loss_total = 0.0
    val_mean_loss_total = 0.0
    val_hard_negative_loss_total = 0.0
    val_sep_loss_total = 0.0
    val_sep_loss_raw_total = 0.0
    val_sep_loss_centered_total = 0.0
    val_routing_correct_total = 0
    val_routing_examples_total = 0
    val_routing_margin_total = 0.0
    val_batches = 0
    valid_losses = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_device = model.get_input_device()
            output_device = model.get_output_device()
            input_ids = batch['input_ids'].to(input_device)
            attention_mask = batch['attention_mask'].to(input_device)
            labels = batch['labels'].to(output_device)

            logits, hidden_states = model(
                input_ids,
                attention_mask,
                return_hidden_states=True,
            )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if hidden_states is not None:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
            else:
                shift_hidden_states = None
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            reserved_token_ids = model.get_reserved_token_tensor(shift_logits.device)

            # Check if there are any valid (non-ignored) labels
            valid_mask = shift_labels != ignore_index
            if valid_mask.sum() > 0:
                lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
                if use_task_loss:
                    task_loss, _ = compute_task_loss(
                        shift_logits,
                        shift_labels,
                        reserved_token_ids,
                    )
                else:
                    task_loss = torch.tensor(0.0, device=shift_logits.device)
                mean_loss, _ = compute_memory_bank_mean_stats(
                    get_memory_bank_embeddings(model),
                )
                if compute_bank_routing_metrics:
                    routing_outputs = compute_bank_only_routing_outputs(
                        shift_hidden_states,
                        shift_labels,
                        reserved_token_ids,
                        get_memory_bank_embeddings(model),
                    )
                else:
                    routing_outputs = None

                if compute_bank_routing_metrics and use_hard_negative_loss and hard_negative_loss_weight != 0.0:
                    hard_negative_loss = compute_hard_negative_loss(
                        routing_outputs["bank_logits"],
                        routing_outputs["bank_targets"],
                        hard_negative_margin=hard_negative_margin,
                    )
                else:
                    hard_negative_loss = torch.tensor(0.0, device=shift_logits.device)

                if use_sep_loss and sep_loss_weight != 0.0:
                    sep_loss_components = compute_separation_loss_components(
                        get_separation_loss_embeddings(model),
                        tau=sep_loss_tau,
                        use_centered_sep=use_centered_sep,
                    )
                    sep_loss = sep_loss_components["sep_loss"]
                    sep_loss_raw = sep_loss_components["sep_loss_raw"]
                    sep_loss_centered = sep_loss_components["sep_loss_centered"]
                else:
                    sep_loss = torch.tensor(0.0, device=shift_logits.device)
                    sep_loss_raw = torch.tensor(0.0, device=shift_logits.device)
                    sep_loss_centered = torch.tensor(0.0, device=shift_logits.device)
                loss = (
                    lm_loss
                    + task_loss_weight * task_loss
                    + sep_loss_weight * sep_loss
                )
                if use_mean_loss and mean_loss_weight != 0.0:
                    loss = loss + mean_loss_weight * mean_loss
                if use_hard_negative_loss and hard_negative_loss_weight != 0.0:
                    loss = loss + hard_negative_loss_weight * hard_negative_loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss_total += loss.item()
                    val_mean_loss_total += mean_loss.item()
                    val_hard_negative_loss_total += hard_negative_loss.item()
                    val_sep_loss_total += sep_loss.item()
                    val_sep_loss_raw_total += sep_loss_raw.item()
                    val_sep_loss_centered_total += sep_loss_centered.item()
                    if compute_bank_routing_metrics:
                        val_routing_correct_total += routing_outputs["routing_correct_count"]
                        val_routing_examples_total += routing_outputs["routing_count"]
                        val_routing_margin_total += routing_outputs["routing_margin_sum"].item()
                    valid_losses += 1
            val_batches += 1

    if valid_losses == 0:
        print(f"Warning: No valid validation losses computed ({val_batches} batches processed)")
        avg_val_loss = float('inf')
        avg_mean_loss = 0.0
        avg_hard_negative_loss = 0.0
        avg_sep_loss = 0.0
        avg_sep_loss_raw = 0.0
        avg_sep_loss_centered = 0.0
        avg_routing_bank_acc = 0.0
        avg_routing_bank_margin = 0.0
    else:
        avg_val_loss = val_loss_total / valid_losses
        avg_mean_loss = val_mean_loss_total / valid_losses
        avg_hard_negative_loss = val_hard_negative_loss_total / valid_losses
        avg_sep_loss = val_sep_loss_total / valid_losses
        avg_sep_loss_raw = val_sep_loss_raw_total / valid_losses
        avg_sep_loss_centered = val_sep_loss_centered_total / valid_losses
        if val_routing_examples_total > 0:
            avg_routing_bank_acc = val_routing_correct_total / val_routing_examples_total
            avg_routing_bank_margin = val_routing_margin_total / val_routing_examples_total
        else:
            avg_routing_bank_acc = 0.0
            avg_routing_bank_margin = 0.0

    if was_training:
        model.train()

    if return_metrics:
        return {
            "avg_val_loss": avg_val_loss,
            "avg_mean_loss": avg_mean_loss,
            "avg_hard_negative_loss": avg_hard_negative_loss,
            "avg_sep_loss": avg_sep_loss,
            "avg_sep_loss_raw": avg_sep_loss_raw,
            "avg_sep_loss_centered": avg_sep_loss_centered,
            "routing_bank_acc": avg_routing_bank_acc,
            "routing_bank_margin_avg": avg_routing_bank_margin,
            "valid_losses": valid_losses,
            "val_batches": val_batches,
        }

    return avg_val_loss

def train_task_calling_model(
    model,
    dataloader,
    val_dataloader=None,
    num_epochs=3,
    lr=0.01,
    gradient_accumulation_steps=1,
    device="cuda",
    timestamp=None,
    save_dir="saved_models",
    validate_every_n_steps=1000,
    use_task_loss=False,
    task_loss_weight=0.0,
    use_mean_loss=True,
    mean_loss_weight=0.01,
    use_hard_negative_loss=True,
    hard_negative_loss_weight=0.1,
    hard_negative_margin=0.2,
    use_sep_loss=True,
    sep_loss_weight=0.0,
    sep_loss_tau=0.2,
    use_centered_sep=False,
    compute_memory_bank_geometry_stats=False,
):
    """Train the task calling model using reserved tokens"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    import torch.nn.functional as F
    
    
    # Get training logger
    training_logger = logging.getLogger('training')
    compute_bank_routing_metrics = should_compute_bank_routing_metrics(
        use_hard_negative_loss=use_hard_negative_loss,
    )
    
    model.train()
    
    # Only train the trainable task embedding parameters 
    trainable_params = model.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.0)
    
    total_steps = len(dataloader) * num_epochs
    
    # Create linear learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_steps
    )
    
    print(f"Training for {num_epochs} epochs, {len(dataloader)} batches per epoch")
    print(f"Total steps: {total_steps}")
    print(f"Learning rate: {lr} (with linear schedule + warmup)")
    print(f"Warmup steps: {total_steps // 10}")
    print(f"Use task loss: {use_task_loss}")
    print(f"Task loss weight: {task_loss_weight}")
    print(f"Use mean loss: {use_mean_loss}")
    print(f"Mean loss weight: {mean_loss_weight}")
    print(f"Use hard-negative routing loss: {use_hard_negative_loss}")
    print(f"Hard-negative routing loss weight: {hard_negative_loss_weight}")
    print(f"Hard-negative routing margin: {hard_negative_margin}")
    print(f"Use separation loss: {use_sep_loss}")
    print(f"Separation loss weight: {sep_loss_weight}")
    print(f"Separation loss tau: {sep_loss_tau}")
    print(f"Use centered separation loss: {use_centered_sep}")
    print(f"Compute memory bank geometry stats: {compute_memory_bank_geometry_stats}")
    print(f"Compute bank routing metrics: {compute_bank_routing_metrics}")
    if model.decouple_embeddings:
        print(f"Training mode: Decoupled embeddings")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} (input: {model.trainable_task_input_embeddings.numel()}, output: {model.trainable_task_output_embeddings.numel()})")
    else:
        print(f"Training mode: Coupled embeddings")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} (shared: {model.trainable_task_embeddings.numel()})")
    print(f"Task token IDs to monitor: {model.reserved_token_ids}")
    print()
    
    # Log training configuration
    training_logger.info(f"TRAINING START - Epochs: {num_epochs}, Batches: {len(dataloader)}, Total steps: {total_steps}")
    training_logger.info(f"Config - LR: {lr}, Warmup: {total_steps // 10}, Mode: {'Decoupled' if model.decouple_embeddings else 'Coupled'}")
    training_logger.info(f"Task loss enabled: {use_task_loss}, Weight: {task_loss_weight}")
    training_logger.info(f"Mean loss enabled: {use_mean_loss}")
    training_logger.info(f"Mean loss weight: {mean_loss_weight}")
    training_logger.info(
        f"Hard-negative loss enabled: {use_hard_negative_loss}, Weight: {hard_negative_loss_weight}, "
        f"Margin: {hard_negative_margin}"
    )
    training_logger.info(f"Separation loss enabled: {use_sep_loss}, Weight: {sep_loss_weight}, Tau: {sep_loss_tau}, Centered: {use_centered_sep}")
    training_logger.info(f"Memory bank geometry stats enabled: {compute_memory_bank_geometry_stats}")
    training_logger.info(f"Bank routing metrics enabled: {compute_bank_routing_metrics}")
    training_logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params)}, Task tokens: {model.reserved_token_ids}")
    training_logger.info(f"PyTorch manual seed: {torch.initial_seed()}")
    
    total_loss = 0
    total_task_loss = 0
    total_mean_loss = 0
    total_hard_negative_loss = 0
    total_sep_loss = 0
    total_sep_loss_raw = 0
    total_sep_loss_centered = 0
    task_token_count = 0
    total_routing_correct = 0
    total_routing_margin = 0.0
    task_loss_batches = 0  # Count of batches that had task tokens
    step = 0

    batch_loss = 0
    batch_task_loss = 0
    batch_mean_loss = 0
    batch_hard_negative_loss = 0
    batch_sep_loss = 0
    batch_sep_loss_raw = 0
    batch_sep_loss_centered = 0
    batch_task_count = 0
    batch_routing_correct = 0
    batch_routing_margin = 0.0
    batches_since_log = 0
    
    # Track best validation loss and model state
    best_val_loss = float('inf')
    best_model_state = None
    best_model_path = None
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_device = model.get_input_device()
            output_device = model.get_output_device()
            input_ids = batch['input_ids'].to(input_device)
            attention_mask = batch['attention_mask'].to(input_device)
            labels = batch['labels'].to(output_device)
            
            # Forward pass
            logits, hidden_states = model(
                input_ids,
                attention_mask,
                return_hidden_states=True,
            )
            # Shift logits and labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            if hidden_states is not None:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
            else:
                shift_hidden_states = None
            # Flatten for cross entropy
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            reserved_token_ids = model.get_reserved_token_tensor(shift_logits.device)
            
            # Calculate overall loss (ignore -100 labels)
            lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            task_loss, current_task_count = compute_task_loss(
                shift_logits,
                shift_labels,
                reserved_token_ids,
            )
            mean_loss, mean_norm = compute_memory_bank_mean_stats(
                get_memory_bank_embeddings(model),
            )
            if compute_bank_routing_metrics:
                routing_outputs = compute_bank_only_routing_outputs(
                    shift_hidden_states,
                    shift_labels,
                    reserved_token_ids,
                    get_memory_bank_embeddings(model),
                )
            else:
                routing_outputs = None
            if compute_bank_routing_metrics and use_hard_negative_loss and hard_negative_loss_weight != 0.0:
                hard_negative_loss = compute_hard_negative_loss(
                    routing_outputs["bank_logits"],
                    routing_outputs["bank_targets"],
                    hard_negative_margin=hard_negative_margin,
                )
            else:
                hard_negative_loss = torch.tensor(0.0, device=shift_logits.device)
            if use_sep_loss and sep_loss_weight != 0.0:
                sep_loss_components = compute_separation_loss_components(
                    get_separation_loss_embeddings(model),
                    tau=sep_loss_tau,
                    use_centered_sep=use_centered_sep,
                )
                sep_loss = sep_loss_components["sep_loss"]
                sep_loss_raw = sep_loss_components["sep_loss_raw"]
                sep_loss_centered = sep_loss_components["sep_loss_centered"]
            else:
                sep_loss = torch.tensor(0.0, device=shift_logits.device)
                sep_loss_raw = torch.tensor(0.0, device=shift_logits.device)
                sep_loss_centered = torch.tensor(0.0, device=shift_logits.device)

            loss = lm_loss
            if use_task_loss:
                loss = loss + task_loss_weight * task_loss
            if use_mean_loss and mean_loss_weight != 0.0:
                loss = loss + mean_loss_weight * mean_loss
            if use_hard_negative_loss and hard_negative_loss_weight != 0.0:
                loss = loss + hard_negative_loss_weight * hard_negative_loss
            if use_sep_loss and sep_loss_weight != 0.0:
                loss = loss + sep_loss_weight * sep_loss
            batch_task_count += current_task_count

            batch_loss += loss.item()
            batch_task_loss += task_loss.item()
            batch_mean_loss += mean_loss.item()
            batch_hard_negative_loss += hard_negative_loss.item()
            batch_sep_loss += sep_loss.item()
            batch_sep_loss_raw += sep_loss_raw.item()
            batch_sep_loss_centered += sep_loss_centered.item()
            if compute_bank_routing_metrics:
                batch_routing_correct += routing_outputs["routing_correct_count"]
                batch_routing_margin += routing_outputs["routing_margin_sum"].item()
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_mean_loss += mean_loss.item()
            total_hard_negative_loss += hard_negative_loss.item()
            total_sep_loss += sep_loss.item()
            total_sep_loss_raw += sep_loss_raw.item()
            total_sep_loss_centered += sep_loss_centered.item()
            task_token_count += current_task_count
            if compute_bank_routing_metrics:
                total_routing_correct += routing_outputs["routing_correct_count"]
                total_routing_margin += routing_outputs["routing_margin_sum"].item()
            if current_task_count > 0:
                task_loss_batches += 1
            
            # Increment batch counter for logging
            batches_since_log += 1

            if (batch_idx + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                # current_lr = optimizer.param_groups[0]['lr']

                # Calculate averages over the accumulated batches (100 batches or remaining)
                avg_loss = batch_loss / batches_since_log
                avg_task_loss = batch_task_loss / batches_since_log if batches_since_log > 0 else 0.0
                avg_mean_loss = batch_mean_loss / batches_since_log if batches_since_log > 0 else 0.0
                avg_hard_negative_loss = batch_hard_negative_loss / batches_since_log if batches_since_log > 0 else 0.0
                avg_sep_loss = batch_sep_loss / batches_since_log if batches_since_log > 0 else 0.0
                avg_sep_loss_raw = batch_sep_loss_raw / batches_since_log if batches_since_log > 0 else 0.0
                avg_sep_loss_centered = batch_sep_loss_centered / batches_since_log if batches_since_log > 0 else 0.0
                avg_task_count = batch_task_count / batches_since_log
                if batch_task_count > 0:
                    avg_routing_bank_acc = batch_routing_correct / batch_task_count
                    avg_routing_bank_margin = batch_routing_margin / batch_task_count
                else:
                    avg_routing_bank_acc = 0.0
                    avg_routing_bank_margin = 0.0
                
                completed_batches = epoch * len(dataloader) + batch_idx + 1
                elapsed_seconds = time.time() - training_start_time
                if completed_batches > 0:
                    total_estimated_seconds = elapsed_seconds * total_steps / completed_batches
                    remaining_seconds = max(0.0, total_estimated_seconds - elapsed_seconds)
                else:
                    total_estimated_seconds = 0.0
                    remaining_seconds = 0.0
                time_summary = (
                    f"Remaining/Total: {_format_duration(remaining_seconds)}/"
                    f"{_format_duration(total_estimated_seconds)}"
                )

                print(
                    format_training_progress_message(
                        epoch=epoch + 1,
                        num_epochs=num_epochs,
                        batch_idx=batch_idx + 1,
                        num_batches=len(dataloader),
                        avg_loss=avg_loss,
                        avg_task_loss=avg_task_loss,
                        avg_mean_loss=avg_mean_loss,
                        avg_hard_negative_loss=avg_hard_negative_loss,
                        avg_sep_loss=avg_sep_loss,
                        avg_task_count=avg_task_count,
                        current_lr=current_lr,
                        time_summary=time_summary,
                        mean_norm=mean_norm.item(),
                        show_hard_negative_loss=use_hard_negative_loss,
                        include_bank_routing_metrics=compute_bank_routing_metrics,
                        avg_routing_bank_acc=avg_routing_bank_acc,
                        avg_routing_bank_margin=avg_routing_bank_margin,
                    )
                )
                training_logger.info(
                    format_training_logger_message(
                        epoch=epoch + 1,
                        num_epochs=num_epochs,
                        batch_idx=batch_idx + 1,
                        num_batches=len(dataloader),
                        avg_loss=avg_loss,
                        avg_task_loss=avg_task_loss,
                        avg_mean_loss=avg_mean_loss,
                        avg_hard_negative_loss=avg_hard_negative_loss,
                        avg_sep_loss=avg_sep_loss,
                        avg_task_count=avg_task_count,
                        current_lr=current_lr,
                        time_summary=time_summary,
                        mean_norm=mean_norm.item(),
                        show_hard_negative_loss=use_hard_negative_loss,
                        include_bank_routing_metrics=compute_bank_routing_metrics,
                        avg_routing_bank_acc=avg_routing_bank_acc,
                        avg_routing_bank_margin=avg_routing_bank_margin,
                    )
                )
                if use_centered_sep:
                    training_logger.info(
                        f"SEP LOSS DETAIL Raw:{avg_sep_loss_raw:.4f} Centered:{avg_sep_loss_centered:.4f}"
                    )

                # Reset accumulators for next logging window
                batch_loss = 0
                batch_task_loss = 0
                batch_mean_loss = 0
                batch_hard_negative_loss = 0
                batch_sep_loss = 0
                batch_sep_loss_raw = 0
                batch_sep_loss_centered = 0
                batch_task_count = 0
                batch_routing_correct = 0
                batch_routing_margin = 0.0
                batches_since_log = 0
            
            step += 1

            # Step-based validation
            if val_dataloader is not None and validate_every_n_steps is not None and validate_every_n_steps > 0:
                if step % validate_every_n_steps == 0:
                    val_metrics = run_validation(
                        model,
                        val_dataloader,
                        device=device,
                        ignore_index=-100,
                        use_task_loss=use_task_loss,
                        task_loss_weight=task_loss_weight,
                        use_mean_loss=use_mean_loss,
                        mean_loss_weight=mean_loss_weight,
                        use_hard_negative_loss=use_hard_negative_loss,
                        hard_negative_loss_weight=hard_negative_loss_weight,
                        hard_negative_margin=hard_negative_margin,
                        use_sep_loss=use_sep_loss,
                        sep_loss_weight=sep_loss_weight,
                        sep_loss_tau=sep_loss_tau,
                        use_centered_sep=use_centered_sep,
                        return_metrics=True,
                    )
                    clear_cuda_cache()
                    avg_val_loss = val_metrics["avg_val_loss"]
                    avg_val_mean_loss = val_metrics["avg_mean_loss"]
                    avg_val_hard_negative_loss = val_metrics["avg_hard_negative_loss"]
                    avg_val_sep_loss = val_metrics["avg_sep_loss"]
                    avg_val_sep_loss_raw = val_metrics["avg_sep_loss_raw"]
                    avg_val_sep_loss_centered = val_metrics["avg_sep_loss_centered"]
                    avg_val_routing_bank_acc = val_metrics["routing_bank_acc"]
                    avg_val_routing_bank_margin = val_metrics["routing_bank_margin_avg"]
                    print(
                        format_validation_message(
                            prefix=f"Step {step}",
                            avg_val_loss=avg_val_loss,
                            avg_val_mean_loss=avg_val_mean_loss,
                            avg_val_hard_negative_loss=avg_val_hard_negative_loss,
                            avg_val_sep_loss=avg_val_sep_loss,
                            show_hard_negative_loss=use_hard_negative_loss,
                            include_bank_routing_metrics=compute_bank_routing_metrics,
                            avg_val_routing_bank_acc=avg_val_routing_bank_acc,
                            avg_val_routing_bank_margin=avg_val_routing_bank_margin,
                        )
                    )
                    training_logger.info(
                        format_validation_logger_message(
                            prefix=f"VALIDATION STEP {step}",
                            avg_val_loss=avg_val_loss,
                            avg_val_mean_loss=avg_val_mean_loss,
                            avg_val_hard_negative_loss=avg_val_hard_negative_loss,
                            avg_val_sep_loss=avg_val_sep_loss,
                            show_hard_negative_loss=use_hard_negative_loss,
                            include_bank_routing_metrics=compute_bank_routing_metrics,
                            avg_val_routing_bank_acc=avg_val_routing_bank_acc,
                            avg_val_routing_bank_margin=avg_val_routing_bank_margin,
                        )
                    )
                    geometry_summary = maybe_get_memory_bank_geometry_summary(
                        get_memory_bank_embeddings(model),
                        compute_geometry_stats=compute_memory_bank_geometry_stats,
                    )
                    if geometry_summary is not None:
                        print(f"VALIDATION STEP {step} MEMORY BANK GEOMETRY {geometry_summary}")
                        training_logger.info(f"VALIDATION STEP {step} MEMORY BANK GEOMETRY {geometry_summary}")
                    if use_centered_sep:
                        training_logger.info(
                            f"VALIDATION STEP {step} SEP LOSS DETAIL Raw:{avg_val_sep_loss_raw:.4f} "
                            f"Centered:{avg_val_sep_loss_centered:.4f}"
                        )
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        # Save best token state for later use
                        best_model_state = extract_trained_token_state(model)
                        best_model_path = save_trained_model(
                            model,
                            save_dir=save_dir,
                            timestamp=timestamp,
                            suffix='best',
                        )
                        training_logger.info(f"NEW BEST VALIDATION LOSS: {best_val_loss:.4f} | Saved: {best_model_path}")

        # After each epoch, run validation to compute average validation loss
        if val_dataloader is not None:
            val_metrics = run_validation(
                model,
                val_dataloader,
                device=device,
                ignore_index=-100,
                use_task_loss=use_task_loss,
                task_loss_weight=task_loss_weight,
                use_mean_loss=use_mean_loss,
                mean_loss_weight=mean_loss_weight,
                use_hard_negative_loss=use_hard_negative_loss,
                hard_negative_loss_weight=hard_negative_loss_weight,
                hard_negative_margin=hard_negative_margin,
                use_sep_loss=use_sep_loss,
                sep_loss_weight=sep_loss_weight,
                sep_loss_tau=sep_loss_tau,
                use_centered_sep=use_centered_sep,
                return_metrics=True,
            )
            clear_cuda_cache()
            avg_val_loss = val_metrics["avg_val_loss"]
            avg_val_mean_loss = val_metrics["avg_mean_loss"]
            avg_val_hard_negative_loss = val_metrics["avg_hard_negative_loss"]
            avg_val_sep_loss = val_metrics["avg_sep_loss"]
            avg_val_sep_loss_raw = val_metrics["avg_sep_loss_raw"]
            avg_val_sep_loss_centered = val_metrics["avg_sep_loss_centered"]
            avg_val_routing_bank_acc = val_metrics["routing_bank_acc"]
            avg_val_routing_bank_margin = val_metrics["routing_bank_margin_avg"]
            print(
                format_validation_message(
                    prefix=f"Epoch {epoch+1}/{num_epochs}",
                    avg_val_loss=avg_val_loss,
                    avg_val_mean_loss=avg_val_mean_loss,
                    avg_val_hard_negative_loss=avg_val_hard_negative_loss,
                    avg_val_sep_loss=avg_val_sep_loss,
                    show_hard_negative_loss=use_hard_negative_loss,
                    include_bank_routing_metrics=compute_bank_routing_metrics,
                    avg_val_routing_bank_acc=avg_val_routing_bank_acc,
                    avg_val_routing_bank_margin=avg_val_routing_bank_margin,
                )
            )
            training_logger.info(
                format_validation_logger_message(
                    prefix=f"VALIDATION E{epoch+1}/{num_epochs}",
                    avg_val_loss=avg_val_loss,
                    avg_val_mean_loss=avg_val_mean_loss,
                    avg_val_hard_negative_loss=avg_val_hard_negative_loss,
                    avg_val_sep_loss=avg_val_sep_loss,
                    show_hard_negative_loss=use_hard_negative_loss,
                    include_bank_routing_metrics=compute_bank_routing_metrics,
                    avg_val_routing_bank_acc=avg_val_routing_bank_acc,
                    avg_val_routing_bank_margin=avg_val_routing_bank_margin,
                )
            )
            geometry_summary = maybe_get_memory_bank_geometry_summary(
                get_memory_bank_embeddings(model),
                compute_geometry_stats=compute_memory_bank_geometry_stats,
            )
            if geometry_summary is not None:
                print(f"VALIDATION E{epoch+1}/{num_epochs} MEMORY BANK GEOMETRY {geometry_summary}")
                training_logger.info(f"VALIDATION E{epoch+1}/{num_epochs} MEMORY BANK GEOMETRY {geometry_summary}")
            if use_centered_sep:
                training_logger.info(
                    f"VALIDATION E{epoch+1}/{num_epochs} SEP LOSS DETAIL Raw:{avg_val_sep_loss_raw:.4f} "
                    f"Centered:{avg_val_sep_loss_centered:.4f}"
                )
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best token state for later use
                best_model_state = extract_trained_token_state(model)
                best_model_path = save_trained_model(
                    model,
                    save_dir=save_dir,
                    timestamp=timestamp,
                    suffix='best',
                )
                training_logger.info(f"NEW BEST VALIDATION LOSS: {best_val_loss:.4f} | Saved: {best_model_path}")
    
    avg_total_loss = total_loss / (len(dataloader) * num_epochs)
    avg_mean_loss = total_mean_loss / (len(dataloader) * num_epochs)
    avg_hard_negative_loss = total_hard_negative_loss / (len(dataloader) * num_epochs)
    avg_sep_loss = total_sep_loss / (len(dataloader) * num_epochs)
    avg_sep_loss_raw = total_sep_loss_raw / (len(dataloader) * num_epochs)
    avg_sep_loss_centered = total_sep_loss_centered / (len(dataloader) * num_epochs)
    
    avg_task_loss = total_task_loss / (len(dataloader) * num_epochs)
    if task_token_count > 0:
        avg_routing_bank_acc = total_routing_correct / task_token_count
        avg_routing_bank_margin = total_routing_margin / task_token_count
    else:
        avg_routing_bank_acc = 0.0
        avg_routing_bank_margin = 0.0

    if task_token_count > 0:
        print(f"\nTraining completed!")
        print(f"Average overall loss: {avg_total_loss:.4f}")
        print(f"Average task token loss: {avg_task_loss:.4f}")
        print(f"Average mean loss: {avg_mean_loss:.4f}")
        if use_hard_negative_loss:
            print(f"Average hard-negative loss: {avg_hard_negative_loss:.4f}")
        print(f"Average separation loss: {avg_sep_loss:.4f}")
        if compute_bank_routing_metrics:
            print(f"Routing bank accuracy: {avg_routing_bank_acc:.4f}")
            print(f"Routing bank margin avg: {avg_routing_bank_margin:.4f}")
        print(f"Total task tokens processed: {task_token_count}")
        print(f"Batches with task tokens: {task_loss_batches}/{step}")
        print(f"Task token accuracy insight: Lower task loss indicates better task selection performance")
        
        # Log training completion
        summary = format_training_completion_logger_message(
            avg_total_loss=avg_total_loss,
            avg_task_loss=avg_task_loss,
            avg_mean_loss=avg_mean_loss,
            avg_hard_negative_loss=avg_hard_negative_loss,
            avg_sep_loss=avg_sep_loss,
            show_hard_negative_loss=use_hard_negative_loss,
            include_bank_routing_metrics=compute_bank_routing_metrics,
            avg_routing_bank_acc=avg_routing_bank_acc,
            avg_routing_bank_margin=avg_routing_bank_margin,
            task_token_count=task_token_count,
            task_loss_batches=task_loss_batches,
            step=step,
        )
        training_logger.info(summary)
    else:
        print(f"\nTraining completed! Average loss: {avg_total_loss:.4f}")
        print(f"Average task token loss: {avg_task_loss:.4f}")
        print(f"Average mean loss: {avg_mean_loss:.4f}")
        if use_hard_negative_loss:
            print(f"Average hard-negative loss: {avg_hard_negative_loss:.4f}")
        print(f"Average separation loss: {avg_sep_loss:.4f}")
        if compute_bank_routing_metrics:
            print(f"Routing bank accuracy: {avg_routing_bank_acc:.4f}")
            print(f"Routing bank margin avg: {avg_routing_bank_margin:.4f}")
        print("Warning: No task tokens found in training data!")
        
        # Log training completion without task tokens
        summary = format_training_completion_logger_message(
            avg_total_loss=avg_total_loss,
            avg_task_loss=avg_task_loss,
            avg_mean_loss=avg_mean_loss,
            avg_hard_negative_loss=avg_hard_negative_loss,
            avg_sep_loss=avg_sep_loss,
            show_hard_negative_loss=use_hard_negative_loss,
            include_bank_routing_metrics=compute_bank_routing_metrics,
            avg_routing_bank_acc=avg_routing_bank_acc,
            avg_routing_bank_margin=avg_routing_bank_margin,
            warning_no_task_tokens=True,
        )
        training_logger.info(summary)

    # Report best validation loss if validation was performed
    if val_dataloader is not None and best_val_loss < float('inf'):
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path}")
        training_logger.info(f"BEST VALIDATION LOSS:{best_val_loss:.4f}")

    final_memory_bank_geometry_stats = None
    final_memory_bank_geometry_stats = maybe_get_memory_bank_geometry_stats(
        get_memory_bank_embeddings(model),
        compute_geometry_stats=compute_memory_bank_geometry_stats,
    )
    if final_memory_bank_geometry_stats is not None:
        final_memory_bank_geometry_summary = format_memory_bank_geometry_stats(
            final_memory_bank_geometry_stats
        )
        print(f"Final {final_memory_bank_geometry_summary}")
        training_logger.info(f"FINAL {final_memory_bank_geometry_summary}")
    
    # Return training results including best model state
    return {
        'avg_total_loss': avg_total_loss,
        'avg_task_loss': avg_task_loss,
        'avg_mean_loss': avg_mean_loss,
        'avg_hard_negative_loss': avg_hard_negative_loss,
        'avg_sep_loss': avg_sep_loss,
        'avg_sep_loss_raw': avg_sep_loss_raw,
        'avg_sep_loss_centered': avg_sep_loss_centered,
        'routing_bank_acc': avg_routing_bank_acc,
        'routing_bank_margin_avg': avg_routing_bank_margin,
        'best_val_loss': best_val_loss if val_dataloader is not None else None,
        'best_model_state': best_model_state,
        'best_model_path': best_model_path,
        'memory_bank_geometry_stats': final_memory_bank_geometry_stats,
    }


def save_trained_model(model, save_dir="saved_models", timestamp=None, suffix=None):
    """Save the trained task tokens"""
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"task_tokens_{timestamp}.pt" if not suffix else f"task_tokens_{timestamp}_{suffix}.pt"
    filepath = os.path.join(save_dir, filename)
    
    model.save_task_tokens(filepath)
    return filepath


def write_jsonl(path, rows):
    """Write rows to a UTF-8 JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def demo_task_calling(model, tokenizer, test_examples, device="cuda", use_ground_truth_tasks=False):
    """Demo of task calling using held-out test examples"""
    
    model.eval()
    mode_desc = "Ground Truth Task Inference" if use_ground_truth_tasks else "Normal Task Prediction"
    print(f"\n=== Task Calling Demo ({mode_desc}) ===")
    print(f"Testing on {len(test_examples)} held-out examples")
    print(f"Available tasks: {model.task_names}")
    print()
    
    for i, example in enumerate(test_examples): 
        instruction = example['instruction']
        expected_tasks = example.get('tasks', ['unknown'])
        expected_responses = example.get('responses', [''])
        
        print(f"=== Test Example {i} ===")
        print(f"Instruction: {instruction}")
        print(f"Expected Task(s): {expected_tasks}")
        print(f"Expected Response(s): {expected_responses}")
        print()
        
        # Tokenize instruction input
        instruction_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        instruction_tokens = tokenizer(instruction_text, return_tensors="pt").to(model.get_input_device())
        
        # Generate with task prediction or ground truth tasks
        if use_ground_truth_tasks:
            # Use ground truth tasks for inference
            results = model.generate_with_ground_truth_tasks(
                instruction_tokens['input_ids'], 
                instruction_tokens['attention_mask'], 
                tokenizer,
                ground_truth_tasks=expected_tasks,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=False,    
            )
            print(f"Mode: Ground truth tasks used ({expected_tasks})")
        else:
            # Normal task prediction
            results = model.generate_with_task_prediction(
                instruction_tokens['input_ids'], 
                instruction_tokens['attention_mask'], 
                tokenizer,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,    
            )
            print(f"Mode: Model predicts tasks")
        
        result = results[0]
        
        # Display multi-task results
        if 'predicted_tasks' in result and result['predicted_tasks']:
            print(f"Generated {len(result['predicted_tasks'])} task(s):")
            for j, (task_info, response) in enumerate(zip(result['predicted_tasks'], result['responses'])):
                print(f"  Task {j+1}: {task_info['task_name']}")
                print(f"  Response {j+1}: {response}")
                print()
        else:
            # Backward compatibility - single task display
            print(f"Predicted Task: {result['predicted_task_name']}")
            print(f"Task Token Used: {result.get('task_token_used', 'N/A')}")
            print(f"Response: {result['response']}")
        
        print(f"Full Generated: {result['full_generated_sequence']}")
        print("-" * 50)
        print()

def eval_task_calling(
    model,
    tokenizer,
    test_dataloader,
    device="cuda",
    use_ground_truth_tasks=False,
    predictions_output_path=None,
    prompt_mode_label=None,
):
    """Comprehensive evaluation of task calling model using Natural Instructions metrics"""
    import time
    from collections import defaultdict
    from natural_instructions_eval import (
        evaluate_predictions,
        exact_match,
        metric_max_over_ground_truths,
        print_evaluation_results,
        rouge_score,
    )
    
    # Get evaluation logger
    eval_logger = logging.getLogger('evaluation')
    
    model.eval()
    
    # Calculate total examples from dataloader
    total_examples = len(test_dataloader.dataset)
    mode_desc = "Ground Truth Task Inference" if use_ground_truth_tasks else "Normal Task Prediction"
    if prompt_mode_label is not None:
        mode_desc = f"{mode_desc} | Prompt: {prompt_mode_label}"
    print(f"\n=== Task Calling Evaluation ({mode_desc}) ===")
    print(f"Evaluating on {total_examples} test examples")
    print()
    
    # Log evaluation start
    eval_logger.info(f"EVALUATION START - Mode:{mode_desc} Examples:{total_examples}")
    eval_logger.info(f"PyTorch manual seed: {torch.initial_seed()}")
    
    # Collect predictions and references for Natural Instructions evaluation
    all_predictions = []
    all_references = []
    all_task_names = []
    prediction_rows = []
    
    # Legacy metrics for compatibility
    task_correct = 0
    task_breakdown = defaultdict(lambda: {'total': 0, 'task_correct': 0})
    
    # Evaluation loop with batches
    start_time = time.time()
    print("🔄 Running batch evaluation...")

    def build_prediction_row(
        example,
        result,
        predicted_tasks,
        predicted_responses,
        expected_tasks,
        expected_responses,
        task_match,
        error=None,
    ):
        expected_task = expected_tasks[0] if expected_tasks else None
        predicted_task = predicted_tasks[0] if predicted_tasks else None
        expected_response = expected_responses[0] if expected_responses else ""
        predicted_response = predicted_responses[0] if predicted_responses else ""
        has_task_prediction = len(predicted_tasks) > 0

        if error is not None:
            prediction_status = "error"
        elif has_task_prediction:
            prediction_status = "task_and_response"
        elif predicted_response:
            prediction_status = "response_only_no_task_token"
        else:
            prediction_status = "empty_prediction"

        response_exact_match = bool(
            metric_max_over_ground_truths(
                lambda prediction, ground_truth, xlingual=False: (
                    1.0 if exact_match(prediction, ground_truth, xlingual) else 0.0
                ),
                predicted_response,
                expected_responses,
                xlingual=False,
            )
        )
        response_rouge_l = metric_max_over_ground_truths(
            rouge_score,
            predicted_response,
            expected_responses,
            xlingual=False,
        )

        row = {
            "example_index": len(prediction_rows),
            "mode": "ground_truth_tasks" if use_ground_truth_tasks else "normal_task_prediction",
            "prompt_mode": prompt_mode_label,
            "prediction_status": prediction_status,
            "instruction": example.get("instruction", ""),
            "query": example.get("query", ""),
            "expected_task": expected_task,
            "predicted_task": predicted_task,
            "expected_tasks": expected_tasks,
            "predicted_tasks": predicted_tasks,
            "has_task_prediction": has_task_prediction,
            "task_match": task_match,
            "expected_response": expected_response,
            "predicted_response": predicted_response,
            "expected_responses": expected_responses,
            "predicted_responses": predicted_responses,
            "response_exact_match": response_exact_match,
            "response_rouge_l": round(response_rouge_l, 4),
            "full_generated_sequence": result.get("full_generated_sequence", ""),
            "task_token_used": result.get("task_token_used"),
            "source_instance_index": example.get("source_instance_index"),
        }
        if error is not None:
            row["error"] = error
        return row
    
    processed_examples = 0
    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch['raw_data'])
        processed_examples += batch_size
        
        if batch_idx % 10 == 0 or processed_examples == total_examples:
            progress_pct = 100 * processed_examples / total_examples
            print(f"   Progress: {processed_examples}/{total_examples} ({progress_pct:.1f}%)")
            eval_logger.info(f"Progress: {processed_examples}/{total_examples} ({progress_pct:.1f}%)")
        
        try:
            # With eval mode dataset, input_ids already contain just instruction input
            input_ids = batch['input_ids'].to(model.get_input_device())
            attention_mask = batch['attention_mask'].to(model.get_input_device())
            
            # Generate with task prediction or ground truth tasks
            if use_ground_truth_tasks:
                # For ground truth task inference, we need to get ground truth tasks for each example
                batch_results = []
                for i in range(batch_size):
                    example = batch['raw_data'][i]
                    expected_tasks = example.get('tasks', ['unknown'])
                    
                    # Generate for single example
                    single_input = input_ids[i:i+1]
                    single_mask = attention_mask[i:i+1]
                    
                    single_result = model.generate_with_ground_truth_tasks(
                        single_input, 
                        single_mask, 
                        tokenizer,
                        ground_truth_tasks=expected_tasks,
                        max_new_tokens=256,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=False,
                    )
                    batch_results.extend(single_result)
            else:
                # Normal task prediction for batch
                batch_results = model.generate_with_task_prediction(
                    input_ids, 
                    attention_mask, 
                    tokenizer,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                )
            
            # Process each example in the batch
            for i in range(batch_size):
                example = batch['raw_data'][i]
                result = batch_results[i]
                
                # Extract expected data
                expected_tasks = example.get('tasks', ['unknown'])
                expected_responses = example.get('responses', [''])
                
                # Extract predicted tasks and responses
                if 'predicted_tasks' in result and result['predicted_tasks']:
                    predicted_tasks = [task_info['task_name'] for task_info in result['predicted_tasks']]
                    predicted_responses = result['responses']
                else:
                    # Backward compatibility
                    predicted_tasks = [result['predicted_task_name']] if result['predicted_task_name'] != 'none' else []
                    predicted_responses = [result['response']] if result['response'] else []
                
                # Evaluate task prediction accuracy (for legacy compatibility)
                task_match = set(predicted_tasks) == set(expected_tasks)
                if task_match:
                    task_correct += 1
                
                # Track task-level accuracy
                for task in expected_tasks:
                    task_breakdown[task]['total'] += 1
                    if task_match:
                        task_breakdown[task]['task_correct'] += 1
                
                # Collect predictions and references for Natural Instructions evaluation
                # Use the first predicted response (or empty string if none)
                pred_text = predicted_responses[0] if predicted_responses else ""
                all_predictions.append(pred_text)
                
                # Natural Instructions supports multiple references
                all_references.append(expected_responses)
                
                # Use the first expected task for per-task breakdown
                task_name = expected_tasks[0] if expected_tasks else "unknown"
                all_task_names.append(task_name)

                prediction_rows.append(
                    build_prediction_row(
                        example=example,
                        result=result,
                        predicted_tasks=predicted_tasks,
                        predicted_responses=predicted_responses,
                        expected_tasks=expected_tasks,
                        expected_responses=expected_responses,
                        task_match=task_match,
                    )
                )
                        
        except Exception as e:
            print(f"   Error processing batch {batch_idx + 1}: {str(e)}")
            # Add empty predictions for failed batch
            for i in range(batch_size):
                example = batch['raw_data'][i]
                all_predictions.append("")
                all_references.append([""])
                all_task_names.append("error")
                prediction_rows.append(
                    build_prediction_row(
                        example=example,
                        result={},
                        predicted_tasks=[],
                        predicted_responses=[],
                        expected_tasks=example.get("tasks", ["unknown"]),
                        expected_responses=example.get("responses", [""]),
                        task_match=False,
                        error=str(e),
                    )
                )
            continue
    
    eval_time = time.time() - start_time
    
    # Use Natural Instructions evaluation
    print("\n🔍 Computing Natural Instructions metrics...")
    ni_results = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names,
        xlingual=False
    )
    
    # Print Natural Instructions evaluation results
    print_evaluation_results(ni_results, f"NATURAL INSTRUCTIONS EVALUATION ({mode_desc})")
    
    # Legacy task accuracy calculation
    task_accuracy = task_correct / total_examples
    
    print(f"\n🎯 TASK PREDICTION ACCURACY: {task_accuracy:.3f} ({task_correct}/{total_examples})")
    print(f"⏱️  Total evaluation time: {eval_time:.2f} seconds")
    
    # Log evaluation results
    eval_logger.info(f"EVALUATION COMPLETE - TaskAcc:{task_accuracy:.3f} ExactMatch:{ni_results['exact_match']:.1f}% RougeL:{ni_results['rougeL']:.1f}% Time:{eval_time:.1f}s")
    
    # Log per-task performance
    if task_breakdown:
        for task_name in sorted(task_breakdown.keys()):
            stats = task_breakdown[task_name]
            task_acc = stats['task_correct'] / stats['total'] if stats['total'] > 0 else 0.0
            
            # Get NI metrics for this task if available
            ni_metrics = ni_results.get('per_task', {}).get(task_name, {})
            exact_match = ni_metrics.get('exact_match', 0)
            rouge_l = ni_metrics.get('rougeL', 0)
            
            eval_logger.info(f"Task:{task_name} TaskAcc:{task_acc:.3f} ExactMatch:{exact_match:.1f}% RougeL:{rouge_l:.1f}% Examples:{stats['total']}")
    
    # Per-task task accuracy breakdown (console output)
    if task_breakdown:
        print(f"\n📊 TASK PREDICTION ACCURACY BY TASK:")
        print("-" * 60)
        for task_name in sorted(task_breakdown.keys()):
            stats = task_breakdown[task_name]
            task_acc = stats['task_correct'] / stats['total'] if stats['total'] > 0 else 0.0
            print(f"   {task_name}: {task_acc:.3f} ({stats['total']} examples)")

    if predictions_output_path is not None:
        write_jsonl(predictions_output_path, prediction_rows)
        print(f"Saved per-example evaluation predictions to: {predictions_output_path}")
        eval_logger.info(f"PREDICTIONS SAVED:{predictions_output_path}")
    
    # Return results compatible with existing code
    return {
        'prompt_mode': prompt_mode_label,
        'exact_accuracy': ni_results['exact_match'] / 100.0,  # Convert percentage to decimal
        'task_accuracy': task_accuracy,
        'avg_response_score': ni_results['rougeL'] / 100.0,  # Use ROUGE-L as response score
        'total_examples': total_examples,
        'ni_exact_match': ni_results['exact_match'],
        'ni_rouge_l': ni_results['rougeL'],
        'ni_per_task': ni_results.get('per_task', {}),
        'task_breakdown': dict(task_breakdown),
        'predictions_output_path': predictions_output_path,
    }
