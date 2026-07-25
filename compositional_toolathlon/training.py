from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .masked_routing import (
    apply_masked_logit_bias,
    mask_backbone_tool_logits,
    mask_tool_head_logits,
)


@dataclass
class StepLoss:
    total_loss: torch.Tensor
    ar_loss: torch.Tensor
    routing_loss: torch.Tensor
    valid_token_count: int
    routing_site_count: int
    assistant_start_site_count: int
    eoc_transition_site_count: int


def gather_assistant_start_boundaries(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    model: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect the one next-tool boundary from every valid step sample."""
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    token_to_tool = getattr(model, "token_id_to_tool_id", {})
    boundary_hidden = []
    targets = []
    batch_indices = []
    time_indices = []

    for batch_index in range(labels.shape[0]):
        valid_positions = torch.nonzero(
            valid_mask[batch_index],
            as_tuple=False,
        ).flatten()
        if valid_positions.numel() == 0:
            continue
        position = int(valid_positions[0].item())
        next_token_id = int(shift_labels[batch_index, position].item())
        target = token_to_tool.get(next_token_id)
        if target is None:
            raise ValueError(
                "the first supervised token of every Toolathlon step must be a memory tool token"
            )
        if labels[batch_index, position].item() != -100:
            raise ValueError("tool boundary must follow loss-masked context")
        boundary_hidden.append(hidden_states[batch_index, position])
        targets.append(int(target))
        batch_indices.append(batch_index)
        time_indices.append(position)

    if not boundary_hidden:
        hidden_width = hidden_states.shape[-1]
        return (
            hidden_states.new_zeros((0, hidden_width)),
            torch.zeros(0, dtype=torch.long, device=hidden_states.device),
            torch.zeros(0, dtype=torch.long, device=hidden_states.device),
            torch.zeros(0, dtype=torch.long, device=hidden_states.device),
        )
    return (
        torch.stack(boundary_hidden),
        torch.tensor(targets, dtype=torch.long, device=hidden_states.device),
        torch.tensor(batch_indices, dtype=torch.long, device=hidden_states.device),
        torch.tensor(time_indices, dtype=torch.long, device=hidden_states.device),
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("weighted mean expects aligned one-dimensional tensors")
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return (values * weights).sum() / denominator


def compute_stepwise_loss(
    *,
    model: Any,
    batch: dict[str, Any],
    use_logit_bias: bool,
    use_logit_train_add: bool,
    detach: bool,
    logit_bias_loss_weight: float,
) -> StepLoss:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    available_mask = batch["available_tool_mask"]
    episode_weight = batch["episode_weight"].to(dtype=torch.float32)

    logits, hidden_states = model(
        input_ids,
        attention_mask,
        return_hidden_states=True,
    )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not bool(valid_mask.any()):
        raise ValueError("batch has no supervised target tokens")

    tool_token_ids = list(getattr(model, "tool_reserved_token_ids", []))
    shift_logits = mask_backbone_tool_logits(
        shift_logits,
        tool_token_ids,
        available_mask,
    )
    (
        boundary_hidden,
        boundary_targets,
        boundary_batch_indices,
        boundary_time_indices,
    ) = gather_assistant_start_boundaries(hidden_states, labels, model)
    if boundary_targets.numel() != input_ids.shape[0]:
        raise ValueError("every step sample must contribute exactly one assistant-start site")

    routing_loss = shift_logits.new_zeros(())
    if use_logit_bias:
        if getattr(model, "logit_bias_head", None) is None:
            raise ValueError("TapMem training requires a logit-bias head")
        head_input = boundary_hidden.detach() if detach else boundary_hidden
        tool_logits = model._get_logit_bias_scores(head_input)
        boundary_available = available_mask.index_select(0, boundary_batch_indices)
        masked_tool_logits = mask_tool_head_logits(tool_logits, boundary_available)
        target_available = boundary_available.gather(
            1,
            boundary_targets[:, None],
        ).squeeze(1)
        if not bool(target_available.all()):
            raise ValueError("a routing target is absent from its available-tool mask")

        routing_per_site = F.cross_entropy(
            masked_tool_logits,
            boundary_targets,
            reduction="none",
        )
        boundary_weights = episode_weight.index_select(0, boundary_batch_indices)
        routing_loss = _weighted_mean(routing_per_site, boundary_weights)

        if use_logit_train_add:
            boundary_vocab_logits = shift_logits[
                boundary_batch_indices,
                boundary_time_indices,
            ]
            biased_boundary_logits = apply_masked_logit_bias(
                boundary_vocab_logits,
                tool_logits,
                tool_token_ids,
                boundary_available,
                scale=float(getattr(model, "logit_bias_scale", 1.0)),
            )
            shift_logits = shift_logits.clone()
            shift_logits[
                boundary_batch_indices,
                boundary_time_indices,
            ] = biased_boundary_logits
    elif use_logit_train_add:
        raise ValueError("use_logit_train_add requires use_logit_bias")

    position_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    valid_counts = valid_mask.sum(dim=1)
    if not bool((valid_counts > 0).all()):
        raise ValueError("every step sample must contain supervised target tokens")
    per_sample_ar = (position_loss * valid_mask).sum(dim=1) / valid_counts
    ar_loss = _weighted_mean(per_sample_ar, episode_weight)
    total_loss = ar_loss + float(logit_bias_loss_weight) * routing_loss
    return StepLoss(
        total_loss=total_loss,
        ar_loss=ar_loss,
        routing_loss=routing_loss,
        valid_token_count=int(valid_mask.sum().item()),
        routing_site_count=int(boundary_targets.numel()),
        assistant_start_site_count=int(boundary_targets.numel()),
        eoc_transition_site_count=0,
    )


def _move_batch(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_stepwise_model(
    *,
    model: Any,
    dataloader: Any,
    num_epochs: int,
    lr: float,
    device: str | torch.device,
    use_logit_bias: bool,
    use_logit_train_add: bool,
    detach: bool = True,
    logit_bias_loss_weight: float = 0.1,
    gradient_accumulation_steps: int = 1,
    validation_dataloader: Any | None = None,
) -> dict[str, Any]:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    if num_epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    if getattr(model, "lora_config", None):
        raise ValueError("Toolathlon TokMem/TapMem protocol forbids LoRA/adaptation")

    if len(dataloader) == 0:
        raise ValueError("Toolathlon training dataloader is empty")
    if validation_dataloader is not None and len(validation_dataloader) == 0:
        raise ValueError("Toolathlon validation dataloader is empty")

    trainable_parameters = list(model.get_trainable_parameters())
    if not trainable_parameters:
        raise ValueError("Toolathlon model exposes no trainable memory parameters")
    optimizer = AdamW(
        [{"params": trainable_parameters, "lr": lr, "weight_decay": 0.0}]
    )
    optimizer_steps_per_epoch = math.ceil(
        len(dataloader) / gradient_accumulation_steps
    )
    total_optimizer_steps = num_epochs * optimizer_steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_optimizer_steps // 10,
        num_training_steps=total_optimizer_steps,
    )

    history = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    global_optimizer_step = 0
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_trainable_state: list[torch.Tensor] | None = None
    for epoch in range(num_epochs):
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        epoch_totals = {
            "total_loss": 0.0,
            "ar_loss": 0.0,
            "routing_loss": 0.0,
            "batches": 0,
            "valid_token_count": 0,
            "routing_site_count": 0,
            "assistant_start_site_count": 0,
            "eoc_transition_site_count": 0,
            "sample_weight": 0.0,
        }

        for batch_index, raw_batch in enumerate(dataloader):
            batch = _move_batch(raw_batch, device)
            losses = compute_stepwise_loss(
                model=model,
                batch=batch,
                use_logit_bias=use_logit_bias,
                use_logit_train_add=use_logit_train_add,
                detach=detach,
                logit_bias_loss_weight=logit_bias_loss_weight,
            )
            if not torch.isfinite(losses.total_loss):
                raise FloatingPointError("non-finite Toolathlon training loss")
            (losses.total_loss / gradient_accumulation_steps).backward()

            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(dataloader)
            )
            if should_step:
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in trainable_parameters
                ):
                    raise FloatingPointError("non-finite Toolathlon training gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_optimizer_step += 1

            batch_weight = float(batch["episode_weight"].sum().item())
            epoch_totals["total_loss"] += (
                float(losses.total_loss.detach().item()) * batch_weight
            )
            epoch_totals["ar_loss"] += (
                float(losses.ar_loss.detach().item()) * batch_weight
            )
            epoch_totals["routing_loss"] += (
                float(losses.routing_loss.detach().item()) * batch_weight
            )
            epoch_totals["batches"] += 1
            epoch_totals["sample_weight"] += batch_weight
            epoch_totals["valid_token_count"] += losses.valid_token_count
            epoch_totals["routing_site_count"] += losses.routing_site_count
            epoch_totals["assistant_start_site_count"] += (
                losses.assistant_start_site_count
            )
            epoch_totals["eoc_transition_site_count"] += (
                losses.eoc_transition_site_count
            )

        denominator = max(
            torch.finfo(torch.float32).eps,
            epoch_totals["sample_weight"],
        )
        history.append(
            {
                "epoch": epoch + 1,
                "avg_total_loss": epoch_totals["total_loss"] / denominator,
                "avg_ar_loss": epoch_totals["ar_loss"] / denominator,
                "avg_routing_loss": epoch_totals["routing_loss"] / denominator,
                "valid_token_count": epoch_totals["valid_token_count"],
                "routing_site_count": epoch_totals["routing_site_count"],
                "assistant_start_site_count": epoch_totals[
                    "assistant_start_site_count"
                ],
                "eoc_transition_site_count": epoch_totals[
                    "eoc_transition_site_count"
                ],
            }
        )
        if validation_dataloader is not None:
            validation_metrics = evaluate_stepwise_loss(
                model=model,
                dataloader=validation_dataloader,
                device=device,
                use_logit_bias=use_logit_bias,
                use_logit_train_add=use_logit_train_add,
                detach=detach,
                logit_bias_loss_weight=logit_bias_loss_weight,
            )
            history[-1]["validation"] = validation_metrics
            validation_loss = float(validation_metrics["avg_total_loss"])
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_trainable_state = [
                    parameter.detach().cpu().clone()
                    for parameter in trainable_parameters
                ]
            model.train()

    if best_trainable_state is not None:
        with torch.no_grad():
            for parameter, saved_parameter in zip(
                trainable_parameters,
                best_trainable_state,
                strict=True,
            ):
                parameter.copy_(
                    saved_parameter.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
    else:
        best_epoch = num_epochs
    return {
        "epochs": history,
        "optimizer_steps": global_optimizer_step,
        "use_logit_bias": use_logit_bias,
        "use_logit_train_add": use_logit_train_add,
        "detach": detach,
        "logit_bias_loss_weight": logit_bias_loss_weight,
        "checkpoint_selection": {
            "metric": (
                "validation.avg_total_loss"
                if validation_dataloader is not None
                else "final_epoch"
            ),
            "best_epoch": best_epoch,
            "best_value": (
                best_validation_loss
                if validation_dataloader is not None
                else None
            ),
        },
    }


def evaluate_stepwise_loss(
    *,
    model: Any,
    dataloader: Any,
    device: str | torch.device,
    use_logit_bias: bool,
    use_logit_train_add: bool,
    detach: bool = True,
    logit_bias_loss_weight: float = 0.1,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "total": 0.0,
        "ar": 0.0,
        "routing": 0.0,
        "batches": 0,
        "sample_weight": 0.0,
    }
    with torch.inference_mode():
        for raw_batch in dataloader:
            batch = _move_batch(raw_batch, device)
            losses = compute_stepwise_loss(
                model=model,
                batch=batch,
                use_logit_bias=use_logit_bias,
                use_logit_train_add=use_logit_train_add,
                detach=detach,
                logit_bias_loss_weight=logit_bias_loss_weight,
            )
            batch_weight = float(batch["episode_weight"].sum().item())
            totals["total"] += float(losses.total_loss.item()) * batch_weight
            totals["ar"] += float(losses.ar_loss.item()) * batch_weight
            totals["routing"] += float(losses.routing_loss.item()) * batch_weight
            totals["batches"] += 1
            totals["sample_weight"] += batch_weight
    if totals["batches"] == 0:
        raise ValueError("Toolathlon evaluation dataloader is empty")
    denominator = max(
        torch.finfo(torch.float32).eps,
        totals["sample_weight"],
    )
    return {
        "avg_total_loss": totals["total"] / denominator,
        "avg_ar_loss": totals["ar"] / denominator,
        "avg_routing_loss": totals["routing"] / denominator,
        "batches": totals["batches"],
    }
