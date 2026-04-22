import logging
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:
    FSDP = None


def _reset_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    return logger


def setup_logging(log_dir="logs", is_main_process=True):
    """Set up training/evaluation loggers. Only the main process writes files."""
    training_logger = _reset_logger("training")
    eval_logger = _reset_logger("evaluation")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log = None
    evaluation_log = None

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        training_log = os.path.join(log_dir, f"training_{timestamp}.log")
        evaluation_log = os.path.join(log_dir, f"evaluation_{timestamp}.log")

        training_handler = logging.FileHandler(training_log)
        training_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        training_logger.addHandler(training_handler)

        eval_handler = logging.FileHandler(evaluation_log)
        eval_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        eval_logger.addHandler(eval_handler)
    else:
        training_logger.addHandler(logging.NullHandler())
        eval_logger.addHandler(logging.NullHandler())

    return training_logger, eval_logger, training_log, evaluation_log, timestamp


def _unwrap_model(model, accelerator=None):
    return accelerator.unwrap_model(model) if accelerator is not None else model


def extract_trained_token_state(model, accelerator=None):
    """Return a minimal CPU state dict containing only trainable task-token parameters."""
    model = _unwrap_model(model, accelerator=accelerator)

    state = {}
    if getattr(model, "decouple_embeddings", False):
        state["trainable_task_input_embeddings"] = (
            model.trainable_task_input_embeddings.detach().cpu().clone()
        )
        state["trainable_task_output_embeddings"] = (
            model.trainable_task_output_embeddings.detach().cpu().clone()
        )
    else:
        state["trainable_task_embeddings"] = (
            model.trainable_task_embeddings.detach().cpu().clone()
        )
    if getattr(model, "use_logit_bias", False) and getattr(model, "logit_bias_head", None) is not None:
        for key, value in model.logit_bias_head.state_dict().items():
            state[f"logit_bias_head.{key}"] = value.detach().cpu().clone()
    return state


def create_optimizer(model, lr=0.01):
    """Create the optimizer for task-token training."""
    trainable_params = model.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    return optimizer


def create_scheduler(optimizer, dataloader, num_epochs=3, gradient_accumulation_steps=1):
    """Create optimizer-step-aware warmup/decay after dataloader sharding is finalized."""
    num_update_steps_per_epoch = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    total_optimizer_steps = num_update_steps_per_epoch * num_epochs
    warmup_steps = total_optimizer_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    return scheduler, total_optimizer_steps, warmup_steps


def compute_logit_bias_loss(model, shift_hidden_states, shift_labels, ignore_index=-100):
    """Compute detached hidden-state supervision for the first-step logit-bias head."""
    import torch.nn.functional as F

    if not getattr(model, "use_logit_bias", False) or shift_hidden_states is None:
        return torch.tensor(0.0, device=shift_labels.device), 0

    reserved_token_tensor = model.reserved_token_tensor.to(shift_labels.device)
    valid_mask = shift_labels != ignore_index
    task_token_mask = torch.isin(shift_labels, reserved_token_tensor) & valid_mask
    routing_count = int(task_token_mask.sum().item())
    if routing_count == 0:
        return torch.tensor(0.0, device=shift_labels.device), 0

    bias_logits = model.compute_task_logit_bias(
        shift_hidden_states[task_token_mask],
        detach_hidden_states=True,
    )
    bias_targets = model.get_task_bias_targets(shift_labels[task_token_mask]).to(bias_logits.device)
    logit_bias_loss = F.cross_entropy(bias_logits, bias_targets)
    return logit_bias_loss, routing_count


def _maybe_move_batch(batch, device, accelerator=None):
    if accelerator is not None:
        return batch["input_ids"], batch["attention_mask"], batch["labels"]
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )


def _reduce_metrics(values, accelerator=None, device=None):
    tensor = torch.tensor(values, device=accelerator.device if accelerator is not None else device)
    if accelerator is not None:
        tensor = accelerator.reduce(tensor, reduction="sum")
    return tensor


def _fsdp_generation_context(model, accelerator=None):
    """Return a full-param context for FSDP generation/eval when needed."""
    if FSDP is None:
        return nullcontext()
    if not isinstance(model, FSDP):
        return nullcontext()
    if accelerator is not None and getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return nullcontext()
    return FSDP.summon_full_params(model, recurse=True, writeback=False)


def _sync_fsdp_ignored_module_gradients(model, accelerator=None):
    """Average gradients for FSDP-ignored replicated modules across ranks."""
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


def run_validation(model, val_dataloader, device="cuda", ignore_index=-100,
                   use_logit_bias=False, logit_bias_loss_weight=0.0, accelerator=None):
    """Run validation and return globally reduced average losses."""
    import torch.nn.functional as F

    base_model = _unwrap_model(model, accelerator=accelerator)
    was_training = model.training
    model.eval()

    val_loss_total = 0.0
    val_logit_bias_loss_total = 0.0
    valid_losses = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = _maybe_move_batch(batch, device, accelerator=accelerator)

            if use_logit_bias:
                logits, hidden_states = model(input_ids, attention_mask, return_hidden_states=True)
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
            else:
                logits = model(input_ids, attention_mask)
                shift_hidden_states = None

            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)

            valid_mask = shift_labels != ignore_index
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
                logit_bias_loss = torch.tensor(0.0, device=shift_labels.device)
                if use_logit_bias:
                    logit_bias_loss, _ = compute_logit_bias_loss(
                        base_model,
                        shift_hidden_states,
                        shift_labels,
                        ignore_index=ignore_index,
                    )
                    loss = loss + logit_bias_loss_weight * logit_bias_loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss_total += loss.item()
                    val_logit_bias_loss_total += logit_bias_loss.item()
                    valid_losses += 1

    metrics = _reduce_metrics(
        [val_loss_total, val_logit_bias_loss_total, float(valid_losses)],
        accelerator=accelerator,
        device=torch.device(device),
    )
    val_loss_total = metrics[0].item()
    val_logit_bias_loss_total = metrics[1].item()
    valid_losses = int(metrics[2].item())

    if valid_losses == 0:
        avg_val_loss = float("inf")
        avg_logit_bias_loss = 0.0
    else:
        avg_val_loss = val_loss_total / valid_losses
        avg_logit_bias_loss = val_logit_bias_loss_total / valid_losses

    if was_training:
        model.train()

    return {
        "avg_val_loss": avg_val_loss,
        "avg_logit_bias_loss": avg_logit_bias_loss,
    }


def train_task_calling_model(model, dataloader, optimizer, scheduler, val_dataloader=None,
                             num_epochs=3, lr=0.01, gradient_accumulation_steps=1,
                             device="cuda", timestamp=None, validate_every_n_steps=1000,
                             use_logit_bias=False, logit_bias_loss_weight=0.0,
                             accelerator=None):
    """Train the task-calling model with optional Accelerate/FSDP orchestration."""
    import torch.nn.functional as F

    training_logger = logging.getLogger("training")
    should_log = accelerator is None or accelerator.is_main_process

    base_model = _unwrap_model(model, accelerator=accelerator)
    model.train()
    trainable_params = base_model.get_trainable_parameters()
    total_optimizer_steps = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps) * num_epochs)

    if should_log:
        print(f"Training for {num_epochs} epochs, {len(dataloader)} batches per epoch")
        print(f"Total optimizer steps: {total_optimizer_steps}")
        print(f"Learning rate: {lr} (with linear schedule + warmup)")
        print(f"Warmup steps: {total_optimizer_steps // 10}")
        print(f"Use logit bias: {use_logit_bias}")
        print(f"Logit bias loss weight: {logit_bias_loss_weight}")
        if base_model.decouple_embeddings:
            print("Training mode: Decoupled embeddings")
            print(
                "Trainable parameters: "
                f"{sum(p.numel() for p in trainable_params)} "
                f"(input: {base_model.trainable_task_input_embeddings.numel()}, "
                f"output: {base_model.trainable_task_output_embeddings.numel()})"
            )
        else:
            print("Training mode: Coupled embeddings")
            print(
                "Trainable parameters: "
                f"{sum(p.numel() for p in trainable_params)} "
                f"(shared: {base_model.trainable_task_embeddings.numel()})"
            )
        print(f"Task token IDs to monitor: {base_model.reserved_token_ids}")
        print()

    training_logger.info(
        f"TRAINING START - Epochs: {num_epochs}, Batches: {len(dataloader)}, Total steps: {total_optimizer_steps}"
    )
    training_logger.info(
        f"Config - LR: {lr}, Warmup: {total_optimizer_steps // 10}, "
        f"Mode: {'Decoupled' if base_model.decouple_embeddings else 'Coupled'}"
    )
    training_logger.info(
        f"Logit bias enabled: {use_logit_bias}, Weight: {logit_bias_loss_weight}"
    )
    training_logger.info(
        f"Trainable params: {sum(p.numel() for p in trainable_params)}, Task tokens: {base_model.reserved_token_ids}"
    )
    training_logger.info(f"PyTorch manual seed: {torch.initial_seed()}")

    total_loss = 0.0
    total_task_loss = 0.0
    total_logit_bias_loss = 0.0
    task_token_count = 0
    task_loss_batches = 0
    processed_batches = 0
    forward_step = 0

    batch_loss = 0.0
    batch_task_loss = 0.0
    batch_logit_bias_loss = 0.0
    batch_task_count = 0
    batches_since_log = 0

    best_val_loss = float("inf")
    best_model_path = None
    metric_device = accelerator.device if accelerator is not None else torch.device(device)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            accumulation_context = accelerator.accumulate(model) if accelerator is not None else nullcontext()
            with accumulation_context:
                input_ids, attention_mask, labels = _maybe_move_batch(batch, device, accelerator=accelerator)

                if use_logit_bias:
                    logits, hidden_states = model(input_ids, attention_mask, return_hidden_states=True)
                    shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                    shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
                else:
                    logits = model(input_ids, attention_mask)
                    shift_hidden_states = None

                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)

                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                batch_loss += loss.item()

                task_mask = torch.isin(shift_labels, base_model.reserved_token_tensor)
                valid_mask = shift_labels != -100
                task_token_mask = task_mask & valid_mask
                current_batch_task_count = int(task_token_mask.sum().item())

                if current_batch_task_count > 0:
                    task_loss = F.cross_entropy(shift_logits[task_token_mask], shift_labels[task_token_mask])
                else:
                    task_loss = torch.tensor(0.0, device=shift_logits.device)
                batch_task_count += current_batch_task_count
                batch_task_loss += task_loss.item()

                if use_logit_bias:
                    logit_bias_loss, _ = compute_logit_bias_loss(
                        base_model,
                        shift_hidden_states,
                        shift_labels,
                    )
                else:
                    logit_bias_loss = torch.tensor(0.0, device=shift_logits.device)
                batch_logit_bias_loss += logit_bias_loss.item()

                total_batch_loss = loss + logit_bias_loss_weight * logit_bias_loss
                if accelerator is not None:
                    accelerator.backward(total_batch_loss)
                    if accelerator.sync_gradients:
                        _sync_fsdp_ignored_module_gradients(model, accelerator=accelerator)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    (total_batch_loss / gradient_accumulation_steps).backward()
                    if (forward_step + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

            total_loss += total_batch_loss.item()
            total_logit_bias_loss += logit_bias_loss.item()
            processed_batches += 1
            if current_batch_task_count > 0:
                total_task_loss += task_loss.item()
                task_token_count += current_batch_task_count
                task_loss_batches += 1

            batches_since_log += 1
            forward_step += 1

            if (batch_idx + 1) % 100 == 0:
                metrics = _reduce_metrics(
                    [
                        batch_loss,
                        batch_task_loss,
                        batch_logit_bias_loss,
                        float(batch_task_count),
                        float(batches_since_log),
                    ],
                    accelerator=accelerator,
                    device=metric_device,
                )
                if should_log:
                    avg_loss = metrics[0].item() / max(metrics[4].item(), 1.0)
                    avg_task_loss = metrics[1].item() / max(metrics[4].item(), 1.0)
                    avg_logit_bias_loss = metrics[2].item() / max(metrics[4].item(), 1.0)
                    avg_task_count = metrics[3].item() / max(metrics[4].item(), 1.0)
                    current_lr = scheduler.get_last_lr()[0]

                    if metrics[3].item() > 0:
                        if use_logit_bias:
                            print(
                                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                                f"Avg Loss: {avg_loss:.4f}, Avg Task Loss: {avg_task_loss:.4f}, "
                                f"Avg Logit Bias Loss: {avg_logit_bias_loss:.4f}, "
                                f"Avg Task Tokens: {avg_task_count:.1f}, LR: {current_lr:.6f}"
                            )
                            training_logger.info(
                                f"E{epoch+1}/{num_epochs} B{batch_idx+1}/{len(dataloader)} "
                                f"AvgLoss:{avg_loss:.4f} AvgTaskLoss:{avg_task_loss:.4f} "
                                f"AvgLogitBiasLoss:{avg_logit_bias_loss:.4f} AvgTokens:{avg_task_count:.1f} "
                                f"LR:{current_lr:.6f}"
                            )
                        else:
                            print(
                                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                                f"Avg Loss: {avg_loss:.4f}, Avg Task Loss: {avg_task_loss:.4f}, "
                                f"Avg Task Tokens: {avg_task_count:.1f}, LR: {current_lr:.6f}"
                            )
                            training_logger.info(
                                f"E{epoch+1}/{num_epochs} B{batch_idx+1}/{len(dataloader)} "
                                f"AvgLoss:{avg_loss:.4f} AvgTaskLoss:{avg_task_loss:.4f} "
                                f"AvgTokens:{avg_task_count:.1f} LR:{current_lr:.6f}"
                            )
                    else:
                        if use_logit_bias:
                            print(
                                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                                f"Avg Loss: {avg_loss:.4f}, Avg Task Loss: N/A (no task tokens in window), "
                                f"Avg Logit Bias Loss: {avg_logit_bias_loss:.4f}, LR: {current_lr:.6f}"
                            )
                            training_logger.info(
                                f"E{epoch+1}/{num_epochs} B{batch_idx+1}/{len(dataloader)} "
                                f"AvgLoss:{avg_loss:.4f} AvgTaskLoss:N/A AvgLogitBiasLoss:{avg_logit_bias_loss:.4f} "
                                f"LR:{current_lr:.6f}"
                            )
                        else:
                            print(
                                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                                f"Avg Loss: {avg_loss:.4f}, Avg Task Loss: N/A (no task tokens in window), "
                                f"LR: {current_lr:.6f}"
                            )
                            training_logger.info(
                                f"E{epoch+1}/{num_epochs} B{batch_idx+1}/{len(dataloader)} "
                                f"AvgLoss:{avg_loss:.4f} AvgTaskLoss:N/A LR:{current_lr:.6f}"
                            )

                batch_loss = 0.0
                batch_task_loss = 0.0
                batch_logit_bias_loss = 0.0
                batch_task_count = 0
                batches_since_log = 0

            if val_dataloader is not None and validate_every_n_steps and forward_step % validate_every_n_steps == 0:
                val_metrics = run_validation(
                    model,
                    val_dataloader,
                    device=device,
                    ignore_index=-100,
                    use_logit_bias=use_logit_bias,
                    logit_bias_loss_weight=logit_bias_loss_weight,
                    accelerator=accelerator,
                )
                avg_val_loss = val_metrics["avg_val_loss"]
                avg_val_logit_bias_loss = val_metrics["avg_logit_bias_loss"]
                if should_log:
                    if use_logit_bias:
                        print(
                            f"Step {forward_step} - Validation Loss: {avg_val_loss:.4f}, "
                            f"Avg Logit Bias Loss: {avg_val_logit_bias_loss:.4f}"
                        )
                        training_logger.info(
                            f"VALIDATION STEP {forward_step} Loss:{avg_val_loss:.4f} "
                            f"AvgLogitBiasLoss:{avg_val_logit_bias_loss:.4f}"
                        )
                    else:
                        print(f"Step {forward_step} - Validation Loss: {avg_val_loss:.4f}")
                        training_logger.info(f"VALIDATION STEP {forward_step} Loss:{avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss and (accelerator is None or accelerator.is_main_process):
                    best_val_loss = avg_val_loss
                    best_model_path = save_trained_model(
                        model,
                        timestamp=timestamp,
                        suffix="best",
                        accelerator=accelerator,
                    )
                    training_logger.info(
                        f"NEW BEST VALIDATION LOSS: {best_val_loss:.4f} | Saved: {best_model_path}"
                    )
                if accelerator is not None:
                    accelerator.wait_for_everyone()

        if val_dataloader is not None:
            val_metrics = run_validation(
                model,
                val_dataloader,
                device=device,
                ignore_index=-100,
                use_logit_bias=use_logit_bias,
                logit_bias_loss_weight=logit_bias_loss_weight,
                accelerator=accelerator,
            )
            avg_val_loss = val_metrics["avg_val_loss"]
            avg_val_logit_bias_loss = val_metrics["avg_logit_bias_loss"]
            if should_log:
                if use_logit_bias:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}, "
                        f"Avg Logit Bias Loss: {avg_val_logit_bias_loss:.4f}"
                    )
                    training_logger.info(
                        f"VALIDATION E{epoch+1}/{num_epochs} Loss:{avg_val_loss:.4f} "
                        f"AvgLogitBiasLoss:{avg_val_logit_bias_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")
                    training_logger.info(
                        f"VALIDATION E{epoch+1}/{num_epochs} Loss:{avg_val_loss:.4f}"
                    )

            if avg_val_loss < best_val_loss and (accelerator is None or accelerator.is_main_process):
                best_val_loss = avg_val_loss
                best_model_path = save_trained_model(
                    model,
                    timestamp=timestamp,
                    suffix="best",
                    accelerator=accelerator,
                )
                training_logger.info(
                    f"NEW BEST VALIDATION LOSS: {best_val_loss:.4f} | Saved: {best_model_path}"
                )
            if accelerator is not None:
                accelerator.wait_for_everyone()

    final_metrics = _reduce_metrics(
        [
            total_loss,
            total_task_loss,
            total_logit_bias_loss,
            float(task_token_count),
            float(task_loss_batches),
            float(processed_batches),
        ],
        accelerator=accelerator,
        device=metric_device,
    )
    processed_batches = max(int(final_metrics[5].item()), 1)
    avg_total_loss = final_metrics[0].item() / processed_batches
    avg_total_logit_bias_loss = final_metrics[2].item() / processed_batches
    task_token_count = int(final_metrics[3].item())
    task_loss_batches = int(final_metrics[4].item())

    if task_token_count > 0:
        avg_task_loss = final_metrics[1].item() / max(task_loss_batches, 1)
        if should_log:
            print("\nTraining completed!")
            print(f"Average overall loss: {avg_total_loss:.4f}")
            print(f"Average task token loss: {avg_task_loss:.4f}")
            if use_logit_bias:
                print(f"Average logit bias loss: {avg_total_logit_bias_loss:.4f}")
            print(f"Total task tokens processed: {task_token_count}")
            print(f"Batches with task tokens: {task_loss_batches}/{processed_batches}")
            print("Task token accuracy insight: Lower task loss indicates better task selection performance")
        if use_logit_bias:
            training_logger.info(
                f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} TaskLoss:{avg_task_loss:.4f} "
                f"AvgLogitBiasLoss:{avg_total_logit_bias_loss:.4f} TaskTokens:{task_token_count} "
                f"TaskBatches:{task_loss_batches}/{processed_batches}"
            )
        else:
            training_logger.info(
                f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} TaskLoss:{avg_task_loss:.4f} "
                f"TaskTokens:{task_token_count} TaskBatches:{task_loss_batches}/{processed_batches}"
            )
    else:
        if should_log:
            print(f"\nTraining completed! Average loss: {avg_total_loss:.4f}")
            if use_logit_bias:
                print(f"Average logit bias loss: {avg_total_logit_bias_loss:.4f}")
            print("Warning: No task tokens found in training data!")
        if use_logit_bias:
            training_logger.info(
                f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} "
                f"AvgLogitBiasLoss:{avg_total_logit_bias_loss:.4f} WARNING:NoTaskTokens"
            )
        else:
            training_logger.info(
                f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} WARNING:NoTaskTokens"
            )

    if val_dataloader is not None and best_val_loss < float("inf") and should_log:
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path}")
        training_logger.info(f"BEST VALIDATION LOSS:{best_val_loss:.4f}")

    return {
        "avg_total_loss": avg_total_loss,
        "avg_logit_bias_loss": avg_total_logit_bias_loss,
        "best_val_loss": best_val_loss if val_dataloader is not None else None,
        "best_model_path": best_model_path,
        "best_model_state": None,
    }


def save_trained_model(model, save_dir="saved_models", timestamp=None, suffix=None, accelerator=None):
    """Save the trained task tokens from the unwrapped model on the main process."""
    if accelerator is not None and not accelerator.is_main_process:
        return None

    os.makedirs(save_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"task_tokens_{timestamp}.pt" if not suffix else f"task_tokens_{timestamp}_{suffix}.pt"
    filepath = os.path.join(save_dir, filename)

    target_model = accelerator.unwrap_model(model) if accelerator is not None else model
    target_model.save_task_tokens(filepath)
    return filepath


def demo_task_calling(model, tokenizer, test_examples, device="cuda", use_ground_truth_tasks=False,
                      accelerator=None):
    """Demo task calling on a few held-out test examples."""
    base_model = _unwrap_model(model, accelerator=accelerator)
    model.eval()
    mode_desc = "Ground Truth Task Inference" if use_ground_truth_tasks else "Normal Task Prediction"
    print(f"\n=== Task Calling Demo ({mode_desc}) ===")
    print(f"Testing on {len(test_examples)} held-out examples")
    print(f"Available tasks: {base_model.task_names}")
    print()

    for i, example in enumerate(test_examples):
        instruction = example["instruction"]
        expected_tasks = example.get("tasks", ["unknown"])
        expected_responses = example.get("responses", [""])

        print(f"=== Test Example {i} ===")
        print(f"Instruction: {instruction}")
        print(f"Expected Task(s): {expected_tasks}")
        print(f"Expected Response(s): {expected_responses}")
        print()

        instruction_text = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        instruction_tokens = tokenizer(instruction_text, return_tensors="pt").to(device)

        with _fsdp_generation_context(model, accelerator=accelerator):
            if use_ground_truth_tasks:
                results = base_model.generate_with_ground_truth_tasks(
                    instruction_tokens["input_ids"],
                    instruction_tokens["attention_mask"],
                    tokenizer,
                    ground_truth_tasks=expected_tasks,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                )
                print(f"Mode: Ground truth tasks used ({expected_tasks})")
            else:
                results = base_model.generate_with_task_prediction(
                    instruction_tokens["input_ids"],
                    instruction_tokens["attention_mask"],
                    tokenizer,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                )
                print("Mode: Model predicts tasks")

        result = results[0]
        if "predicted_tasks" in result and result["predicted_tasks"]:
            print(f"Generated {len(result['predicted_tasks'])} task(s):")
            for j, (task_info, response) in enumerate(zip(result["predicted_tasks"], result["responses"])):
                print(f"  Task {j+1}: {task_info['task_name']}")
                print(f"  Response {j+1}: {response}")
                print()
        else:
            print(f"Predicted Task: {result['predicted_task_name']}")
            print(f"Task Token Used: {result.get('task_token_used', 'N/A')}")
            print(f"Response: {result['response']}")

        print(f"Full Generated: {result['full_generated_sequence']}")
        print("-" * 50)
        print()


def eval_task_calling(model, tokenizer, test_dataloader, device="cuda", use_ground_truth_tasks=False,
                      accelerator=None):
    """Run evaluation with optional distributed object gathering through Accelerate."""
    from natural_instructions_eval import evaluate_predictions, print_evaluation_results

    eval_logger = logging.getLogger("evaluation")
    base_model = _unwrap_model(model, accelerator=accelerator)
    generation_model = base_model
    should_log = accelerator is None or accelerator.is_main_process

    model.eval()
    total_examples = len(test_dataloader.dataset)
    mode_desc = "Ground Truth Task Inference" if use_ground_truth_tasks else "Normal Task Prediction"
    if should_log:
        print(f"\n=== Task Calling Evaluation ({mode_desc}) ===")
        print(f"Evaluating on {total_examples} test examples")
        print()
        print("🔄 Running batch evaluation...")

    eval_logger.info(f"EVALUATION START - Mode:{mode_desc} Examples:{total_examples}")
    eval_logger.info(f"PyTorch manual seed: {torch.initial_seed()}")

    all_predictions = []
    all_references = []
    all_task_names = []
    task_correct = 0
    task_breakdown = defaultdict(lambda: {"total": 0, "task_correct": 0})

    start_time = time.time()
    processed_examples = 0

    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch["raw_data"])
        if accelerator is None:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        with _fsdp_generation_context(model, accelerator=accelerator):
            if use_ground_truth_tasks:
                batch_results = []
                for i in range(batch_size):
                    example = batch["raw_data"][i]
                    expected_tasks = example.get("tasks", ["unknown"])
                    single_result = generation_model.generate_with_ground_truth_tasks(
                        input_ids[i:i+1],
                        attention_mask[i:i+1],
                        tokenizer,
                        ground_truth_tasks=expected_tasks,
                        max_new_tokens=256,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=False,
                    )
                    batch_results.extend(single_result)
            else:
                batch_results = generation_model.generate_with_task_prediction(
                    input_ids,
                    attention_mask,
                    tokenizer,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                )

        raw_data = batch["raw_data"]
        if accelerator is not None:
            batch_results = accelerator.gather_for_metrics(batch_results, use_gather_object=True)
            raw_data = accelerator.gather_for_metrics(raw_data, use_gather_object=True)

        if not should_log:
            continue

        for example, result in zip(raw_data, batch_results):
            expected_tasks = example.get("tasks", ["unknown"])
            expected_responses = example.get("responses", [""])

            if "predicted_tasks" in result and result["predicted_tasks"]:
                predicted_tasks = [task_info["task_name"] for task_info in result["predicted_tasks"]]
                predicted_responses = result["responses"]
            else:
                predicted_tasks = (
                    [result["predicted_task_name"]]
                    if result["predicted_task_name"] != "none"
                    else []
                )
                predicted_responses = [result["response"]] if result["response"] else []

            task_match = set(predicted_tasks) == set(expected_tasks)
            if task_match:
                task_correct += 1

            for task in expected_tasks:
                task_breakdown[task]["total"] += 1
                if task_match:
                    task_breakdown[task]["task_correct"] += 1

            all_predictions.append(predicted_responses[0] if predicted_responses else "")
            all_references.append(expected_responses)
            all_task_names.append(expected_tasks[0] if expected_tasks else "unknown")

        processed_examples = len(all_predictions)
        if should_log and ((batch_idx % 10 == 0) or processed_examples == total_examples):
            progress_pct = 100 * processed_examples / max(total_examples, 1)
            print(f"   Progress: {processed_examples}/{total_examples} ({progress_pct:.1f}%)")
            eval_logger.info(f"Progress: {processed_examples}/{total_examples} ({progress_pct:.1f}%)")

    if not should_log:
        return None

    eval_time = time.time() - start_time
    print("\n🔍 Computing Natural Instructions metrics...")
    ni_results = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names,
        xlingual=False,
    )
    print_evaluation_results(ni_results, f"NATURAL INSTRUCTIONS EVALUATION ({mode_desc})")

    task_accuracy = task_correct / max(total_examples, 1)
    print(f"\n🎯 TASK PREDICTION ACCURACY: {task_accuracy:.3f} ({task_correct}/{total_examples})")
    print(f"⏱️  Total evaluation time: {eval_time:.2f} seconds")

    eval_logger.info(
        f"EVALUATION COMPLETE - TaskAcc:{task_accuracy:.3f} "
        f"ExactMatch:{ni_results['exact_match']:.1f}% RougeL:{ni_results['rougeL']:.1f}% "
        f"Time:{eval_time:.1f}s"
    )

    if task_breakdown:
        for task_name in sorted(task_breakdown.keys()):
            stats = task_breakdown[task_name]
            task_acc = stats["task_correct"] / stats["total"] if stats["total"] > 0 else 0.0
            ni_metrics = ni_results.get("per_task", {}).get(task_name, {})
            exact_match = ni_metrics.get("exact_match", 0)
            rouge_l = ni_metrics.get("rougeL", 0)
            eval_logger.info(
                f"Task:{task_name} TaskAcc:{task_acc:.3f} ExactMatch:{exact_match:.1f}% "
                f"RougeL:{rouge_l:.1f}% Examples:{stats['total']}"
            )

    if task_breakdown:
        print("\n📊 TASK PREDICTION ACCURACY BY TASK:")
        print("-" * 60)
        for task_name in sorted(task_breakdown.keys()):
            stats = task_breakdown[task_name]
            task_acc = stats["task_correct"] / stats["total"] if stats["total"] > 0 else 0.0
            print(f"   {task_name}: {task_acc:.3f} ({stats['total']} examples)")

    return {
        "exact_accuracy": ni_results["exact_match"] / 100.0,
        "task_accuracy": task_accuracy,
        "avg_response_score": ni_results["rougeL"] / 100.0,
        "total_examples": total_examples,
        "ni_exact_match": ni_results["exact_match"],
        "ni_rouge_l": ni_results["rougeL"],
        "ni_per_task": ni_results.get("per_task", {}),
        "task_breakdown": dict(task_breakdown),
    }
