import inspect
from collections import defaultdict

import torch
import torch.nn.functional as F


def _resolve_mode_flags(model, use_eoc=None, use_gate=None):
    resolved_use_eoc = bool(getattr(model, "use_eoc", False) if use_eoc is None else use_eoc)
    resolved_use_gate = bool(getattr(model, "use_gate", False) if use_gate is None else use_gate)

    if resolved_use_gate and not resolved_use_eoc:
        raise ValueError("use_gate=True requires use_eoc=True")

    return resolved_use_eoc, resolved_use_gate


def _call_method_with_supported_kwargs(method, **kwargs):
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)

    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return method(**supported_kwargs)


def _ground_truth_gate_supported(model):
    return any(
        callable(getattr(model, method_name, None))
        for method_name in ("generate_with_ground_truth_tools_and_gate", "generate_with_gated_ground_truth_tools")
    )


def _is_tool_token_id(token_id, model):
    return token_id in getattr(model, "token_id_to_tool_id", {})


def compute_tool_subset_targets(shift_labels, model):
    """Map shifted labels to tool indices for tool-only CE on the reserved-token subset."""
    tool_targets = torch.full_like(shift_labels, -100)
    token_id_to_tool_id = getattr(model, "token_id_to_tool_id", {})

    for token_id, tool_id in token_id_to_tool_id.items():
        tool_targets[shift_labels == token_id] = tool_id

    tool_mask = tool_targets != -100
    return tool_targets, tool_mask


def build_shift_supervision_masks(shift_labels, model, use_eoc=False, use_gate=False):
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
        "use_gate": use_gate,
    }


def gather_gate_examples(hidden_states, labels, model):
    """Collect hidden states and binary labels for gate supervision."""
    eoc_token_id = getattr(model, "eoc_token_id", None)
    if eoc_token_id is None:
        empty_hidden = hidden_states.new_zeros((0, hidden_states.size(-1)))
        empty_targets = hidden_states.new_zeros((0,), dtype=torch.float32)
        return empty_hidden, empty_targets, 0, 0

    shift_hidden_states = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100

    gate_hidden_states = []
    gate_targets = []
    initial_sites = 0
    eoc_sites = 0

    for batch_idx in range(labels.size(0)):
        valid_positions = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        first_valid_pos = int(valid_positions[0].item())
        initial_pos = first_valid_pos
        if (
            initial_pos >= 0
            and initial_pos < shift_hidden_states.size(1)
            and labels[batch_idx, initial_pos].item() == -100
        ):
            next_token_id = int(shift_labels[batch_idx, initial_pos].item())
            if next_token_id != -100:
                gate_hidden_states.append(shift_hidden_states[batch_idx, initial_pos])
                gate_targets.append(1.0 if _is_tool_token_id(next_token_id, model) else 0.0)
                initial_sites += 1

        eoc_positions = torch.nonzero(
            (labels[batch_idx, :-1] == eoc_token_id) & valid_mask[batch_idx],
            as_tuple=False,
        ).flatten()
        for position in eoc_positions.tolist():
            next_token_id = int(shift_labels[batch_idx, position].item())
            if next_token_id == -100:
                continue
            gate_hidden_states.append(shift_hidden_states[batch_idx, position])
            gate_targets.append(1.0 if _is_tool_token_id(next_token_id, model) else 0.0)
            eoc_sites += 1

    if not gate_hidden_states:
        empty_hidden = hidden_states.new_zeros((0, hidden_states.size(-1)))
        empty_targets = hidden_states.new_zeros((0,), dtype=torch.float32)
        return empty_hidden, empty_targets, initial_sites, eoc_sites

    return (
        torch.stack(gate_hidden_states, dim=0),
        torch.tensor(gate_targets, dtype=torch.float32, device=hidden_states.device),
        initial_sites,
        eoc_sites,
    )


def _forward_with_optional_hidden_states(model, input_ids, attention_mask, output_hidden_states=False):
    forward_model = getattr(model, "model", model)
    return forward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )


def _generate_results(
    model,
    tokenizer,
    user_tokens,
    user_mask,
    use_gate=False,
    use_eoc=False,
    gate_threshold=0.5,
    use_ground_truth_tools=False,
    ground_truth_tools=None,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.9,
    do_sample=False,
):
    candidate_methods = []
    if use_ground_truth_tools:
        if use_gate:
            candidate_methods.extend(
                [
                    "generate_with_ground_truth_tools_and_gate",
                    "generate_with_gated_ground_truth_tools",
                    "generate_with_ground_truth_tools",
                ]
            )
        else:
            candidate_methods.append("generate_with_ground_truth_tools")
    else:
        if use_gate:
            candidate_methods.extend(
                [
                    "generate_with_optional_gate",
                    "generate_with_gate",
                    "generate_with_tool_prediction",
                ]
            )
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
        "gate_threshold": gate_threshold,
        "use_gate": use_gate,
        "use_eoc": use_eoc,
    }
    if ground_truth_tools is not None:
        generation_kwargs["ground_truth_tools"] = ground_truth_tools

    for method_name in candidate_methods:
        method = getattr(model, method_name, None)
        if callable(method):
            return _call_method_with_supported_kwargs(method, **generation_kwargs)

    raise AttributeError(
        f"Model {type(model).__name__} does not expose a compatible generation method "
        f"for use_gate={use_gate}, use_ground_truth_tools={use_ground_truth_tools}."
    )


def train_native_function_calling_model(
    model,
    dataloader,
    num_epochs=3,
    lr=0.01,
    gradient_accumulation_steps=1,
    device="cuda",
    lora_lr=None,
    active_tool_ids=None,
    renorm_active_rows=False,
    use_eoc=None,
    use_gate=None,
    use_tool_loss=False,
    eoc_loss_weight=0.1,
    tool_loss_weight=0.1,
    gate_loss_weight=0.1,
    gate_threshold=0.5,
):
    """Train the native function calling model using reserved tokens."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    model.train()
    resolved_use_eoc, resolved_use_gate = _resolve_mode_flags(model, use_eoc, use_gate)

    # Set up optimizer with different learning rates for embeddings and LoRA.
    if model.lora_config and lora_lr is not None:
        embedding_params, lora_params = model.get_trainable_parameters(separate_lora=True)
        known_param_ids = {id(param) for param in embedding_params + lora_params}
        extra_trainable_params = [
            param
            for param in model.parameters()
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
        print(f"Using separate learning rates: embeddings={lr}, LoRA={lora_lr} (wd: emb=0.0, lora=0.01)")
    else:
        embedding_params = model.get_trainable_parameters()
        known_param_ids = {id(param) for param in embedding_params}
        extra_trainable_params = [
            param
            for param in model.parameters()
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

    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    print(f"Training for {num_epochs} epochs, {len(dataloader)} batches per epoch")
    print(f"Total steps: {total_steps}")
    if model.lora_config and lora_lr is not None:
        print(f"Learning rates: embeddings={lr}, LoRA={lora_lr} (with linear schedule + warmup)")
    else:
        print(f"Learning rate: {lr} (with linear schedule + warmup)")
    print(f"Warmup steps: {total_steps // 10}")
    print(
        "Mode: "
        f"use_eoc={resolved_use_eoc}, use_gate={resolved_use_gate}, use_tool_loss={use_tool_loss}, "
        f"eoc_loss_weight={eoc_loss_weight}, tool_loss_weight={tool_loss_weight}, "
        f"gate_loss_weight={gate_loss_weight}, gate_threshold={gate_threshold}"
    )

    all_trainable_params = model.get_trainable_parameters()
    total_trainable = sum(param.numel() for param in all_trainable_params)

    if model.decouple_embeddings:
        print("Training mode: Decoupled embeddings")
        print(
            "Trainable parameters: "
            f"{total_trainable:,} "
            f"(input: {model.trainable_tool_input_embeddings.numel():,}, "
            f"output: {model.trainable_tool_output_embeddings.numel():,})"
        )
    else:
        print("Training mode: Coupled embeddings")
        print(f"Trainable parameters: {total_trainable:,} (shared: {model.trainable_tool_embeddings.numel():,})")
    print(f"Tool token IDs to monitor: {model.reserved_token_ids}")
    print()

    total_loss = 0.0
    total_ar_loss = 0.0
    total_eoc_loss = 0.0
    total_tool_loss = 0.0
    total_gate_loss = 0.0
    total_valid_positions = 0
    total_eoc_positions = 0
    total_tool_positions = 0
    total_gate_positions = 0
    total_gate_initial_positions = 0
    total_gate_eoc_positions = 0
    successful_steps = 0
    step = 0

    window_batches = 0
    window_total_loss = 0.0
    window_ar_loss = 0.0
    window_eoc_loss = 0.0
    window_tool_loss = 0.0
    window_gate_loss = 0.0
    window_valid_positions = 0
    window_eoc_positions = 0
    window_tool_positions = 0
    window_gate_positions = 0

    zero = torch.tensor(0.0, device=device)
    tool_token_ids = torch.tensor(model.reserved_token_ids, device=device, dtype=torch.long)
    has_gate_head = hasattr(model, "gate_mlp") and model.gate_mlp is not None
    if resolved_use_gate and not has_gate_head:
        print("Warning: use_gate=True but model has no gate_mlp; gate loss will stay zero until the model-side hook lands.")

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = _forward_with_optional_hidden_states(
                model,
                input_ids,
                attention_mask,
                output_hidden_states=resolved_use_gate,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            hidden_states = None
            if resolved_use_gate and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]

            if not torch.isfinite(logits).all():
                print(
                    f"Warning: skipping non-finite logits at epoch {epoch + 1}, "
                    f"batch {batch_idx + 1}/{len(dataloader)}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels != -100

            if valid_mask.sum() == 0:
                print(
                    f"Warning: skipping batch with no valid targets at epoch {epoch + 1}, "
                    f"batch {batch_idx + 1}/{len(dataloader)}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            ar_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            masks = build_shift_supervision_masks(
                shift_labels,
                model,
                use_eoc=resolved_use_eoc,
                use_gate=resolved_use_gate,
            )

            eoc_loss = zero
            tool_loss = zero
            gate_loss = zero

            eoc_count = int(masks["eoc_mask"].sum().item())
            tool_count = int(masks["tool_mask"].sum().item())

            if resolved_use_eoc and eoc_count > 0:
                eoc_loss = F.cross_entropy(shift_logits[masks["eoc_mask"]], shift_labels[masks["eoc_mask"]])

            if resolved_use_eoc and use_tool_loss and tool_count > 0:
                tool_logits = shift_logits[masks["tool_mask"]][:, tool_token_ids]
                tool_targets = masks["tool_targets"][masks["tool_mask"]]
                tool_loss = F.cross_entropy(tool_logits, tool_targets)

            gate_hidden_states = None
            gate_targets = None
            gate_initial_count = 0
            gate_eoc_count = 0
            if resolved_use_gate and hidden_states is not None:
                gate_hidden_states, gate_targets, gate_initial_count, gate_eoc_count = gather_gate_examples(
                    hidden_states,
                    labels,
                    model,
                )
                if gate_targets.numel() > 0 and has_gate_head:
                    gate_logits = model.gate_mlp(gate_hidden_states).squeeze(-1)
                    gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_targets)

            loss = ar_loss
            if resolved_use_eoc:
                loss = loss + eoc_loss_weight * eoc_loss
                if use_tool_loss:
                    loss = loss + tool_loss_weight * tool_loss
            if resolved_use_gate:
                loss = loss + gate_loss_weight * gate_loss

            if not torch.isfinite(loss):
                print(
                    f"Warning: skipping non-finite loss at epoch {epoch + 1}, "
                    f"batch {batch_idx + 1}/{len(dataloader)}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            loss.backward()

            trainable_params = model.get_trainable_parameters()
            if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in trainable_params):
                print(
                    f"Warning: skipping optimizer step with non-finite gradients at epoch {epoch + 1}, "
                    f"batch {batch_idx + 1}/{len(dataloader)}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            if active_tool_ids is not None:
                try:
                    total_tools = len(model.reserved_token_ids)
                    total_rows = model.trainable_tool_input_embeddings.size(0)
                    device_for_mask = model.trainable_tool_input_embeddings.device
                    active_mask = torch.zeros(total_rows, dtype=torch.bool, device=device_for_mask)
                    active_mask[active_tool_ids] = True
                    if total_rows > total_tools:
                        active_mask[total_tools:] = True

                    params_to_mask = []
                    if hasattr(model, "trainable_tool_input_embeddings") and model.trainable_tool_input_embeddings is not None:
                        params_to_mask.append(model.trainable_tool_input_embeddings)
                    if hasattr(model, "trainable_tool_output_embeddings") and model.trainable_tool_output_embeddings is not None:
                        if model.trainable_tool_output_embeddings is not model.trainable_tool_input_embeddings:
                            params_to_mask.append(model.trainable_tool_output_embeddings)

                    for param in params_to_mask:
                        if param.grad is not None:
                            inactive_rows = ~active_mask
                            param.grad[inactive_rows] = 0
                except Exception as exc:
                    print(f"Warning: failed to apply active tool grad mask: {exc}")

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

                if renorm_active_rows and active_tool_ids is not None:
                    try:
                        total_tools = len(model.reserved_token_ids)
                        device_for_mask = model.trainable_tool_input_embeddings.device
                        active_mask = torch.zeros(total_tools, dtype=torch.bool, device=device_for_mask)
                        active_mask[active_tool_ids] = True
                        inactive_mask = ~active_mask

                        params_to_norm = []
                        if hasattr(model, "trainable_tool_output_embeddings") and model.trainable_tool_output_embeddings is not None:
                            params_to_norm.append(model.trainable_tool_output_embeddings)
                        elif hasattr(model, "trainable_tool_embeddings") and model.trainable_tool_embeddings is not None:
                            params_to_norm.append(model.trainable_tool_embeddings)

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
                        print(f"Warning: failed to apply post-step renorm: {exc}")

                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            total_ar_loss += ar_loss.item()
            total_eoc_loss += eoc_loss.item()
            total_tool_loss += tool_loss.item()
            total_gate_loss += gate_loss.item()
            total_valid_positions += int(valid_mask.sum().item())
            total_eoc_positions += eoc_count
            total_tool_positions += tool_count
            total_gate_positions += int(gate_targets.numel()) if gate_targets is not None else 0
            total_gate_initial_positions += gate_initial_count
            total_gate_eoc_positions += gate_eoc_count
            successful_steps += 1

            window_batches += 1
            window_total_loss += loss.item()
            window_ar_loss += ar_loss.item()
            window_eoc_loss += eoc_loss.item()
            window_tool_loss += tool_loss.item()
            window_gate_loss += gate_loss.item()
            window_valid_positions += int(valid_mask.sum().item())
            window_eoc_positions += eoc_count
            window_tool_positions += tool_count
            window_gate_positions += int(gate_targets.numel()) if gate_targets is not None else 0

            if (batch_idx + 1) % 10 == 0:
                lr_values = scheduler.get_last_lr()
                if model.lora_config and lora_lr is not None:
                    lr_info = f"LR(emb/lora): {lr_values[0]:.6f}/{lr_values[1]:.6f}"
                else:
                    lr_info = f"LR: {lr_values[0]:.6f}"

                window_denom = max(1, window_batches)
                tool_loss_fragment = (
                    f"Tool: {(window_tool_loss / window_denom):.4f}, "
                    if use_tool_loss
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {window_total_loss / window_denom:.4f}, "
                    f"AR: {window_ar_loss / window_denom:.4f}, "
                    f"EOC: {window_eoc_loss / window_denom:.4f}, "
                    f"{tool_loss_fragment}"
                    f"Gate: {window_gate_loss / window_denom:.4f}, "
                    f"Sites(valid/eoc/tool/gate): {window_valid_positions}/{window_eoc_positions}/"
                    f"{window_tool_positions}/{window_gate_positions}, "
                    f"{lr_info}"
                )

                window_batches = 0
                window_total_loss = 0.0
                window_ar_loss = 0.0
                window_eoc_loss = 0.0
                window_tool_loss = 0.0
                window_gate_loss = 0.0
                window_valid_positions = 0
                window_eoc_positions = 0
                window_tool_positions = 0
                window_gate_positions = 0

            step += 1

    avg_total_loss = total_loss / successful_steps if successful_steps > 0 else float("nan")
    avg_ar_loss = total_ar_loss / successful_steps if successful_steps > 0 else float("nan")
    avg_eoc_loss = total_eoc_loss / successful_steps if successful_steps > 0 else float("nan")
    avg_tool_loss = total_tool_loss / successful_steps if successful_steps > 0 else float("nan")
    avg_gate_loss = total_gate_loss / successful_steps if successful_steps > 0 else float("nan")

    print("\nTraining completed!")
    print(f"Average total loss: {avg_total_loss:.4f}")
    print(f"Average AR loss:    {avg_ar_loss:.4f}")
    if resolved_use_eoc:
        print(f"Average EOC loss:   {avg_eoc_loss:.4f}")
        if use_tool_loss:
            print(f"Average Tool loss:  {avg_tool_loss:.4f}")
    if resolved_use_gate:
        print(f"Average Gate loss:   {avg_gate_loss:.4f}")
    print(f"Total valid supervised positions: {total_valid_positions}")
    if resolved_use_eoc:
        print(f"Total EOC positions: {total_eoc_positions}")
        print(f"Total tool positions: {total_tool_positions}")
    if resolved_use_gate:
        print(f"Total gate positions: {total_gate_positions}")
        print(f"Gate sites from assistant-start positions: {total_gate_initial_positions}")
        print(f"Gate sites from EOC positions: {total_gate_eoc_positions}")
    print(f"Successful optimizer steps: {successful_steps}")

    return {
        "avg_total_loss": avg_total_loss,
        "avg_ar_loss": avg_ar_loss,
        "avg_eoc_loss": avg_eoc_loss,
        "avg_tool_loss": avg_tool_loss,
        "avg_gate_loss": avg_gate_loss,
        "total_valid_positions": total_valid_positions,
        "total_eoc_positions": total_eoc_positions,
        "total_tool_positions": total_tool_positions,
        "total_gate_positions": total_gate_positions,
        "successful_steps": successful_steps,
        "use_eoc": resolved_use_eoc,
        "use_gate": resolved_use_gate,
        "use_tool_loss": use_tool_loss,
    }


def demo_native_function_calling(
    model,
    tokenizer,
    test_examples,
    device="cuda",
    use_ground_truth_tools=False,
    use_eoc=None,
    use_gate=None,
    gate_threshold=0.5,
):
    """Demo of native function calling using held-out test examples."""
    model.eval()
    resolved_use_eoc, resolved_use_gate = _resolve_mode_flags(model, use_eoc, use_gate)
    mode_desc = "Ground Truth Tool Inference" if use_ground_truth_tools else "Normal Tool Prediction"
    ground_truth_gate_active = use_ground_truth_tools and resolved_use_gate and _ground_truth_gate_supported(model)
    if resolved_use_gate and (not use_ground_truth_tools or ground_truth_gate_active):
        mode_desc += " + gate"
    elif use_ground_truth_tools and resolved_use_gate:
        mode_desc += " (gate bypassed by ground-truth tool forcing)"

    print(f"\n=== Native Function Calling Demo ({mode_desc}) ===")
    print(f"Testing on {len(test_examples)} held-out examples")
    print(f"Available tools: {model.tool_names}")
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
        user_tokens = tokenizer(user_text, return_tensors="pt").to(device)

        results = _generate_results(
            model,
            tokenizer,
            user_tokens["input_ids"],
            user_tokens["attention_mask"],
            use_gate=resolved_use_gate,
            use_eoc=resolved_use_eoc,
            gate_threshold=gate_threshold,
            use_ground_truth_tools=use_ground_truth_tools,
            ground_truth_tools=expected_tools if use_ground_truth_tools else None,
            max_new_tokens=150,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
        )
        mode_line = "Ground truth tools used" if use_ground_truth_tools else "Model predicts tools"
        if resolved_use_gate and (not use_ground_truth_tools or ground_truth_gate_active):
            mode_line += " + gate"
        elif use_ground_truth_tools and resolved_use_gate:
            mode_line += " (gate bypassed by ground-truth tool forcing)"
        print(f"Mode: {mode_line}")

        result = results[0]

        if "predicted_tools" in result and result["predicted_tools"]:
            print(f"Generated {len(result['predicted_tools'])} tool(s):")
            for j, (tool_info, func_call) in enumerate(zip(result["predicted_tools"], result["function_calls"])):
                print(f"  Tool {j + 1}: {tool_info['tool_name']}")
                print(f"  Function Call {j + 1}: {func_call}")

                parsed = model.parse_function_call(func_call)
                print(f"  Parsed {j + 1}: {parsed}")
                print()
        else:
            print(f"Predicted Tool: {result['predicted_tool_name']}")
            print(f"Tool Token Used: {result.get('tool_token_used', 'N/A')}")
            print(f"Function Call: {result['function_call']}")

            parsed = model.parse_function_call(result["function_call"])
            print(f"Parsed: {parsed}")

        print(f"Full Generated: {result['full_generated_sequence']}")
        print("-" * 50)
        print()


def eval_native_function_calling(
    model,
    tokenizer,
    test_dataloader,
    device="cuda",
    use_ground_truth_tools=False,
    use_eoc=None,
    use_gate=None,
    gate_threshold=0.5,
):
    """Comprehensive evaluation of native function calling model using batch processing."""
    from eval import compare_function_calls_advanced
    import time

    model.eval()
    resolved_use_eoc, resolved_use_gate = _resolve_mode_flags(model, use_eoc, use_gate)

    total_examples = len(test_dataloader.dataset)
    mode_desc = "Ground Truth Tool Inference" if use_ground_truth_tools else "Normal Tool Prediction"
    ground_truth_gate_active = use_ground_truth_tools and resolved_use_gate and _ground_truth_gate_supported(model)
    if resolved_use_gate and (not use_ground_truth_tools or ground_truth_gate_active):
        mode_desc += " + gate"
    elif use_ground_truth_tools and resolved_use_gate:
        mode_desc += " (gate bypassed by ground-truth tool forcing)"

    print(f"\n=== Native Function Calling Evaluation ({mode_desc}) ===")
    print(f"Evaluating on {total_examples} test examples")
    print()

    exact_matches = 0
    tool_correct = 0
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
            "tool_correct": 0,
            "f1_scores": [],
            "precision_scores": [],
            "recall_scores": [],
            "tool_f1_scores": [],
            "tool_precision_scores": [],
            "tool_recall_scores": [],
            "parse_errors": 0,
        }
    )

    start_time = time.time()
    print("🔄 Running batch evaluation...")

    processed_examples = 0
    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch["raw_data"])
        processed_examples += batch_size

        if batch_idx % 10 == 0 or processed_examples == total_examples:
            print(f"   Progress: {processed_examples}/{total_examples} ({100 * processed_examples / total_examples:.1f}%)")

        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if use_ground_truth_tools:
                batch_results = []
                for i in range(batch_size):
                    example = batch["raw_data"][i]
                    expected_tools = example.get("tools", [example.get("tool_name", "unknown")])
                    single_input = input_ids[i : i + 1]
                    single_mask = attention_mask[i : i + 1]
                    single_result = _generate_results(
                        model,
                        tokenizer,
                        single_input,
                        single_mask,
                        use_gate=resolved_use_gate,
                        use_eoc=resolved_use_eoc,
                        gate_threshold=gate_threshold,
                        use_ground_truth_tools=True,
                        ground_truth_tools=expected_tools,
                        max_new_tokens=256,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=False,
                    )
                    batch_results.extend(single_result)
            else:
                batch_results = _generate_results(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    use_gate=resolved_use_gate,
                    use_eoc=resolved_use_eoc,
                    gate_threshold=gate_threshold,
                    use_ground_truth_tools=False,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                )

            for i in range(batch_size):
                example = batch["raw_data"][i]
                result = batch_results[i]

                expected_tools = example.get("tools", [example.get("tool_name", "unknown")])
                expected_calls = example.get("function_calls", [example.get("function_call", "{}")])
                expected_call_count = len(expected_calls)

                if "predicted_tools" in result and result["predicted_tools"]:
                    predicted_tools = [tool_info["tool_name"] for tool_info in result["predicted_tools"]]
                    predicted_calls = result["function_calls"]
                else:
                    predicted_tools = [result["predicted_tool_name"]] if result["predicted_tool_name"] != "none" else []
                    predicted_calls = [result["function_call"]] if result["function_call"] else []

                from collections import Counter
                from eval import calculate_f1_score

                tool_f1_result = calculate_f1_score(predicted_tools, expected_tools)
                tool_f1_scores.append(tool_f1_result["f1_score"])
                tool_precision_scores.append(tool_f1_result["precision"])
                tool_recall_scores.append(tool_f1_result["recall"])

                tool_match = Counter(predicted_tools) == Counter(expected_tools)
                if tool_match:
                    tool_correct += 1

                eval_result = compare_function_calls_advanced(
                    predicted_calls,
                    expected_calls,
                    ignore_order=True,
                )

                if eval_result.exact_match:
                    exact_matches += 1

                f1_scores.append(eval_result.f1_score)
                precision_scores.append(eval_result.precision)
                recall_scores.append(eval_result.recall)

                call_count_breakdown[expected_call_count]["total"] += 1
                if eval_result.exact_match:
                    call_count_breakdown[expected_call_count]["exact_matches"] += 1
                if tool_match:
                    call_count_breakdown[expected_call_count]["tool_correct"] += 1
                call_count_breakdown[expected_call_count]["f1_scores"].append(eval_result.f1_score)
                call_count_breakdown[expected_call_count]["precision_scores"].append(eval_result.precision)
                call_count_breakdown[expected_call_count]["recall_scores"].append(eval_result.recall)
                call_count_breakdown[expected_call_count]["tool_f1_scores"].append(tool_f1_result["f1_score"])
                call_count_breakdown[expected_call_count]["tool_precision_scores"].append(tool_f1_result["precision"])
                call_count_breakdown[expected_call_count]["tool_recall_scores"].append(tool_f1_result["recall"])

                if "parse_errors" in eval_result.details:
                    current_parse_errors = eval_result.details["parse_errors"]["outputs"]
                    parse_errors += current_parse_errors
                    call_count_breakdown[expected_call_count]["parse_errors"] += current_parse_errors

        except Exception as exc:
            print(f"   Error processing batch {batch_idx + 1}: {str(exc)}")
            for _ in range(batch_size):
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                tool_f1_scores.append(0.0)
                tool_precision_scores.append(0.0)
                tool_recall_scores.append(0.0)
            continue

    eval_time = time.time() - start_time

    exact_accuracy = exact_matches / total_examples
    tool_accuracy = tool_correct / total_examples
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
    print(f"   Tool Prediction Accuracy: {tool_accuracy:.3f} ({tool_correct}/{total_examples})")
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

    print("\n📊 TOOL PREDICTION ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        tool_accuracy = stats["tool_correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"   {call_count} call(s): {tool_accuracy:.3f} ({stats['tool_correct']}/{stats['total']})")
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
        "tool_accuracy": tool_accuracy,
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
