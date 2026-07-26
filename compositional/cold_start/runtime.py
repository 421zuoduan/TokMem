"""Runtime-only model expansion; the original model implementation stays untouched."""

import math
import types

import torch
import torch.nn as nn


COLD_START_FORMAT = "compositional_cold_start_delta"
COLD_START_VERSION = 1


class _OrthogonalInitializationTarget:
    def __init__(self, parameter):
        self.decouple_embeddings = False
        self.trainable_tool_embeddings = parameter


def make_orthogonal_new_embeddings(model, new_tool_count, seed):
    """Use the compositional experiment's exact orthogonal initialization."""
    if model.decouple_embeddings:
        raise ValueError("The Llama cold-start experiment uses coupled embeddings")

    old_tool_count = int(model.num_tools)
    total_tool_count = old_tool_count + int(new_tool_count)
    old_embeddings = model.trainable_tool_embeddings[:old_tool_count]
    temporary = nn.Parameter(
        torch.empty(
            total_tool_count,
            old_embeddings.shape[1],
            dtype=old_embeddings.dtype,
            device=old_embeddings.device,
        ),
        requires_grad=False,
    )

    from main_sequential import apply_orthogonal_init_all_tools

    cuda_devices = []
    if old_embeddings.is_cuda:
        cuda_devices = [old_embeddings.device.index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if old_embeddings.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        target = _OrthogonalInitializationTarget(temporary)
        apply_orthogonal_init_all_tools(target, total_tool_count)

    return temporary.detach()[old_tool_count:].clone()


def make_convex_new_embeddings(model, aggregation_weights):
    """Aggregate learned old-tool embeddings with document-derived weights."""
    old_embeddings = model.trainable_tool_embeddings[: model.num_tools]
    combined = (
        aggregation_weights.to(device=old_embeddings.device).float()
        @ old_embeddings.detach().float()
    )
    return combined.to(dtype=old_embeddings.dtype)


def bounded_affine_weights(convex_weights, negative_mass_cap):
    """Extrapolate away from the old-tool mean with bounded negative mass."""
    weights = convex_weights.detach().float()
    old_tool_count = weights.shape[1]
    uniform = torch.full_like(weights, 1.0 / old_tool_count)
    direction = weights - uniform
    cap = float(negative_mass_cap)
    if cap < 0.0:
        raise ValueError("negative_mass_cap must be non-negative")

    gains = torch.ones(weights.shape[0], dtype=torch.float32)
    if cap == 0.0:
        return weights.clone(), gains, torch.zeros_like(gains)

    for row_index in range(weights.shape[0]):
        base = weights[row_index]
        row_direction = direction[row_index]

        def negative_mass(extrapolation):
            candidate = base + extrapolation * row_direction
            return candidate.clamp_max(0.0).neg().sum().item()

        lower = 0.0
        upper = 1.0
        while negative_mass(upper) < cap:
            upper *= 2.0
        for _ in range(40):
            middle = 0.5 * (lower + upper)
            if negative_mass(middle) < cap:
                lower = middle
            else:
                upper = middle
        gains[row_index] = 1.0 + 0.5 * (lower + upper)

    affine = uniform + gains[:, None] * (weights - uniform)
    negative_mass = affine.clamp_max(0.0).neg().sum(dim=1)
    return affine, gains, negative_mass


def make_affine_new_embeddings(model, affine_weights):
    """Apply shared affine coefficients to learned old-tool embeddings."""
    old_embeddings = model.trainable_tool_embeddings[: model.num_tools]
    combined = (
        affine_weights.to(device=old_embeddings.device).float()
        @ old_embeddings.detach().float()
    )
    return combined.to(dtype=old_embeddings.dtype)


def partial_renorm_new_embeddings(model, new_embeddings, strength):
    """Interpolate each new row norm toward the learned old-tool mean."""
    old_embeddings = model.trainable_tool_embeddings[: model.num_tools]
    target_norm = old_embeddings.detach().float().norm(dim=1).mean().clamp(min=1e-6)
    new_float = new_embeddings.float()
    new_norms = new_float.norm(dim=1, keepdim=True).clamp(min=1e-6)
    scale = 1.0 + float(strength) * (target_norm / new_norms - 1.0)
    renormalized = new_float * scale
    return renormalized.to(dtype=old_embeddings.dtype), target_norm.detach().cpu()


def coherence_partial_renorm_new_embeddings(
    model,
    new_embeddings,
    aggregation_weights,
    strength,
):
    """Restore norm in proportion to agreement among each tool's donors."""
    old_embeddings = model.trainable_tool_embeddings[: model.num_tools]
    old_norms = old_embeddings.detach().float().norm(dim=1)
    donor_target_norms = (
        aggregation_weights.to(device=old_embeddings.device).float()
        @ old_norms
    ).clamp(min=1e-6)
    new_float = new_embeddings.float()
    new_norms = new_float.norm(dim=1).clamp(min=1e-6)
    coherence = (new_norms / donor_target_norms).clamp(min=0.0, max=1.0)
    scale = 1.0 + float(strength) * (1.0 - coherence)
    renormalized = new_float * scale[:, None]
    return (
        renormalized.to(dtype=old_embeddings.dtype),
        donor_target_norms.detach().cpu(),
        coherence.detach().cpu(),
    )


def squared_coherence_renorm_new_embeddings(
    model,
    new_embeddings,
    aggregation_weights,
    strength=1.0,
):
    """Use a parameter-free second-order correction for donor cancellation."""
    old_embeddings = model.trainable_tool_embeddings[: model.num_tools]
    old_norms = old_embeddings.detach().float().norm(dim=1)
    donor_target_norms = (
        aggregation_weights.to(device=old_embeddings.device).float()
        @ old_norms
    ).clamp(min=1e-6)
    new_float = new_embeddings.float()
    new_norms = new_float.norm(dim=1).clamp(min=1e-6)
    coherence = (new_norms / donor_target_norms).clamp(min=0.0, max=1.0)
    scale = 1.0 + float(strength) * (1.0 - coherence).square()
    renormalized = new_float * scale[:, None]
    return (
        renormalized.to(dtype=old_embeddings.dtype),
        donor_target_norms.detach().cpu(),
        coherence.detach().cpu(),
    )


def renorm_new_embeddings_to_old_mean(model, new_embeddings):
    """Match every new row to the mean norm of the learned old-tool rows."""
    return partial_renorm_new_embeddings(model, new_embeddings, strength=1.0)


def _available_reserved_tokens(model, count):
    reserved = sorted(
        (
            (name, token_id)
            for name, token_id in model.tokenizer.get_vocab().items()
            if "reserved_special_token_" in name
        ),
        key=lambda item: item[1],
    )
    used_ids = set(model.tool_reserved_token_ids)
    if model.eoc_token_id is not None:
        used_ids.add(model.eoc_token_id)
    available = [item for item in reserved if item[1] not in used_ids]
    if len(available) < count:
        raise ValueError(
            f"Need {count} unused reserved tokens, but only found {len(available)}"
        )
    return available[:count]


def _replace_memory_parameter(model, new_embeddings):
    old_tool_count = int(model.num_tools)
    old_parameter = model.trainable_tool_embeddings
    old_tool_embeddings = old_parameter[:old_tool_count].detach().clone()
    pieces = [
        old_tool_embeddings,
        new_embeddings.to(
            device=old_parameter.device,
            dtype=old_parameter.dtype,
        ),
    ]
    if model.use_eoc:
        pieces.append(old_parameter[old_tool_count : old_tool_count + 1].detach().clone())

    expanded = nn.Parameter(torch.cat(pieces, dim=0), requires_grad=False)
    model.trainable_tool_embeddings = expanded
    model.trainable_tool_input_embeddings = expanded
    model.trainable_tool_output_embeddings = expanded


def _replace_registry(model, new_tool_names, new_reserved_tokens):
    old_tool_names = list(model.tool_names)
    old_token_names = list(model.tool_reserved_token_names)
    old_token_ids = list(model.tool_reserved_token_ids)

    new_token_names = [name for name, _ in new_reserved_tokens]
    new_token_ids = [token_id for _, token_id in new_reserved_tokens]
    all_tool_names = old_tool_names + list(new_tool_names)
    all_token_names = old_token_names + new_token_names
    all_token_ids = old_token_ids + new_token_ids

    model.num_tools = len(all_tool_names)
    model.num_reserved_slots = model.num_tools + (1 if model.use_eoc else 0)
    model.tool_names = all_tool_names
    model.tool_name_to_id = {
        name: index for index, name in enumerate(all_tool_names)
    }
    model.tool_id_to_name = {
        index: name for index, name in enumerate(all_tool_names)
    }

    model.tool_reserved_token_names = all_token_names
    model.tool_reserved_token_ids = all_token_ids
    model.reserved_token_names = list(all_token_names)
    model.reserved_token_ids = list(all_token_ids)
    model.tool_id_to_token_id = {
        index: token_id for index, token_id in enumerate(all_token_ids)
    }
    model.token_id_to_tool_id = {
        token_id: index for index, token_id in enumerate(all_token_ids)
    }

    trainable_names = list(all_token_names)
    trainable_ids = list(all_token_ids)
    if model.use_eoc:
        trainable_names.append(model.eoc_token_name)
        trainable_ids.append(model.eoc_token_id)
    model.trainable_reserved_token_names = trainable_names
    model.trainable_reserved_token_ids = trainable_ids

    device = model.trainable_tool_embeddings.device
    trainable_tensor = torch.tensor(
        trainable_ids,
        dtype=torch.long,
        device=device,
    )
    tool_tensor = torch.tensor(
        all_token_ids,
        dtype=torch.long,
        device=device,
    )
    lookup = torch.full(
        (int(trainable_tensor.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=device,
    )
    lookup[trainable_tensor] = torch.arange(
        trainable_tensor.numel(),
        dtype=torch.long,
        device=device,
    )
    model._trainable_reserved_token_id_tensor = trainable_tensor
    model._tool_reserved_token_id_tensor = tool_tensor
    model._trainable_reserved_index_lookup = lookup
    model._min_trainable_reserved_token_id = int(trainable_tensor.min().item())
    model._max_trainable_reserved_token_id = int(trainable_tensor.max().item())


def _replace_linear_tcra(model, old_tool_count, new_weight, new_bias):
    old_head = model.logit_bias_head
    if not isinstance(old_head, nn.Linear):
        raise ValueError("Cold-start TCRA expansion currently supports the linear head")

    new_weight = new_weight.to(
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    new_bias = new_bias.to(
        device=old_head.bias.device,
        dtype=old_head.bias.dtype,
    )
    expanded_head = nn.Linear(
        old_head.in_features,
        old_tool_count + new_weight.shape[0],
        bias=True,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    with torch.no_grad():
        expanded_head.weight[:old_tool_count].copy_(old_head.weight)
        expanded_head.bias[:old_tool_count].copy_(old_head.bias)
        expanded_head.weight[old_tool_count:].copy_(new_weight)
        expanded_head.bias[old_tool_count:].copy_(new_bias)
    expanded_head.requires_grad_(False)
    expanded_head.eval()
    model.logit_bias_head = expanded_head


def _cold_start_logit_bias(
    self,
    logits,
    hidden_states,
    active_decision_rows,
):
    if not active_decision_rows.any():
        return logits
    active_indices = active_decision_rows.nonzero(as_tuple=False).squeeze(-1)
    old_count = int(self._cold_start_old_tool_count)
    tool_token_ids = self._get_tool_reserved_token_ids_tensor(logits.device)
    identity_offsets = getattr(
        self,
        "_cold_start_new_identity_offsets",
        None,
    )
    if self.logit_bias_head is None:
        if identity_offsets is None:
            return logits
        identity_offsets = identity_offsets.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        new_token_ids = tool_token_ids[old_count:]
        original_new_logits = logits[
            active_indices[:, None],
            new_token_ids[None, :],
        ]
        identity_logits = original_new_logits - identity_offsets.unsqueeze(0)
        calibrated_new_logits = (
            original_new_logits.max(dim=1, keepdim=True).values
            + identity_logits
            - identity_logits.max(dim=1, keepdim=True).values
        )
        updated_logits = logits.clone()
        updated_logits[
            active_indices[:, None],
            new_token_ids[None, :],
        ] = calibrated_new_logits
        return updated_logits
    if hidden_states is None:
        raise ValueError("Boundary hidden states are required for TCRA")

    scores = self._get_logit_bias_scores(hidden_states[active_indices]).float()
    old_log_probabilities = torch.log_softmax(scores[:, :old_count], dim=-1)
    old_bias = old_log_probabilities + math.log(old_count)
    old_reference = (
        torch.logsumexp(scores[:, :old_count], dim=-1, keepdim=True)
        - math.log(old_count)
    )
    new_gain = float(getattr(self, "_cold_start_new_tcra_gain", 1.0))
    new_bias = (scores[:, old_count:] - old_reference) * new_gain
    tool_bias = torch.cat([old_bias, new_bias], dim=-1)
    tool_bias = (tool_bias * self.logit_bias_scale).to(dtype=logits.dtype)
    if identity_offsets is not None:
        identity_offsets = identity_offsets.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        new_token_ids = tool_token_ids[old_count:]
        original_new_logits = (
            logits[active_indices[:, None], new_token_ids[None, :]]
            + tool_bias[:, old_count:]
        )
        identity_logits = original_new_logits - identity_offsets.unsqueeze(0)
        calibrated_new_logits = (
            original_new_logits.max(dim=1, keepdim=True).values
            + identity_logits
            - identity_logits.max(dim=1, keepdim=True).values
        )
        tool_bias[:, old_count:] += (
            calibrated_new_logits - original_new_logits
        )
    new_penalty = float(
        getattr(self, "_cold_start_new_logit_penalty", 0.0)
    )
    if new_penalty:
        tool_bias[:, old_count:] -= new_penalty

    updated_logits = logits.clone()
    updated_logits[
        active_indices[:, None],
        tool_token_ids[None, :],
    ] += tool_bias
    return updated_logits


def append_cold_start_tools(
    model,
    new_tool_names,
    new_embeddings,
    new_tcra_weight=None,
    new_tcra_bias=None,
):
    """Append frozen tool rows after strict loading while preserving the old EOC."""
    old_tool_count = int(model.num_tools)
    new_tool_names = list(new_tool_names)
    new_reserved_tokens = _available_reserved_tokens(
        model,
        len(new_tool_names),
    )

    _replace_memory_parameter(model, new_embeddings)
    _replace_registry(model, new_tool_names, new_reserved_tokens)

    if model.logit_bias_head is not None:
        if new_tcra_weight is None or new_tcra_bias is None:
            raise ValueError("TapMem expansion requires new TCRA weight and bias")
        _replace_linear_tcra(
            model,
            old_tool_count,
            new_tcra_weight,
            new_tcra_bias,
        )

    model._cold_start_old_tool_count = old_tool_count
    model._cold_start_new_tool_count = len(new_tool_names)
    model._apply_logit_bias_to_logits = types.MethodType(
        _cold_start_logit_bias,
        model,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    return {
        "old_tool_count": old_tool_count,
        "new_tool_count": len(new_tool_names),
        "tool_names": list(model.tool_names),
        "tool_token_names": list(model.tool_reserved_token_names),
        "tool_token_ids": list(model.tool_reserved_token_ids),
        "eoc_token_name": model.eoc_token_name,
        "eoc_token_id": model.eoc_token_id,
    }


def apply_cold_start_delta(model, delta):
    if delta.get("format") != COLD_START_FORMAT:
        raise ValueError(f"Unknown cold-start format: {delta.get('format')!r}")
    if delta.get("version") != COLD_START_VERSION:
        raise ValueError(f"Unsupported cold-start version: {delta.get('version')!r}")
    if list(model.tool_names) != list(delta["base_tool_names"]):
        raise ValueError("Delta base-tool order does not match the loaded checkpoint")

    tcra = delta.get("tcra")
    registry = append_cold_start_tools(
        model,
        delta["new_tool_names"],
        delta["new_embeddings"],
        None if tcra is None else tcra["new_weight"],
        None if tcra is None else tcra["new_bias"],
    )
    calibration = delta.get("routing_calibration") or {}
    model._cold_start_new_tcra_gain = float(
        calibration.get("new_tcra_gain", 1.0)
    )
    model._cold_start_new_logit_penalty = float(
        calibration.get("new_logit_penalty", 0.0)
    )
    identity_offsets = calibration.get("new_tool_identity_offsets")
    model._cold_start_new_identity_offsets = (
        None
        if identity_offsets is None
        else torch.as_tensor(identity_offsets).float().cpu()
    )
    return registry
