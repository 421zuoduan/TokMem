from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def build_available_tool_mask(
    tool_names: Sequence[str],
    available_tool_ids: Sequence[str],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    index = {tool_name: position for position, tool_name in enumerate(tool_names)}
    unknown = sorted(set(available_tool_ids) - set(index))
    if unknown:
        raise ValueError(f"available tool IDs are missing from the manifest: {unknown}")
    mask = torch.zeros(len(tool_names), dtype=torch.bool, device=device)
    for tool_id in available_tool_ids:
        mask[index[tool_id]] = True
    if not mask.any():
        raise ValueError("available tool mask cannot be empty")
    return mask


def _broadcast_mask(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape[-1] != logits.shape[-1]:
        raise ValueError(
            f"mask width {mask.shape[-1]} does not match logits width {logits.shape[-1]}"
        )
    if mask.ndim == 1:
        mask = mask.reshape(*([1] * (logits.ndim - 1)), mask.shape[-1])
    elif mask.ndim < logits.ndim:
        if mask.shape[0] != logits.shape[0]:
            raise ValueError(
                "batch-specific mask first dimension must match logits batch size"
            )
        mask = mask.reshape(
            mask.shape[0],
            *([1] * (logits.ndim - mask.ndim)),
            *mask.shape[1:],
        )
    try:
        return torch.broadcast_to(mask, logits.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to logits {tuple(logits.shape)}"
        ) from exc


def mask_tool_head_logits(
    tool_logits: torch.Tensor,
    available_tool_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _broadcast_mask(available_tool_mask, tool_logits)
    return tool_logits.masked_fill(~mask, float("-inf"))


def mask_backbone_tool_logits(
    vocab_logits: torch.Tensor,
    tool_token_ids: Sequence[int] | torch.Tensor,
    available_tool_mask: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.as_tensor(
        tool_token_ids,
        dtype=torch.long,
        device=vocab_logits.device,
    )
    if token_ids.numel() != available_tool_mask.shape[-1]:
        raise ValueError("tool token count must match available mask width")
    masked = vocab_logits.clone()
    selected = masked.index_select(-1, token_ids)
    mask = _broadcast_mask(available_tool_mask, selected)
    selected = selected.masked_fill(~mask, float("-inf"))
    masked[..., token_ids] = selected
    return masked


def masked_routing_cross_entropy(
    tool_logits: torch.Tensor,
    targets: torch.Tensor,
    available_tool_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = mask_tool_head_logits(tool_logits, available_tool_mask)
    flat_targets = targets.reshape(-1)
    flat_mask = _broadcast_mask(available_tool_mask, tool_logits).reshape(
        -1, tool_logits.shape[-1]
    )
    target_available = flat_mask.gather(1, flat_targets[:, None]).squeeze(1)
    if not bool(target_available.all()):
        bad = torch.nonzero(~target_available, as_tuple=False).flatten().tolist()
        raise ValueError(f"routing targets are unavailable for rows: {bad}")
    return F.cross_entropy(masked_logits.reshape(-1, tool_logits.shape[-1]), flat_targets)


def masked_centered_log_prior(
    tool_logits: torch.Tensor,
    available_tool_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _broadcast_mask(available_tool_mask, tool_logits)
    masked_logits = tool_logits.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits.float(), dim=-1)
    available_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
    centered = log_probs + torch.log(available_counts)
    return centered.masked_fill(~mask, 0.0)


def apply_masked_logit_bias(
    vocab_logits: torch.Tensor,
    tool_logits: torch.Tensor,
    tool_token_ids: Sequence[int] | torch.Tensor,
    available_tool_mask: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    masked_vocab = mask_backbone_tool_logits(
        vocab_logits,
        tool_token_ids,
        available_tool_mask,
    )
    token_ids = torch.as_tensor(
        tool_token_ids,
        dtype=torch.long,
        device=vocab_logits.device,
    )
    bias = masked_centered_log_prior(tool_logits, available_tool_mask)
    bias = bias.to(dtype=vocab_logits.dtype) * float(scale)
    selected = masked_vocab.index_select(-1, token_ids) + bias
    masked_vocab[..., token_ids] = selected
    return masked_vocab
