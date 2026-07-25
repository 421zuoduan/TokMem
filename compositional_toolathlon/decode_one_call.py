from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from compositional.backbone_prompting import response_end_token_ids

from .masked_routing import (
    apply_masked_logit_bias,
    mask_backbone_tool_logits,
)


@dataclass(frozen=True)
class DecodedAction:
    tool_id: str
    arguments: dict[str, Any]
    stop_reason: str
    generated_token_ids: tuple[int, ...]


class ActionDecodeError(RuntimeError):
    pass


def _ends_with(values: Sequence[int], suffix: Sequence[int]) -> bool:
    return len(values) >= len(suffix) and list(values[-len(suffix) :]) == list(suffix)


def _parse_arguments(tokenizer: Any, token_ids: list[int]) -> dict[str, Any]:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionDecodeError(f"arguments are not one valid JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionDecodeError("tool arguments must decode to a JSON object")
    return payload


def decode_one_call(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    available_tool_mask: torch.Tensor,
    use_eoc: bool,
    use_logit_bias: bool,
    max_new_tokens: int = 768,
    native_response_end_ids: Sequence[int] | None = None,
) -> DecodedAction:
    """Greedily decode exactly one tool call without parameter repair."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("single-call decoding currently requires batch size 1")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    tool_names = list(getattr(model, "tool_names", []))
    tool_token_ids = list(getattr(model, "tool_reserved_token_ids", []))
    if len(tool_names) != len(tool_token_ids) or not tool_names:
        raise ValueError("model must expose aligned tool_names and tool_reserved_token_ids")
    if available_tool_mask.shape[-1] != len(tool_names):
        raise ValueError("available_tool_mask width must equal the model tool count")
    if not bool(available_tool_mask.any()):
        raise ValueError("available_tool_mask cannot be empty")

    end_ids = list(
        native_response_end_ids
        if native_response_end_ids is not None
        else response_end_token_ids(tokenizer, model=model)
    )
    if not end_ids:
        raise ValueError("native response-end token sequence cannot be empty")

    eoc_token_id = getattr(model, "eoc_token_id", None) if use_eoc else None
    if use_eoc and eoc_token_id is None:
        raise ValueError("EOC decoding requires model.eoc_token_id")
    if use_logit_bias and not use_eoc:
        raise ValueError("TapMem logit bias requires EOC")

    tool_token_to_position = {
        int(token_id): position for position, token_id in enumerate(tool_token_ids)
    }
    generated: list[int] = []
    argument_token_ids: list[int] = []
    chosen_tool_position: int | None = None
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    step_input_ids = input_ids
    past_key_values = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            need_hidden = use_logit_bias and step == 0
            next_logits, hidden_states, past_key_values = model._generation_forward_step(
                input_ids=step_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                return_last_hidden_state=need_hidden,
            )
            selection_logits = next_logits

            if step == 0:
                selection_logits = mask_backbone_tool_logits(
                    selection_logits,
                    tool_token_ids,
                    available_tool_mask,
                )
                if use_logit_bias:
                    if hidden_states is None:
                        raise ActionDecodeError(
                            "TapMem decoding did not receive boundary hidden states"
                        )
                    tool_logits = model._get_logit_bias_scores(hidden_states)
                    if tool_logits is None:
                        raise ActionDecodeError("TapMem model has no logit-bias head")
                    selection_logits = apply_masked_logit_bias(
                        selection_logits,
                        tool_logits,
                        tool_token_ids,
                        available_tool_mask,
                        scale=float(getattr(model, "logit_bias_scale", 1.0)),
                    )
            else:
                selection_logits = selection_logits.clone()
                selection_logits[..., torch.as_tensor(
                    tool_token_ids,
                    dtype=torch.long,
                    device=selection_logits.device,
                )] = float("-inf")

            next_token = int(torch.argmax(selection_logits, dim=-1).item())
            generated.append(next_token)

            if step == 0:
                chosen_tool_position = tool_token_to_position.get(next_token)
                if chosen_tool_position is None:
                    raise ActionDecodeError(
                        "first generated token is not an available memory tool token"
                    )
                if not bool(available_tool_mask[chosen_tool_position]):
                    raise ActionDecodeError("decoder selected a masked tool token")
            else:
                if use_eoc and next_token == int(eoc_token_id):
                    arguments = _parse_arguments(tokenizer, argument_token_ids)
                    return DecodedAction(
                        tool_id=tool_names[int(chosen_tool_position)],
                        arguments=arguments,
                        stop_reason="eoc",
                        generated_token_ids=tuple(generated),
                    )
                argument_token_ids.append(next_token)
                if _ends_with(argument_token_ids, end_ids):
                    if use_eoc:
                        raise ActionDecodeError("response ended before EOC")
                    arguments = _parse_arguments(
                        tokenizer,
                        argument_token_ids[: -len(end_ids)],
                    )
                    return DecodedAction(
                        tool_id=tool_names[int(chosen_tool_position)],
                        arguments=arguments,
                        stop_reason="native_response_end",
                        generated_token_ids=tuple(generated),
                    )

            next_tensor = torch.tensor(
                [[next_token]],
                dtype=current_input_ids.dtype,
                device=current_input_ids.device,
            )
            current_input_ids = torch.cat([current_input_ids, next_tensor], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            step_input_ids = next_tensor

    raise ActionDecodeError(f"single tool call exceeded {max_new_tokens} generated tokens")
