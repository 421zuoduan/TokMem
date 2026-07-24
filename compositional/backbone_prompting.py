"""Backbone-specific prompt and response-boundary formatting.

Legacy backbones keep the repository's existing Llama serialization exactly.
Qwen3.5 uses its native chat template and tokenizer EOS without changing the
semantic experiment sequence.
"""


QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}


def resolve_model_type(model=None, model_type=None):
    """Resolve the outer or text model type without loading another config."""
    if model_type:
        return model_type

    full_config = getattr(model, "full_config", None)
    if full_config is not None and getattr(full_config, "model_type", None):
        return full_config.model_type

    config = getattr(model, "config", None)
    if config is not None and getattr(config, "model_type", None):
        return config.model_type

    return None


def uses_qwen35_prompting(model=None, model_type=None):
    """Return whether the Qwen3.5 conversation protocol should be used."""
    return resolve_model_type(model=model, model_type=model_type) in QWEN35_MODEL_TYPES


def format_user_assistant_prompt(
    tokenizer,
    user_input,
    *,
    model=None,
    model_type=None,
    legacy_assistant_newline=False,
):
    """Format a user message followed by an assistant generation prompt."""
    if uses_qwen35_prompting(model=model, model_type=model_type):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    assistant_suffix = "\n" if legacy_assistant_newline else ""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        f"{assistant_suffix}"
    )


def format_system_user_assistant_prompt(
    tokenizer,
    system_content,
    user_input,
    *,
    model=None,
    model_type=None,
    legacy_prompt=None,
):
    """Format system/user messages followed by an assistant prompt."""
    if uses_qwen35_prompting(model=model, model_type=model_type):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    if legacy_prompt is None:
        raise ValueError("legacy_prompt is required for non-Qwen3.5 backbones")
    return legacy_prompt


def response_end_token_ids(tokenizer, *, model=None, model_type=None):
    """Return the supervised response-ending token sequence."""
    if uses_qwen35_prompting(model=model, model_type=model_type):
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen3.5 tokenizer must define eos_token_id")
        return [int(tokenizer.eos_token_id)]

    return tokenizer("<|eot_id|>", add_special_tokens=False)["input_ids"]


def response_end_text(tokenizer, *, model=None, model_type=None):
    """Return the text appended to a supervised assistant response."""
    if uses_qwen35_prompting(model=model, model_type=model_type):
        if tokenizer.eos_token is None:
            raise ValueError("Qwen3.5 tokenizer must define eos_token")
        return tokenizer.eos_token
    return "<|eot_id|>"


def qwen35_generation_token_ids(tokenizer, *, model=None, model_type=None):
    """Return explicit (pad, eos) IDs for Qwen3.5 native generation."""
    if not uses_qwen35_prompting(model=model, model_type=model_type):
        return None

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Qwen3.5 tokenizer must define eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return int(pad_token_id), int(eos_token_id)
