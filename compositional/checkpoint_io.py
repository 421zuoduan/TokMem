"""Versioned checkpoint helpers for compositional TokMem-family models."""

import os
from copy import deepcopy

import torch
from peft import (
    PeftConfig,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


FULL_CHECKPOINT_FORMAT = "full"
TRAINABLE_CHECKPOINT_FORMAT = "compositional_trainable"
TRAINABLE_CHECKPOINT_VERSION = 1
CHECKPOINT_FORMAT_CHOICES = (FULL_CHECKPOINT_FORMAT, "trainable_only")


def _cpu_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Checkpoint state values must be tensors, got {type(tensor).__name__}")
    return tensor.detach().cpu().clone()


def _cpu_state_dict(state_dict):
    return {name: _cpu_tensor(tensor) for name, tensor in state_dict.items()}


def _model_metadata(model, base_model_name):
    config = getattr(model, "config", None)
    return {
        "base_model": base_model_name,
        "wrapper_class": type(model).__name__,
        "model_type": getattr(config, "model_type", None),
        "num_tools": int(model.num_tools),
        "decouple_embeddings": bool(model.decouple_embeddings),
        "use_eoc": bool(model.use_eoc),
        "use_logit_bias": bool(model.use_logit_bias),
        "use_tool_head_replacement": bool(model.use_tool_head_replacement),
        "use_memory_bank_constraint": bool(
            getattr(model, "use_memory_bank_constraint", False)
        ),
        "memory_bank_probability_threshold": float(
            getattr(model, "memory_bank_probability_threshold", 0.5)
        ),
        "logit_bias_network": model.logit_bias_network,
        "logit_bias_scale": float(model.logit_bias_scale),
        "lora_config": deepcopy(model.lora_config),
        "reserved_token_names": list(model.trainable_reserved_token_names),
        "reserved_token_ids": list(model.trainable_reserved_token_ids),
        "eoc_token_name": model.eoc_token_name,
        "eoc_token_id": model.eoc_token_id,
    }


def _memory_state(model):
    if model.decouple_embeddings:
        return {
            "mode": "decoupled",
            "input_embeddings": _cpu_tensor(
                model.trainable_tool_input_embeddings
            ),
            "output_embeddings": _cpu_tensor(
                model.trainable_tool_output_embeddings
            ),
        }
    return {
        "mode": "coupled",
        "embeddings": _cpu_tensor(model.trainable_tool_embeddings),
    }


def _tool_head_state(model):
    if model.logit_bias_head is None:
        return None
    return _cpu_state_dict(model.logit_bias_head.state_dict())


def _adapter_state(model):
    if not model.lora_config:
        return None
    state_dict = get_peft_model_state_dict(
        model.model,
        adapter_name="default",
        save_embedding_layers=False,
    )
    if not state_dict:
        raise ValueError("LoRA is configured, but PEFT returned an empty adapter state")
    return {
        "adapter_name": "default",
        "state_dict": _cpu_state_dict(state_dict),
    }


def build_checkpoint_payload(
    model,
    round_number,
    round_tools,
    results,
    *,
    checkpoint_format=FULL_CHECKPOINT_FORMAT,
    base_model_name=None,
):
    """Build either the legacy full payload or a versioned trainable-only payload."""
    if checkpoint_format == FULL_CHECKPOINT_FORMAT:
        # Preserve the historical payload exactly for backward compatibility.
        return {
            "round": round_number,
            "tools": list(round_tools),
            "model_state_dict": model.state_dict(),
            "results": results,
        }
    if checkpoint_format != "trainable_only":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    if not base_model_name:
        raise ValueError("base_model_name is required for trainable-only checkpoints")

    return {
        "checkpoint_format": TRAINABLE_CHECKPOINT_FORMAT,
        "checkpoint_version": TRAINABLE_CHECKPOINT_VERSION,
        "round": round_number,
        # Keep the historical field as the current round's tools.
        "tools": list(round_tools),
        # The model is initialized with every round's tools, in this exact order.
        "tool_names": list(model.tool_names),
        "model_metadata": _model_metadata(model, base_model_name),
        "trainable_state": {
            "memory": _memory_state(model),
            "tool_head": _tool_head_state(model),
            # Use PEFT's semantic adapter extraction even if LoRA is now frozen.
            "lora": _adapter_state(model),
        },
        "results": results,
    }


def checkpoint_tool_names(checkpoint, fallback=None):
    """Return the complete ordered tool list needed to reconstruct a checkpoint."""
    if checkpoint.get("checkpoint_format") == TRAINABLE_CHECKPOINT_FORMAT:
        tool_names = checkpoint.get("tool_names")
        if not isinstance(tool_names, list) or not tool_names:
            raise ValueError(
                "Trainable-only checkpoint is missing its complete ordered tool_names"
            )
        return list(tool_names)
    if fallback is not None:
        return list(fallback)
    tools = checkpoint.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Checkpoint is missing tools")
    return list(tools)


def _validate_equal(label, checkpoint_value, model_value):
    if checkpoint_value != model_value:
        raise ValueError(
            f"Checkpoint {label} mismatch: saved={checkpoint_value!r}, "
            f"model={model_value!r}"
        )


def _validate_trainable_metadata(model, checkpoint):
    metadata = checkpoint.get("model_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Trainable-only checkpoint is missing model_metadata")

    _validate_equal(
        "base_model",
        metadata.get("base_model"),
        getattr(model, "model_name", None),
    )
    _validate_equal(
        "wrapper_class",
        metadata.get("wrapper_class"),
        type(model).__name__,
    )
    _validate_equal(
        "model_type",
        metadata.get("model_type"),
        getattr(getattr(model, "config", None), "model_type", None),
    )
    _validate_equal("num_tools", metadata.get("num_tools"), int(model.num_tools))
    _validate_equal(
        "tool_names",
        checkpoint_tool_names(checkpoint),
        list(model.tool_names),
    )
    _validate_equal(
        "decouple_embeddings",
        metadata.get("decouple_embeddings"),
        bool(model.decouple_embeddings),
    )
    for name in ("use_eoc", "use_logit_bias", "use_tool_head_replacement"):
        _validate_equal(name, metadata.get(name), bool(getattr(model, name)))
    _validate_equal(
        "use_memory_bank_constraint",
        metadata.get("use_memory_bank_constraint"),
        bool(getattr(model, "use_memory_bank_constraint", False)),
    )
    _validate_equal(
        "memory_bank_probability_threshold",
        metadata.get("memory_bank_probability_threshold"),
        float(getattr(model, "memory_bank_probability_threshold", 0.5)),
    )
    _validate_equal(
        "logit_bias_network",
        metadata.get("logit_bias_network"),
        model.logit_bias_network,
    )
    _validate_equal(
        "logit_bias_scale",
        metadata.get("logit_bias_scale"),
        float(model.logit_bias_scale),
    )
    _validate_equal(
        "lora_config",
        metadata.get("lora_config"),
        model.lora_config,
    )
    _validate_equal(
        "reserved_token_names",
        metadata.get("reserved_token_names"),
        list(model.trainable_reserved_token_names),
    )
    _validate_equal(
        "reserved_token_ids",
        metadata.get("reserved_token_ids"),
        list(model.trainable_reserved_token_ids),
    )
    _validate_equal("eoc_token_name", metadata.get("eoc_token_name"), model.eoc_token_name)
    _validate_equal("eoc_token_id", metadata.get("eoc_token_id"), model.eoc_token_id)


def _copy_tensor(target, source, label):
    if not isinstance(source, torch.Tensor):
        raise TypeError(f"Checkpoint {label} must be a tensor")
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"Checkpoint {label} shape mismatch: saved={tuple(source.shape)}, "
            f"model={tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def _validate_memory_state(model, memory_state):
    if not isinstance(memory_state, dict):
        raise ValueError("Trainable-only checkpoint is missing memory state")
    expected_mode = "decoupled" if model.decouple_embeddings else "coupled"
    _validate_equal("memory mode", memory_state.get("mode"), expected_mode)

    if expected_mode == "coupled":
        expected_keys = {"mode", "embeddings"}
        if set(memory_state) != expected_keys:
            raise ValueError(
                f"Unexpected coupled memory keys: {sorted(set(memory_state) - expected_keys)}"
            )
        source = memory_state["embeddings"]
        if not isinstance(source, torch.Tensor):
            raise TypeError("Checkpoint memory embeddings must be a tensor")
        if tuple(model.trainable_tool_embeddings.shape) != tuple(source.shape):
            raise ValueError(
                "Checkpoint memory embeddings shape mismatch: "
                f"saved={tuple(source.shape)}, "
                f"model={tuple(model.trainable_tool_embeddings.shape)}"
            )
        return

    expected_keys = {"mode", "input_embeddings", "output_embeddings"}
    if set(memory_state) != expected_keys:
        raise ValueError(
            f"Unexpected decoupled memory keys: {sorted(set(memory_state) - expected_keys)}"
        )
    for key, target, label in (
        (
            "input_embeddings",
            model.trainable_tool_input_embeddings,
            "input memory embeddings",
        ),
        (
            "output_embeddings",
            model.trainable_tool_output_embeddings,
            "output memory embeddings",
        ),
    ):
        source = memory_state[key]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"Checkpoint {label} must be a tensor")
        if tuple(target.shape) != tuple(source.shape):
            raise ValueError(
                f"Checkpoint {label} shape mismatch: "
                f"saved={tuple(source.shape)}, model={tuple(target.shape)}"
            )


def _load_memory_state(model, memory_state):
    if model.decouple_embeddings:
        _copy_tensor(
            model.trainable_tool_input_embeddings,
            memory_state["input_embeddings"],
            "input memory embeddings",
        )
        _copy_tensor(
            model.trainable_tool_output_embeddings,
            memory_state["output_embeddings"],
            "output memory embeddings",
        )
        return
    _copy_tensor(
        model.trainable_tool_embeddings,
        memory_state["embeddings"],
        "memory embeddings",
    )


def _validate_tool_head_state(model, head_state):
    if model.logit_bias_head is None:
        if head_state is not None:
            raise ValueError("Checkpoint contains a tool head, but the model does not")
        return
    if not isinstance(head_state, dict):
        raise ValueError("Model requires a tool head, but the checkpoint does not contain one")
    expected_state = model.logit_bias_head.state_dict()
    if set(head_state) != set(expected_state):
        missing = sorted(set(expected_state) - set(head_state))
        unexpected = sorted(set(head_state) - set(expected_state))
        raise ValueError(
            f"Tool-head keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, tensor in head_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Checkpoint tool-head value {name} must be a tensor")
        if tuple(tensor.shape) != tuple(expected_state[name].shape):
            raise ValueError(
                f"Tool-head shape mismatch for {name}: "
                f"saved={tuple(tensor.shape)}, "
                f"model={tuple(expected_state[name].shape)}"
            )


def _load_tool_head_state(model, head_state):
    if model.logit_bias_head is None:
        return
    model.logit_bias_head.load_state_dict(head_state, strict=True)


def _validate_adapter_state(model, adapter_state):
    if not model.lora_config:
        if adapter_state is not None:
            raise ValueError("Checkpoint contains LoRA state, but the model has no LoRA adapter")
        return
    if not isinstance(adapter_state, dict):
        raise ValueError("Model requires LoRA state, but the checkpoint does not contain it")
    adapter_name = adapter_state.get("adapter_name")
    state_dict = adapter_state.get("state_dict")
    if adapter_name != "default" or not isinstance(state_dict, dict):
        raise ValueError("Invalid LoRA adapter checkpoint payload")

    expected_state = get_peft_model_state_dict(
        model.model,
        adapter_name=adapter_name,
        save_embedding_layers=False,
    )
    if set(state_dict) != set(expected_state):
        missing = sorted(set(expected_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected_state))
        raise ValueError(
            f"LoRA adapter keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, tensor in state_dict.items():
        if tuple(tensor.shape) != tuple(expected_state[name].shape):
            raise ValueError(
                f"LoRA adapter shape mismatch for {name}: "
                f"saved={tuple(tensor.shape)}, model={tuple(expected_state[name].shape)}"
            )
    return adapter_name, state_dict


def _load_adapter_state(model, adapter_state):
    if not model.lora_config:
        return
    adapter_name = adapter_state["adapter_name"]
    state_dict = adapter_state["state_dict"]
    set_peft_model_state_dict(
        model.model,
        state_dict,
        adapter_name=adapter_name,
    )


def load_checkpoint_into_model(model, checkpoint):
    """Load legacy full checkpoints or explicit base-plus-delta checkpoints."""
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return FULL_CHECKPOINT_FORMAT

    checkpoint_format = checkpoint.get("checkpoint_format")
    if checkpoint_format != TRAINABLE_CHECKPOINT_FORMAT:
        raise ValueError(f"Unknown checkpoint format: {checkpoint_format!r}")
    version = checkpoint.get("checkpoint_version")
    if version != TRAINABLE_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported {TRAINABLE_CHECKPOINT_FORMAT} version: {version!r}"
        )

    _validate_trainable_metadata(model, checkpoint)
    state = checkpoint.get("trainable_state")
    if not isinstance(state, dict):
        raise ValueError("Trainable-only checkpoint is missing trainable_state")
    expected_keys = {"memory", "tool_head", "lora"}
    if set(state) != expected_keys:
        raise ValueError(
            f"Unexpected trainable state keys: {sorted(set(state) - expected_keys)}"
        )

    # Validate every component before mutating the freshly loaded base model.
    _validate_memory_state(model, state["memory"])
    _validate_tool_head_state(model, state["tool_head"])
    _validate_adapter_state(model, state["lora"])

    # PEFT is the only external loader; apply it first after successful validation.
    _load_adapter_state(model, state["lora"])
    _load_tool_head_state(model, state["tool_head"])
    _load_memory_state(model, state["memory"])
    return "trainable_only"


def _normalized_model_reference(reference):
    if reference is None:
        return None
    reference = os.fspath(reference)
    if os.path.isabs(reference) or os.path.exists(reference):
        return os.path.realpath(reference)
    return reference


def load_lora_adapter(
    base_model,
    adapter_checkpoint,
    *,
    expected_base_model_name=None,
    is_trainable=False,
):
    """Load a standalone PEFT adapter on top of an already loaded base model."""
    adapter_config = PeftConfig.from_pretrained(
        adapter_checkpoint,
        local_files_only=True,
    )
    if expected_base_model_name is not None:
        saved_base_model = _normalized_model_reference(
            adapter_config.base_model_name_or_path
        )
        expected_base_model = _normalized_model_reference(
            expected_base_model_name
        )
        if saved_base_model != expected_base_model:
            raise ValueError(
                "LoRA adapter base model mismatch: "
                f"saved={adapter_config.base_model_name_or_path!r}, "
                f"expected={expected_base_model_name!r}"
            )
    return PeftModel.from_pretrained(
        base_model,
        adapter_checkpoint,
        config=adapter_config,
        is_trainable=is_trainable,
        local_files_only=True,
    )
