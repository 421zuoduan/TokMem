"""Qwen3.5-specific TokMem model wrapper.

The existing ``FunctionCallingModel`` remains the implementation used by all
previous backbones.  This subclass keeps its public interface and reuses its
training, decoding, and parsing methods while isolating the Qwen3.5 model
construction differences here.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from backbone_prompting import qwen35_generation_token_ids
from model import FunctionCallingModel, build_logit_bias_head


def resolve_qwen35_lora_targets(model, config, layer_indices, base_modules):
    """Resolve exact Qwen3.5 LoRA targets for selected hybrid layers."""
    num_layers = config.num_hidden_layers
    actual_layer_indices = []
    for layer_index in layer_indices:
        actual_index = num_layers + layer_index if layer_index < 0 else layer_index
        if actual_index < 0 or actual_index >= num_layers:
            raise ValueError(
                f"Invalid layer index {layer_index}. Model has {num_layers} "
                f"layers (0 to {num_layers - 1})"
            )
        actual_layer_indices.append(actual_index)

    available_modules = dict(model.named_modules())
    target_modules = []
    unmatched_layers = []
    for layer_index in actual_layer_indices:
        layer_targets = []
        for attention_module in ("self_attn", "linear_attn"):
            for module_name in base_modules:
                target_name = (
                    f"model.layers.{layer_index}.{attention_module}.{module_name}"
                )
                if target_name in available_modules:
                    layer_targets.append(target_name)
        if layer_targets:
            target_modules.extend(layer_targets)
        else:
            unmatched_layers.append(layer_index)

    if not target_modules:
        raise ValueError(
            "No Qwen3.5 LoRA modules matched "
            f"layers={actual_layer_indices}, target_modules={base_modules}"
        )

    return target_modules, actual_layer_indices, unmatched_layers


class Qwen35FunctionCallingModel(FunctionCallingModel):
    """Function-calling wrapper for the Qwen3.5 text backbone."""

    def __init__(
        self,
        model_name,
        num_tools=100,
        tool_names=None,
        tokenizer=None,
        device="cuda",
        dtype=torch.bfloat16,
        decouple_embeddings=False,
        lora_config=None,
        use_eoc=False,
        use_logit_bias=False,
        use_tool_head_replacement=False,
        logit_bias_network="linear",
        logit_bias_scale=1.0,
        use_memory_bank_constraint=False,
        memory_bank_probability_threshold=0.5,
    ):
        nn.Module.__init__(self)

        self.full_config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=True,
        )
        if self.full_config.model_type != "qwen3_5":
            raise ValueError(
                "Qwen35FunctionCallingModel requires model_type='qwen3_5', "
                f"got {self.full_config.model_type!r}"
            )

        self.use_eoc = use_eoc
        self.use_logit_bias = use_logit_bias
        self.use_tool_head_replacement = use_tool_head_replacement
        self.use_memory_bank_constraint = use_memory_bank_constraint
        self.memory_bank_probability_threshold = float(
            memory_bank_probability_threshold
        )
        self.logit_bias_network = logit_bias_network
        self.logit_bias_scale = logit_bias_scale

        if self.use_logit_bias and self.use_tool_head_replacement:
            raise ValueError(
                "--use_logit_bias and --use_tool_head_replacement are "
                "decode-time alternatives"
            )
        if self.use_memory_bank_constraint and self.use_tool_head_replacement:
            raise ValueError(
                "--use_memory_bank_constraint cannot be combined with "
                "--use_tool_head_replacement"
            )
        if (
            self.use_logit_bias or self.use_tool_head_replacement
        ) and not self.use_eoc:
            raise ValueError(
                "--use_logit_bias and --use_tool_head_replacement require --use_eoc"
            )
        if not 0.0 <= self.memory_bank_probability_threshold <= 1.0:
            raise ValueError(
                "memory_bank_probability_threshold must be between 0 and 1"
            )
        if self.logit_bias_network not in {"linear", "mlp"}:
            raise ValueError(
                f"Unsupported logit_bias_network: {self.logit_bias_network}"
            )

        # Keep the same memory-token budget and mapping behavior as the existing
        # FunctionCallingModel.
        self.max_reserved_tokens = 248
        self.num_tools = min(
            num_tools,
            self.max_reserved_tokens - (1 if self.use_eoc else 0),
        )
        self.num_reserved_slots = self.num_tools + (1 if self.use_eoc else 0)
        self.added_reserved_token_count = 0
        if self.num_tools != num_tools:
            print(
                f"Adjusted num_tools from {num_tools} to {self.num_tools} to fit "
                f"the reserved-token budget with use_eoc={self.use_eoc}"
            )

        self.device = device
        self.dtype = dtype
        self.decouple_embeddings = decouple_embeddings
        self.lora_config = lora_config
        self.tokenizer = tokenizer
        (
            self._native_generation_pad_token_id,
            self._native_generation_eos_token_id,
        ) = qwen35_generation_token_ids(
            tokenizer,
            model_type=self.full_config.model_type,
        )

        self._setup_reserved_tokens()

        # Recent Transformers releases automatically select the text-only
        # Qwen3_5ForCausalLM and map the language-model weights out of the full
        # multimodal checkpoint.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
        self.config = self.model.config
        if self.config.model_type != "qwen3_5_text":
            raise ValueError(
                "Expected the Qwen3.5 text config after causal-LM loading, "
                f"got {self.config.model_type!r}"
            )

        self._ensure_model_token_capacity()

        if self.lora_config:
            print(f"Applying Qwen3.5 LoRA with config: {self.lora_config}")
            layer_indices = self.lora_config.get("layer_indices")
            base_modules = self.lora_config.get("target_modules", ["o_proj"])

            if layer_indices is not None:
                if isinstance(layer_indices, int):
                    layer_indices = [layer_indices]
                (
                    target_modules,
                    actual_layer_indices,
                    unmatched_layers,
                ) = resolve_qwen35_lora_targets(
                    self.model,
                    self.config,
                    layer_indices,
                    base_modules,
                )
                print(
                    "Applying Qwen3.5 LoRA to layers "
                    f"{actual_layer_indices} (indices={layer_indices})"
                )
                if unmatched_layers:
                    print(
                        "Qwen3.5 LoRA targets were not present in layers "
                        f"{unmatched_layers}; those layers are skipped"
                    )
            else:
                target_modules = base_modules
                print("Applying Qwen3.5 LoRA to all matching layers")

            lora_peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_config.get("r", 1),
                lora_alpha=self.lora_config.get("alpha", 32),
                lora_dropout=self.lora_config.get("dropout", 0.1),
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_peft_config)
            print("Qwen3.5 LoRA applied successfully")
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        core_model = self._get_qwen35_core_model()
        input_embeddings = core_model.get_input_embeddings()
        output_embeddings = core_model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            raise ValueError(
                "Qwen3.5 causal LM must expose input and output embeddings"
            )
        original_embeddings = input_embeddings.weight.data

        reserved_token_indices = torch.tensor(
            self.trainable_reserved_token_ids,
            device=device,
        )
        if int(reserved_token_indices.max().item()) >= input_embeddings.num_embeddings:
            raise ValueError(
                "Qwen3.5 reserved token ID exceeds the input embedding capacity"
            )
        if int(reserved_token_indices.max().item()) >= output_embeddings.out_features:
            raise ValueError(
                "Qwen3.5 reserved token ID exceeds the output embedding capacity"
            )

        self.register_buffer(
            "_trainable_reserved_token_id_tensor",
            reserved_token_indices.clone().long(),
            persistent=False,
        )
        tool_token_indices = torch.tensor(
            self.tool_reserved_token_ids,
            device=device,
            dtype=torch.long,
        )
        self.register_buffer(
            "_tool_reserved_token_id_tensor",
            tool_token_indices,
            persistent=False,
        )
        reserved_index_lookup = torch.full(
            (int(reserved_token_indices.max().item()) + 1,),
            -1,
            device=device,
            dtype=torch.long,
        )
        reserved_index_lookup[reserved_token_indices.long()] = torch.arange(
            reserved_token_indices.numel(),
            device=device,
            dtype=torch.long,
        )
        self.register_buffer(
            "_trainable_reserved_index_lookup",
            reserved_index_lookup,
            persistent=False,
        )
        self._min_trainable_reserved_token_id = int(
            reserved_token_indices.min().item()
        )
        self._max_trainable_reserved_token_id = int(
            reserved_token_indices.max().item()
        )

        if self.decouple_embeddings:
            self.trainable_tool_input_embeddings = nn.Parameter(
                original_embeddings[reserved_token_indices].clone()
            )
            self.trainable_tool_output_embeddings = nn.Parameter(
                original_embeddings[reserved_token_indices].clone()
            )
        else:
            self.trainable_tool_embeddings = nn.Parameter(
                original_embeddings[reserved_token_indices].clone()
            )
            self.trainable_tool_input_embeddings = self.trainable_tool_embeddings
            self.trainable_tool_output_embeddings = self.trainable_tool_embeddings

        if self.use_logit_bias or self.use_tool_head_replacement:
            self.logit_bias_head = build_logit_bias_head(
                self.config.hidden_size,
                self.num_tools,
                self.logit_bias_network,
            ).to(device=device, dtype=original_embeddings.dtype)
        else:
            self.logit_bias_head = None

        provided_tool_names = tool_names or [
            f"tool_{index}" for index in range(self.num_tools)
        ]
        self.tool_names = list(provided_tool_names[: self.num_tools])
        self.tool_name_to_id = {
            name: index for index, name in enumerate(self.tool_names)
        }
        self.tool_id_to_name = {
            index: name for index, name in enumerate(self.tool_names)
        }
        self.tool_id_to_token_id = {
            index: self.tool_reserved_token_ids[index]
            for index in range(self.num_tools)
        }
        self.token_id_to_tool_id = {
            self.tool_reserved_token_ids[index]: index
            for index in range(self.num_tools)
        }

        self._setup_model_override()
        self._print_parameter_breakdown()

    def _get_qwen35_core_model(self):
        if self.lora_config:
            return self.model.get_base_model()
        return self.model

    def _setup_model_override(self):
        """Bind TokMem embedding/logit overrides through public model accessors."""
        core_model = self._get_qwen35_core_model()
        input_embedding_module = core_model.get_input_embeddings()
        output_embedding_module = core_model.get_output_embeddings()
        if input_embedding_module is None or output_embedding_module is None:
            raise ValueError(
                "Qwen3.5 causal LM must expose input and output embeddings"
            )

        self.original_embed_forward = input_embedding_module.forward
        self.original_lm_head_forward = output_embedding_module.forward

        def custom_embed_forward(input_ids):
            embeddings = self.original_embed_forward(input_ids)

            flattened_input_ids = input_ids.reshape(-1)
            candidate_mask = (
                (flattened_input_ids >= self._min_trainable_reserved_token_id)
                & (flattened_input_ids <= self._max_trainable_reserved_token_id)
            )
            if candidate_mask.any():
                candidate_positions = candidate_mask.nonzero(
                    as_tuple=False
                ).squeeze(-1)
                lookup = self._get_trainable_reserved_index_lookup(input_ids.device)
                reserved_indices = lookup[
                    flattened_input_ids[candidate_positions]
                ]
                reserved_mask = reserved_indices >= 0
                reserved_positions = candidate_positions[reserved_mask]
                reserved_indices = reserved_indices[reserved_mask]
            else:
                reserved_positions = None

            if reserved_positions is not None and reserved_positions.numel() > 0:
                flattened_embeddings = embeddings.reshape(
                    -1,
                    embeddings.size(-1),
                )
                flattened_embeddings[reserved_positions] = (
                    self.trainable_tool_input_embeddings[reserved_indices]
                )

            return embeddings

        def custom_lm_head_forward(hidden_states):
            logits = self.original_lm_head_forward(hidden_states)
            reserved_token_ids = self._get_trainable_reserved_token_ids_tensor(
                logits.device
            )
            reserved_token_logits = torch.matmul(
                hidden_states,
                self.trainable_tool_output_embeddings.transpose(0, 1),
            )
            logits[..., reserved_token_ids] = reserved_token_logits
            return logits

        input_embedding_module.forward = custom_embed_forward
        output_embedding_module.forward = custom_lm_head_forward

    def restore_original_model(self):
        """Restore the Qwen3.5 embedding and output-head forward methods."""
        core_model = self._get_qwen35_core_model()
        if hasattr(self, "original_embed_forward"):
            core_model.get_input_embeddings().forward = self.original_embed_forward
        if hasattr(self, "original_lm_head_forward"):
            core_model.get_output_embeddings().forward = (
                self.original_lm_head_forward
            )
