import math
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import json

from fsdp_utils import is_rank_zero

def build_logit_bias_head(hidden_size, num_tools, network_type):
    """Build the external prior head used to bias tool-token logits."""
    if network_type == "linear":
        return nn.Linear(hidden_size, num_tools)
    if network_type == "mlp":
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_tools),
        )
    raise ValueError(f"Unsupported logit_bias_network: {network_type}")

JS_TRUNCATION_THRESHOLD = 0.6


def compute_js_divergence_against_final(logits_by_layer, eps=1e-12):
    logits_by_layer = logits_by_layer.float()
    log_probs = torch.log_softmax(logits_by_layer, dim=-1)
    probs = log_probs.exp()

    final_probs = probs[-1:].expand_as(probs)
    final_log_probs = log_probs[-1:].expand_as(log_probs)

    mixture = 0.5 * (probs + final_probs)
    log_mixture = torch.log(mixture.clamp_min(eps))

    kl_layer = (probs * (log_probs - log_mixture)).sum(dim=-1)
    kl_final = (final_probs * (final_log_probs - log_mixture)).sum(dim=-1)
    return 0.5 * (kl_layer + kl_final)

def count_parameters(model):
    """Count trainable and total parameters in a model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def print_model_info(model, model_name):
    """Print model information including parameter counts"""
    trainable_params, total_params = count_parameters(model)
    print(f"  {model_name}:")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable ratio: {trainable_params/total_params*100:.4f}%")


class ToolTokenModule(nn.Module):
    """Keep token-memory parameters in a small module that can stay replicated under FSDP."""

    def __init__(
        self,
        original_embeddings,
        reserved_token_indices,
        hidden_size,
        num_tools,
        decouple_embeddings=False,
        use_logit_bias=False,
        logit_bias_network="linear",
    ):
        super().__init__()
        selected_embeddings = original_embeddings[reserved_token_indices].clone()
        if decouple_embeddings:
            self.input_embeddings = nn.Parameter(selected_embeddings.clone())
            self.output_embeddings = nn.Parameter(selected_embeddings.clone())
        else:
            shared_embeddings = nn.Parameter(selected_embeddings.clone())
            self.input_embeddings = shared_embeddings
            self.output_embeddings = shared_embeddings

        if use_logit_bias:
            self.logit_bias_head = build_logit_bias_head(
                hidden_size,
                num_tools,
                logit_bias_network,
            ).to(dtype=original_embeddings.dtype, device=original_embeddings.device)
        else:
            self.logit_bias_head = None

class FunctionCallingModel(nn.Module):
    """
    Function calling using Llama's native reserved special tokens as tool tokens.
    
    This model transparently overrides the embedding and lm_head layers to use 
    trainable embeddings for reserved tokens, with optional coupling/decoupling 
    of input and output layer weights. Optionally supports LoRA fine-tuning
    of the base model for hybrid tokenized memory + LoRA approach.
    
    Args:
        model_name: HuggingFace model name/path (default: "meta-llama/Llama-3.2-1B-Instruct")
        num_tools: Number of tool tokens to use (max 248, default: 100)
        tool_names: List of tool names (default: auto-generated)
        tokenizer: Pre-loaded tokenizer (required)
        device: Device to load model on (default: "cuda")
        dtype: Model data type (default: torch.bfloat16)
        decouple_embeddings: If True, use separate weights for input/output layers.
                           If False, share weights between input/output (default: False)
        lora_config: LoRA configuration dict or None. If provided, applies LoRA to base model.
                    Example: {"r": 1, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "v_proj"],
                             "layer_indices": [21, 22]}  # Optional: specific layers to apply LoRA to
    """
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct",
                 num_tools=100, tool_names=None, tokenizer=None, device="cuda", dtype=torch.bfloat16, 
                 decouple_embeddings=False, lora_config=None, use_eoc=False, use_js_trunc=False,
                 use_logit_bias=False, logit_bias_network="linear", logit_bias_scale=1.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.use_eoc = use_eoc
        self.use_js_trunc = use_js_trunc
        self.use_logit_bias = use_logit_bias
        self.logit_bias_network = logit_bias_network
        self.logit_bias_scale = logit_bias_scale
        if self.use_js_trunc and not self.use_eoc:
            raise ValueError("--use_js_trunc requires --use_eoc")
        if self.use_logit_bias and not self.use_eoc:
            raise ValueError("--use_logit_bias requires --use_eoc")
        if self.logit_bias_network not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported logit_bias_network: {self.logit_bias_network}")

        self.max_reserved_tokens = 248  # Native Llama reserved_special_token_* count
        self.num_tools = min(num_tools, self.max_reserved_tokens - (1 if self.use_eoc else 0))
        self.num_reserved_slots = self.num_tools + (1 if self.use_eoc else 0)
        if self.num_tools != num_tools:
            print(
                f"Adjusted num_tools from {num_tools} to {self.num_tools} to fit the "
                f"reserved-token budget with use_eoc={self.use_eoc}"
            )
        self.device = device
        self.dtype = dtype
        self.decouple_embeddings = decouple_embeddings
        self.lora_config = lora_config
        
        # Load tokenizer to get reserved token mappings
        self.tokenizer = tokenizer
        
        # Get reserved special token mappings
        self._setup_reserved_tokens()
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        
        # Apply LoRA if configured
        if self.lora_config:
            print(f"Applying LoRA with config: {self.lora_config}")
            
            # Handle target modules configuration
            layer_indices = self.lora_config.get("layer_indices", None)
            base_modules = self.lora_config.get("target_modules", ["o_proj"])
            
            if layer_indices is not None:
                # Convert to list if single value provided (backward compatibility)
                if isinstance(layer_indices, int):
                    layer_indices = [layer_indices]
                
                # Apply to specific layers
                num_layers = self.config.num_hidden_layers
                actual_layer_indices = []
                
                for idx in layer_indices:
                    # Handle negative indexing like Python lists
                    if idx < 0:
                        actual_idx = num_layers + idx
                    else:
                        actual_idx = idx
                    
                    # Validate layer index
                    if actual_idx < 0 or actual_idx >= num_layers:
                        raise ValueError(f"Invalid layer index {idx}. Model has {num_layers} layers (0 to {num_layers-1})")
                    
                    actual_layer_indices.append(actual_idx)
                
                # Specify modules only in the specified layers
                # For Llama models: model.layers.{layer_idx}.self_attn.{module_name}
                target_modules = []
                for layer_idx in actual_layer_indices:
                    target_modules.extend([f"model.layers.{layer_idx}.self_attn.{mod}" for mod in base_modules])
                
                print(f"Applying LoRA to layers {actual_layer_indices} (indices={layer_indices})")
            else:
                # Apply to all layers
                target_modules = base_modules
                print(f"Applying LoRA to all layers")
            
            lora_peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_config.get("r", 1),
                lora_alpha=self.lora_config.get("alpha", 32),
                lora_dropout=self.lora_config.get("dropout", 0.1),
                target_modules=target_modules,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_peft_config)
            target_dtype = next(self.model.parameters()).dtype
            for param in self.model.parameters():
                if param.requires_grad and param.dtype != target_dtype:
                    param.data = param.data.to(dtype=target_dtype)
            print("LoRA applied successfully")
        else:
            # Freeze all base model parameters (original tokenized memory approach)
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get original embedding parameters for initialization
        if self.lora_config:
            # When LoRA is applied, access base model embeddings
            original_embeddings = self.model.get_base_model().model.embed_tokens.weight.data
        else:
            # Direct access for non-LoRA model
            original_embeddings = self.model.model.embed_tokens.weight.data
        
        reserved_token_indices = torch.tensor(self.trainable_reserved_token_ids, device=device)
        self.tool_token_module = ToolTokenModule(
            original_embeddings=original_embeddings,
            reserved_token_indices=reserved_token_indices,
            hidden_size=self.config.hidden_size,
            num_tools=self.num_tools,
            decouple_embeddings=self.decouple_embeddings,
            use_logit_bias=self.use_logit_bias,
            logit_bias_network=self.logit_bias_network,
        )
        
        # Tool name mapping
        provided_tool_names = tool_names or [f"tool_{i}" for i in range(self.num_tools)]
        self.tool_names = list(provided_tool_names[:self.num_tools])
        self.tool_name_to_id = {name: i for i, name in enumerate(self.tool_names)}
        self.tool_id_to_name = {i: name for i, name in enumerate(self.tool_names)}
        
        # Tool token mappings (using reserved tokens)
        self.tool_id_to_token_id = {i: self.tool_reserved_token_ids[i] for i in range(self.num_tools)}
        self.token_id_to_tool_id = {self.tool_reserved_token_ids[i]: i for i in range(self.num_tools)}
        
        # Override the model's forward method to use our custom embeddings and logits
        self._setup_model_override()
        
        # Print parameter breakdown after everything is initialized
        self._print_parameter_breakdown()

    @property
    def trainable_tool_input_embeddings(self):
        return self.tool_token_module.input_embeddings

    @property
    def trainable_tool_output_embeddings(self):
        return self.tool_token_module.output_embeddings

    @property
    def trainable_tool_embeddings(self):
        return self.tool_token_module.input_embeddings

    @property
    def logit_bias_head(self):
        return self.tool_token_module.logit_bias_head

    def _get_backbone_model(self):
        return self.model.get_base_model() if self.lora_config else self.model

    def _get_embed_module(self):
        return self._get_backbone_model().model.embed_tokens
    
    def _setup_model_override(self):
        """Keep initialization flow stable; FSDP-safe custom logic runs in this wrapper."""
        self.original_embed_forward = self._get_embed_module().forward
        self.original_lm_head_forward = self._get_lm_head_module().forward
    
    def restore_original_model(self):
        """No-op: the FSDP-safe path does not override submodule forwards."""
        return

    def _setup_reserved_tokens(self):
        """Extract reserved special token IDs from tokenizer"""
        vocab = self.tokenizer.get_vocab()
        reserved_tokens = {k: v for k, v in vocab.items() if 'reserved_special_token_' in k}
        
        # Sort by token ID to get consistent ordering
        sorted_reserved = sorted(reserved_tokens.items(), key=lambda x: x[1])
        if len(sorted_reserved) < self.num_reserved_slots:
            raise ValueError(
                f"Tokenizer only exposes {len(sorted_reserved)} reserved_special_token_* entries, "
                f"but {self.num_reserved_slots} are required"
            )

        # Take the first num_tools reserved tokens as tool tokens
        self.tool_reserved_token_names = [token for token, _ in sorted_reserved[:self.num_tools]]
        self.tool_reserved_token_ids = [token_id for _, token_id in sorted_reserved[:self.num_tools]]

        # Keep backward-compatible tool-only aliases
        self.reserved_token_names = list(self.tool_reserved_token_names)
        self.reserved_token_ids = list(self.tool_reserved_token_ids)

        # Optional EOC token occupies the next reserved slot
        if self.use_eoc:
            self.eoc_token_name = sorted_reserved[self.num_tools][0]
            self.eoc_token_id = sorted_reserved[self.num_tools][1]
        else:
            self.eoc_token_name = None
            self.eoc_token_id = None

        self.trainable_reserved_token_names = list(self.tool_reserved_token_names)
        self.trainable_reserved_token_ids = list(self.tool_reserved_token_ids)
        if self.use_eoc:
            self.trainable_reserved_token_names.append(self.eoc_token_name)
            self.trainable_reserved_token_ids.append(self.eoc_token_id)
        
        print(f"Using {len(self.tool_reserved_token_ids)} reserved tokens as tool tokens:")
        for name, token_id in zip(self.tool_reserved_token_names[:5], self.tool_reserved_token_ids[:5]):
            print(f"  {name}: {token_id}")
        if len(self.tool_reserved_token_ids) > 5:
            print(f"  ... and {len(self.tool_reserved_token_ids) - 5} more")
        if self.use_eoc:
            print(f"Using reserved token as eoc: {self.eoc_token_name} -> {self.eoc_token_id}")
        print(f"Trainable reserved slots: {self.num_reserved_slots}")
        print(f"Embedding coupling mode: {'Decoupled' if self.decouple_embeddings else 'Coupled'}")
    
    def _print_parameter_breakdown(self):
        """Print breakdown of trainable parameters"""
        # Count tokenized memory parameters
        if self.decouple_embeddings:
            token_params = self.trainable_tool_input_embeddings.numel() + self.trainable_tool_output_embeddings.numel()
            print(f"Tokenized memory parameters: {token_params:,} (input: {self.trainable_tool_input_embeddings.numel():,}, output: {self.trainable_tool_output_embeddings.numel():,})")
        else:
            token_params = self.trainable_tool_embeddings.numel()
            print(f"Tokenized memory parameters: {token_params:,} (shared)")

        logit_bias_params = 0
        if self.logit_bias_head is not None:
            logit_bias_params = sum(p.numel() for p in self.logit_bias_head.parameters())
            print(f"Logit bias head parameters: {logit_bias_params:,}")

        # Count LoRA parameters if using LoRA
        if self.lora_config:
            lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"LoRA parameters: {lora_params:,}")
            print(
                f"Total trainable: {token_params + logit_bias_params + lora_params:,} "
                f"({(token_params + logit_bias_params + lora_params)/total_params*100:.4f}%)"
            )
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Total trainable: {token_params + logit_bias_params:,} "
                f"({(token_params + logit_bias_params)/total_params*100:.4f}%)"
            )
    
    def forward(self, input_ids, attention_mask, return_hidden_states=False):
        """
        Forward pass for function calling training
        
        Sequence: [User] [Reserved_Tool_Token] [Function_Call] <|eot_id|>
        """
        outputs, final_hidden_states, hidden_states = self._forward_with_captured_final_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            capture_all_hidden_states=return_hidden_states,
        )
        if return_hidden_states:
            if hidden_states is not None:
                return outputs.logits, hidden_states[-1]
            return outputs.logits, final_hidden_states
        return outputs.logits

    def get_eoc_token_id(self):
        """Get the reserved token ID for eoc when enabled."""
        return self.eoc_token_id

    def is_tool_token_id(self, token_id):
        """Check whether a token ID belongs to the tool token set."""
        return token_id in self.token_id_to_tool_id

    def mask_logits_to_tool_tokens(self, logits):
        """Mask logits so only tool token IDs remain valid."""
        if not self.tool_reserved_token_ids:
            return logits
        tool_token_ids = torch.tensor(self.tool_reserved_token_ids, device=logits.device)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits[..., tool_token_ids] = logits[..., tool_token_ids]
        return masked_logits

    def _get_logit_bias_scores(self, hidden_states):
        """Project boundary hidden states to tool-selection logits."""
        if self.logit_bias_head is None:
            return None
        return self.logit_bias_head(hidden_states)

    def _apply_logit_bias_to_logits(self, logits, hidden_states, active_decision_rows):
        """Add soft tool priors to the active boundary rows of the full-vocab logits."""
        if self.logit_bias_head is None or not active_decision_rows.any():
            return logits
        if hidden_states is None:
            raise ValueError("Boundary hidden states are required when use_logit_bias is enabled")

        active_indices = active_decision_rows.nonzero(as_tuple=False).squeeze(-1)
        tool_logits = self._get_logit_bias_scores(hidden_states[active_indices])
        tool_log_probs = torch.log_softmax(tool_logits.float(), dim=-1)
        uniform_tool_log_prob = math.log(max(1, len(self.tool_reserved_token_ids)))
        tool_bias = (tool_log_probs + uniform_tool_log_prob) * self.logit_bias_scale
        tool_bias = tool_bias.to(dtype=logits.dtype)

        tool_token_ids = torch.tensor(self.tool_reserved_token_ids, device=logits.device)
        logits = logits.clone()
        logits[active_indices[:, None], tool_token_ids[None, :]] += tool_bias
        return logits

    def _get_js_core_model(self):
        """Return the underlying causal LM used for JS-style logit lens scoring."""
        return self._get_backbone_model()

    def _get_final_norm_module(self):
        """Locate the model's final norm module."""
        core_model = self._get_js_core_model()
        if hasattr(core_model, "model") and hasattr(core_model.model, "norm"):
            return core_model.model.norm
        return None

    def _get_lm_head_module(self):
        """Locate the model's lm_head or output embeddings."""
        core_model = self._get_js_core_model()
        if hasattr(core_model, "lm_head"):
            return core_model.lm_head
        if hasattr(core_model, "get_output_embeddings"):
            output_embeddings = core_model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings
        raise ValueError("Unable to locate lm_head / output embeddings on the model")

    def _compute_input_embeddings(self, input_ids):
        """Build input embeddings with token-memory rows swapped in for reserved tool tokens."""
        embeddings = self._get_embed_module()(input_ids)
        reserved_token_ids = torch.as_tensor(
            self.trainable_reserved_token_ids,
            device=input_ids.device,
        )
        flattened_input_ids = input_ids.reshape(-1)
        reserved_matches = (flattened_input_ids[:, None] == reserved_token_ids[None, :]).nonzero(as_tuple=False)
        if reserved_matches.numel() > 0:
            flattened_embeddings = embeddings.reshape(-1, embeddings.size(-1))
            replacement_rows = self.trainable_tool_input_embeddings[
                reserved_matches[:, 1]
            ].to(
                device=flattened_embeddings.device,
                dtype=flattened_embeddings.dtype,
            )
            flattened_embeddings[reserved_matches[:, 0]] = replacement_rows
        return embeddings

    def _compute_logits_from_hidden_states(self, hidden_states):
        """Project hidden states to vocab logits and overwrite tool-token columns."""
        logits = self._get_lm_head_module()(hidden_states)
        reserved_token_ids = torch.as_tensor(
            self.trainable_reserved_token_ids,
            device=logits.device,
        )
        reserved_token_logits = torch.matmul(
            hidden_states,
            self.trainable_tool_output_embeddings.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).transpose(0, 1),
        )
        logits = logits.clone()
        logits[..., reserved_token_ids] = reserved_token_logits
        return logits

    def _prepare_js_layer_hidden_states(self, hidden_states, final_norm_module):
        """Stack per-layer last-token states with js_explore's final-norm semantics."""
        if len(hidden_states) < 2:
            raise ValueError("Expected hidden_states to include embeddings and layer outputs")

        layer_outputs = list(hidden_states[1:])
        if final_norm_module is None or len(layer_outputs) <= 1:
            return torch.stack([tensor[:, -1, :] for tensor in layer_outputs], dim=0)

        intermediate_layers = layer_outputs[:-1]
        final_layer = layer_outputs[-1][:, -1, :].unsqueeze(0)
        if intermediate_layers:
            intermediate_tensor = torch.stack([tensor[:, -1, :] for tensor in intermediate_layers], dim=0)
            intermediate_tensor = final_norm_module(intermediate_tensor)
            return torch.cat([intermediate_tensor, final_layer], dim=0)
        return final_layer

    def _compute_js_curve_from_hidden_states(self, hidden_states):
        """Compute the layer-to-final JS curve for the last generated position."""
        final_norm_module = self._get_final_norm_module()
        layer_hidden_states = self._prepare_js_layer_hidden_states(hidden_states, final_norm_module)
        logits_by_layer = self._compute_logits_from_hidden_states(layer_hidden_states)
        return compute_js_divergence_against_final(logits_by_layer)

    def _compute_js_mean_from_hidden_states(self, hidden_states):
        """Compute the per-example mean JS divergence against the final layer."""
        js_curve = self._compute_js_curve_from_hidden_states(hidden_states)
        return js_curve.mean(dim=0)

    def _sample_next_tokens(self, logits, temperature=0.6, top_p=0.9, do_sample=False):
        """Sample or decode greedily from a batch of logits."""
        if do_sample:
            if temperature <= 0:
                raise ValueError("temperature must be positive when sampling")
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                logits = logits.clone()
                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        return torch.argmax(logits, dim=-1)

    def _forward_with_captured_final_hidden_states(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=False,
        capture_all_hidden_states=False,
    ):
        """Run the LM and capture final-layer hidden states without returning every layer unless requested."""
        inputs_embeds = self._compute_input_embeddings(input_ids)
        if capture_all_hidden_states:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
            )
            final_hidden_states = outputs.hidden_states[-1]
            outputs.logits = self._compute_logits_from_hidden_states(final_hidden_states)
            return outputs, final_hidden_states, outputs.hidden_states

        lm_head_module = self._get_lm_head_module()
        captured = {}

        def _capture_hidden_states(module, args):
            if args:
                captured["final_hidden_states"] = args[0]

        handle = lm_head_module.register_forward_pre_hook(_capture_hidden_states)
        try:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        final_hidden_states = captured.get("final_hidden_states")
        outputs.logits = self._compute_logits_from_hidden_states(final_hidden_states)
        return outputs, final_hidden_states, None

    def forward_with_final_hidden_states(self, input_ids, attention_mask):
        """Return logits plus final-layer hidden states without materializing all transformer layers."""
        outputs, final_hidden_states, _ = self._forward_with_captured_final_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            capture_all_hidden_states=False,
        )
        return outputs.logits, final_hidden_states

    def _generation_forward_step(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        return_last_hidden_state=False,
        return_all_hidden_states=False,
    ):
        """Run one cached generation step and return only the newest position outputs."""
        outputs, final_hidden_states, hidden_states = self._forward_with_captured_final_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            capture_all_hidden_states=return_all_hidden_states,
        )

        next_logits = outputs.logits[:, -1, :]
        last_hidden_states = None
        if return_all_hidden_states or return_last_hidden_state:
            if final_hidden_states is None:
                raise ValueError("Final hidden states were requested but not captured")
            last_hidden_states = final_hidden_states[:, -1, :]

        return next_logits, last_hidden_states, outputs.past_key_values, hidden_states

    def _build_decision_context(self, input_ids, batch_size, device, step, use_eoc=True):
        """Return rows whose current token should use routing-probe logic."""
        decision_context = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if step == 0:
            decision_context[:] = True
        elif use_eoc and self.eoc_token_id is not None:
            decision_context = input_ids[:, -1] == self.eoc_token_id
        return decision_context

    def generate_with_tool_prediction(
        self,
        user_tokens,
        user_mask,
        tokenizer,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        do_sample=False,
        use_logit_bias=None,
    ):
        """Generate complete sequence: predict tool, then generate function call"""
        self.eval()
        resolved_use_logit_bias = self.use_logit_bias if use_logit_bias is None else use_logit_bias
        
        with torch.no_grad():
            batch_size = user_tokens.shape[0]
            device = user_tokens.device
            input_ids = user_tokens.clone()
            attention_mask = user_mask.clone()
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            step_input_ids = user_tokens
            past_key_values = None

            for step in range(max_new_tokens):
                decision_context = self._build_decision_context(
                    input_ids=input_ids,
                    batch_size=batch_size,
                    device=device,
                    step=step,
                )
                active_decision_rows = decision_context & ~finished
                need_all_hidden_states = bool(self.use_js_trunc and active_decision_rows.any())
                need_last_hidden_state = bool(
                    resolved_use_logit_bias and active_decision_rows.any()
                )
                next_logits, last_hidden_states, past_key_values, hidden_states = self._generation_forward_step(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_last_hidden_state=need_last_hidden_state,
                    return_all_hidden_states=need_all_hidden_states,
                )
                next_logits = next_logits.clone()

                selection_logits = next_logits
                if resolved_use_logit_bias and active_decision_rows.any():
                    selection_logits = self._apply_logit_bias_to_logits(
                        logits=next_logits,
                        hidden_states=last_hidden_states,
                        active_decision_rows=active_decision_rows,
                    )
                if self.use_js_trunc:
                    if active_decision_rows.any():
                        active_indices = active_decision_rows.nonzero(as_tuple=False).squeeze(-1)
                        active_hidden_states = tuple(layer_hidden[active_indices] for layer_hidden in hidden_states)
                        js_mean = self._compute_js_mean_from_hidden_states(active_hidden_states)
                        js_positive = js_mean > JS_TRUNCATION_THRESHOLD
                        if js_positive.any():
                            positive_indices = active_indices[js_positive]
                            selection_logits[positive_indices] = self.mask_logits_to_tool_tokens(selection_logits[positive_indices])
                next_tokens = self._sample_next_tokens(
                    selection_logits,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                next_tokens = next_tokens.masked_fill(finished, tokenizer.eos_token_id)

                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device)], dim=-1)
                finished = finished | (next_tokens == tokenizer.eos_token_id)
                if finished.all():
                    break
                step_input_ids = next_tokens.unsqueeze(-1)
            
            # Parse the generated sequences
            return self._parse_generated_sequences(input_ids, user_tokens, tokenizer)

    def generate_with_ground_truth_tools(
        self,
        user_tokens,
        user_mask,
        tokenizer,
        ground_truth_tools,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        do_sample=False,
        use_js_trunc=None,
        use_eoc=None,
        use_logit_bias=None,
    ):
        """Generate sequence where predicted tool tokens are replaced with ground truth tool tokens"""
        self.eval()

        resolved_use_eoc = self.use_eoc if use_eoc is None else use_eoc
        resolved_use_js_trunc = self.use_js_trunc if use_js_trunc is None else use_js_trunc
        resolved_use_logit_bias = self.use_logit_bias if use_logit_bias is None else use_logit_bias
        
        # Convert ground truth tool names to token IDs
        gt_tool_token_ids = []
        for tool_name in ground_truth_tools:
            if tool_name in self.tool_name_to_id:
                tool_id = self.tool_name_to_id[tool_name]
                token_id = self.tool_id_to_token_id[tool_id]
                gt_tool_token_ids.append(token_id)
            else:
                print(f"Warning: Unknown tool name '{tool_name}', skipping")
        
        with torch.no_grad():
            batch_size = user_tokens.shape[0]
            device = user_tokens.device
            
            # Initialize generation state
            input_ids = user_tokens.clone()
            attention_mask = user_mask.clone()
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            step_input_ids = user_tokens
            past_key_values = None
            tool_replacement_count = [0] * batch_size  # Track how many tools replaced per example
            
            for step in range(max_new_tokens):
                decision_context = self._build_decision_context(
                    input_ids=input_ids,
                    batch_size=batch_size,
                    device=device,
                    step=step,
                    use_eoc=resolved_use_eoc,
                )
                active_decision_rows = decision_context & ~finished
                force_ground_truth_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)
                for i in range(batch_size):
                    if active_decision_rows[i] and tool_replacement_count[i] < len(gt_tool_token_ids):
                        force_ground_truth_rows[i] = True

                analysis_rows = active_decision_rows & ~force_ground_truth_rows
                need_all_hidden_states = bool(resolved_use_js_trunc and analysis_rows.any())
                need_last_hidden_state = bool(
                    resolved_use_logit_bias and analysis_rows.any()
                )
                logits, last_hidden_states, past_key_values, hidden_states = self._generation_forward_step(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_last_hidden_state=need_last_hidden_state,
                    return_all_hidden_states=need_all_hidden_states,
                )
                logits = logits.clone()

                next_tokens = None
                selection_logits = logits
                if resolved_use_logit_bias and analysis_rows.any():
                    selection_logits = self._apply_logit_bias_to_logits(
                        logits=logits,
                        hidden_states=last_hidden_states,
                        active_decision_rows=analysis_rows,
                    )
                if resolved_use_js_trunc and analysis_rows.any():
                    active_indices = analysis_rows.nonzero(as_tuple=False).squeeze(-1)
                    active_hidden_states = tuple(layer_hidden[active_indices] for layer_hidden in hidden_states)
                    js_mean = self._compute_js_mean_from_hidden_states(active_hidden_states)
                    js_positive = js_mean > JS_TRUNCATION_THRESHOLD
                    force_ground_truth_rows[active_indices] = js_positive
                
                # Sample/select next tokens
                if next_tokens is None:
                    next_tokens = self._sample_next_tokens(
                        selection_logits,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )

                for i in range(batch_size):
                    if force_ground_truth_rows[i] and tool_replacement_count[i] < len(gt_tool_token_ids):
                        next_tokens[i] = gt_tool_token_ids[tool_replacement_count[i]]
                        tool_replacement_count[i] += 1
                
                # Check if any predicted tokens are tool tokens and replace them
                for i in range(batch_size):
                    if force_ground_truth_rows[i]:
                        continue
                    predicted_token = next_tokens[i].item()
                    
                    # If predicted token is a tool token and we have ground truth tools left
                    if (predicted_token in self.token_id_to_tool_id and 
                        tool_replacement_count[i] < len(gt_tool_token_ids)):
                        # Replace with ground truth tool token
                        next_tokens[i] = gt_tool_token_ids[tool_replacement_count[i]]
                        tool_replacement_count[i] += 1

                next_tokens = next_tokens.masked_fill(finished, tokenizer.eos_token_id)
                finished = finished | (next_tokens == tokenizer.eos_token_id)
                
                # Check for EOS tokens
                if finished.all():
                    break
                
                # Append new tokens to input_ids and update attention_mask
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device)], dim=-1)
                step_input_ids = next_tokens.unsqueeze(-1)
            
            # Parse the generated sequences
            return self._parse_generated_sequences(input_ids, user_tokens, tokenizer)
    
    def _parse_generated_sequences(self, generated_tokens, input_tokens, tokenizer):
        """Parse generated sequences to extract multiple tool predictions and function calls"""
        results = []
        batch_size = generated_tokens.shape[0]
        input_length = input_tokens.shape[1]
        
        for i in range(batch_size):
            # Extract only the newly generated part
            generated_seq = generated_tokens[i, input_length:]
            
            # Remove EOS tokens and convert to list
            valid_tokens = []
            for token_id in generated_seq:
                if token_id.item() == tokenizer.eos_token_id:
                    break
                valid_tokens.append(token_id.item())

            if self.use_eoc and self.eoc_token_id is not None:
                display_tokens = [token_id for token_id in valid_tokens if token_id != self.eoc_token_id]
            else:
                display_tokens = list(valid_tokens)

            if not display_tokens:
                results.append({
                    'predicted_tools': [],
                    'function_calls': [],
                    'full_generated_sequence': '',
                    # Backward compatibility
                    'predicted_tool_id': None,
                    'predicted_tool_name': 'none',
                    'function_call': ''
                })
                continue
            
            # Find all tool tokens and their positions
            tool_positions = []
            for j, token_id in enumerate(valid_tokens):
                if self.is_tool_token_id(token_id):
                    tool_id = self.token_id_to_tool_id[token_id]
                    tool_name = self.tool_id_to_name[tool_id]
                    tool_positions.append({
                        'position': j,
                        'token_id': token_id,
                        'tool_id': tool_id,
                        'tool_name': tool_name
                    })
            
            predicted_tools = []
            function_calls = []
            
            # Extract function call for each tool
            for idx, tool_info in enumerate(tool_positions):
                start_pos = tool_info['position'] + 1  # Position after tool token
                
                # Find end position. Prefer eoc when enabled, then fall back to the next tool or end of sequence.
                end_pos = len(valid_tokens)
                if self.use_eoc and self.eoc_token_id is not None:
                    for j in range(start_pos, len(valid_tokens)):
                        if valid_tokens[j] == self.eoc_token_id:
                            end_pos = j
                            break
                if end_pos == len(valid_tokens) and idx + 1 < len(tool_positions):
                    end_pos = tool_positions[idx + 1]['position']
                if end_pos == len(valid_tokens):
                    # Last tool - function call goes to end of sequence (excluding EOT if present)
                    eot_tokens = tokenizer('<|eot_id|>', add_special_tokens=False)['input_ids']
                    if len(eot_tokens) > 0 and end_pos >= len(eot_tokens):
                        if valid_tokens[-len(eot_tokens):] == eot_tokens:
                            end_pos -= len(eot_tokens)
                
                # Extract function call tokens
                if start_pos < end_pos:
                    function_call_tokens = valid_tokens[start_pos:end_pos]
                    function_call = tokenizer.decode(function_call_tokens, skip_special_tokens=True).strip()
                else:
                    function_call = ""
                
                predicted_tools.append({
                    'tool_id': tool_info['tool_id'],
                    'tool_name': tool_info['tool_name'],
                    'token_id': tool_info['token_id']
                })
                function_calls.append(function_call)
            
            # If no tool tokens found, treat entire sequence as function call
            if not tool_positions:
                function_call = tokenizer.decode(display_tokens, skip_special_tokens=True).strip()
                predicted_tools = []
                function_calls = [function_call] if function_call else []
            
            # Prepare result with both new multi-tool format and backward compatibility
            result = {
                'predicted_tools': predicted_tools,
                'function_calls': function_calls,
                'full_generated_sequence': tokenizer.decode(display_tokens, skip_special_tokens=True),
                # Backward compatibility - use first tool if available
                'predicted_tool_id': predicted_tools[0]['tool_id'] if predicted_tools else None,
                'predicted_tool_name': predicted_tools[0]['tool_name'] if predicted_tools else 'none',
                'function_call': function_calls[0] if function_calls else '',
                'tool_token_used': tokenizer.decode([predicted_tools[0]['token_id']]) if predicted_tools else None
            }
            
            results.append(result)
        
        return results
    
    def parse_function_call(self, function_call_text):
        """Parse generated function call text into structured format"""
        try:
            if '{' in function_call_text and '}' in function_call_text:
                start = function_call_text.find('{')
                end = function_call_text.rfind('}') + 1
                json_str = function_call_text[start:end]
                return json.loads(json_str)
            else:
                return {'raw_text': function_call_text}
        except json.JSONDecodeError:
            return {'raw_text': function_call_text, 'parse_error': True}
    
    def get_tool_token_id(self, tool_name):
        """Get the reserved token ID for a tool name"""
        if tool_name in self.tool_name_to_id:
            tool_id = self.tool_name_to_id[tool_name]
            return self.tool_id_to_token_id[tool_id]
        return None
    
    def get_tool_name_from_token_id(self, token_id):
        """Get tool name from reserved token ID"""
        if token_id in self.token_id_to_tool_id:
            tool_id = self.token_id_to_tool_id[token_id]
            return self.tool_id_to_name[tool_id]
        return None

    def get_fsdp_trainable_modules(self):
        """Return small helper modules that should stay replicated under FSDP."""
        return [self.tool_token_module]

    def build_trainable_state_payload(self):
        """Build a compact checkpoint payload for token-memory state and optional LoRA adapters."""
        payload = {
            "tool_names": self.tool_names,
            "num_tools": self.num_tools,
            "decouple_embeddings": self.decouple_embeddings,
            "tool_name_to_id": self.tool_name_to_id,
            "reserved_token_ids": self.reserved_token_ids,
            "tool_reserved_token_ids": self.tool_reserved_token_ids,
            "trainable_reserved_token_ids": self.trainable_reserved_token_ids,
            "use_eoc": self.use_eoc,
            "eoc_token_id": self.eoc_token_id,
            "use_logit_bias": self.use_logit_bias,
            "logit_bias_network": self.logit_bias_network,
            "logit_bias_scale": self.logit_bias_scale,
            "lora_config": self.lora_config,
        }

        if self.decouple_embeddings:
            payload["input_embeddings"] = self.trainable_tool_input_embeddings.detach().cpu().clone()
            payload["output_embeddings"] = self.trainable_tool_output_embeddings.detach().cpu().clone()
        else:
            payload["embeddings"] = self.trainable_tool_embeddings.detach().cpu().clone()

        if self.logit_bias_head is not None:
            payload["logit_bias_head"] = {
                key: value.detach().cpu().clone()
                for key, value in self.logit_bias_head.state_dict().items()
            }

        if self.lora_config:
            payload["lora_state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(self.model).items()
            }

        return payload

    def save_trainable_state(self, filepath):
        """Save token-memory state and optional LoRA adapter weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.build_trainable_state_payload(), filepath)
        if is_rank_zero():
            print(f"Saved trainable state to {filepath}")

    def load_trainable_state(self, filepath):
        """Load token-memory state and optional LoRA adapter weights."""
        payload = torch.load(filepath, map_location="cpu")

        if payload["tool_names"] != self.tool_names:
            raise ValueError("Tool names do not match the saved trainable state")
        if payload["decouple_embeddings"] != self.decouple_embeddings:
            raise ValueError("Embedding coupling mode does not match the saved trainable state")

        if self.decouple_embeddings:
            self.trainable_tool_input_embeddings.data.copy_(
                payload["input_embeddings"].to(self.trainable_tool_input_embeddings.device)
            )
            self.trainable_tool_output_embeddings.data.copy_(
                payload["output_embeddings"].to(self.trainable_tool_output_embeddings.device)
            )
        else:
            self.trainable_tool_embeddings.data.copy_(
                payload["embeddings"].to(self.trainable_tool_embeddings.device)
            )

        saved_logit_bias_head = payload.get("logit_bias_head")
        if self.use_logit_bias and self.logit_bias_head is not None and saved_logit_bias_head is not None:
            self.logit_bias_head.load_state_dict(saved_logit_bias_head)

        saved_lora_state = payload.get("lora_state_dict")
        if self.lora_config and saved_lora_state is not None:
            set_peft_model_state_dict(self.model, saved_lora_state)
    
    def get_trainable_parameters(self, separate_lora=False):
        """Get trainable parameters for optimizer initialization
        
        Args:
            separate_lora: If True, return separate groups for embedding and LoRA params
        
        Returns:
            If separate_lora=False: List of all parameters
            If separate_lora=True: Tuple of (embedding_params, lora_params)
        """
        # Get tokenized memory parameters
        embedding_params = []
        if self.decouple_embeddings:
            embedding_params.extend([self.trainable_tool_input_embeddings, self.trainable_tool_output_embeddings])
        else:
            embedding_params.append(self.trainable_tool_embeddings)

        if self.logit_bias_head is not None:
            embedding_params.extend(list(self.logit_bias_head.parameters()))
        
        # Get LoRA parameters if using LoRA
        lora_params = []
        if self.lora_config:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]

        if separate_lora and self.lora_config:
            return embedding_params, lora_params
        else:
            # Return all parameters combined (backward compatibility)
            all_params = embedding_params + lora_params
            return all_params

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoints while dropping legacy gate/toolmix probe weights."""
        filtered_state_dict = dict(state_dict)
        legacy_prefixes = ("routing_probe.", "toolmix_head.", "gate_mlp.")
        for key in list(filtered_state_dict.keys()):
            if key.startswith(legacy_prefixes):
                del filtered_state_dict[key]
        return super().load_state_dict(filtered_state_dict, strict=strict)
    
 
