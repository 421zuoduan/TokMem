import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json


def build_routing_probe(hidden_size, gate_network):
    """Build the configurable routing probe used for boundary decisions."""
    if gate_network == "linear":
        return nn.Linear(hidden_size, 1)
    if gate_network == "mlp":
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
    raise ValueError(f"Unsupported gate_network: {gate_network}")

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
                 decouple_embeddings=False, lora_config=None, use_eoc=False, use_gate=False,
                 gate_threshold=0.5, gate_network="mlp", use_toolmix=False,
                 enable_routing_probe=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.use_eoc = use_eoc
        self.use_gate = use_gate
        self.use_toolmix = use_toolmix
        self.enable_routing_probe = (
            bool(enable_routing_probe)
            if enable_routing_probe is not None
            else (self.use_gate or self.use_toolmix)
        )
        self.gate_threshold = gate_threshold
        self.gate_network = gate_network
        self.toolmix_alpha = None
        if self.use_gate and not self.use_eoc:
            raise ValueError("--use_gate requires --use_eoc")
        if self.use_toolmix and not self.use_eoc:
            raise ValueError("--use_toolmix requires --use_eoc")
        if self.enable_routing_probe and not self.use_eoc:
            raise ValueError("--enable_routing_probe requires --use_eoc")

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
        
        # Create trainable parameters for tool tokens (coupled or decoupled)
        # Clone the embeddings for the trainable reserved tokens and make them parameters
        reserved_token_indices = torch.tensor(self.trainable_reserved_token_ids, device=device)
        
        if self.decouple_embeddings:
            # Separate parameters for input and output layers
            self.trainable_tool_input_embeddings = nn.Parameter(
                original_embeddings[reserved_token_indices].clone()
            )
            self.trainable_tool_output_embeddings = nn.Parameter(
                original_embeddings[reserved_token_indices].clone()
            )
        else:
            # Shared parameter for both input and output layers
            self.trainable_tool_embeddings = nn.Parameter(
                original_embeddings[reserved_token_indices].clone()
            )
            # Create aliases for backward compatibility
            self.trainable_tool_input_embeddings = self.trainable_tool_embeddings
            self.trainable_tool_output_embeddings = self.trainable_tool_embeddings

        if self.enable_routing_probe:
            self.routing_probe = build_routing_probe(self.config.hidden_size, self.gate_network).to(
                device=device,
                dtype=original_embeddings.dtype,
            )
        else:
            self.routing_probe = None
        
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
    
    def _setup_model_override(self):
        """Set up model override to make embedding/logit modifications transparent"""
        # Store original methods (handle LoRA vs non-LoRA)
        if self.lora_config:
            # When LoRA is applied, access base model methods
            base_model = self.model.get_base_model()
            self.original_embed_forward = base_model.model.embed_tokens.forward
            self.original_lm_head_forward = base_model.lm_head.forward
        else:
            # Direct access for non-LoRA model
            self.original_embed_forward = self.model.model.embed_tokens.forward
            self.original_lm_head_forward = self.model.lm_head.forward
        
        # Override embedding layer
        def custom_embed_forward(input_ids):
            # Get standard embeddings
            embeddings = self.original_embed_forward(input_ids)
            
            # Replace reserved token embeddings with trainable input embeddings
            for i, reserved_token_id in enumerate(self.trainable_reserved_token_ids):
                mask = (input_ids == reserved_token_id)
                if mask.any():
                    embeddings[mask] = self.trainable_tool_input_embeddings[i]
            
            return embeddings
        
        # Override lm_head
        def custom_lm_head_forward(hidden_states):
            # Get standard logits
            logits = self.original_lm_head_forward(hidden_states)
            
            # Replace reserved token logits with trainable output projections
            for i, reserved_token_id in enumerate(self.trainable_reserved_token_ids):
                tool_logits = torch.matmul(hidden_states, self.trainable_tool_output_embeddings[i])
                logits[..., reserved_token_id] = tool_logits
            
            return logits
        
        # Apply overrides (handle LoRA vs non-LoRA)
        if self.lora_config:
            # When LoRA is applied, override base model methods
            base_model = self.model.get_base_model()
            base_model.model.embed_tokens.forward = custom_embed_forward
            base_model.lm_head.forward = custom_lm_head_forward
        else:
            # Direct override for non-LoRA model
            self.model.model.embed_tokens.forward = custom_embed_forward
            self.model.lm_head.forward = custom_lm_head_forward
    
    def restore_original_model(self):
        """Restore the original model methods (useful for cleanup or debugging)"""
        if hasattr(self, 'original_embed_forward'):
            if self.lora_config:
                base_model = self.model.get_base_model()
                base_model.model.embed_tokens.forward = self.original_embed_forward
            else:
                self.model.model.embed_tokens.forward = self.original_embed_forward
        if hasattr(self, 'original_lm_head_forward'):
            if self.lora_config:
                base_model = self.model.get_base_model()
                base_model.lm_head.forward = self.original_lm_head_forward
            else:
                self.model.lm_head.forward = self.original_lm_head_forward

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

        routing_probe_params = 0
        if self.routing_probe is not None:
            routing_probe_params = sum(p.numel() for p in self.routing_probe.parameters())
            print(f"Routing probe parameters: {routing_probe_params:,}")

        # Count LoRA parameters if using LoRA
        if self.lora_config:
            lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"LoRA parameters: {lora_params:,}")
            print(
                f"Total trainable: {token_params + routing_probe_params + lora_params:,} "
                f"({(token_params + routing_probe_params + lora_params)/total_params*100:.4f}%)"
            )
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Total trainable: {token_params + routing_probe_params:,} "
                f"({(token_params + routing_probe_params)/total_params*100:.4f}%)"
            )
    
    def forward(self, input_ids, attention_mask, return_hidden_states=False):
        """
        Forward pass for function calling training
        
        Sequence: [User] [Reserved_Tool_Token] [Function_Call] <|eot_id|>
        """
        # Use the model's forward method directly (our overrides handle the custom embeddings/logits)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
            return_dict=True
        )
        if return_hidden_states:
            return outputs.logits, outputs.hidden_states[-1]
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

    def _get_routing_probe_scores(self, hidden_states):
        """Project hidden states to scalar routing-probe logits."""
        if self.routing_probe is None:
            return None
        return self.routing_probe(hidden_states).squeeze(-1)

    def _get_gate_scores(self, hidden_states):
        """Backward-compatible wrapper for older analysis utilities."""
        return self._get_routing_probe_scores(hidden_states)

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

    def _generation_forward_step(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        return_hidden_states=False,
    ):
        """Run one cached generation step and return only the newest position outputs."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )

        next_logits = outputs.logits[:, -1, :]
        last_hidden_states = None
        if return_hidden_states:
            last_hidden_states = outputs.hidden_states[-1][:, -1, :]

        return next_logits, last_hidden_states, outputs.past_key_values

    
    def generate_with_tool_prediction(self, user_tokens, user_mask, tokenizer, 
                                     max_new_tokens=256, temperature=0.6, top_p=0.9, do_sample=False):
        """Generate complete sequence: predict tool, then generate function call"""
        self.eval()
        
        with torch.no_grad():
            if not self.use_gate:
                # Use native generation - our overrides make the custom embeddings/logits transparent
                generated = self.model.generate(
                    input_ids=user_tokens,
                    attention_mask=user_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                return self._parse_generated_sequences(generated, user_tokens, tokenizer)

            batch_size = user_tokens.shape[0]
            device = user_tokens.device
            input_ids = user_tokens.clone()
            attention_mask = user_mask.clone()
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            step_input_ids = user_tokens
            past_key_values = None

            for step in range(max_new_tokens):
                next_logits, last_hidden_states, past_key_values = self._generation_forward_step(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_hidden_states=True
                )
                next_logits = next_logits.clone()

                gate_context = torch.zeros(batch_size, dtype=torch.bool, device=device)
                if step == 0:
                    gate_context[:] = True
                else:
                    gate_context = input_ids[:, -1] == self.eoc_token_id

                active_gate_rows = gate_context & ~finished
                if active_gate_rows.any():
                    gate_scores = self._get_routing_probe_scores(last_hidden_states[active_gate_rows])
                    gate_positive = torch.sigmoid(gate_scores) >= self.gate_threshold
                    if gate_positive.any():
                        active_indices = active_gate_rows.nonzero(as_tuple=False).squeeze(-1)
                        positive_indices = active_indices[gate_positive]
                        next_logits[positive_indices] = self.mask_logits_to_tool_tokens(next_logits[positive_indices])

                next_tokens = self._sample_next_tokens(
                    next_logits,
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

    def generate_with_ground_truth_tools(self, user_tokens, user_mask, tokenizer, ground_truth_tools,
                                       max_new_tokens=256, temperature=0.6, top_p=0.9, do_sample=False):
        """Generate sequence where predicted tool tokens are replaced with ground truth tool tokens"""
        self.eval()
        
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
            tool_replacement_count = [0] * batch_size  # Track how many tools replaced per example
            
            for step in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Get logits for last position
                
                # Sample/select next tokens
                if do_sample:
                    # Apply temperature
                    logits = logits / temperature
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            logits[i][indices_to_remove] = -float('inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                
                # Check if any predicted tokens are tool tokens and replace them
                for i in range(batch_size):
                    predicted_token = next_tokens[i].item()
                    
                    # If predicted token is a tool token and we have ground truth tools left
                    if (predicted_token in self.token_id_to_tool_id and 
                        tool_replacement_count[i] < len(gt_tool_token_ids)):
                        # Replace with ground truth tool token
                        next_tokens[i] = gt_tool_token_ids[tool_replacement_count[i]]
                        tool_replacement_count[i] += 1
                
                # Check for EOS tokens
                if (next_tokens == tokenizer.eos_token_id).all():
                    break
                
                # Append new tokens to input_ids and update attention_mask
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device)], dim=-1)
            
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

        if self.routing_probe is not None:
            embedding_params.extend(list(self.routing_probe.parameters()))
        
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
        """Load checkpoints with backward compatibility for legacy probe keys."""
        remapped_state_dict = state_dict
        legacy_prefixes = ("toolmix_head.", "gate_mlp.")
        if any(key.startswith(prefix) for prefix in legacy_prefixes):
            remapped_state_dict = dict(state_dict)
            for key in list(remapped_state_dict.keys()):
                for legacy_prefix in legacy_prefixes:
                    if not key.startswith(legacy_prefix):
                        continue
                    mapped_key = f"routing_probe.{key[len(legacy_prefix):]}"
                    remapped_state_dict.setdefault(mapped_key, remapped_state_dict[key])
                    del remapped_state_dict[key]
                    break
        return super().load_state_dict(remapped_state_dict, strict=strict)
    
 
