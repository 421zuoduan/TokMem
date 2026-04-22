import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json
import math
import os
from contextlib import contextmanager, nullcontext

def is_rank_zero():
    return os.environ.get("RANK", "0") == "0"

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


def build_task_logit_bias_head(hidden_size, num_tasks, network_type):
    """Build the optional task-token logit-bias head."""
    if network_type == "mlp":
        head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_tasks),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    head = nn.Linear(hidden_size, num_tasks)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class TaskTokenParameterModule(nn.Module):
    """Own the trainable task-token parameters separately from the frozen HF backbone."""

    def __init__(
        self,
        input_embeddings,
        output_embeddings,
        decouple_embeddings,
        use_logit_bias,
        hidden_size,
        logit_bias_network,
        head_device,
        head_dtype,
    ):
        super().__init__()
        self.decouple_embeddings = decouple_embeddings

        if decouple_embeddings:
            self._input_embeddings = nn.Parameter(input_embeddings.clone())
            self._output_embeddings = nn.Parameter(output_embeddings.clone())
            self.shared_embeddings = None
        else:
            self.shared_embeddings = nn.Parameter(input_embeddings.clone())
            self._input_embeddings = None
            self._output_embeddings = None

        if use_logit_bias:
            head = build_task_logit_bias_head(hidden_size, input_embeddings.shape[0], logit_bias_network)
            self.logit_bias_head = head.to(device=head_device, dtype=head_dtype)
        else:
            self.logit_bias_head = None

    @property
    def input_embeddings(self):
        if self.decouple_embeddings:
            return self._input_embeddings
        return self.shared_embeddings

    @property
    def output_embeddings(self):
        if self.decouple_embeddings:
            return self._output_embeddings
        return self.shared_embeddings

    def get_embedding_parameters(self):
        if self.decouple_embeddings:
            return [self.input_embeddings, self.output_embeddings]
        return [self.shared_embeddings]

    def get_all_trainable_parameters(self):
        params = list(self.get_embedding_parameters())
        if self.logit_bias_head is not None:
            params.extend(self.logit_bias_head.parameters())
        return params

    def get_serializable_coupled_embeddings(self):
        if self.decouple_embeddings:
            raise AttributeError("Coupled embeddings are only available in coupled mode")
        return self.shared_embeddings

class TaskCallingModel(nn.Module):
    """
    Task calling using Llama's native reserved special tokens as task tokens.
    
    This model transparently overrides the embedding and lm_head layers to use 
    trainable embeddings for reserved tokens, with optional coupling/decoupling 
    of input and output layer weights. This enables the use of native generation 
    methods while maintaining custom task token behavior.
    
    Args:
        model_name: HuggingFace model name/path (default: "meta-llama/Llama-3.2-1B-Instruct")
        num_tasks: Number of task tokens to use (max 248, default: 100)
        task_names: List of task names (default: auto-generated)
        tokenizer: Pre-loaded tokenizer (required)
        device: Device to load model on (default: "cuda")
        dtype: Model data type (default: torch.bfloat16)
        decouple_embeddings: If True, use separate weights for input/output layers.
                           If False, share weights between input/output (default: True)
    """
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct",
                 num_tasks=100, task_names=None, tokenizer=None, device="cuda", dtype=torch.bfloat16, 
                 decouple_embeddings=False, is_extended=False, use_logit_bias=False,
                 logit_bias_network="linear", logit_bias_scale=1.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # self.num_tasks = min(num_tasks, 248)  # Max 248 reserved tokens available
        self.num_tasks = num_tasks
        self.device = device
        self.dtype = dtype
        self.decouple_embeddings = decouple_embeddings
        self.use_logit_bias = use_logit_bias
        self.logit_bias_network = logit_bias_network
        self.logit_bias_scale = logit_bias_scale
        
        # Load tokenizer to get reserved token mappings
        self.tokenizer = tokenizer
        
        # Get reserved special token mappings
        self._setup_reserved_tokens()
        
        # Load frozen base model (all parameters frozen)
        model_load_kwargs = {"dtype": dtype}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
        self.model = self.model.to(device)
        if is_extended:
            self.model.resize_token_embeddings(len(tokenizer))

        self.input_device = self.model.model.embed_tokens.weight.device
        self.output_device = self.model.lm_head.weight.device
        self.runtime_device = self.input_device
            
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get original embedding parameters for initialization
        original_input_embeddings = self.model.model.embed_tokens.weight.data
        original_output_embeddings = self.model.lm_head.weight.data
        
        # Create trainable parameters for task tokens (coupled or decoupled)
        # Clone the embeddings for the reserved tokens and make them parameters
        # Pre-create tensors for efficient task-token lookup during training/inference.
        self.register_buffer(
            "reserved_token_tensor",
            torch.tensor(self.reserved_token_ids, device=self.input_device),
            persistent=False,
        )
        task_token_lookup = torch.full(
            (original_input_embeddings.shape[0],),
            -1,
            dtype=torch.long,
            device=self.input_device,
        )
        task_token_lookup[self.reserved_token_tensor] = torch.arange(
            self.num_tasks,
            dtype=torch.long,
            device=self.input_device,
        )
        self.register_buffer(
            "task_token_lookup",
            task_token_lookup,
            persistent=False,
        )

        reserved_token_tensor_on_output = self.reserved_token_tensor.to(self.output_device)
        input_task_embeddings = original_input_embeddings[self.reserved_token_tensor].clone()
        coupled_output_embeddings = input_task_embeddings.to(self.output_device)
        self.task_token_module = TaskTokenParameterModule(
            input_embeddings=(
                input_task_embeddings
                if self.decouple_embeddings
                else input_task_embeddings
            ),
            output_embeddings=(
                original_output_embeddings[reserved_token_tensor_on_output].clone()
                if self.decouple_embeddings
                else coupled_output_embeddings
            ),
            decouple_embeddings=self.decouple_embeddings,
            use_logit_bias=self.use_logit_bias,
            hidden_size=self.config.hidden_size,
            logit_bias_network=self.logit_bias_network,
            head_device=self.output_device,
            head_dtype=self.model.lm_head.weight.dtype,
        )
        
        # Task name mapping
        self.task_names = task_names or [f"task_{i}" for i in range(self.num_tasks)]
        self.task_name_to_id = {name: i for i, name in enumerate(self.task_names)}
        self.task_id_to_name = {i: name for i, name in enumerate(self.task_names)}
        
        # Task token mappings (using reserved tokens)
        self.task_id_to_token_id = {i: self.reserved_token_ids[i] for i in range(self.num_tasks)}
        self.token_id_to_task_id = {self.reserved_token_ids[i]: i for i in range(self.num_tasks)} 

        # Override the model's forward method to use our custom embeddings and logits
        self._setup_model_override()

    def _setup_model_override(self):
        """Set up model override to make embedding/logit modifications transparent"""
        # Store original methods
        self.original_embed_forward = self.model.model.embed_tokens.forward
        self.original_lm_head_forward = self.model.lm_head.forward
        
        # Override embedding layer
        def custom_embed_forward(input_ids):
            # Get standard embeddings
            embeddings = self.original_embed_forward(input_ids)
            task_token_lookup = self.task_token_lookup.to(input_ids.device)
            task_token_rows = task_token_lookup[input_ids]
            is_reserved = task_token_rows >= 0
            if is_reserved.any():
                task_input_embedding_rows = self.task_token_module.input_embeddings
                embeddings[is_reserved] = task_input_embedding_rows[
                    task_token_rows[is_reserved]
                ]

            return embeddings
        
        # Override lm_head  
        def custom_lm_head_forward(hidden_states):
            # Get standard logits
            logits = self.original_lm_head_forward(hidden_states)
            reserved_token_tensor = self.reserved_token_tensor.to(logits.device)
            task_output_embedding_rows = self.task_token_module.output_embeddings

            # Efficiently replace reserved token logits using batch matmul
            # Shape: hidden_states (..., hidden_dim), task_embeddings (num_tasks, hidden_dim)
            task_logits = torch.matmul(hidden_states, task_output_embedding_rows.T)  # (..., num_tasks)
            if task_logits.device != logits.device:
                task_logits = task_logits.to(logits.device)
            
            # Replace the specific reserved token positions with our computed logits
            logits[..., reserved_token_tensor] = task_logits
            
            return logits
        
        # Apply overrides
        self.model.model.embed_tokens.forward = custom_embed_forward
        self.model.lm_head.forward = custom_lm_head_forward
    
    def restore_original_model(self):
        """Restore the original model methods (useful for cleanup or debugging)"""
        if hasattr(self, 'original_embed_forward'):
            self.model.model.embed_tokens.forward = self.original_embed_forward
        if hasattr(self, 'original_lm_head_forward'):
            self.model.lm_head.forward = self.original_lm_head_forward

    def _setup_reserved_tokens(self):
        """Extract reserved special token IDs from tokenizer"""
        vocab = self.tokenizer.get_vocab()
        reserved_tokens = {k: v for k, v in vocab.items() if 'reserved_special_token_' in k}
        
        # Sort by token ID to get consistent ordering
        sorted_reserved = sorted(reserved_tokens.items(), key=lambda x: x[1])
        
        # Take first num_tasks reserved tokens
        self.reserved_token_names = [token for token, _ in sorted_reserved[:self.num_tasks]]
        self.reserved_token_ids = [token_id for _, token_id in sorted_reserved[:self.num_tasks]]
        
        if is_rank_zero():
            print(f"Using {len(self.reserved_token_ids)} reserved tokens as task tokens:")
            for i, (name, token_id) in enumerate(zip(self.reserved_token_names[:5], self.reserved_token_ids[:5])):
                print(f"  {name}: {token_id}")
            if len(self.reserved_token_ids) > 5:
                print(f"  ... and {len(self.reserved_token_ids) - 5} more")
            print(f"Embedding coupling mode: {'Decoupled' if self.decouple_embeddings else 'Coupled'}")
            if self.use_logit_bias:
                print(
                    f"Logit-bias head: {self.logit_bias_network}, "
                    f"scale={self.logit_bias_scale}"
                )
    
    def forward(self, input_ids, attention_mask, return_hidden_states=False):
        """
        Forward pass for task calling training
        
        Sequence: [Instruction] [Reserved_Task_Token] [Response] <|eot_id|>
        """
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.model.lm_head(last_hidden_state)
        if return_hidden_states:
            if last_hidden_state.device != logits.device:
                last_hidden_state = last_hidden_state.to(logits.device)
            return logits, last_hidden_state
        return logits

    def get_task_bias_targets(self, token_ids):
        """Map reserved task token IDs to 0..num_tasks-1 class labels."""
        task_token_lookup = self.task_token_lookup.to(token_ids.device)
        task_targets = task_token_lookup[token_ids]
        if (task_targets < 0).any():
            raise ValueError("Task bias targets require reserved task token IDs")
        return task_targets

    def compute_task_logit_bias(self, hidden_states, detach_hidden_states=True):
        """Project hidden states to task-token bias logits."""
        if not self.use_logit_bias or self.logit_bias_head is None:
            raise RuntimeError("Logit bias head is not enabled on this model")

        if detach_hidden_states:
            hidden_states = hidden_states.detach()
        head_param = next(self.logit_bias_head.parameters())
        hidden_states = hidden_states.to(
            device=head_param.device,
            dtype=head_param.dtype,
        )
        return self.logit_bias_head(hidden_states)

    def _apply_first_step_logit_bias(self, next_token_logits, hidden_states):
        """Add task-token bias only to the first generated token logits."""
        if not self.use_logit_bias:
            return next_token_logits

        bias_logits = self.compute_task_logit_bias(hidden_states, detach_hidden_states=True)
        bias_log_probs = torch.log_softmax(bias_logits.float(), dim=-1)
        uniform_log_prob = math.log(max(1, self.num_tasks))
        centered_bias = (bias_log_probs + uniform_log_prob) * self.logit_bias_scale
        centered_bias = centered_bias.to(next_token_logits.device, dtype=next_token_logits.dtype)
        biased_logits = next_token_logits.clone()
        reserved_token_tensor = self.reserved_token_tensor.to(next_token_logits.device)
        biased_logits[:, reserved_token_tensor] += centered_bias
        return biased_logits

    def _sample_next_tokens(self, logits, temperature=0.6, top_p=0.9, do_sample=False):
        """Sample or greedily select the next tokens from logits."""
        if do_sample:
            logits = logits / max(float(temperature), 1e-5)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i][indices_to_remove] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        return torch.argmax(logits, dim=-1)

    @contextmanager
    def eval_generation_context(self, generation_context=None):
        """Compose an optional caller-supplied eval context around generation."""
        with generation_context if generation_context is not None else nullcontext():
            yield

    def _prefill_with_cache(self, input_ids, attention_mask):
        """Run the prompt once, returning next-token logits, last hidden state, and KV cache."""
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
        )
        last_hidden_states = outputs.last_hidden_state[:, -1, :]
        next_token_logits = self.model.lm_head(last_hidden_states.unsqueeze(1))[:, -1, :]
        if last_hidden_states.device != next_token_logits.device:
            last_hidden_states = last_hidden_states.to(next_token_logits.device)
        return next_token_logits, last_hidden_states, outputs.past_key_values

    def _generate_from_past_key_values(
        self,
        instruction_tokens,
        instruction_mask,
        first_step_tokens,
        tokenizer,
        past_key_values,
        remaining_new_tokens,
        temperature,
        top_p,
        do_sample,
    ):
        """Continue generation from an existing prompt KV cache."""
        generated = torch.cat([instruction_tokens, first_step_tokens.unsqueeze(-1)], dim=-1)
        generated_attention_mask = torch.cat(
            [
                instruction_mask,
                torch.ones(
                    instruction_mask.shape[0],
                    1,
                    device=instruction_mask.device,
                    dtype=instruction_mask.dtype,
                ),
            ],
            dim=-1,
        )

        if remaining_new_tokens <= 0:
            return generated

        next_input_ids = first_step_tokens.unsqueeze(-1)
        finished = first_step_tokens == tokenizer.eos_token_id

        for _ in range(remaining_new_tokens):
            local_all_finished = finished.all()
            if dist.is_available() and dist.is_initialized():
                global_finished = local_all_finished.to(dtype=torch.int32)
                dist.all_reduce(global_finished, op=dist.ReduceOp.MIN)
                should_stop = bool(global_finished.item())
            else:
                should_stop = bool(local_all_finished.item())

            if should_stop:
                break

            outputs = self.model.model(
                input_ids=next_input_ids,
                attention_mask=generated_attention_mask,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = self.model.lm_head(outputs.last_hidden_state)[:, -1, :]
            next_tokens = self._sample_next_tokens(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            if finished.any():
                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, tokenizer.eos_token_id),
                    next_tokens,
                )

            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            generated_attention_mask = torch.cat(
                [
                    generated_attention_mask,
                    torch.ones(
                        generated_attention_mask.shape[0],
                        1,
                        device=generated_attention_mask.device,
                        dtype=generated_attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            finished = finished | (next_tokens == tokenizer.eos_token_id)
            next_input_ids = next_tokens.unsqueeze(-1)

        return generated

    def generate_with_task_prediction(self, instruction_tokens, instruction_mask, tokenizer, 
                                     max_new_tokens=256, temperature=0.6, top_p=0.9, do_sample=False,
                                     generation_context=None):
        """Generate complete sequence: predict task, then generate response"""
        self.eval()

        with torch.no_grad():
            with self.eval_generation_context(generation_context):
                if self.use_logit_bias and max_new_tokens > 0:
                    next_token_logits, last_hidden_states, past_key_values = self._prefill_with_cache(
                        instruction_tokens,
                        instruction_mask,
                    )
                    next_token_logits = self._apply_first_step_logit_bias(
                        next_token_logits,
                        last_hidden_states,
                    )
                    first_step_tokens = self._sample_next_tokens(
                        next_token_logits,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    remaining_new_tokens = max_new_tokens - 1
                    generated = self._generate_from_past_key_values(
                        instruction_tokens=instruction_tokens,
                        instruction_mask=instruction_mask,
                        first_step_tokens=first_step_tokens,
                        tokenizer=tokenizer,
                        past_key_values=past_key_values,
                        remaining_new_tokens=remaining_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    return self._parse_generated_sequences(generated, instruction_tokens, tokenizer)

                # Use native generation - our overrides make the custom embeddings/logits transparent
                generated = self.model.generate(
                    input_ids=instruction_tokens,
                    attention_mask=instruction_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

                # Parse the generated sequences
                return self._parse_generated_sequences(generated, instruction_tokens, tokenizer)

    def generate_with_ground_truth_tasks(self, instruction_tokens, instruction_mask, tokenizer, ground_truth_tasks,
                                       max_new_tokens=256, temperature=0.6, top_p=0.9, do_sample=False,
                                       generation_context=None):
        """Generate sequence where predicted task tokens are replaced with ground truth task tokens"""
        self.eval()

        # Convert ground truth task names to token IDs
        gt_task_token_ids = []
        for task_name in ground_truth_tasks:
            if task_name in self.task_name_to_id:
                task_id = self.task_name_to_id[task_name]
                token_id = self.task_id_to_token_id[task_id]
                gt_task_token_ids.append(token_id)
            else:
                print(f"Warning: Unknown task name '{task_name}', skipping")

        with torch.no_grad():
            with self.eval_generation_context(generation_context):
                batch_size = instruction_tokens.shape[0]
                device = instruction_tokens.device

                # Initialize generation state
                input_ids = instruction_tokens.clone()
                attention_mask = instruction_mask.clone()
                task_replacement_count = [0] * batch_size  # Track how many tasks replaced per example

                for step in range(max_new_tokens):
                    # Get model predictions
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])[:, -1, :]

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

                    # Check if any predicted tokens are task tokens and replace them
                    for i in range(batch_size):
                        predicted_token = next_tokens[i].item()

                        # If predicted token is a task token and we have ground truth tasks left
                        if (predicted_token in self.token_id_to_task_id and
                            task_replacement_count[i] < len(gt_task_token_ids)):
                            # Replace with ground truth task token
                            next_tokens[i] = gt_task_token_ids[task_replacement_count[i]]
                            task_replacement_count[i] += 1

                    # Check for EOS tokens
                    if (next_tokens == tokenizer.eos_token_id).all():
                        break

                    # Append new tokens to input_ids and update attention_mask
                    input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype),
                        ],
                        dim=-1,
                    )

                # Parse the generated sequences
                return self._parse_generated_sequences(input_ids, instruction_tokens, tokenizer)

    def _parse_generated_sequences(self, generated_tokens, input_tokens, tokenizer):
        """Parse generated sequences to extract task predictions and responses"""
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

            if not valid_tokens:
                results.append({
                    'predicted_tasks': [],
                    'responses': [],
                    'full_generated_sequence': '',
                    # Backward compatibility
                    'predicted_task_id': None,
                    'predicted_task_name': 'none',
                    'response': ''
                })
                continue

            # Find all task tokens and their positions
            task_positions = []
            for j, token_id in enumerate(valid_tokens):
                if token_id in self.token_id_to_task_id:
                    task_id = self.token_id_to_task_id[token_id]
                    task_name = self.task_id_to_name[task_id]
                    task_positions.append({
                        'position': j,
                        'token_id': token_id,
                        'task_id': task_id,
                        'task_name': task_name
                    })

            predicted_tasks = []
            responses = []

            # Extract response for each task
            for idx, task_info in enumerate(task_positions):
                start_pos = task_info['position'] + 1  # Position after task token

                # Find end position (next task token or end of sequence)
                if idx + 1 < len(task_positions):
                    end_pos = task_positions[idx + 1]['position']
                else:
                    # Last task - response goes to end of sequence (excluding EOT if present)
                    end_pos = len(valid_tokens)
                    # Check if last tokens are EOT tokens and exclude them
                    eot_tokens = tokenizer(tokenizer.eos_token, add_special_tokens=False)['input_ids']
                    if len(eot_tokens) > 0 and end_pos >= len(eot_tokens):
                        # Check if the last tokens match EOT
                        if valid_tokens[-len(eot_tokens):] == eot_tokens:
                            end_pos -= len(eot_tokens)

                # Extract response tokens
                if start_pos < end_pos:
                    response_tokens = valid_tokens[start_pos:end_pos]
                    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                else:
                    response = ""

                predicted_tasks.append({
                    'task_id': task_info['task_id'],
                    'task_name': task_info['task_name'],
                    'token_id': task_info['token_id']
                })
                responses.append(response)

            # If no task tokens found, treat entire sequence as response
            if not task_positions:
                response = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                predicted_tasks = []
                responses = [response] if response else []

            # Prepare result with both new multi-task format and backward compatibility
            result = {
                'predicted_tasks': predicted_tasks,
                'responses': responses,
                'full_generated_sequence': tokenizer.decode(valid_tokens, skip_special_tokens=True),
                # Backward compatibility - use first task if available
                'predicted_task_id': predicted_tasks[0]['task_id'] if predicted_tasks else None,
                'predicted_task_name': predicted_tasks[0]['task_name'] if predicted_tasks else 'none',
                'response': responses[0] if responses else '',
                'task_token_used': tokenizer.decode([predicted_tasks[0]['token_id']]) if predicted_tasks else None
            }

            results.append(result)

        return results
    
    def get_task_token_id(self, task_name):
        """Get the reserved token ID for a task name"""
        if task_name in self.task_name_to_id:
            task_id = self.task_name_to_id[task_name]
            return self.task_id_to_token_id[task_id]
        return None
    
    def get_task_name_from_token_id(self, token_id):
        """Get task name from reserved token ID"""
        if token_id in self.token_id_to_task_id:
            task_id = self.token_id_to_task_id[token_id]
            return self.task_id_to_name[task_id]
        return None

    @property
    def trainable_task_input_embeddings(self):
        return self.task_token_module.input_embeddings

    @property
    def trainable_task_output_embeddings(self):
        return self.task_token_module.output_embeddings

    @property
    def trainable_task_embeddings(self):
        return self.task_token_module.get_serializable_coupled_embeddings()

    @property
    def logit_bias_head(self):
        return self.task_token_module.logit_bias_head

    def get_fsdp_trainable_modules(self):
        """Return small trainable modules that should stay replicated under FSDP."""
        return [self.task_token_module]
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters for optimizer initialization"""
        return self.task_token_module.get_all_trainable_parameters()
    
    def save_task_tokens(self, filepath):
        """Save trained task token embeddings to file"""
        def validate_task_tensor(name, tensor):
            if tensor.ndim != 2 or tensor.shape[0] != self.num_tasks:
                raise ValueError(
                    f"{name} must be a full 2D task-token tensor with first dimension "
                    f"{self.num_tasks}, got shape {tuple(tensor.shape)}. "
                    "This usually means the save path is reading an FSDP shard instead of "
                    "the replicated task-token module."
                )

        save_data = {
            'task_names': self.task_names,
            'num_tasks': self.num_tasks,
            'decouple_embeddings': self.decouple_embeddings,
            'task_name_to_id': self.task_name_to_id,
            'reserved_token_ids': self.reserved_token_ids,
            'use_logit_bias': self.use_logit_bias,
            'logit_bias_network': self.logit_bias_network,
            'logit_bias_scale': self.logit_bias_scale,
        }
        
        if self.decouple_embeddings:
            validate_task_tensor('input_embeddings', self.trainable_task_input_embeddings)
            validate_task_tensor('output_embeddings', self.trainable_task_output_embeddings)
            save_data['input_embeddings'] = self.trainable_task_input_embeddings.detach().cpu()
            save_data['output_embeddings'] = self.trainable_task_output_embeddings.detach().cpu()
        else:
            validate_task_tensor('embeddings', self.trainable_task_embeddings)
            save_data['embeddings'] = self.trainable_task_embeddings.detach().cpu()
        if self.use_logit_bias and self.logit_bias_head is not None:
            save_data['logit_bias_head'] = {
                key: value.detach().cpu()
                for key, value in self.logit_bias_head.state_dict().items()
            }
        
        torch.save(save_data, filepath)
        if is_rank_zero():
            print(f"Task tokens saved to {filepath}")
    
    def load_task_tokens(self, filepath):
        """Load task token embeddings from file"""
        data = torch.load(filepath, map_location='cpu')
        
        # Basic checks
        if data['task_names'] != self.task_names:
            raise ValueError(f"Task names don't match saved file")
        
        if data['decouple_embeddings'] != self.decouple_embeddings:
            raise ValueError(f"Embedding mode doesn't match")
        
        # Load embeddings
        if self.decouple_embeddings:
            input_device = self.trainable_task_input_embeddings.device
            output_device = self.trainable_task_output_embeddings.device
            self.trainable_task_input_embeddings.data = data['input_embeddings'].to(input_device)
            self.trainable_task_output_embeddings.data = data['output_embeddings'].to(output_device)
        else:
            shared_device = self.trainable_task_embeddings.device
            self.trainable_task_embeddings.data = data['embeddings'].to(shared_device)

        saved_logit_bias_head = data.get('logit_bias_head')
        if self.use_logit_bias and self.logit_bias_head is not None and saved_logit_bias_head is not None:
            self.logit_bias_head.load_state_dict(saved_logit_bias_head)
        elif self.use_logit_bias and self.logit_bias_head is not None:
            print("Loaded checkpoint has no logit-bias head. Keeping current bias-head initialization.")
        
        if is_rank_zero():
            print(f"Task tokens loaded from {filepath}")
        return data
