"""Expandable procedural-memory tokens for TokMem and TapMem.

The base language model is frozen.  Procedure tokens use a separate trainable
embedding matrix for both input lookup and output logits.  When the procedure
inventory exceeds the tokenizer's native reserved slots, this module adds
special tokens and resizes the base embedding tables; it never truncates K.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = {"tokmem", "eoc_only", "tapmem"}
TCRA_TRIGGER_POLICY = "assistant_start_or_generated_eoc_with_mass_constraint_v4"
TCRA_ADDITIVE_ONLY_TRIGGER_POLICY = (
    "assistant_start_or_generated_eoc_train_inference_aligned_v3"
)
_RESERVED_PATTERN = re.compile(r"reserved_special_token_\d+")


def _vocabulary_hash(vocabulary: Mapping[str, int]) -> str:
    payload = json.dumps(
        sorted((str(token), int(token_id)) for token, token_id in vocabulary.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_without_specials(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    value = tokenizer(text, add_special_tokens=False)
    return list(value["input_ids"])


def _append_special_tokens(tokenizer, token_names: Sequence[str]) -> None:
    missing = [name for name in token_names if name not in tokenizer.get_vocab()]
    if not missing:
        return
    try:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        existing = [
            str(value) for value in getattr(tokenizer, "additional_special_tokens", ())
        ]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [*existing, *missing]}
        )


@dataclass(frozen=True)
class MemoryTokenRegistry:
    num_procedures: int
    procedure_token_names: tuple[str, ...]
    procedure_token_ids: tuple[int, ...]
    eoc_token_name: str
    eoc_token_id: int
    native_reserved_count: int
    native_procedure_count: int
    added_token_names: tuple[str, ...]
    base_vocab_hash: str
    augmented_vocab_hash: str

    @classmethod
    def build(cls, tokenizer, num_procedures: int) -> "MemoryTokenRegistry":
        if num_procedures <= 0:
            raise ValueError("num_procedures must be positive")
        base_vocab = tokenizer.get_vocab()
        base_vocab_hash = _vocabulary_hash(base_vocab)
        native = sorted(
            (
                (name, int(token_id))
                for name, token_id in base_vocab.items()
                if _RESERVED_PATTERN.search(name)
            ),
            key=lambda item: item[1],
        )

        # One slot is kept for EOC in the registry shared by both methods.
        native_procedure_capacity = max(0, len(native) - 1)
        native_procedure_count = min(num_procedures, native_procedure_capacity)
        procedure_names = [name for name, _token_id in native[:native_procedure_count]]
        for piece_id in range(native_procedure_count, num_procedures):
            procedure_names.append(f"<|procedure_memory_{piece_id:06d}|>")

        if native:
            eoc_name = native[native_procedure_count][0]
        else:
            eoc_name = "<|procedure_eoc|>"
        names_to_make_valid = [*procedure_names, eoc_name]
        added_names = tuple(
            name for name in names_to_make_valid if name not in base_vocab
        )
        _append_special_tokens(tokenizer, names_to_make_valid)
        augmented_vocab = tokenizer.get_vocab()

        missing = [name for name in names_to_make_valid if name not in augmented_vocab]
        if missing:
            raise RuntimeError(f"Tokenizer failed to add control tokens: {missing}")
        procedure_ids = tuple(int(augmented_vocab[name]) for name in procedure_names)
        eoc_id = int(augmented_vocab[eoc_name])
        if len(set(procedure_ids)) != num_procedures:
            raise AssertionError("Procedure token IDs are not unique")
        if eoc_id in procedure_ids:
            raise AssertionError("EOC token aliases a procedure token")
        for name, token_id in zip(procedure_names, procedure_ids):
            encoded = _encode_without_specials(tokenizer, name)
            if encoded != [token_id]:
                raise ValueError(
                    f"Procedure control token is not atomic: {name!r} -> {encoded}"
                )
        if _encode_without_specials(tokenizer, eoc_name) != [eoc_id]:
            raise ValueError(f"EOC control token is not atomic: {eoc_name!r}")

        return cls(
            num_procedures=num_procedures,
            procedure_token_names=tuple(procedure_names),
            procedure_token_ids=procedure_ids,
            eoc_token_name=eoc_name,
            eoc_token_id=eoc_id,
            native_reserved_count=len(native),
            native_procedure_count=native_procedure_count,
            added_token_names=added_names,
            base_vocab_hash=base_vocab_hash,
            augmented_vocab_hash=_vocabulary_hash(augmented_vocab),
        )

    @classmethod
    def restore(cls, tokenizer, manifest: Mapping[str, Any]) -> "MemoryTokenRegistry":
        _append_special_tokens(tokenizer, manifest["added_token_names"])
        vocabulary = tokenizer.get_vocab()
        names = tuple(str(value) for value in manifest["procedure_token_names"])
        ids = tuple(int(vocabulary[name]) for name in names)
        eoc_name = str(manifest["eoc_token_name"])
        restored = cls(
            num_procedures=int(manifest["num_procedures"]),
            procedure_token_names=names,
            procedure_token_ids=ids,
            eoc_token_name=eoc_name,
            eoc_token_id=int(vocabulary[eoc_name]),
            native_reserved_count=int(manifest["native_reserved_count"]),
            native_procedure_count=int(manifest["native_procedure_count"]),
            added_token_names=tuple(str(value) for value in manifest["added_token_names"]),
            base_vocab_hash=str(manifest["base_vocab_hash"]),
            augmented_vocab_hash=_vocabulary_hash(vocabulary),
        )
        expected_ids = tuple(int(value) for value in manifest["procedure_token_ids"])
        if restored.procedure_token_ids != expected_ids:
            raise ValueError(
                "Restored procedure token IDs differ from checkpoint: "
                f"{restored.procedure_token_ids} != {expected_ids}"
            )
        if restored.eoc_token_id != int(manifest["eoc_token_id"]):
            raise ValueError("Restored EOC token ID differs from checkpoint")
        if (
            manifest.get("augmented_vocab_hash")
            and restored.augmented_vocab_hash != manifest["augmented_vocab_hash"]
        ):
            raise ValueError("Restored augmented tokenizer vocabulary hash differs")
        return restored

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "procedure_token_names",
            "procedure_token_ids",
            "added_token_names",
        ):
            value[key] = list(value[key])
        return value


def orthogonal_memory_rows(
    num_procedures: int,
    hidden_size: int,
    *,
    seed: int,
    target_norm: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Initialize all procedure rows together as a pairwise orthogonal set."""

    if num_procedures > hidden_size:
        raise ValueError(
            f"K={num_procedures} exceeds hidden_size={hidden_size}; "
            "strict pairwise orthogonal procedure embeddings do not exist"
        )
    if num_procedures <= 0 or hidden_size <= 0:
        raise ValueError("num_procedures and hidden_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_columns = torch.randn(
        hidden_size,
        num_procedures,
        generator=generator,
        dtype=torch.float32,
    )
    orthonormal_columns, _ = torch.linalg.qr(random_columns, mode="reduced")
    rows = orthonormal_columns.transpose(0, 1).contiguous() * float(target_norm)
    normalized = F.normalize(rows, dim=1)
    gram = normalized @ normalized.transpose(0, 1)
    error = float((gram - torch.eye(num_procedures)).abs().max().item())
    if error > 1e-5:
        raise RuntimeError(f"Float32 orthogonal initialization failed, max Gram error={error}")
    return rows.to(device=device, dtype=dtype)


@dataclass
class MemoryForwardOutput:
    logits: torch.Tensor
    final_hidden_state: torch.Tensor
    past_key_values: Any
    base_output: Any


def resize_token_embeddings(base_model: nn.Module, new_size: int) -> None:
    """Expand frozen tokenizer rows without costly distribution estimation.

    Procedure rows are initialized separately by ``orthogonal_memory_rows`` and
    override the frozen placeholder rows during every forward pass.
    """

    try:
        base_model.resize_token_embeddings(new_size, mean_resizing=False)
    except TypeError:
        # Keep the small test model and older Transformers releases compatible.
        base_model.resize_token_embeddings(new_size)


class ProceduralMemoryModel(nn.Module):
    """Frozen causal LM with coupled, expandable procedure-memory embeddings."""

    def __init__(
        self,
        base_model: nn.Module,
        registry: MemoryTokenRegistry,
        *,
        method: str,
        initialization_seed: int,
        logit_bias_scale: float = 1.0,
        memory_bank_probability_threshold: float = 0.5,
        enable_memory_bank_constraint: bool = False,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
        self.base_model = base_model
        self.registry = registry
        self.method = method
        self.use_eoc = method in {"eoc_only", "tapmem"}
        self.use_logit_bias = method == "tapmem"
        self.use_memory_bank_constraint = (
            method == "tapmem" and bool(enable_memory_bank_constraint)
        )
        self.logit_bias_scale = float(logit_bias_scale)
        self.memory_bank_probability_threshold = float(
            memory_bank_probability_threshold
        )
        if not 0.0 <= self.memory_bank_probability_threshold <= 1.0:
            raise ValueError(
                "memory_bank_probability_threshold must be between 0 and 1"
            )

        if hasattr(base_model, "resize_token_embeddings"):
            resize_token_embeddings(
                base_model,
                max(
                    len(base_model.get_input_embeddings().weight),
                    max((*registry.procedure_token_ids, registry.eoc_token_id)) + 1,
                ),
            )
        for parameter in base_model.parameters():
            parameter.requires_grad = False

        input_embedding = base_model.get_input_embeddings()
        if input_embedding is None or not hasattr(input_embedding, "weight"):
            raise TypeError("Base model must expose get_input_embeddings().weight")
        hidden_size = int(input_embedding.weight.shape[1])
        config_hidden_size = int(getattr(base_model.config, "hidden_size", hidden_size))
        if hidden_size != config_hidden_size:
            raise ValueError(
                f"Embedding width {hidden_size} != config.hidden_size {config_hidden_size}"
            )
        self.hidden_size = hidden_size
        base_dtype = input_embedding.weight.dtype
        base_device = input_embedding.weight.device
        procedure_rows = orthogonal_memory_rows(
            registry.num_procedures,
            hidden_size,
            seed=initialization_seed,
            target_norm=1.0,
            dtype=base_dtype,
            device=base_device,
        )
        self.procedure_embeddings = nn.Parameter(procedure_rows)
        if self.use_eoc:
            eoc_row = input_embedding.weight.detach()[registry.eoc_token_id].clone()
            self.eoc_embedding = nn.Parameter(eoc_row.unsqueeze(0))
        else:
            self.register_parameter("eoc_embedding", None)
        if self.use_logit_bias:
            self.routing_head = nn.Linear(
                hidden_size,
                registry.num_procedures,
                bias=True,
                device=base_device,
                dtype=base_dtype,
            )
        else:
            self.routing_head = None

        trainable_ids = list(registry.procedure_token_ids)
        if self.use_eoc:
            trainable_ids.append(registry.eoc_token_id)
        self.register_buffer(
            "_trainable_token_ids",
            torch.tensor(trainable_ids, dtype=torch.long, device=base_device),
            persistent=False,
        )
        self.register_buffer(
            "_procedure_token_ids",
            torch.tensor(
                registry.procedure_token_ids,
                dtype=torch.long,
                device=base_device,
            ),
            persistent=False,
        )
        lookup_size = max(trainable_ids) + 1
        lookup = torch.full(
            (lookup_size,),
            -1,
            dtype=torch.long,
            device=base_device,
        )
        lookup[self._trainable_token_ids] = torch.arange(
            len(trainable_ids),
            dtype=torch.long,
            device=base_device,
        )
        self.register_buffer("_control_lookup", lookup, persistent=False)

        output_embedding = base_model.get_output_embeddings()
        if output_embedding is None:
            raise TypeError("Base model must expose an output embedding/lm_head module")
        self._captured_final_hidden: torch.Tensor | None = None
        self._output_pre_hook = output_embedding.register_forward_pre_hook(
            self._capture_output_input
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tokenizer,
        num_procedures: int,
        *,
        method: str,
        initialization_seed: int,
        revision: str | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
        logit_bias_scale: float = 1.0,
        memory_bank_probability_threshold: float = 0.5,
        enable_memory_bank_constraint: bool = False,
    ) -> "ProceduralMemoryModel":
        from transformers import AutoModelForCausalLM

        registry = MemoryTokenRegistry.build(tokenizer, num_procedures)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        resize_token_embeddings(base_model, len(tokenizer))
        base_model.to(device)
        return cls(
            base_model,
            registry,
            method=method,
            initialization_seed=initialization_seed,
            logit_bias_scale=logit_bias_scale,
            memory_bank_probability_threshold=memory_bank_probability_threshold,
            enable_memory_bank_constraint=enable_memory_bank_constraint,
        )

    def _capture_output_input(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("LM output module received no hidden-state input")
        self._captured_final_hidden = inputs[0]

    def _all_trainable_rows(self) -> torch.Tensor:
        if self.eoc_embedding is None:
            return self.procedure_embeddings
        return torch.cat([self.procedure_embeddings, self.eoc_embedding], dim=0)

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        lookup = self._control_lookup
        if input_ids.device != lookup.device:
            lookup = lookup.to(input_ids.device)
        in_range = input_ids < lookup.numel()
        safe_ids = torch.where(in_range, input_ids, torch.zeros_like(input_ids))
        row_indices = lookup[safe_ids]
        mask = in_range & (row_indices >= 0)
        if not mask.any():
            return embeddings
        safe_rows = row_indices.clamp_min(0)
        replacements = self._all_trainable_rows()[safe_rows]
        return torch.where(mask.unsqueeze(-1), replacements, embeddings)

    def _replace_output_logits(
        self,
        logits: torch.Tensor,
        final_hidden: torch.Tensor,
    ) -> torch.Tensor:
        rows = self._all_trainable_rows()
        control_logits = torch.matmul(final_hidden, rows.transpose(0, 1))
        token_ids = self._trainable_token_ids
        if token_ids.device != logits.device:
            token_ids = token_ids.to(logits.device)
        return torch.index_copy(logits, -1, token_ids, control_logits)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> MemoryForwardOutput:
        self._captured_final_hidden = None
        inputs_embeds = self._input_embeddings(input_ids)
        output = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        final_hidden = self._captured_final_hidden
        if final_hidden is None:
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError(
                    "Could not capture final hidden states from the base LM output module"
                )
            final_hidden = hidden_states[-1]
        logits = self._replace_output_logits(output.logits, final_hidden)
        return MemoryForwardOutput(
            logits=logits,
            final_hidden_state=final_hidden,
            past_key_values=getattr(output, "past_key_values", None),
            base_output=output,
        )

    def routing_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.routing_head is None:
            raise RuntimeError("Routing scores are only available for TapMem")
        return self.routing_head(hidden_states)

    def add_routing_bias(
        self,
        full_logits: torch.Tensor,
        boundary_hidden: torch.Tensor,
        *,
        detach_hidden: bool = True,
    ) -> torch.Tensor:
        if self.routing_head is None:
            return full_logits
        head_input = boundary_hidden.detach() if detach_hidden else boundary_hidden
        route_logits = self.routing_scores(head_input)
        bias = (
            F.log_softmax(route_logits.float(), dim=-1)
            + math.log(self.registry.num_procedures)
        ) * self.logit_bias_scale
        bias = bias.to(full_logits.dtype)
        token_ids = self._procedure_token_ids
        if token_ids.device != full_logits.device:
            token_ids = token_ids.to(full_logits.device)
        selected = full_logits.index_select(-1, token_ids) + bias
        return torch.index_copy(full_logits, -1, token_ids, selected)

    def tcra_trigger_masks(
        self,
        raw_logits: torch.Tensor,
        decision_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply TCRA only at assistant-start and generated-EOC boundaries."""

        if raw_logits.shape[:-1] != decision_context.shape:
            raise ValueError("decision_context must match the logits prefix shape")
        decision_context = decision_context.to(
            device=raw_logits.device,
            dtype=torch.bool,
        )
        token_ids = self._procedure_token_ids
        if token_ids.device != raw_logits.device:
            token_ids = token_ids.to(raw_logits.device)
        constraint_mask = torch.zeros_like(decision_context)
        if self.use_memory_bank_constraint:
            float_logits = raw_logits.float()
            memory_logits = float_logits.index_select(-1, token_ids)
            memory_mass = torch.exp(
                torch.logsumexp(memory_logits, dim=-1)
                - torch.logsumexp(float_logits, dim=-1)
            )
            constraint_mask = (
                memory_mass >= self.memory_bank_probability_threshold
            )
        constraint_mask = decision_context & constraint_mask
        routing_mask = decision_context
        return routing_mask, constraint_mask

    @torch.inference_mode()
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        response_end_sequences: Sequence[Sequence[int]],
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("InterCode generation currently requires batch size one")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        generated: list[int] = []
        output = self.forward(
            input_ids,
            attention_mask,
            use_cache=True,
        )
        past = output.past_key_values
        next_logits = output.logits[:, -1, :]
        boundary_hidden = output.final_hidden_state[:, -1, :]

        terminated = False
        inside_procedure = False
        procedure_token_ids = set(self.registry.procedure_token_ids)
        constraint_trigger_count = 0
        constraint_changed_token_count = 0
        for _step in range(max_new_tokens):
            raw_next_token = int(torch.argmax(next_logits, dim=-1).item())
            at_decision_boundary = (
                _step == 0
                or (
                    bool(generated)
                    and generated[-1] == self.registry.eoc_token_id
                )
            )
            routing_mask, constraint_mask = self.tcra_trigger_masks(
                next_logits,
                torch.tensor(
                    [at_decision_boundary and not inside_procedure],
                    dtype=torch.bool,
                    device=next_logits.device,
                ),
            )
            apply_routing_bias = bool(routing_mask.item())
            constrain_to_memory_bank = bool(constraint_mask.item())
            decision_logits = next_logits
            if self.use_logit_bias and apply_routing_bias:
                decision_logits = self.add_routing_bias(
                    decision_logits,
                    boundary_hidden,
                    detach_hidden=True,
                )
            if constrain_to_memory_bank:
                memory_logits = decision_logits.index_select(
                    -1,
                    self._procedure_token_ids,
                )
                memory_index = int(
                    torch.argmax(memory_logits, dim=-1).item()
                )
                next_token = int(self.registry.procedure_token_ids[memory_index])
                constraint_trigger_count += 1
                constraint_changed_token_count += int(
                    next_token != raw_next_token
                )
            else:
                next_token = int(torch.argmax(decision_logits, dim=-1).item())
            generated.append(next_token)
            if not inside_procedure and next_token in procedure_token_ids:
                inside_procedure = True
            elif (
                inside_procedure
                and self.use_eoc
                and next_token == self.registry.eoc_token_id
            ):
                inside_procedure = False
            if any(
                len(sequence) <= len(generated)
                and generated[-len(sequence) :] == list(sequence)
                for sequence in response_end_sequences
                if sequence
            ):
                terminated = True
                break

            token_tensor = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=input_ids.device,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            output = self.forward(
                token_tensor,
                attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            next_logits = output.logits[:, -1, :]
            boundary_hidden = output.final_hidden_state[:, -1, :]
        return {
            "generated_ids": generated,
            "terminated": terminated,
            "missing_terminator": not terminated,
            "memory_bank_constraint_trigger_count": constraint_trigger_count,
            "memory_bank_constraint_changed_token_count": (
                constraint_changed_token_count
            ),
        }

    def orthogonality_report(self) -> dict[str, float | int]:
        normalized = F.normalize(self.procedure_embeddings.detach().float(), dim=1)
        gram = normalized @ normalized.transpose(0, 1)
        identity = torch.eye(gram.shape[0], device=gram.device)
        return {
            "K": self.registry.num_procedures,
            "hidden_size": self.hidden_size,
            "max_absolute_gram_error": float((gram - identity).abs().max().item()),
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def close(self) -> None:
        self._output_pre_hook.remove()
