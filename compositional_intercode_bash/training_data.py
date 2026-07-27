"""Prompt formatting, lossless targets, collation, and TapMem route sites."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .atomizer import atomize_command
from .memory_model import METHODS, MemoryTokenRegistry, ProceduralMemoryModel
from .unigram import ProcedureUnigramModel


SYSTEM_PROMPT = (
    "You are an agent operating a Bourne-compatible Bash shell. "
    "Return exactly one Bash command to advance the task. "
    "Do not add Markdown fences or explanations. "
    "Use the execution result from earlier turns when it is available."
)
TARGET_SERIALIZATION_POLICY = "procedure_surface_connector_eoc_v3"
ROUTING_TARGET_MODES = frozenset(
    {"one_hot", "fixed_count_posterior", "residual_set_margin"}
)


def response_end_token_ids(tokenizer) -> list[int]:
    vocabulary = tokenizer.get_vocab()
    for token_name in ("<|eot_id|>", "<|im_end|>"):
        if token_name in vocabulary:
            encoded = tokenizer.encode(token_name, add_special_tokens=False)
            if encoded:
                return [int(value) for value in encoded]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no known response-end token")
    return [int(tokenizer.eos_token_id)]


def apply_chat_template(tokenizer, messages: Sequence[Mapping[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        value = tokenizer.apply_chat_template(
            list(messages),
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        value = tokenizer.apply_chat_template(list(messages), **kwargs)
    if isinstance(value, torch.Tensor):
        value = value.flatten().tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return [int(token_id) for token_id in value]


def initial_messages(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def _encode_chunk(tokenizer, chunk: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer.encode(chunk, add_special_tokens=False)
    ]


def _validate_segments(
    segments: Sequence[Mapping[str, Any]],
    number_of_chunks: int,
    number_of_procedures: int,
) -> None:
    position = 0
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        piece_id = int(segment["piece_id"])
        if start != position or end <= start or end > number_of_chunks:
            raise ValueError(f"Segments do not form a contiguous partition: {segments}")
        if not 0 <= piece_id < number_of_procedures:
            raise ValueError(f"Invalid piece_id={piece_id}")
        position = end
    if position != number_of_chunks:
        raise ValueError("Segments do not cover every base atom chunk")


def strip_control_token_ids(
    generated_ids: Sequence[int],
    *,
    procedure_token_ids: Sequence[int],
    eoc_token_id: int | None,
    response_end_sequences: Sequence[Sequence[int]],
) -> tuple[list[int], bool]:
    """Truncate at the first complete terminator, then remove exact controls."""

    values = [int(value) for value in generated_ids]
    end_index: int | None = None
    for start in range(len(values)):
        for sequence in response_end_sequences:
            sequence = [int(value) for value in sequence]
            if sequence and values[start : start + len(sequence)] == sequence:
                end_index = start
                break
        if end_index is not None:
            break
    missing_terminator = end_index is None
    prefix = values if end_index is None else values[:end_index]
    controls = {int(value) for value in procedure_token_ids}
    if eoc_token_id is not None:
        controls.add(int(eoc_token_id))
    return [value for value in prefix if value not in controls], missing_terminator


@dataclass
class SerializedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    target_length: int
    ordinary_token_ids: list[int]
    memory_token_count: int
    eoc_count: int
    route_site_count: int
    routing_target_probabilities: list[list[float]] | None
    routing_valid_piece_ids: list[list[int]] | None


def fixed_count_routing_targets(
    view: Mapping[str, Any],
    procedure_model: ProcedureUnigramModel,
) -> list[list[float]]:
    """Return a conditional next-procedure distribution at each chosen boundary."""

    chunks = [str(value) for value in view["base_chunks"]]
    segments = list(view["segments"])
    _validate_segments(segments, len(chunks), procedure_model.size)
    signatures: list[tuple[str, str] | None] = [None] * len(chunks)
    for segment in segments:
        piece_id = int(segment["piece_id"])
        start = int(segment["start"])
        end = int(segment["end"])
        piece = procedure_model.pieces[piece_id]
        if len(piece) != end - start:
            raise ValueError(
                f"Segment span does not match procedure {piece_id}: {segment}"
            )
        for position, signature in enumerate(piece, start=start):
            incumbent = signatures[position]
            if incumbent is not None and incumbent != signature:
                raise ValueError("Selected procedures imply inconsistent atom signatures")
            signatures[position] = signature
    if any(signature is None for signature in signatures):
        raise ValueError("Selected procedures do not recover every atom signature")
    sequence = tuple(signature for signature in signatures if signature is not None)

    targets: list[list[float]] = []
    for segment_index, segment in enumerate(segments):
        start = int(segment["start"])
        remaining = len(segments) - segment_index
        distribution = procedure_model.fixed_count_next_piece_distribution(
            sequence,
            start,
            remaining,
        )
        dense = [0.0] * procedure_model.size
        for piece_id, probability in distribution:
            dense[piece_id] = probability
        selected_piece = int(segment["piece_id"])
        if dense[selected_piece] <= 0.0:
            raise ValueError(
                f"Selected procedure {selected_piece} is not a legal continuation"
            )
        targets.append(dense)
    return targets


def residual_routing_valid_piece_ids(
    view: Mapping[str, Any],
    procedure_model: ProcedureUnigramModel,
    *,
    cumulative_mass: float,
) -> list[list[int]]:
    """Build per-boundary valid sets without softening artificial anchors."""

    if not 0.0 < cumulative_mass <= 1.0:
        raise ValueError("cumulative_mass must be in (0, 1]")
    segments = list(view["segments"])
    if view.get("presentation_kind") != "boundary_sample":
        return [[int(segment["piece_id"])] for segment in segments]

    probabilities = fixed_count_routing_targets(view, procedure_model)
    valid_sets: list[list[int]] = []
    for segment, distribution in zip(segments, probabilities):
        candidates = sorted(
            (
                (probability, piece_id)
                for piece_id, probability in enumerate(distribution)
                if probability > 0.0
            ),
            key=lambda value: (-value[0], value[1]),
        )
        selected: list[int] = []
        mass = 0.0
        for probability, piece_id in candidates:
            selected.append(piece_id)
            mass += probability
            if mass >= cumulative_mass:
                break
        gold_piece = int(segment["piece_id"])
        if gold_piece not in selected:
            selected.append(gold_piece)
        valid_sets.append(sorted(selected))
    return valid_sets


def serialize_view(
    tokenizer,
    registry: MemoryTokenRegistry,
    view: Mapping[str, Any],
    *,
    method: str,
    max_length: int,
    explicit_response_end_ids: Sequence[int] | None = None,
    procedure_model: ProcedureUnigramModel | None = None,
    routing_target_mode: str = "one_hot",
    routing_candidate_mass: float = 0.9,
) -> SerializedExample:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if routing_target_mode not in ROUTING_TARGET_MODES:
        raise ValueError(f"Unknown routing_target_mode: {routing_target_mode}")
    if routing_target_mode != "one_hot":
        if method != "tapmem":
            raise ValueError("Posterior routing targets are TapMem-only")
        if procedure_model is None:
            raise ValueError("Posterior routing targets require a procedure model")
        if procedure_model.size != registry.num_procedures:
            raise ValueError("Procedure model and memory registry sizes differ")
    chunks = [str(value) for value in view["base_chunks"]]
    segments = list(view["segments"])
    _validate_segments(segments, len(chunks), registry.num_procedures)
    command_raw = str(view["command_raw"])
    atomized = atomize_command(command_raw)
    if list(atomized.base_chunks) != chunks:
        raise ValueError(
            f"Stored atom chunks differ from reparsed command for "
            f"{view.get('presentation_id', '<unknown>')}"
        )

    use_eoc = method in {"eoc_only", "tapmem"}
    segment_bounds: list[tuple[int, int, int]] = []
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        core_start = atomized.atoms[start].char_start
        core_end = atomized.atoms[end - 1].char_end
        segment_bounds.append((int(segment["piece_id"]), core_start, core_end))

    target: list[int] = []
    if use_eoc:
        surface_start = 0
        for segment_index, (piece_id, _core_start, _core_end) in enumerate(
            segment_bounds
        ):
            surface_end = (
                segment_bounds[segment_index + 1][1]
                if segment_index + 1 < len(segment_bounds)
                else len(command_raw)
            )
            target.append(registry.procedure_token_ids[piece_id])
            target.extend(
                _encode_chunk(
                    tokenizer,
                    command_raw[surface_start:surface_end],
                )
            )
            target.append(registry.eoc_token_id)
            surface_start = surface_end
    else:
        cursor = 0
        for piece_id, core_start, core_end in segment_bounds:
            target.extend(_encode_chunk(tokenizer, command_raw[cursor:core_start]))
            target.append(registry.procedure_token_ids[piece_id])
            target.extend(_encode_chunk(tokenizer, command_raw[core_start:core_end]))
            cursor = core_end
        target.extend(_encode_chunk(tokenizer, command_raw[cursor:]))
    end_ids = (
        [int(value) for value in explicit_response_end_ids]
        if explicit_response_end_ids is not None
        else response_end_token_ids(tokenizer)
    )
    target.extend(end_ids)

    recovered, missing = strip_control_token_ids(
        target,
        procedure_token_ids=registry.procedure_token_ids,
        eoc_token_id=registry.eoc_token_id,
        response_end_sequences=[end_ids],
    )
    if missing:
        raise AssertionError("Serialized target lacks its response terminator")
    decoded = tokenizer.decode(
        recovered,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded.encode("utf-8") != command_raw.encode("utf-8"):
        raise ValueError(
            f"Tokenizer byte round trip failed for {view.get('presentation_id', '<unknown>')}"
        )

    prompt_ids = apply_chat_template(
        tokenizer,
        initial_messages(str(view["instruction"])),
    )
    input_ids = [*prompt_ids, *target]
    if len(input_ids) > max_length:
        identity = view.get("presentation_id", view.get("sample_id", "<unknown>"))
        raise ValueError(
            f"Serialized training example {identity} has length {len(input_ids)}, "
            f"exceeding max_length={max_length}; truncation is forbidden"
        )
    labels = [-100] * len(prompt_ids) + target
    return SerializedExample(
        input_ids=input_ids,
        labels=labels,
        prompt_length=len(prompt_ids),
        target_length=len(target),
        ordinary_token_ids=recovered,
        memory_token_count=len(segments),
        eoc_count=len(segments) if use_eoc else 0,
        route_site_count=len(segments) if method == "tapmem" else 0,
        routing_target_probabilities=(
            fixed_count_routing_targets(view, procedure_model)
            if routing_target_mode == "fixed_count_posterior"
            and procedure_model is not None
            else None
        ),
        routing_valid_piece_ids=(
            residual_routing_valid_piece_ids(
                view,
                procedure_model,
                cumulative_mass=routing_candidate_mass,
            )
            if routing_target_mode == "residual_set_margin"
            and procedure_model is not None
            else None
        ),
    )


class BoundaryViewDataset(Dataset):
    def __init__(
        self,
        views: Sequence[Mapping[str, Any]],
        tokenizer,
        registry: MemoryTokenRegistry,
        *,
        method: str,
        max_length: int,
        procedure_model: ProcedureUnigramModel | None = None,
        routing_target_mode: str = "one_hot",
        routing_candidate_mass: float = 0.9,
    ) -> None:
        self.views = list(views)
        self.routing_target_mode = routing_target_mode
        self.routing_num_procedures = registry.num_procedures
        end_token_ids = response_end_token_ids(tokenizer)
        self.examples = [
            serialize_view(
                tokenizer,
                registry,
                view,
                method=method,
                max_length=max_length,
                explicit_response_end_ids=end_token_ids,
                procedure_model=procedure_model,
                routing_target_mode=routing_target_mode,
                routing_candidate_mass=routing_candidate_mass,
            )
            for view in self.views
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        view = self.views[index]
        return {
            "input_ids": example.input_ids,
            "labels": example.labels,
            "presentation_id": view.get("presentation_id"),
            "view_id": view.get("view_id"),
            "epoch": view.get("epoch"),
            "effective_batch_id": view.get("effective_batch_id"),
            "position_in_batch": view.get("position_in_batch"),
            "pool": view.get("pool"),
            "routing_target_probabilities": (
                example.routing_target_probabilities
            ),
            "routing_valid_piece_ids": example.routing_valid_piece_ids,
            "routing_num_procedures": (
                self.routing_num_procedures
                if example.routing_valid_piece_ids is not None
                else None
            ),
        }

    def exposure_report(self) -> dict[str, int]:
        return {
            "presentations": len(self.examples),
            "supervised_tokens": sum(example.target_length for example in self.examples),
            "ordinary_tokens": sum(
                len(example.ordinary_token_ids) for example in self.examples
            ),
            "memory_tokens": sum(
                example.memory_token_count for example in self.examples
            ),
            "eoc_tokens": sum(example.eoc_count for example in self.examples),
            "route_sites": sum(example.route_site_count for example in self.examples),
        }


def left_pad_collate(
    records: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    maximum = max(len(record["input_ids"]) for record in records)
    input_rows = []
    label_rows = []
    mask_rows = []
    for record in records:
        padding = maximum - len(record["input_ids"])
        input_rows.append([pad_token_id] * padding + list(record["input_ids"]))
        label_rows.append([-100] * padding + list(record["labels"]))
        mask_rows.append([0] * padding + [1] * len(record["input_ids"]))
    routing_targets = [
        target
        for record in records
        for target in (record.get("routing_target_probabilities") or [])
    ]
    posterior_records = [
        record.get("routing_target_probabilities") is not None
        for record in records
    ]
    if any(posterior_records) and not all(posterior_records):
        raise ValueError("A batch mixes one-hot and posterior routing targets")
    valid_set_records = [
        record.get("routing_valid_piece_ids") is not None
        for record in records
    ]
    if any(valid_set_records) and not all(valid_set_records):
        raise ValueError("A batch mixes residual-set and other routing targets")
    if any(posterior_records) and any(valid_set_records):
        raise ValueError("A batch has two routing target modes")
    routing_valid_mask = None
    if all(valid_set_records):
        procedure_counts = {
            int(record["routing_num_procedures"]) for record in records
        }
        if len(procedure_counts) != 1:
            raise ValueError("A batch mixes procedure inventory sizes")
        number_of_procedures = next(iter(procedure_counts))
        valid_piece_ids = [
            piece_ids
            for record in records
            for piece_ids in record["routing_valid_piece_ids"]
        ]
        routing_valid_mask = torch.zeros(
            (len(valid_piece_ids), number_of_procedures),
            dtype=torch.bool,
        )
        for row_index, piece_ids in enumerate(valid_piece_ids):
            routing_valid_mask[row_index, piece_ids] = True
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "presentation_ids": [record.get("presentation_id") for record in records],
        "view_ids": [record.get("view_id") for record in records],
        "epochs": [record.get("epoch") for record in records],
        "effective_batch_ids": [
            record.get("effective_batch_id") for record in records
        ],
        "positions_in_batch": [
            record.get("position_in_batch") for record in records
        ],
        "pools": [record.get("pool") for record in records],
        "routing_target_probabilities": (
            torch.tensor(routing_targets, dtype=torch.float32)
            if all(posterior_records)
            else None
        ),
        "routing_valid_mask": routing_valid_mask,
    }


@dataclass
class RoutingSites:
    batch_indices: torch.Tensor
    time_indices: torch.Tensor
    targets: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.targets.numel())


@dataclass
class TCRASites:
    batch_indices: torch.Tensor
    time_indices: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.batch_indices.numel())


def gather_routing_sites(
    labels: torch.Tensor,
    registry: MemoryTokenRegistry,
) -> RoutingSites:
    """Find causal hidden positions that predict the next procedure token."""

    token_to_piece = {
        int(token_id): piece_id
        for piece_id, token_id in enumerate(registry.procedure_token_ids)
    }
    batches: list[int] = []
    times: list[int] = []
    targets: list[int] = []
    for batch_index in range(labels.shape[0]):
        valid_positions = torch.nonzero(
            labels[batch_index] != -100,
            as_tuple=False,
        ).flatten()
        for position_tensor in valid_positions:
            position = int(position_tensor.item())
            token = int(labels[batch_index, position].item())
            if token not in token_to_piece:
                continue
            if position == 0:
                raise ValueError("No causal position exists before a procedure token")
            batches.append(batch_index)
            times.append(position - 1)
            targets.append(token_to_piece[token])
    device = labels.device
    return RoutingSites(
        batch_indices=torch.tensor(batches, dtype=torch.long, device=device),
        time_indices=torch.tensor(times, dtype=torch.long, device=device),
        targets=torch.tensor(targets, dtype=torch.long, device=device),
    )


def gather_training_tcra_sites(
    labels: torch.Tensor,
    registry: MemoryTokenRegistry,
) -> TCRASites:
    """Collect gold procedure starts at assistant-start or EOC boundaries."""

    routing_sites = gather_routing_sites(labels, registry)
    if routing_sites.count == 0:
        return TCRASites(
            batch_indices=routing_sites.batch_indices,
            time_indices=routing_sites.time_indices,
        )
    preceding_labels = labels[
        routing_sites.batch_indices,
        routing_sites.time_indices,
    ]
    valid_boundaries = (preceding_labels == -100) | (
        preceding_labels == registry.eoc_token_id
    )
    if not bool(valid_boundaries.all()):
        invalid_count = int((~valid_boundaries).sum().item())
        raise ValueError(
            f"{invalid_count} procedure tokens are not at assistant-start or EOC boundaries"
        )
    return TCRASites(
        batch_indices=routing_sites.batch_indices,
        time_indices=routing_sites.time_indices,
    )


def add_train_routing_bias(
    model: ProceduralMemoryModel,
    shift_logits: torch.Tensor,
    boundary_hidden: torch.Tensor,
    sites: RoutingSites | TCRASites,
    *,
    detach_hidden: bool = True,
) -> torch.Tensor:
    if sites.count == 0:
        return shift_logits
    route_input = boundary_hidden.detach() if detach_hidden else boundary_hidden
    head_logits = model.routing_scores(route_input)
    bias = (
        F.log_softmax(head_logits.float(), dim=-1)
        + math.log(model.registry.num_procedures)
    ) * model.logit_bias_scale
    bias = bias.to(shift_logits.dtype)
    token_ids = torch.tensor(
        model.registry.procedure_token_ids,
        dtype=torch.long,
        device=shift_logits.device,
    )
    output = shift_logits.clone()
    selected = output[
        sites.batch_indices[:, None],
        sites.time_indices[:, None],
        token_ids[None, :],
    ]
    output[
        sites.batch_indices[:, None],
        sites.time_indices[:, None],
        token_ids[None, :],
    ] = selected + bias
    return output


def residual_set_margin_routing_loss(
    raw_memory_logits: torch.Tensor,
    route_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    logit_bias_scale: float,
    margin: float,
) -> torch.Tensor:
    """Train TCRA only when the raw memory ranking needs correction."""

    if raw_memory_logits.ndim != 2:
        raise ValueError("raw_memory_logits must have shape [sites, procedures]")
    if route_logits.shape != raw_memory_logits.shape:
        raise ValueError("route_logits and raw_memory_logits must have equal shape")
    if valid_mask.shape != raw_memory_logits.shape:
        raise ValueError("valid_mask and raw_memory_logits must have equal shape")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if not valid_mask.any(dim=-1).all():
        raise ValueError("Every routing site needs at least one valid procedure")
    if valid_mask.all(dim=-1).any():
        raise ValueError("Every routing site needs at least one invalid procedure")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")

    raw_scores = raw_memory_logits.detach().float()
    correction = (
        F.log_softmax(route_logits.float(), dim=-1)
        + math.log(raw_scores.shape[-1])
    ) * float(logit_bias_scale)
    calibrated_scores = raw_scores + correction
    raw_top = raw_scores.argmax(dim=-1, keepdim=True)
    raw_top_is_valid = valid_mask.gather(-1, raw_top).squeeze(-1)

    valid_score = calibrated_scores.masked_fill(
        ~valid_mask,
        -torch.inf,
    ).amax(dim=-1)
    invalid_score = calibrated_scores.masked_fill(
        valid_mask,
        -torch.inf,
    ).amax(dim=-1)
    fix_loss = F.softplus(float(margin) + invalid_score - valid_score)

    per_site = torch.where(
        raw_top_is_valid,
        torch.zeros_like(fix_loss),
        fix_loss,
    )
    return per_site.sum()


@dataclass
class TrainingLossSums:
    """Differentiable loss numerators for one microbatch."""

    ar_loss_sum: torch.Tensor
    supervised_tokens: int
    route_loss_sum: torch.Tensor
    route_sites: int


def normalized_microbatch_loss(
    component: TrainingLossSums,
    *,
    effective_batch_supervised_tokens: int,
    effective_batch_route_sites: int,
    route_loss_weight: float,
) -> torch.Tensor:
    """Scale one microbatch by the denominators of its complete 2A+2R group."""

    if effective_batch_supervised_tokens <= 0:
        raise ValueError("Effective-batch supervised-token count must be positive")
    if not 0 <= component.supervised_tokens <= effective_batch_supervised_tokens:
        raise ValueError("Microbatch supervised-token count exceeds its group")
    if not 0 <= component.route_sites <= effective_batch_route_sites:
        raise ValueError("Microbatch route-site count exceeds its group")
    loss = component.ar_loss_sum / effective_batch_supervised_tokens
    if effective_batch_route_sites:
        loss = loss + (
            float(route_loss_weight)
            * component.route_loss_sum
            / effective_batch_route_sites
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite normalized microbatch loss: {loss.detach().item()}"
        )
    return loss


def compute_training_loss_sums(
    model: ProceduralMemoryModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    routing_target_probabilities: torch.Tensor | None = None,
    routing_valid_mask: torch.Tensor | None = None,
    routing_margin: float = 0.5,
) -> TrainingLossSums:
    """Return unnormalized CE sums so microbatching cannot change the objective."""

    position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    if not torch.isfinite(output.logits).all():
        raise FloatingPointError("Model produced non-finite full-vocabulary logits")
    shift_logits = output.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    route_loss = shift_logits.new_zeros(())
    route_sites = 0
    if model.use_logit_bias:
        sites = gather_routing_sites(labels, model.registry)
        route_sites = sites.count
        boundary_hidden = output.final_hidden_state[
            sites.batch_indices,
            sites.time_indices,
        ]
        route_logits = model.routing_scores(boundary_hidden.detach())
        if (
            routing_target_probabilities is not None
            and routing_valid_mask is not None
        ):
            raise ValueError("A microbatch has two routing target modes")
        if routing_valid_mask is not None:
            token_ids = torch.tensor(
                model.registry.procedure_token_ids,
                dtype=torch.long,
                device=shift_logits.device,
            )
            raw_memory_logits = shift_logits[
                sites.batch_indices[:, None],
                sites.time_indices[:, None],
                token_ids[None, :],
            ]
            route_loss = residual_set_margin_routing_loss(
                raw_memory_logits,
                route_logits,
                routing_valid_mask.to(device=route_logits.device),
                logit_bias_scale=model.logit_bias_scale,
                margin=routing_margin,
            )
        elif routing_target_probabilities is None:
            route_loss = F.cross_entropy(
                route_logits.float(),
                sites.targets,
                reduction="sum",
            )
        else:
            expected_shape = (
                sites.count,
                model.registry.num_procedures,
            )
            if tuple(routing_target_probabilities.shape) != expected_shape:
                raise ValueError(
                    "Posterior routing targets have shape "
                    f"{tuple(routing_target_probabilities.shape)}, "
                    f"expected {expected_shape}"
                )
            target_probabilities = routing_target_probabilities.to(
                device=route_logits.device,
                dtype=torch.float32,
            )
            row_sums = target_probabilities.sum(dim=-1)
            if not torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("Posterior routing targets must sum to one")
            route_loss = -(
                target_probabilities
                * F.log_softmax(route_logits.float(), dim=-1)
            ).sum()
        tcra_sites = gather_training_tcra_sites(
            labels,
            model.registry,
        )
        tcra_hidden = output.final_hidden_state[
            tcra_sites.batch_indices,
            tcra_sites.time_indices,
        ]
        shift_logits = add_train_routing_bias(
            model,
            shift_logits,
            tcra_hidden,
            tcra_sites,
            detach_hidden=True,
        )
    supervised_tokens = int((shift_labels != -100).sum().item())
    if supervised_tokens <= 0:
        raise ValueError("A training microbatch has no supervised tokens")
    ar_loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    for name, value in (
        ("ar_loss_sum", ar_loss),
        ("route_loss_sum", route_loss),
    ):
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite {name}: {value.detach().item()}")
    return TrainingLossSums(
        ar_loss_sum=ar_loss,
        supervised_tokens=supervised_tokens,
        route_loss_sum=route_loss,
        route_sites=route_sites,
    )


def combine_training_loss_sums(
    components: Sequence[TrainingLossSums],
    *,
    route_loss_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Normalize one complete effective batch by its actual token/site counts."""

    if not components:
        raise ValueError("Cannot combine an empty effective batch")
    supervised_tokens = sum(item.supervised_tokens for item in components)
    route_sites = sum(item.route_sites for item in components)
    if supervised_tokens <= 0:
        raise ValueError("An effective batch has no supervised tokens")
    ar_loss_sum = torch.stack([item.ar_loss_sum for item in components]).sum()
    route_loss_sum = torch.stack(
        [item.route_loss_sum for item in components]
    ).sum()
    ar_loss = ar_loss_sum / supervised_tokens
    if route_sites:
        route_loss = route_loss_sum / route_sites
    else:
        route_loss = route_loss_sum
    total_loss = ar_loss + float(route_loss_weight) * route_loss
    for name, value in (
        ("ar_loss", ar_loss),
        ("route_loss", route_loss),
        ("total_loss", total_loss),
    ):
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite {name}: {value.detach().item()}")
    return total_loss, {
        "total_loss": float(total_loss.detach().item()),
        "ar_loss": float(ar_loss.detach().item()),
        "route_loss": float(route_loss.detach().item()),
        "ar_loss_sum": float(ar_loss_sum.detach().item()),
        "route_loss_sum": float(route_loss_sum.detach().item()),
        "supervised_tokens": supervised_tokens,
        "route_sites": route_sites,
    }


def compute_training_loss(
    model: ProceduralMemoryModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    route_loss_weight: float = 0.1,
    routing_target_probabilities: torch.Tensor | None = None,
    routing_valid_mask: torch.Tensor | None = None,
    routing_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Convenience wrapper for a single, already-complete effective batch."""

    component = compute_training_loss_sums(
        model,
        input_ids,
        attention_mask,
        labels,
        routing_target_probabilities,
        routing_valid_mask,
        routing_margin,
    )
    return combine_training_loss_sums(
        [component],
        route_loss_weight=route_loss_weight,
    )
