from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from compositional.backbone_prompting import (
    format_user_assistant_prompt,
    response_end_token_ids,
)

from .context import SENTINEL_PATTERN, ObservationPolicy, render_step_context
from .manifest import canonical_json, load_manifest, validate_tool_arguments
from .masked_routing import build_available_tool_mask


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: step must be an object")
            records.append(record)
    return records


def truncate_preserving_supervised_target(
    input_ids: list[int],
    labels: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same length")
    if len(input_ids) <= max_length:
        return input_ids, labels
    supervised_positions = [
        position for position, label in enumerate(labels) if label != -100
    ]
    if not supervised_positions:
        return input_ids[-max_length:], labels[-max_length:]
    first_target = supervised_positions[0]
    target_length = len(input_ids) - first_target
    if target_length >= max_length:
        raise ValueError(
            "supervised target does not fit max_length with the required "
            "assistant-start context token; increase max_length or reject the sample"
        )
    retained_context = max_length - target_length
    start = max(0, first_target - retained_context)
    truncated_input_ids = input_ids[start:]
    truncated_labels = labels[start:]
    if truncated_labels[0] != -100:
        raise AssertionError("target truncation removed the assistant-start boundary")
    if [label for label in truncated_labels if label != -100] != [
        label for label in labels if label != -100
    ]:
        raise AssertionError("target truncation changed supervised tokens")
    return truncated_input_ids, truncated_labels


def tokenize_context_with_memory_slots(
    context_text: str,
    *,
    tokenizer: Any,
    model: Any,
    slot_to_tool: dict[int, str],
) -> list[int]:
    formatted = format_user_assistant_prompt(
        tokenizer,
        context_text,
        model=model,
    )
    token_ids: list[int] = []
    cursor = 0
    for match in SENTINEL_PATTERN.finditer(formatted):
        token_ids.extend(
            tokenizer(
                formatted[cursor : match.start()],
                add_special_tokens=False,
            )["input_ids"]
        )
        slot = int(match.group(1))
        tool_id = slot_to_tool.get(slot)
        if tool_id is None:
            raise ValueError(f"context references unknown memory slot {slot}")
        tool_token_id = model.get_tool_token_id(tool_id)
        if tool_token_id is None:
            raise ValueError(f"model has no token for manifest tool {tool_id}")
        token_ids.append(int(tool_token_id))
        cursor = match.end()
    token_ids.extend(
        tokenizer(formatted[cursor:], add_special_tokens=False)["input_ids"]
    )
    return token_ids


class ToolathlonStepDataset(Dataset):
    """One verified tool action per sample with observation-aware context."""

    def __init__(
        self,
        data_path: str | Path,
        manifest_path: str | Path,
        tokenizer: Any,
        model: Any,
        *,
        max_length: int = 4096,
        use_eoc: bool = False,
        mode: str = "train",
        observation_policy: ObservationPolicy = ObservationPolicy(),
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'")
        self.data = read_jsonl(data_path)
        self.manifest = load_manifest(manifest_path)
        self.tool_names = [record["stable_id"] for record in self.manifest["tools"]]
        self.tool_slots = {
            record["stable_id"]: int(record["memory_slot"])
            for record in self.manifest["tools"]
        }
        self.slot_to_tool = {slot: tool for tool, slot in self.tool_slots.items()}
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = int(max_length)
        self.use_eoc = bool(use_eoc)
        self.mode = mode
        self.observation_policy = observation_policy

        model_tool_names = list(getattr(model, "tool_names", []))
        if model_tool_names != self.tool_names:
            raise ValueError(
                "model tool_names must exactly match the ordered manifest stable IDs"
            )
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        known_tools = set(self.tool_names)
        records_by_id = {
            record["stable_id"]: record for record in self.manifest["tools"]
        }
        for record in self.data:
            if record.get("tool_manifest_hash") != self.manifest["manifest_hash"]:
                raise ValueError(
                    f"step {record.get('sample_id')} uses a different tool manifest"
                )
            available = record.get("available_tool_ids")
            if not isinstance(available, list) or not set(available) <= known_tools:
                raise ValueError(
                    f"step {record.get('sample_id')} has unknown available tools"
                )
            if record.get("target_tool_id") not in set(available):
                raise ValueError(
                    f"step {record.get('sample_id')} target is not available"
                )
            validate_tool_arguments(
                records_by_id[record["target_tool_id"]],
                record.get("target_arguments"),
            )

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize_context(self, context_text: str) -> list[int]:
        return tokenize_context_with_memory_slots(
            context_text,
            tokenizer=self.tokenizer,
            model=self.model,
            slot_to_tool=self.slot_to_tool,
        )

    def _eoc_token_id(self) -> int:
        if hasattr(self.model, "get_eoc_token_id"):
            token_id = self.model.get_eoc_token_id()
        else:
            token_id = getattr(self.model, "eoc_token_id", None)
        if token_id is None:
            raise ValueError("use_eoc=True requires an EOC token on the model")
        return int(token_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        step = self.data[index]
        context_text = render_step_context(
            step,
            self.tool_slots,
            observation_policy=self.observation_policy,
        )
        context_ids = self._tokenize_context(context_text)
        input_ids = list(context_ids)
        labels = [-100] * len(context_ids)

        if self.mode == "train":
            target_tool = step["target_tool_id"]
            tool_token_id = self.model.get_tool_token_id(target_tool)
            if tool_token_id is None:
                raise ValueError(f"model has no token for target tool {target_tool}")
            argument_ids = self.tokenizer(
                canonical_json(step.get("target_arguments", {})),
                add_special_tokens=False,
            )["input_ids"]
            target_ids = [int(tool_token_id), *argument_ids]
            if self.use_eoc:
                target_ids.append(self._eoc_token_id())
            target_ids.extend(response_end_token_ids(self.tokenizer, model=self.model))
            input_ids.extend(target_ids)
            labels.extend(target_ids)

        input_ids, labels = truncate_preserving_supervised_target(
            input_ids,
            labels,
            self.max_length,
        )

        available_mask = build_available_tool_mask(
            self.tool_names,
            step["available_tool_ids"],
        )
        target_position = self.tool_names.index(step["target_tool_id"])
        if not bool(available_mask[target_position]):
            raise ValueError("target tool is not available for this sample")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "available_tool_mask": available_mask,
            "episode_weight": torch.tensor(
                1.0,
                dtype=torch.float32,
            ),
            "raw_data": step,
        }


class EpisodeBalancedSampler(Sampler[int]):
    """Sample episodes uniformly while cycling through each episode's steps."""

    def __init__(self, records: list[dict[str, Any]], seed: int = 42) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("every step requires a non-empty episode_id")
            grouped[episode_id].append(index)
        if not grouped:
            raise ValueError("episode-balanced sampler requires at least one step")
        self.groups = dict(sorted(grouped.items()))
        self.seed = int(seed)
        self.epoch = 0
        self.sample_count = len(records)

    def __len__(self) -> int:
        return self.sample_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        episode_ids = list(self.groups)
        queues = {episode_id: list(indices) for episode_id, indices in self.groups.items()}
        positions = {episode_id: 0 for episode_id in episode_ids}
        for queue in queues.values():
            rng.shuffle(queue)

        yielded = 0
        while yielded < self.sample_count:
            rng.shuffle(episode_ids)
            for episode_id in episode_ids:
                if yielded >= self.sample_count:
                    break
                queue = queues[episode_id]
                position = positions[episode_id]
                if position >= len(queue):
                    rng.shuffle(queue)
                    position = 0
                yield queue[position]
                positions[episode_id] = position + 1
                yielded += 1


def collate_step_batch(batch: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    def left_pad(values: list[torch.Tensor], fill: int | float) -> torch.Tensor:
        width = max(value.numel() for value in values)
        result = torch.full(
            (len(values), width),
            fill_value=fill,
            dtype=values[0].dtype,
        )
        for row, value in enumerate(values):
            result[row, width - value.numel() :] = value
        return result

    return {
        "input_ids": left_pad(
            [item["input_ids"] for item in batch],
            tokenizer.pad_token_id,
        ),
        "attention_mask": left_pad(
            [item["attention_mask"] for item in batch],
            0,
        ),
        "labels": left_pad([item["labels"] for item in batch], -100),
        "available_tool_mask": torch.stack(
            [item["available_tool_mask"] for item in batch]
        ),
        "episode_weight": torch.stack([item["episode_weight"] for item in batch]),
        "raw_data": [item["raw_data"] for item in batch],
    }


def create_step_dataloader(
    data_path: str | Path,
    manifest_path: str | Path,
    tokenizer: Any,
    model: Any,
    *,
    batch_size: int,
    max_length: int,
    use_eoc: bool,
    mode: str = "train",
    shuffle: bool = False,
    balance_by_episode: bool = True,
    sampler_seed: int = 42,
) -> DataLoader:
    dataset = ToolathlonStepDataset(
        data_path,
        manifest_path,
        tokenizer,
        model,
        max_length=max_length,
        use_eoc=use_eoc,
        mode=mode,
    )
    sampler = None
    if mode == "train" and balance_by_episode:
        if shuffle:
            raise ValueError("shuffle and balance_by_episode cannot both be enabled")
        sampler = EpisodeBalancedSampler(dataset.data, seed=sampler_seed)
    shuffle_generator = None
    if shuffle:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(int(sampler_seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=shuffle_generator,
        collate_fn=lambda batch: collate_step_batch(batch, tokenizer),
    )
