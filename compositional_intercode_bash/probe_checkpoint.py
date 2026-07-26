"""Measure a partial TapMem checkpoint on disjoint TRAIN template groups."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .io_utils import write_json
from .route_probe_split import (
    checkpoint_split_record,
    load_route_probe_manifest,
)
from .training_data import (
    BoundaryViewDataset,
    add_train_routing_bias,
    gather_routing_sites,
    left_pad_collate,
)


FIXED_MEMORY_BANK_THRESHOLD = 0.5


@dataclass
class ProbeMetricSums:
    ar_loss_sum: float = 0.0
    supervised_tokens: int = 0
    route_loss_sum: float = 0.0
    route_sites: int = 0
    outside_memory_sites: int = 0
    outside_memory_triggers: int = 0
    outside_ordinary_sites: int = 0
    outside_ordinary_triggers: int = 0
    triggered_memory_bank_top1_correct: int = 0

    def add(self, other: "ProbeMetricSums") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(
                self,
                field_name,
                getattr(self, field_name) + getattr(other, field_name),
            )

    def report(self) -> dict[str, float | int | None]:
        return {
            "supervised_tokens": self.supervised_tokens,
            "route_sites": self.route_sites,
            "ar_nll": (
                self.ar_loss_sum / self.supervised_tokens
                if self.supervised_tokens
                else None
            ),
            "route_nll": (
                self.route_loss_sum / self.route_sites
                if self.route_sites
                else None
            ),
            "gold_outside_memory_sites": self.outside_memory_sites,
            "gold_outside_memory_triggers": self.outside_memory_triggers,
            "gold_outside_trigger_recall": (
                self.outside_memory_triggers / self.outside_memory_sites
                if self.outside_memory_sites
                else None
            ),
            "gold_outside_ordinary_sites": self.outside_ordinary_sites,
            "gold_outside_ordinary_false_triggers": (
                self.outside_ordinary_triggers
            ),
            "gold_outside_ordinary_false_trigger_rate": (
                self.outside_ordinary_triggers / self.outside_ordinary_sites
                if self.outside_ordinary_sites
                else None
            ),
            "triggered_memory_bank_top1_correct": (
                self.triggered_memory_bank_top1_correct
            ),
            "triggered_tcra_bank_top1_accuracy": (
                self.triggered_memory_bank_top1_correct
                / self.outside_memory_triggers
                if self.outside_memory_triggers
                else None
            ),
        }


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _batch_probe_metrics(
    model,
    batch: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, ProbeMetricSums]:
    """One teacher-forced pass with metrics allocated to boundary pools."""

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    raw_shift_logits = output.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    sites = gather_routing_sites(labels, model.registry)
    boundary_hidden = output.final_hidden_state[
        sites.batch_indices,
        sites.time_indices,
    ]
    route_logits = model.routing_scores(boundary_hidden.detach())
    route_losses = F.cross_entropy(
        route_logits.float(),
        sites.targets,
        reduction="none",
    )
    biased_shift_logits = add_train_routing_bias(
        model,
        raw_shift_logits,
        boundary_hidden,
        sites,
        detach_hidden=True,
    )
    ar_losses = F.cross_entropy(
        biased_shift_logits.float().view(-1, biased_shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    pools = [str(value) for value in batch["pools"]]
    sums = {"A": ProbeMetricSums(), "R": ProbeMetricSums()}
    for batch_index, pool in enumerate(pools):
        valid = shift_labels[batch_index] != -100
        sums[pool].ar_loss_sum += float(
            ar_losses[batch_index][valid].sum().item()
        )
        sums[pool].supervised_tokens += int(valid.sum().item())
    for site_index, batch_index_tensor in enumerate(sites.batch_indices):
        pool = pools[int(batch_index_tensor.item())]
        sums[pool].route_loss_sum += float(route_losses[site_index].item())
        sums[pool].route_sites += 1

    procedure_ids = {
        int(token_id): piece_id
        for piece_id, token_id in enumerate(model.registry.procedure_token_ids)
    }
    memory_token_ids = model._procedure_token_ids
    if memory_token_ids.device != raw_shift_logits.device:
        memory_token_ids = memory_token_ids.to(raw_shift_logits.device)
    threshold = float(model.memory_bank_probability_threshold)
    for batch_index, pool in enumerate(pools):
        inside_procedure = False
        valid_target_positions = torch.nonzero(
            labels[batch_index] != -100,
            as_tuple=False,
        ).flatten()
        for position_tensor in valid_target_positions:
            target_position = int(position_tensor.item())
            if target_position == 0:
                raise ValueError("Probe target has no causal prediction position")
            target_token = int(labels[batch_index, target_position].item())
            target_piece_id = procedure_ids.get(target_token)
            if not inside_procedure:
                prediction_time = target_position - 1
                full_logits = raw_shift_logits[
                    batch_index,
                    prediction_time,
                ].float()
                memory_logits = full_logits.index_select(0, memory_token_ids)
                memory_mass = torch.exp(
                    torch.logsumexp(memory_logits, dim=0)
                    - torch.logsumexp(full_logits, dim=0)
                )
                triggered = bool(memory_mass.item() >= threshold)
                if target_piece_id is not None:
                    sums[pool].outside_memory_sites += 1
                    sums[pool].outside_memory_triggers += int(triggered)
                    if triggered:
                        post_tcra_memory_logits = biased_shift_logits[
                            batch_index,
                            prediction_time,
                        ].index_select(0, memory_token_ids)
                        predicted_piece_id = int(
                            torch.argmax(post_tcra_memory_logits).item()
                        )
                        sums[pool].triggered_memory_bank_top1_correct += int(
                            predicted_piece_id == target_piece_id
                        )
                else:
                    sums[pool].outside_ordinary_sites += 1
                    sums[pool].outside_ordinary_triggers += int(triggered)

            if not inside_procedure and target_piece_id is not None:
                inside_procedure = True
            elif (
                inside_procedure
                and target_token == model.registry.eoc_token_id
            ):
                inside_procedure = False
    return sums


def _validate_checkpoint_split(
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if metadata.get("formal_ready") is not False:
        raise ValueError("Routing-LR probes require an exploratory partial checkpoint")
    summary = metadata.get("training_summary", {})
    recorded = summary.get("route_probe_split")
    expected = checkpoint_split_record(manifest)
    if recorded != expected:
        raise ValueError(
            "Checkpoint routing-LR split identity differs from the probe manifest"
        )
    if int(summary.get("run_total_updates", -1)) != int(
        manifest["fit_updates"]
    ):
        raise ValueError("Checkpoint update count differs from the fit manifest")
    source_updates = int(manifest["source_schedule_updates"])
    if int(summary.get("scheduler_total_updates", -1)) != source_updates:
        raise ValueError(
            "Checkpoint scheduler length differs from the full source schedule"
        )
    if int(summary.get("warmup_steps", -1)) != source_updates // 10:
        raise ValueError(
            "Checkpoint warmup differs from the full source schedule"
        )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    manifest, _fit_views, probe_views = load_route_probe_manifest(
        args.split_manifest,
        views_path=args.views,
    )
    model, tokenizer, metadata = load_checkpoint(
        args.checkpoint,
        device=args.device,
        dtype=_dtype(args.dtype),
        local_files_only=True,
    )
    try:
        if metadata["method"] != "tapmem":
            raise ValueError("Routing probes require a TapMem checkpoint")
        _validate_checkpoint_split(metadata, manifest)
        threshold = float(model.memory_bank_probability_threshold)
        if threshold != FIXED_MEMORY_BANK_THRESHOLD:
            raise ValueError(
                "Routing-LR comparison fixes the TapMem memory-bank threshold "
                f"at {FIXED_MEMORY_BANK_THRESHOLD}, got {threshold}"
            )
        dataset = BoundaryViewDataset(
            probe_views,
            tokenizer,
            model.registry,
            method="tapmem",
            max_length=args.max_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda rows: left_pad_collate(
                rows,
                pad_token_id=int(tokenizer.pad_token_id),
            ),
        )
        model.eval()
        model.base_model.eval()
        by_pool = {"A": ProbeMetricSums(), "R": ProbeMetricSums()}
        with torch.no_grad():
            for batch in loader:
                batch_sums = _batch_probe_metrics(
                    model,
                    batch,
                    device=args.device,
                )
                for pool in ("A", "R"):
                    by_pool[pool].add(batch_sums[pool])
        overall = ProbeMetricSums()
        for pool in ("A", "R"):
            overall.add(by_pool[pool])
        if overall.route_sites <= 0:
            raise ValueError("Probe views contain no routing sites")
        row_norms = model.routing_head.weight.detach().float().norm(dim=1)
        result: dict[str, object] = {
            "schema": "intercode_bash_route_lr_probe_v2",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "views": str(Path(args.views).resolve()),
            "split_manifest": str(Path(args.split_manifest).resolve()),
            "split_sha256": manifest["split_sha256"],
            "memory_bank_probability_threshold": threshold,
            "presentations": len(probe_views),
            "template_groups": len(probe_views),
            "presentations_by_pool": manifest["probe_presentations_by_pool"],
            "metrics": overall.report(),
            "metrics_by_pool": {
                pool: by_pool[pool].report() for pool in ("A", "R")
            },
            "routing_head_row_norm_mean": float(row_norms.mean().item()),
            "routing_head_row_norm_max": float(row_norms.max().item()),
            "memory_learning_rate": metadata["training_config"]["learning_rate"],
            "routing_learning_rate": metadata["training_config"][
                "routing_learning_rate"
            ],
        }
        write_json(args.output, result)
        return result
    finally:
        model.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    result = run_probe(parse_args(argv))
    print(result)


if __name__ == "__main__":
    main()
