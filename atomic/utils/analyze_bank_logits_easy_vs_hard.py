#!/usr/bin/env python3
"""Analyze best-checkpoint bank-only routing logits for easy vs hard tasks.

This utility loads an archived atomic run, restores the best task-token
checkpoint, replays the test split in teacher-forced task-token format, and
collects bank-only routing statistics for every routed task-token position.

Outputs are written into the run directory by default:
- bank_margin_analysis.json
- bank_margin_analysis.md
- bank_margin_easy_vs_hard_margin.png
- bank_margin_easy_vs_hard_prob.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


SCRIPT_PATH = Path(__file__).resolve()
ATOMIC_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = ATOMIC_DIR.parent

for path in (str(ATOMIC_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from main_in_domain_fixed_split import add_reserved_special_tokens, load_split_cache  # noqa: E402
from main_base_model_fixed_split import summarize_metrics  # noqa: E402
from natural_instructions_eval import evaluate_predictions  # noqa: E402
from run_layout import write_json  # noqa: E402
from task_dataset import NaturalInstructionsTaskDataset, collate_fn  # noqa: E402
from task_model import TaskCallingModel  # noqa: E402
from task_training import get_memory_bank_embeddings  # noqa: E402


@dataclass
class RoutingRecord:
    task_name: str
    task_accuracy: float
    task_group: str
    example_index: int
    routed_position: int
    positive_logit: float
    hardest_negative_logit: float
    logit_margin: float
    positive_prob: float
    hardest_negative_prob: float
    predicted_task_id: int
    target_task_id: int
    hardest_negative_task_id: int


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: True or False")


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def choose_checkpoint(run_summary: Dict, run_dir: Path) -> Path:
    candidate_keys = [
        "best_task_tokens_path",
        "task_tokens_path",
        "best_model_path",
        "final_task_tokens_path",
    ]
    for key in candidate_keys:
        candidate = run_summary.get(key)
        if candidate:
            resolved = resolve_path(run_dir, candidate)
            if resolved and resolved.exists():
                return resolved

    candidates = sorted(run_dir.glob("task_tokens_*_best.pt"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find a best task-token checkpoint in {run_dir}. "
        "Expected best_task_tokens_path in run_summary.json or a task_tokens_*_best.pt file."
    )


def load_run_metadata(run_dir: Path) -> Tuple[Dict, Dict]:
    run_config_path = run_dir / "run_config.json"
    run_summary_path = run_dir / "run_summary.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {run_dir}")
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json in {run_dir}")
    return load_json(run_config_path), load_json(run_summary_path)


def build_eval_args(run_config: Dict) -> SimpleNamespace:
    args = run_config.get("args", {})
    return SimpleNamespace(
        tasks_dir=args.get("tasks_dir"),
        model_name=args.get("model_name"),
        num_tasks=int(args.get("num_tasks", 0)),
        train_size=int(args.get("train_size", 500)),
        val_size=int(args.get("val_size", 10)),
        test_size=int(args.get("test_size", 50)),
        few_shot=bool(args.get("few_shot", False)),
        seed=int(args.get("seed", 42)),
        split_cache_path=run_config.get("split_cache_path") or args.get("split_cache_path"),
        device_map=args.get("device_map"),
        decouple_embeddings=bool(args.get("decouple_embeddings", False)),
        max_length=int(args.get("max_length", 1024)),
        batch_size=int(args.get("test_batch_size", 400)),
        test_batch_size=int(args.get("test_batch_size", 400)),
        include_instruction_in_prompt=True,
    )


def build_data_loader(test_data, model, tokenizer, max_length, test_batch_size):
    test_dataset = NaturalInstructionsTaskDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        model=model,
        mode="train",
        include_instruction_in_prompt=True,
    )
    return DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )


def get_task_group(task_name: str, task_accuracy: float, easy_threshold: float, hard_threshold: float) -> str:
    if task_accuracy >= easy_threshold:
        return "easy"
    if task_accuracy < hard_threshold:
        return "hard"
    return "middle"


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def compute_bank_logits_for_batch(model, batch, example_offset: int):
    input_device = model.get_input_device()
    output_device = model.get_output_device()
    input_ids = batch["input_ids"].to(input_device)
    attention_mask = batch["attention_mask"].to(input_device)
    labels = batch["labels"].to(output_device)

    logits, hidden_states = model(
        input_ids,
        attention_mask,
        return_hidden_states=True,
    )

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_hidden_states = hidden_states[..., :-1, :].contiguous()

    shift_logits = shift_logits.to(dtype=torch.float32)
    shift_hidden_states = shift_hidden_states.to(dtype=torch.float32)
    shift_labels = shift_labels.to(device=shift_hidden_states.device)
    reserved_token_ids = model.get_reserved_token_tensor(shift_hidden_states.device)
    routing_mask = (shift_labels != -100) & torch.isin(shift_labels, reserved_token_ids)

    bank_embeddings = get_memory_bank_embeddings(model).detach().to(dtype=torch.float32)
    normalized_bank = F.normalize(bank_embeddings, p=2, dim=-1, eps=1e-12)

    batch_records: List[RoutingRecord] = []

    for batch_index in range(shift_labels.shape[0]):
        positions = torch.nonzero(routing_mask[batch_index], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue

        task_name = batch["raw_data"][batch_index].get("tasks", ["unknown"])[0]
        for pos in positions.tolist():
            hidden = shift_hidden_states[batch_index, pos].unsqueeze(0)
            normalized_hidden = F.normalize(hidden, p=2, dim=-1, eps=1e-12)
            bank_logits = torch.matmul(normalized_hidden, normalized_bank.transpose(0, 1)).squeeze(0)
            bank_probs = torch.softmax(bank_logits, dim=-1)

            target_token_id = int(shift_labels[batch_index, pos].item())
            target_task_id = int(torch.searchsorted(reserved_token_ids, torch.tensor(target_token_id, device=reserved_token_ids.device)).item())
            positive_logit = float(bank_logits[target_task_id].item())
            positive_prob = float(bank_probs[target_task_id].item())

            negative_mask = torch.ones_like(bank_logits, dtype=torch.bool)
            negative_mask[target_task_id] = False
            hardest_negative_task_id = int(torch.argmax(bank_logits.masked_fill(~negative_mask, float("-inf"))).item())
            hardest_negative_logit = float(bank_logits[hardest_negative_task_id].item())
            hardest_negative_prob = float(bank_probs[hardest_negative_task_id].item())
            logit_margin = positive_logit - hardest_negative_logit

            predicted_task_id = int(torch.argmax(bank_logits).item())

            batch_records.append(
                RoutingRecord(
                    task_name=task_name,
                    task_accuracy=0.0,
                    task_group="middle",
                    example_index=int(example_offset + batch_index),
                    routed_position=int(pos),
                    positive_logit=positive_logit,
                    hardest_negative_logit=hardest_negative_logit,
                    logit_margin=logit_margin,
                    positive_prob=positive_prob,
                    hardest_negative_prob=hardest_negative_prob,
                    predicted_task_id=predicted_task_id,
                    target_task_id=target_task_id,
                    hardest_negative_task_id=hardest_negative_task_id,
                )
            )

    return batch_records


def compute_analysis(
    model,
    dataloader,
    task_groups: Dict[str, str],
    task_accuracies: Dict[str, float],
    max_batches: Optional[int] = None,
) -> List[RoutingRecord]:
    records: List[RoutingRecord] = []
    model.eval()
    example_offset = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch_records = compute_bank_logits_for_batch(model, batch, example_offset=example_offset)
            for record in batch_records:
                record.task_accuracy = float(task_accuracies.get(record.task_name, 0.0))
                record.task_group = task_groups.get(record.task_name, "middle")
            records.extend(batch_records)
            example_offset += len(batch["raw_data"])

    return records


def records_to_dicts(records: Sequence[RoutingRecord]) -> List[Dict]:
    return [asdict(record) for record in records]


def group_records(records: Sequence[RoutingRecord]) -> Dict[str, List[RoutingRecord]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.task_group].append(record)
    return dict(grouped)


def compute_task_stats(records: Sequence[RoutingRecord]) -> Dict[str, Dict]:
    task_groups = defaultdict(list)
    for record in records:
        task_groups[record.task_name].append(record)

    task_stats = {}
    for task_name, task_records in task_groups.items():
        task_stats[task_name] = {
            "task_group": task_records[0].task_group if task_records else "middle",
            "task_accuracy": task_records[0].task_accuracy if task_records else 0.0,
            "num_routed_positions": len(task_records),
            "positive_logit": summarize([r.positive_logit for r in task_records]),
            "hardest_negative_logit": summarize([r.hardest_negative_logit for r in task_records]),
            "logit_margin": summarize([r.logit_margin for r in task_records]),
            "positive_prob": summarize([r.positive_prob for r in task_records]),
            "hardest_negative_prob": summarize([r.hardest_negative_prob for r in task_records]),
        }
    return task_stats


def compute_group_stats(records: Sequence[RoutingRecord]) -> Dict[str, Dict]:
    grouped = group_records(records)
    group_stats = {}
    for group_name in ["easy", "hard", "middle"]:
        group_records_list = grouped.get(group_name, [])
        group_stats[group_name] = {
            "num_records": len(group_records_list),
            "task_count": len({r.task_name for r in group_records_list}),
            "positive_logit": summarize([r.positive_logit for r in group_records_list]),
            "hardest_negative_logit": summarize([r.hardest_negative_logit for r in group_records_list]),
            "logit_margin": summarize([r.logit_margin for r in group_records_list]),
            "positive_prob": summarize([r.positive_prob for r in group_records_list]),
            "hardest_negative_prob": summarize([r.hardest_negative_prob for r in group_records_list]),
        }
    return group_stats


def _value_to_hex(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    channel = int(round(255 * (1.0 - value)))
    return f"#{channel:02x}{channel:02x}ff"


def _histogram(values_by_group: Dict[str, Sequence[float]], bins: int = 24):
    groups = ["hard", "easy"]
    all_values = [value for values in values_by_group.values() for value in values]
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        if math.isclose(min_val, max_val):
            max_val = min_val + 1e-6
    else:
        min_val, max_val = 0.0, 1.0

    histograms = {}
    max_count = 0
    for group in groups:
        values = np.asarray(values_by_group.get(group, []), dtype=np.float64)
        if values.size == 0:
            hist = np.zeros(bins, dtype=np.int64)
        else:
            hist, _ = np.histogram(values, bins=bins, range=(min_val, max_val))
        histograms[group] = hist
        max_count = max(max_count, int(hist.max()) if hist.size > 0 else 0)

    return histograms, (float(min_val), float(max_val)), max_count


def _stat_lines(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_footer(parts: List[str], path: Path) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _draw_hist_panel(
    parts: List[str],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    group_name: str,
    hist: np.ndarray,
    value_range: Tuple[float, float],
    max_count: int,
    stat_values: Dict[str, float],
) -> None:
    axis_left = x + 56
    axis_right = x + width - 20
    axis_top = y + 34
    axis_bottom = y + height - 44
    plot_width = axis_right - axis_left
    plot_height = axis_bottom - axis_top
    bins = len(hist)
    bin_width = plot_width / max(bins, 1)

    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + 18}" text-anchor="middle" font-size="16" font-family="monospace">{title}</text>')
    parts.append(f'<text x="{x + 10}" y="{y + 18}" text-anchor="start" font-size="13" font-family="monospace">{group_name}</text>')

    parts.append(f'<line x1="{axis_left}" y1="{axis_bottom}" x2="{axis_right}" y2="{axis_bottom}" stroke="#4a4a4a" stroke-width="1"/>')
    parts.append(f'<line x1="{axis_left}" y1="{axis_top}" x2="{axis_left}" y2="{axis_bottom}" stroke="#4a4a4a" stroke-width="1"/>')

    for tick_index in range(5):
        y_value = axis_bottom - tick_index * (plot_height / 4)
        count_value = (max_count * tick_index / 4) if max_count > 0 else 0
        parts.append(f'<line x1="{axis_left - 4}" y1="{y_value:.1f}" x2="{axis_left}" y2="{y_value:.1f}" stroke="#4a4a4a" stroke-width="1"/>')
        parts.append(f'<text x="{axis_left - 8}" y="{y_value + 4:.1f}" text-anchor="end" font-size="11" font-family="monospace">{count_value:.0f}</text>')

    min_val, max_val = value_range
    for tick_index in range(5):
        x_value = axis_left + tick_index * (plot_width / 4)
        label_value = min_val + (max_val - min_val) * tick_index / 4
        parts.append(f'<line x1="{x_value:.1f}" y1="{axis_bottom}" x2="{x_value:.1f}" y2="{axis_bottom + 4}" stroke="#4a4a4a" stroke-width="1"/>')
        parts.append(f'<text x="{x_value:.1f}" y="{axis_bottom + 18}" text-anchor="middle" font-size="11" font-family="monospace">{label_value:.3f}</text>')

    if bins > 0:
        for index, count in enumerate(hist):
            x0 = axis_left + index * bin_width + 1
            bar_width = max(bin_width - 2, 1)
            normalized = (count / max_count) if max_count > 0 else 0.0
            bar_height = normalized * plot_height
            y0 = axis_bottom - bar_height
            parts.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{_value_to_hex(normalized)}" stroke="#8a8aff" stroke-width="0.3"/>'
            )

    if max_val > min_val:
        for name, color in (("mean", "#1f3fff"), ("median", "#5a7cff")):
            value = stat_values[name]
            x_value = axis_left + (value - min_val) / (max_val - min_val) * plot_width
            parts.append(
                f'<line x1="{x_value:.1f}" y1="{axis_top}" x2="{x_value:.1f}" y2="{axis_bottom}" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>'
            )

    legend_y = y + height - 10
    parts.append(f'<text x="{axis_left}" y="{legend_y}" font-size="11" font-family="monospace" fill="#1f3fff">Mean: {stat_values["mean"]:.4f}</text>')
    parts.append(f'<text x="{axis_left + 210}" y="{legend_y}" font-size="11" font-family="monospace" fill="#5a7cff">Median: {stat_values["median"]:.4f}</text>')


def plot_group_histograms_svg(values_by_group: Dict[str, Sequence[float]], metric_label: str, output_path: Path) -> None:
    histograms, value_range, max_count = _histogram(values_by_group)
    width = 1100
    panel_height = 290
    height = 80 + panel_height * 2
    parts = _svg_header(width, height)
    parts.append(
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="monospace">'
        f'Easy vs Hard Task {metric_label}</text>'
    )

    for panel_index, group in enumerate(("hard", "easy")):
        panel_y = 55 + panel_index * panel_height
        _draw_hist_panel(
            parts,
            x=30,
            y=panel_y,
            width=width - 60,
            height=panel_height - 10,
            title=f"{group} distribution",
            group_name=group,
            hist=histograms[group],
            value_range=value_range,
            max_count=max_count,
            stat_values=_stat_lines(values_by_group.get(group, [])),
        )

    _svg_footer(parts, output_path)


def plot_probability_histograms_svg(
    positive_values_by_group: Dict[str, Sequence[float]],
    negative_values_by_group: Dict[str, Sequence[float]],
    output_path: Path,
) -> None:
    width = 2200
    height = 660
    parts = _svg_header(width, height)
    parts.append(
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="monospace">'
        'Positive Probability vs Hardest-Negative Probability</text>'
    )

    left_origin = 30
    right_origin = width // 2 + 10
    panel_width = width // 2 - 40
    panel_height = 290

    pos_histograms, pos_range, pos_max = _histogram(positive_values_by_group)
    neg_histograms, neg_range, neg_max = _histogram(negative_values_by_group)

    for panel_index, group in enumerate(("hard", "easy")):
        panel_y = 55 + panel_index * panel_height
        _draw_hist_panel(
            parts,
            x=left_origin,
            y=panel_y,
            width=panel_width,
            height=panel_height - 10,
            title="Positive Probability",
            group_name=group,
            hist=pos_histograms[group],
            value_range=pos_range,
            max_count=pos_max,
            stat_values=_stat_lines(positive_values_by_group.get(group, [])),
        )
        _draw_hist_panel(
            parts,
            x=right_origin,
            y=panel_y,
            width=panel_width,
            height=panel_height - 10,
            title="Hardest-Negative Probability",
            group_name=group,
            hist=neg_histograms[group],
            value_range=neg_range,
            max_count=neg_max,
            stat_values=_stat_lines(negative_values_by_group.get(group, [])),
        )

    _svg_footer(parts, output_path)


def build_markdown_report(
    run_dir: Path,
    run_config: Dict,
    run_summary: Dict,
    analysis_payload: Dict,
) -> str:
    group_stats = analysis_payload["group_stats"]
    overall = analysis_payload["overall"]
    task_groups = analysis_payload["task_groups"]

    lines = []
    lines.append("# Bank Logit 分析")
    lines.append("")
    lines.append(f"- Run 目录：`{run_dir}`")
    lines.append(f"- checkpoint：`{analysis_payload['checkpoint_path']}`")
    lines.append(f"- easy 阈值：`{analysis_payload['easy_threshold']}`")
    lines.append(f"- hard 阈值：`{analysis_payload['hard_threshold']}`")
    lines.append(f"- partial analysis：`{analysis_payload['partial_analysis']}`")
    if analysis_payload["max_batches"] is not None:
        lines.append(f"- max_batches：`{analysis_payload['max_batches']}`")
    lines.append("")
    lines.append("## 总体")
    lines.append("")
    lines.append(f"- routed positions: `{overall['num_records']}`")
    lines.append(f"- tasks covered: `{overall['task_count']}`")
    lines.append(f"- logit margin mean: `{overall['logit_margin']['mean']:.4f}`")
    lines.append(f"- logit margin median: `{overall['logit_margin']['median']:.4f}`")
    lines.append(f"- positive prob mean: `{overall['positive_prob']['mean']:.4f}`")
    lines.append(f"- hardest negative prob mean: `{overall['hardest_negative_prob']['mean']:.4f}`")
    lines.append("")
    lines.append("## easy vs hard")
    lines.append("")
    lines.append("| group | records | tasks | margin mean | margin median | pos prob mean | neg prob mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in ["easy", "hard", "middle"]:
        stats = group_stats[group]
        lines.append(
            f"| {group} | {stats['num_records']} | {stats['task_count']} | "
            f"{stats['logit_margin']['mean']:.4f} | {stats['logit_margin']['median']:.4f} | "
            f"{stats['positive_prob']['mean']:.4f} | {stats['hardest_negative_prob']['mean']:.4f} |"
        )
    lines.append("")
    lines.append("## 任务分组")
    lines.append("")
    lines.append(f"- easy tasks: `{len(task_groups['easy'])}`")
    lines.append(f"- hard tasks: `{len(task_groups['hard'])}`")
    lines.append(f"- middle tasks: `{len(task_groups['middle'])}`")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "- 这里的 margin 定义为 `positive_logit - hardest_negative_logit`，"
        "与训练中的 hard-negative margin 保持一致。"
    )
    lines.append(
        "- 这个分析使用 test split 的 teacher-forced task-token 位置，因此可直接对比 "
        "easy / hard task 的 routing 分布。"
    )
    lines.append("")
    lines.append("## 运行摘要")
    lines.append("")
    lines.append(f"- run_name: `{run_summary.get('run_name')}`")
    lines.append(f"- model_name: `{run_config.get('args', {}).get('model_name')}`")
    lines.append(f"- split_cache_path: `{run_summary.get('split_cache_path', run_config.get('split_cache_path'))}`")
    lines.append("")
    return "\n".join(lines)


def compute_task_groups_from_evaluation(evaluation_results: Dict, easy_threshold: float, hard_threshold: float):
    task_breakdown = evaluation_results.get("task_breakdown", {})
    task_accuracies = {}
    task_groups = {}

    for task_name, stats in task_breakdown.items():
        total = float(stats.get("total", 0))
        task_correct = float(stats.get("task_correct", 0))
        task_accuracy = task_correct / total if total > 0 else 0.0
        task_accuracies[task_name] = task_accuracy
        if task_accuracy >= easy_threshold:
            task_groups[task_name] = "easy"
        elif task_accuracy < hard_threshold:
            task_groups[task_name] = "hard"
        else:
            task_groups[task_name] = "middle"

    return task_accuracies, task_groups


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best-checkpoint bank-only routing logits for easy vs hard tasks."
    )
    parser.add_argument("run_dir", type=str, help="Run directory containing run_config.json and run_summary.json")
    parser.add_argument(
        "--easy-threshold",
        type=float,
        default=0.98,
        help="Task accuracy threshold for easy tasks (inclusive)",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.9,
        help="Task accuracy threshold for hard tasks (exclusive)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to the run directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for the analysis dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-task-limit",
        type=int,
        default=25,
        help="How many tasks to list per group in the markdown summary.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional batch cap for quick smoke tests.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config, run_summary = load_run_metadata(run_dir)
    eval_args = build_eval_args(run_config)
    checkpoint_path = choose_checkpoint(run_summary, run_dir)
    evaluation_results_path = resolve_path(run_dir, run_summary.get("evaluation_results_path")) or (run_dir / "evaluation_results.json")
    if not evaluation_results_path.exists():
        raise FileNotFoundError(f"Missing evaluation results: {evaluation_results_path}")
    evaluation_results = load_json(evaluation_results_path)

    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer, is_extended = add_reserved_special_tokens(tokenizer, eval_args.num_tasks)

    split_cache_path = resolve_path(run_dir, run_config.get("split_cache_path") or eval_args.split_cache_path)
    if split_cache_path is None or not split_cache_path.exists():
        raise FileNotFoundError(f"Split cache not found: {split_cache_path}")
    eval_args.split_cache_path = str(split_cache_path)

    train_data, val_data, test_data, task_names, split_cache_metadata = load_split_cache(eval_args)

    model = TaskCallingModel(
        model_name=eval_args.model_name,
        num_tasks=len(task_names),
        task_names=task_names,
        tokenizer=tokenizer,
        device=args.device,
        decouple_embeddings=eval_args.decouple_embeddings,
        is_extended=is_extended,
        device_map=eval_args.device_map,
        generation_routing=run_config.get("args", {}).get("generation_routing", "full_vocab_generation"),
    )
    model.load_task_tokens(str(checkpoint_path))
    model.eval()

    test_dataloader = build_data_loader(
        test_data=test_data,
        model=model,
        tokenizer=tokenizer,
        max_length=eval_args.max_length,
        test_batch_size=args.batch_size or eval_args.test_batch_size,
    )

    task_accuracies, task_groups = compute_task_groups_from_evaluation(
        evaluation_results,
        easy_threshold=args.easy_threshold,
        hard_threshold=args.hard_threshold,
    )

    records = compute_analysis(
        model,
        test_dataloader,
        task_groups,
        task_accuracies,
        max_batches=args.max_batches,
    )

    overall = {
        "num_records": len(records),
        "task_count": len({r.task_name for r in records}),
        "positive_logit": summarize([r.positive_logit for r in records]),
        "hardest_negative_logit": summarize([r.hardest_negative_logit for r in records]),
        "logit_margin": summarize([r.logit_margin for r in records]),
        "positive_prob": summarize([r.positive_prob for r in records]),
        "hardest_negative_prob": summarize([r.hardest_negative_prob for r in records]),
    }
    group_stats = compute_group_stats(records)
    task_stats = compute_task_stats(records)

    grouped = group_records(records)
    plot_group_histograms_svg(
        {k: [r.logit_margin for r in grouped.get(k, [])] for k in ("easy", "hard")},
        metric_label="Logit Margin",
        output_path=output_dir / "bank_margin_easy_vs_hard_margin.svg",
    )
    plot_probability_histograms_svg(
        positive_values_by_group={k: [r.positive_prob for r in grouped.get(k, [])] for k in ("easy", "hard")},
        negative_values_by_group={k: [r.hardest_negative_prob for r in grouped.get(k, [])] for k in ("easy", "hard")},
        output_path=output_dir / "bank_margin_easy_vs_hard_prob.svg",
    )

    task_ranked = sorted(
        task_stats.items(),
        key=lambda item: (item[1]["task_accuracy"], item[1]["logit_margin"]["mean"]),
        reverse=True,
    )
    task_ranked_by_hard = sorted(
        task_stats.items(),
        key=lambda item: (item[1]["task_accuracy"], item[1]["logit_margin"]["mean"]),
        reverse=False,
    )

    analysis_payload = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "split_cache_path": str(split_cache_path),
        "evaluation_results_path": str(evaluation_results_path),
        "easy_threshold": args.easy_threshold,
        "hard_threshold": args.hard_threshold,
        "max_batches": args.max_batches,
        "partial_analysis": args.max_batches is not None,
        "overall": overall,
        "group_stats": group_stats,
        "task_stats": task_stats,
        "task_groups": {
            "easy": sorted([task for task, group in task_groups.items() if group == "easy"]),
            "hard": sorted([task for task, group in task_groups.items() if group == "hard"]),
            "middle": sorted([task for task, group in task_groups.items() if group == "middle"]),
        },
        "records": records_to_dicts(records),
        "evaluation_summary": summarize_metrics(
            {
                "prompt_mode": evaluation_results.get("prompt_mode"),
                "task_accuracy": evaluation_results.get("task_accuracy"),
                "ni_exact_match": evaluation_results.get("ni_exact_match"),
                "ni_rouge_l": evaluation_results.get("ni_rouge_l"),
                "total_examples": evaluation_results.get("total_examples"),
            }
        ),
        "top_easy_tasks": [
            {"task_name": task, **stats}
            for task, stats in task_ranked[: args.top_task_limit]
        ],
        "top_hard_tasks": [
            {"task_name": task, **stats}
            for task, stats in task_ranked_by_hard[: args.top_task_limit]
        ],
        "run_summary": {
            "run_name": run_summary.get("run_name"),
            "run_dir": run_summary.get("run_dir"),
            "model_name": run_config.get("args", {}).get("model_name"),
            "num_tasks": run_config.get("args", {}).get("num_tasks"),
            "metrics": evaluation_results.get("prompt_mode"),
        },
    }

    json_path = output_dir / "bank_margin_analysis.json"
    md_path = output_dir / "bank_margin_analysis.md"
    write_json(str(json_path), analysis_payload)

    markdown = build_markdown_report(run_dir, run_config, run_summary, analysis_payload)
    md_path.write_text(markdown, encoding="utf-8")

    print(f"Saved analysis JSON to: {json_path}")
    print(f"Saved analysis Markdown to: {md_path}")
    print(f"Saved margin plot to: {output_dir / 'bank_margin_easy_vs_hard_margin.svg'}")
    print(f"Saved probability plot to: {output_dir / 'bank_margin_easy_vs_hard_prob.svg'}")


if __name__ == "__main__":
    main()
