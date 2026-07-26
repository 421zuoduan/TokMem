#!/usr/bin/env python3
"""Evaluate the first greedy tool-routing decision on a fixed test prefix."""

import argparse
import json
import sys
from pathlib import Path

import torch


COMPOSITIONAL_DIR = Path(__file__).resolve().parents[1]
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from backbone_prompting import format_user_assistant_prompt  # noqa: E402
from cold_start.common import (  # noqa: E402
    load_checkpoint_model,
    load_json,
    torch_dtype,
)
from cold_start.probes import (  # noqa: E402
    _final_hidden_states_without_logits,
    _right_pad,
)
from cold_start.runtime import apply_cold_start_delta  # noqa: E402


DEFAULT_DATA = (
    COMPOSITIONAL_DIR
    / "data"
    / "test"
    / "function_calling_test_tools51-100_plus_cold20_4calls.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the first full-vocabulary greedy tool decision."
    )
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _encode_prompts(model, tokenizer, items):
    sequences = []
    for item in items:
        prompt = format_user_assistant_prompt(
            tokenizer,
            item["user_input"],
            model=model,
        )
        sequences.append(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
    return sequences


def _predict_batches(model, tokenizer, sequences, batch_size, progress_every):
    predictions = []
    device = model.trainable_tool_input_embeddings.device
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        input_ids, attention_mask, lengths = _right_pad(
            batch,
            tokenizer.pad_token_id,
            device,
        )
        with torch.inference_mode():
            hidden_states = _final_hidden_states_without_logits(
                model,
                input_ids,
                attention_mask,
            )
            row_indices = torch.arange(len(batch), device=device)
            end_indices = torch.tensor(lengths, device=device) - 1
            last_hidden_states = hidden_states[row_indices, end_indices]
            logits = model._get_lm_head_module()(last_hidden_states)
            active_rows = torch.ones(
                len(batch),
                dtype=torch.bool,
                device=device,
            )
            logits = model._apply_logit_bias_to_logits(
                logits,
                last_hidden_states,
                active_rows,
            )
            predictions.extend(
                torch.argmax(logits, dim=-1).detach().cpu().tolist()
            )
        completed = min(start + len(batch), len(sequences))
        if progress_every > 0 and (
            completed == len(sequences) or completed % progress_every == 0
        ):
            print(f"Evaluated {completed}/{len(sequences)} first decisions")
    return predictions


def _partition(records, expected_is_new):
    selected = [
        record
        for record in records
        if record["expected_is_new"] is expected_is_new
    ]
    correct = sum(record["correct"] for record in selected)
    return {
        "examples": len(selected),
        "correct": correct,
        "accuracy": correct / max(1, len(selected)),
    }


def main():
    args = parse_args()
    _, tokenizer, model = load_checkpoint_model(
        args.run_config,
        args.checkpoint,
        args.device,
        torch_dtype(args.dtype),
    )
    delta = torch.load(
        args.delta,
        map_location="cpu",
        weights_only=False,
    )
    apply_cold_start_delta(model, delta)

    items = load_json(args.data_path)
    if args.limit is not None:
        items = items[: args.limit]
    sequences = _encode_prompts(model, tokenizer, items)
    predicted_token_ids = _predict_batches(
        model,
        tokenizer,
        sequences,
        args.batch_size,
        args.progress_every,
    )

    token_id_to_tool = {
        token_id: model.tool_id_to_name[tool_id]
        for token_id, tool_id in model.token_id_to_tool_id.items()
    }
    new_tool_set = set(delta["new_tool_names"])
    records = []
    for index, (item, predicted_token_id) in enumerate(
        zip(items, predicted_token_ids)
    ):
        expected_tool = item["tools"][0]
        predicted_tool = token_id_to_tool.get(predicted_token_id)
        records.append(
            {
                "index": index,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "predicted_token_id": predicted_token_id,
                "predicted_tool_token": predicted_tool is not None,
                "expected_is_new": expected_tool in new_tool_set,
                "correct": predicted_tool == expected_tool,
            }
        )

    correct = sum(record["correct"] for record in records)
    predicted_tool_tokens = sum(
        record["predicted_tool_token"] for record in records
    )
    summary = {
        "metric": "first_full_vocabulary_greedy_tool_routing",
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / max(1, len(records)),
        "new_tools": _partition(records, True),
        "old_tools": _partition(records, False),
        "tool_token_predictions": predicted_tool_tokens,
        "tool_token_prediction_rate": (
            predicted_tool_tokens / max(1, len(records))
        ),
        "run_config": str(Path(args.run_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "delta": str(Path(args.delta).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"Wrote first-tool routing results to {output_path}")


if __name__ == "__main__":
    main()
