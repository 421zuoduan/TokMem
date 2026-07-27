#!/usr/bin/env python3
"""Run free-generation and oracle-boundary diagnostics for TapMem."""

import argparse
import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
UTILS_DIR = COMPOSITIONAL_DIR / "utils"
for import_dir in (COMPOSITIONAL_DIR, UTILS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from analyze_generation_eoc_boundaries import boundary_counts  # noqa: E402
from run_error_type_transition_analysis import (  # noqa: E402
    CHECKPOINT_RUN_DIRS,
    DEFAULT_DATA_PATH,
)
from run_train4_checkpoint_eval_10calls import (  # noqa: E402
    build_model,
    build_user_text,
    final_checkpoint_name,
    load_json,
    prediction_record,
    torch_dtype,
)


DEFAULT_OUTPUT_DIR = (
    COMPOSITIONAL_DIR / "rebuttal" / "results" / "tapmem_failure_diagnostics"
)
MODEL_BATCH_SIZES = {
    "llama1b": 8,
    "llama3b": 4,
    "llama8b": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace TapMem free decoding and probe routing at teacher-forced "
            "gold procedure boundaries."
        )
    )
    parser.add_argument("--models", default="llama1b,llama8b")
    parser.add_argument("--trial-id", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--oracle-batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def raw_generated_token_ids(generated_tokens, input_length, eos_token_id):
    sequences = []
    for row in generated_tokens:
        tokens = []
        for token_id in row[input_length:].tolist():
            if token_id == eos_token_id:
                break
            tokens.append(int(token_id))
        sequences.append(tokens)
    return sequences


def ranked_tools(scores, model, top_k):
    import torch

    count = min(top_k, scores.numel())
    values, indices = torch.topk(scores.float(), k=count)
    probabilities = torch.softmax(scores.float(), dim=-1)
    rows = []
    for value, index in zip(values.tolist(), indices.tolist()):
        rows.append(
            {
                "tool": model.tool_id_to_name[int(index)],
                "score": float(value),
                "candidate_probability": float(probabilities[int(index)].item()),
            }
        )
    return rows


def gold_diagnostics(scores, expected_tool, model):
    import torch

    if expected_tool is None or expected_tool not in model.tool_name_to_id:
        return {
            "gold_rank": None,
            "gold_margin": None,
            "gold_candidate_probability": None,
        }
    gold_index = int(model.tool_name_to_id[expected_tool])
    float_scores = scores.float()
    gold_score = float_scores[gold_index]
    rank = int((float_scores > gold_score).sum().item()) + 1
    other_mask = torch.ones_like(float_scores, dtype=torch.bool)
    other_mask[gold_index] = False
    strongest_other = float_scores[other_mask].max()
    return {
        "gold_rank": rank,
        "gold_margin": float((gold_score - strongest_other).item()),
        "gold_candidate_probability": float(
            torch.softmax(float_scores, dim=-1)[gold_index].item()
        ),
    }


def decision_snapshot(
    model,
    tokenizer,
    base_logits,
    fused_logits,
    hidden_state,
    expected_tool,
    top_k,
):
    import torch

    tool_token_ids = model._get_tool_reserved_token_ids_tensor(base_logits.device)
    base_tool_scores = base_logits.index_select(-1, tool_token_ids)
    fused_tool_scores = fused_logits.index_select(-1, tool_token_ids)
    with torch.inference_mode():
        head_scores = model._get_logit_bias_scores(hidden_state.unsqueeze(0))[0]
    snapshot = {
        "expected_tool": expected_tool,
        "base_top_tools": ranked_tools(base_tool_scores, model, top_k),
        "head_top_tools": ranked_tools(head_scores, model, top_k),
        "fused_top_tools": ranked_tools(fused_tool_scores, model, top_k),
        "base": gold_diagnostics(base_tool_scores, expected_tool, model),
        "head": gold_diagnostics(head_scores, expected_tool, model),
        "fused": gold_diagnostics(fused_tool_scores, expected_tool, model),
    }
    return snapshot


def attach_selected_token(snapshot, selected_token_id, model, tokenizer):
    selected_tool = model.get_tool_name_from_token_id(int(selected_token_id))
    snapshot.update(
        {
            "selected_token_id": int(selected_token_id),
            "selected_token": tokenizer.decode([int(selected_token_id)]),
            "selected_tool": selected_tool,
            "selected_is_tool": selected_tool is not None,
            "selected_is_eos": int(selected_token_id) == tokenizer.eos_token_id,
            "selected_matches_expected": (
                selected_tool == snapshot["expected_tool"]
                if snapshot["expected_tool"] is not None
                else int(selected_token_id) == tokenizer.eos_token_id
            ),
        }
    )


def generate_free_batch_with_trace(
    model,
    tokenizer,
    items,
    device,
    max_new_tokens,
    top_k,
):
    import torch

    encoded = tokenizer(
        [build_user_text(item, tokenizer, model) for item in items],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    user_tokens = encoded["input_ids"]
    user_mask = encoded["attention_mask"]
    batch_size = user_tokens.shape[0]
    input_ids = user_tokens.clone()
    attention_mask = user_mask.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    step_input_ids = user_tokens
    past_key_values = None
    boundary_ordinals = [0] * batch_size
    traces = [[] for _ in range(batch_size)]

    with torch.inference_mode():
        for step in range(max_new_tokens):
            decision_context = model._build_decision_context(
                input_ids=input_ids,
                batch_size=batch_size,
                device=device,
                step=step,
                use_eoc=True,
            )
            active_rows = decision_context & ~finished
            next_logits, hidden_states, past_key_values = model._generation_forward_step(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                return_last_hidden_state=bool(active_rows.any()),
            )
            fused_logits = next_logits
            if active_rows.any():
                fused_logits = model._apply_logit_bias_to_logits(
                    logits=next_logits,
                    hidden_states=hidden_states,
                    active_decision_rows=active_rows,
                )
            next_tokens = model._sample_next_tokens(
                fused_logits,
                temperature=0.6,
                top_p=0.9,
                do_sample=False,
            )
            next_tokens = next_tokens.masked_fill(finished, tokenizer.eos_token_id)

            for row in active_rows.nonzero(as_tuple=False).squeeze(-1).tolist():
                ordinal = boundary_ordinals[row]
                expected_tools = items[row]["tools"]
                expected_tool = (
                    expected_tools[ordinal] if ordinal < len(expected_tools) else None
                )
                snapshot = decision_snapshot(
                    model=model,
                    tokenizer=tokenizer,
                    base_logits=next_logits[row],
                    fused_logits=fused_logits[row],
                    hidden_state=hidden_states[row],
                    expected_tool=expected_tool,
                    top_k=top_k,
                )
                snapshot.update(
                    {
                        "generation_step": step,
                        "boundary_ordinal": ordinal,
                        "boundary_type": "initial" if step == 0 else "after_generated_eoc",
                    }
                )
                attach_selected_token(
                    snapshot,
                    int(next_tokens[row].item()),
                    model,
                    tokenizer,
                )
                traces[row].append(snapshot)
                boundary_ordinals[row] += 1

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        batch_size,
                        1,
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )
            finished = finished | (next_tokens == tokenizer.eos_token_id)
            if finished.all():
                break
            step_input_ids = next_tokens.unsqueeze(-1)

    parsed = model._parse_generated_sequences(input_ids, user_tokens, tokenizer)
    raw_sequences = raw_generated_token_ids(
        input_ids,
        user_tokens.shape[1],
        tokenizer.eos_token_id,
    )
    return parsed, raw_sequences, traces


def free_prediction_records(
    model,
    tokenizer,
    data,
    model_name,
    trial_id,
    run_name,
    device,
    batch_size,
    max_new_tokens,
    top_k,
    progress_every,
):
    records = []
    candidate_tools = list(model.tool_names)
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        parsed, raw_sequences, traces = generate_free_batch_with_trace(
            model=model,
            tokenizer=tokenizer,
            items=batch,
            device=device,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
        for offset, (item, result, raw_ids, trace) in enumerate(
            zip(batch, parsed, raw_sequences, traces)
        ):
            index = start + offset
            record = prediction_record(
                index=index,
                item=item,
                result=result,
                tokenizer=tokenizer,
                method="tapmem",
                candidate_tools=candidate_tools,
            )
            record.update(
                {
                    "model": model_name,
                    "trial": trial_id,
                    "run_name": run_name,
                    "raw_generated_token_ids": raw_ids,
                    "raw_generated_tokens": [
                        tokenizer.decode([token_id]) for token_id in raw_ids
                    ],
                    "eoc_token_id": int(model.eoc_token_id),
                    "eoc_token": tokenizer.decode([model.eoc_token_id]),
                    "boundary_trace": trace,
                    **boundary_counts(raw_ids, model, len(item["tools"])),
                }
            )
            records.append(record)
        completed = start + len(batch)
        if progress_every and completed % progress_every < batch_size:
            print(f"[{model_name}] free generation {completed}/{len(data)}", flush=True)
    return records


def oracle_cases(data, tokenizer, model):
    cases = []
    for index, item in enumerate(data):
        prefix = tokenizer(
            build_user_text(item, tokenizer, model),
            add_special_tokens=False,
        )["input_ids"]
        tools = list(item["tools"])
        calls = list(item["function_calls"])
        cases.append(
            {
                "index": index,
                "boundary_ordinal": 0,
                "boundary_type": "initial",
                "expected_tool": tools[0],
                "prefix_ids": list(prefix),
            }
        )
        for procedure_index, (tool, call) in enumerate(zip(tools, calls)):
            tool_token_id = model.get_tool_token_id(tool)
            if tool_token_id is None:
                raise ValueError(f"Unknown tool in oracle prefix: {tool}")
            prefix.append(int(tool_token_id))
            prefix.extend(
                tokenizer(call, add_special_tokens=False)["input_ids"]
            )
            prefix.append(int(model.eoc_token_id))
            next_ordinal = procedure_index + 1
            cases.append(
                {
                    "index": index,
                    "boundary_ordinal": next_ordinal,
                    "boundary_type": (
                        "transition" if next_ordinal < len(tools) else "terminal"
                    ),
                    "expected_tool": (
                        tools[next_ordinal] if next_ordinal < len(tools) else None
                    ),
                    "prefix_ids": list(prefix),
                }
            )
    return cases


def oracle_probe_records(
    model,
    tokenizer,
    data,
    model_name,
    trial_id,
    run_name,
    device,
    batch_size,
    top_k,
    progress_every,
):
    import torch

    cases = oracle_cases(data, tokenizer, model)
    records = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        max_length = max(len(case["prefix_ids"]) for case in batch)
        input_ids = torch.full(
            (len(batch), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, case in enumerate(batch):
            prefix = torch.tensor(case["prefix_ids"], dtype=torch.long, device=device)
            input_ids[row, -prefix.numel() :] = prefix
            attention_mask[row, -prefix.numel() :] = 1

        with torch.inference_mode():
            base_logits, hidden_states, _past = model._generation_forward_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                return_last_hidden_state=True,
            )
            active_rows = torch.ones(
                len(batch),
                dtype=torch.bool,
                device=device,
            )
            fused_logits = model._apply_logit_bias_to_logits(
                logits=base_logits,
                hidden_states=hidden_states,
                active_decision_rows=active_rows,
            )
            selected_tokens = torch.argmax(fused_logits, dim=-1)

        for row, case in enumerate(batch):
            snapshot = decision_snapshot(
                model=model,
                tokenizer=tokenizer,
                base_logits=base_logits[row],
                fused_logits=fused_logits[row],
                hidden_state=hidden_states[row],
                expected_tool=case["expected_tool"],
                top_k=top_k,
            )
            attach_selected_token(
                snapshot,
                int(selected_tokens[row].item()),
                model,
                tokenizer,
            )
            snapshot.update(
                {
                    "index": case["index"],
                    "model": model_name,
                    "trial": trial_id,
                    "run_name": run_name,
                    "boundary_ordinal": case["boundary_ordinal"],
                    "boundary_type": case["boundary_type"],
                }
            )
            records.append(snapshot)

        completed = start + len(batch)
        if progress_every and completed % progress_every < batch_size:
            print(f"[{model_name}] oracle probes {completed}/{len(cases)}", flush=True)
    return records


def selected_run(model_name, trial_id):
    run_dir = CHECKPOINT_RUN_DIRS[(model_name, "tapmem")][trial_id - 1]
    run_config_path = run_dir / "run_config.json"
    run_config = load_json(run_config_path)
    checkpoint_path = run_dir / final_checkpoint_name(run_config)
    return run_dir, run_config_path, run_config, checkpoint_path


def run_model(args, model_name, data, output_dir):
    import torch
    from transformers import AutoTokenizer

    run_dir, run_config_path, run_config, checkpoint_path = selected_run(
        model_name,
        args.trial_id,
    )
    free_path = (
        output_dir
        / "free_generation"
        / model_name
        / f"trial{args.trial_id}.jsonl"
    )
    oracle_path = (
        output_dir
        / "oracle_probes"
        / model_name
        / f"trial{args.trial_id}.jsonl"
    )
    manifest_path = (
        output_dir
        / "manifests"
        / model_name
        / f"trial{args.trial_id}.json"
    )
    if (
        not args.force
        and free_path.exists()
        and oracle_path.exists()
        and manifest_path.exists()
    ):
        print(f"[{model_name}] diagnostics already complete; skipping")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(
        run_config["args"]["model_name"],
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = build_model(
        run_config,
        checkpoint,
        tokenizer,
        args.device,
        torch_dtype(args.dtype),
    )
    if not model.use_eoc or not model.use_logit_bias:
        raise ValueError(f"Selected checkpoint is not TapMem: {run_dir}")
    del checkpoint
    gc.collect()

    batch_size = args.eval_batch_size or MODEL_BATCH_SIZES[model_name]
    oracle_batch_size = args.oracle_batch_size or max(batch_size, 8)
    free_records = free_prediction_records(
        model=model,
        tokenizer=tokenizer,
        data=data,
        model_name=model_name,
        trial_id=args.trial_id,
        run_name=run_dir.name,
        device=args.device,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        progress_every=args.progress_every,
    )
    write_jsonl(free_path, free_records)

    oracle_records = oracle_probe_records(
        model=model,
        tokenizer=tokenizer,
        data=data,
        model_name=model_name,
        trial_id=args.trial_id,
        run_name=run_dir.name,
        device=args.device,
        batch_size=oracle_batch_size,
        top_k=args.top_k,
        progress_every=args.progress_every,
    )
    write_jsonl(oracle_path, oracle_records)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "trial": args.trial_id,
        "run_name": run_dir.name,
        "run_config_path": str(run_config_path),
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(Path(args.data_path).resolve()),
        "samples": len(data),
        "free_generation_path": str(free_path.resolve()),
        "oracle_probe_path": str(oracle_path.resolve()),
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "eval_batch_size": batch_size,
            "oracle_batch_size": oracle_batch_size,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{model_name}] wrote {len(free_records)} free records and "
        f"{len(oracle_records)} oracle probes",
        flush=True,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    models = split_csv(args.models)
    unknown = [model for model in models if model not in MODEL_BATCH_SIZES]
    if unknown:
        raise SystemExit(f"Unsupported models: {unknown}")
    data = load_json(Path(args.data_path))
    if args.limit is not None:
        data = data[: args.limit]
    output_dir = Path(args.output_dir)
    for model_name in models:
        run_model(args, model_name, data, output_dir)


if __name__ == "__main__":
    main()
