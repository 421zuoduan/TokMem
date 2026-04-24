#!/usr/bin/env python3
"""Shared generation utilities for atomic base and RAG entrypoints."""

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from main_in_domain import load_split_cache
from natural_instructions_eval import (
    evaluate_predictions,
    exact_match,
    metric_max_over_ground_truths,
    print_evaluation_results,
    rouge_score,
)
from task_dataset import sample_natural_instructions_tasks
from task_training import setup_logging


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducible data selection and generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_content(example: Dict, include_instruction: bool = True) -> str:
    instruction = example.get("instruction", "")
    query = example.get("query", "")
    if include_instruction:
        return f"{instruction}\n\n{query}"
    return str(query)


def normalize_demo_output(output) -> str:
    if isinstance(output, list):
        return str(output[0]) if output else ""
    return str(output)


def build_demo_input(example: Dict, include_instruction: bool = True) -> str:
    if include_instruction:
        return (
            f"Instruction:\n{example.get('instruction', '')}\n\n"
            f"Input:\n{example.get('query', '')}"
        )
    return str(example.get("query", ""))


def build_chat_prompt(
    example: Dict,
    tokenizer,
    demos: Optional[Sequence[Dict]] = None,
    include_instruction: bool = True,
    include_demo_instruction: bool = True,
) -> str:
    """Build a chat prompt matching the current atomic Qwen/Llama prompt style."""
    demos = list(demos or [])
    is_qwen = "qwen" in str(tokenizer.name_or_path).lower()

    if is_qwen:
        parts = []
        for demo in demos:
            parts.append(
                f"<|im_start|>user\n{build_demo_input(demo, include_demo_instruction)}<|im_end|>\n"
            )
            parts.append(
                f"<|im_start|>assistant\n{normalize_demo_output(demo.get('responses', ['']))}<|im_end|>\n"
            )
        parts.append(f"<|im_start|>user\n{build_user_content(example, include_instruction)}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    parts = ["<|begin_of_text|>"]
    for demo in demos:
        parts.append(
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{build_demo_input(demo, include_demo_instruction)}<|eot_id|>"
        )
        parts.append(
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{normalize_demo_output(demo.get('responses', ['']))}<|eot_id|>"
        )
    parts.append(
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{build_user_content(example, include_instruction)}<|eot_id|>"
    )
    parts.append("<|start_header_id|>assistant<|end_header_id|>")
    return "".join(parts)


def extract_response_text(text: str, prompt_text: str = "") -> str:
    response = text
    for marker in ("<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>"):
        if marker in response:
            response = response.rsplit(marker, 1)[-1]
    if prompt_text and response.startswith(prompt_text):
        response = response[len(prompt_text):]
    for stop_marker in ("<|im_end|>", "<|eot_id|>"):
        if stop_marker in response:
            response = response.split(stop_marker, 1)[0]
    return response.strip()


class PromptDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict],
        tokenizer,
        max_length: int,
        include_instruction: bool = True,
        include_demo_instruction: bool = True,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_instruction = include_instruction
        self.include_demo_instruction = include_demo_instruction

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        demos = example.get("retrieved_examples", example.get("few_shot_examples", []))
        prompt_text = build_chat_prompt(
            example,
            self.tokenizer,
            demos=demos,
            include_instruction=self.include_instruction,
            include_demo_instruction=self.include_demo_instruction,
        )
        encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "raw_data": example,
            "prompt_text": prompt_text,
        }


def prompt_collate_fn(batch: Sequence[Dict], tokenizer) -> Dict:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        width = item["input_ids"].size(0)
        input_ids[row, -width:] = item["input_ids"]
        attention_mask[row, -width:] = item["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "raw_data": [item["raw_data"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
    }


def create_prompt_dataloader(
    examples: Sequence[Dict],
    tokenizer,
    max_length: int,
    batch_size: int,
    include_instruction: bool = True,
    include_demo_instruction: bool = True,
) -> DataLoader:
    dataset = PromptDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        include_instruction=include_instruction,
        include_demo_instruction=include_demo_instruction,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )


def get_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def decode_continuation(tokenizer, token_ids: Sequence[int]) -> str:
    valid_ids = []
    eos_token_id = tokenizer.eos_token_id
    for token_id in token_ids:
        value = int(token_id)
        if eos_token_id is not None and value == eos_token_id:
            break
        valid_ids.append(value)
    return tokenizer.decode(valid_ids, skip_special_tokens=False)


def build_prediction_row(
    example: Dict,
    predicted_response: str,
    full_generated_sequence: str,
    mode: str,
    prompt_mode: str,
    retrievals: Optional[Sequence[Dict]] = None,
    error: Optional[str] = None,
) -> Dict:
    expected_tasks = example.get("tasks", ["unknown"])
    expected_responses = example.get("responses", [""])
    response_exact_match = bool(
        metric_max_over_ground_truths(
            lambda prediction, ground_truth, xlingual=False: (
                1.0 if exact_match(prediction, ground_truth, xlingual) else 0.0
            ),
            predicted_response,
            expected_responses,
            xlingual=False,
        )
    )
    response_rouge_l = metric_max_over_ground_truths(
        rouge_score,
        predicted_response,
        expected_responses,
        xlingual=False,
    )
    row = {
        "example_index": None,
        "mode": mode,
        "prompt_mode": prompt_mode,
        "prediction_status": "response_only_generation" if predicted_response else "empty_prediction",
        "instruction": example.get("instruction", ""),
        "query": example.get("query", ""),
        "expected_task": expected_tasks[0] if expected_tasks else None,
        "predicted_task": None,
        "expected_tasks": expected_tasks,
        "predicted_tasks": [],
        "has_task_prediction": False,
        "task_match": None,
        "expected_response": expected_responses[0] if expected_responses else "",
        "predicted_response": predicted_response,
        "expected_responses": expected_responses,
        "predicted_responses": [predicted_response] if predicted_response else [],
        "response_exact_match": response_exact_match,
        "response_rouge_l": round(response_rouge_l, 4),
        "full_generated_sequence": full_generated_sequence,
        "task_token_used": None,
    }
    if retrievals is not None:
        row["retrieved_examples"] = list(retrievals)
    if error is not None:
        row["error"] = error
    return row


def evaluate_generation(
    model,
    tokenizer,
    dataloader: DataLoader,
    mode: str,
    prompt_mode: str,
    predictions_output_path: Optional[str] = None,
    max_new_tokens: int = 256,
) -> Dict:
    eval_logger = logging.getLogger("evaluation")
    model.eval()

    total_examples = len(dataloader.dataset)
    print(f"\n=== Atomic Generation Evaluation ({mode}, {prompt_mode}) ===")
    print(f"Evaluating on {total_examples} test examples")
    eval_logger.info(f"GENERATION EVALUATION START - Mode:{mode} Prompt:{prompt_mode} Examples:{total_examples}")

    all_predictions = []
    all_references = []
    all_task_names = []
    prediction_rows = []
    input_device = get_input_device(model)
    start_time = time.time()

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.use_cache = True
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    for batch_idx, batch in enumerate(dataloader):
        raw_examples = batch["raw_data"]
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        prompt_width = input_ids.shape[1]

        if batch_idx % 10 == 0 or (batch_idx + 1) * len(raw_examples) >= total_examples:
            processed = min((batch_idx + 1) * len(raw_examples), total_examples)
            print(f"   Progress: {processed}/{total_examples} ({processed / total_examples * 100:.1f}%)")

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
            continuations = outputs[:, prompt_width:]
            for example, continuation in zip(raw_examples, continuations):
                full_generated_sequence = decode_continuation(tokenizer, continuation.tolist())
                predicted_response = extract_response_text(full_generated_sequence)
                all_predictions.append(predicted_response)
                all_references.append(example.get("responses", [""]))
                all_task_names.append(example.get("tasks", ["unknown"])[0])
                row = build_prediction_row(
                    example=example,
                    predicted_response=predicted_response,
                    full_generated_sequence=full_generated_sequence,
                    mode=mode,
                    prompt_mode=prompt_mode,
                    retrievals=example.get("retrieval_metadata"),
                )
                row["example_index"] = len(prediction_rows)
                prediction_rows.append(row)
        except Exception as exc:
            error_message = str(exc)
            print(f"   Error processing batch {batch_idx + 1}: {error_message}")
            eval_logger.info(f"BATCH ERROR - Batch:{batch_idx + 1} Error:{error_message}")
            for example in raw_examples:
                all_predictions.append("")
                all_references.append(example.get("responses", [""]))
                all_task_names.append(example.get("tasks", ["unknown"])[0])
                row = build_prediction_row(
                    example=example,
                    predicted_response="",
                    full_generated_sequence="",
                    mode=mode,
                    prompt_mode=prompt_mode,
                    retrievals=example.get("retrieval_metadata"),
                    error=error_message,
                )
                row["example_index"] = len(prediction_rows)
                prediction_rows.append(row)

    print("\nComputing Natural Instructions metrics...")
    ni_results = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names,
        xlingual=False,
    )
    print_evaluation_results(ni_results, f"ATOMIC GENERATION EVALUATION ({mode})")

    eval_time = time.time() - start_time
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    eval_logger.info(
        "GENERATION EVALUATION COMPLETE - "
        f"ExactMatch:{ni_results['exact_match']:.1f}% RougeL:{ni_results['rougeL']:.1f}% "
        f"Time:{eval_time:.1f}s"
    )

    if predictions_output_path is not None:
        write_jsonl(predictions_output_path, prediction_rows)
        print(f"Saved per-example evaluation predictions to: {predictions_output_path}")

    return {
        "mode": mode,
        "prompt_mode": prompt_mode,
        "exact_accuracy": ni_results["exact_match"] / 100.0,
        "task_accuracy": None,
        "avg_response_score": ni_results["rougeL"] / 100.0,
        "total_examples": total_examples,
        "ni_exact_match": ni_results["exact_match"],
        "ni_rouge_l": ni_results["rougeL"],
        "ni_per_task": ni_results.get("per_task", {}),
        "task_breakdown": {},
        "predictions_output_path": predictions_output_path,
    }


def load_tokenizer(model_name: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}, padding side: {tokenizer.padding_side}")
    print()
    return tokenizer


def load_model(model_name: str, device: str, device_map: Optional[str], torch_dtype: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    print("Loading base model...")
    kwargs = {"torch_dtype": dtype_map[torch_dtype]}
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device_map:
        model = model.to(device)
    print("Base model loaded")
    print()
    return model


def load_atomic_split(args, tokenizer):
    if args.split_cache_path:
        print(f"Loading cached split from: {os.path.abspath(args.split_cache_path)}")
        train_data, val_data, test_data, task_names, cache_path = load_split_cache(args)
        split_source = os.path.abspath(cache_path)
    else:
        print(f"Sampling {args.num_tasks} tasks from Natural Instructions dataset...")
        train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(
            tasks_dir=args.tasks_dir,
            num_tasks=args.num_tasks,
            max_instruction_tokens=args.max_instruction_tokens,
            tokenizer=tokenizer,
            stable_test_split=True,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            few_shot=args.few_shot,
        )
        split_source = "runtime_sampling"

    print(
        f"Loaded split. Train: {len(train_data)}, Val: {len(val_data)}, "
        f"Test: {len(test_data)}, Tasks: {len(task_names)}"
    )
    print()
    return train_data, val_data, test_data, task_names, split_source


@dataclass
class RunArtifacts:
    output_dir: str
    timestamp: str
    training_log: str
    evaluation_log: str


def setup_generation_run(output_dir: Optional[str]) -> RunArtifacts:
    log_dir = output_dir or "logs"
    _, _, training_log, evaluation_log, timestamp = setup_logging(log_dir=log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return RunArtifacts(
        output_dir=log_dir,
        timestamp=timestamp,
        training_log=training_log,
        evaluation_log=evaluation_log,
    )


def write_run_outputs(
    artifacts: RunArtifacts,
    args,
    results: Dict,
    split_source: str,
    task_names: Sequence[str],
    train_count: int,
    val_count: int,
    test_count: int,
) -> None:
    write_json(
        os.path.join(artifacts.output_dir, "run_config.json"),
        {
            "args": vars(args),
            "split_source": split_source,
            "task_names": list(task_names),
            "dataset_summary": {
                "train_examples": train_count,
                "val_examples": val_count,
                "test_examples": test_count,
                "task_count": len(task_names),
            },
            "logs": {
                "training_log": artifacts.training_log,
                "evaluation_log": artifacts.evaluation_log,
            },
        },
    )
    write_json(os.path.join(artifacts.output_dir, "evaluation_results.json"), results)
    write_json(
        os.path.join(artifacts.output_dir, "run_summary.json"),
        {
            "mode": results.get("mode"),
            "prompt_mode": results.get("prompt_mode"),
            "split_source": split_source,
            "num_tasks": len(task_names),
            "test_examples": test_count,
            "metrics": {
                "task_accuracy": results.get("task_accuracy"),
                "exact_accuracy": results.get("exact_accuracy"),
                "avg_response_score": results.get("avg_response_score"),
                "ni_exact_match": results.get("ni_exact_match"),
                "ni_rouge_l": results.get("ni_rouge_l"),
            },
            "evaluation_predictions_path": results.get("predictions_output_path"),
            "evaluation_results_path": os.path.join(artifacts.output_dir, "evaluation_results.json"),
        },
    )
