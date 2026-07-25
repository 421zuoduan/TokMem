#!/usr/bin/env python3
"""Run one real Qwen3.5-9B TapMem optimizer step and audit gradients."""

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[4]
COMPOSITIONAL = REPO / "compositional"
MODEL_PATH = REPO / "models" / "Qwen3.5-9B"
TRAIN_PATH = (
    REPO
    / "results"
    / "compositional"
    / "all_methods"
    / "data"
    / "training"
    / "function_calling_train_tools51-100_4calls.json"
)
TEST_PATH = (
    REPO
    / "results"
    / "compositional"
    / "all_methods"
    / "data"
    / "test"
    / "function_calling_test_tools51-100_4calls.json"
)
SEED = 42
MEMORY_LR = 7e-3

sys.path.insert(0, str(COMPOSITIONAL))

from dataset import (  # noqa: E402
    NativeFunctionCallingDataset,
    collate_fn,
    discover_available_tools,
)
from main_sequential import apply_orthogonal_init_all_tools  # noqa: E402
from qwen35_model import Qwen35FunctionCallingModel  # noqa: E402
from training import (  # noqa: E402
    _forward_with_optional_hidden_states,
    apply_logit_train_add,
    build_shift_supervision_masks,
    compute_logit_bias_loss,
    gather_logit_bias_examples,
)


def tensor_snapshot(parameters):
    return [
        parameter.detach().float().cpu().clone()
        for parameter in parameters
    ]


def snapshot_delta(before, parameters):
    deltas = [
        float(
            (
                parameter.detach().float().cpu() - old
            ).abs().max().item()
        )
        for old, parameter in zip(before, parameters)
    ]
    return max(deltas, default=0.0), sum(
        delta > 0.0 for delta in deltas
    )


def grad_audit(named_parameters):
    missing = []
    nonfinite = []
    zero = []
    norms = {}
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
            continue
        norm = float(gradient.float().norm().item())
        norms[name] = norm
        if norm == 0.0:
            zero.append(name)
    return missing, nonfinite, zero, norms


def main():
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    assert torch.cuda.is_available(), "CUDA is unavailable"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"torch={torch.__version__}")
    print(f"transformers={__import__('transformers').__version__}")
    print(f"cuda_device={torch.cuda.get_device_name(device)}")
    print(f"model_path={MODEL_PATH}")
    print(
        "formal_mode="
        "use_eoc=True,use_logit_bias=True,use_logit_train_add=True,"
        "detach=True,loss_weight=0.1,network=linear,scale=1.0"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    tool_names = discover_available_tools(
        str(TRAIN_PATH),
        str(TEST_PATH),
    )
    assert len(tool_names) == 50, (
        f"Expected 50 tools, got {len(tool_names)}"
    )

    model = Qwen35FunctionCallingModel(
        model_name=str(MODEL_PATH),
        num_tools=len(tool_names),
        tool_names=tool_names,
        tokenizer=tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        decouple_embeddings=False,
        lora_config=None,
        use_eoc=True,
        use_logit_bias=True,
        use_tool_head_replacement=False,
        logit_bias_network="linear",
        logit_bias_scale=1.0,
        use_memory_bank_constraint=False,
        memory_bank_probability_threshold=0.5,
    )
    apply_orthogonal_init_all_tools(model, len(tool_names))
    model.train()

    base_named_parameters = list(model.model.named_parameters())
    base_requires_grad = [
        name
        for name, parameter in base_named_parameters
        if parameter.requires_grad
    ]
    assert not base_requires_grad, (
        "Base LM unexpectedly has trainable parameters: "
        + ", ".join(base_requires_grad[:10])
    )

    with TRAIN_PATH.open("r", encoding="utf-8") as stream:
        raw_examples = json.load(stream)
    four_call_examples = [
        example
        for example in raw_examples
        if len(example.get("function_calls", [])) == 4
    ]
    assert four_call_examples, "No four-call training example found"

    selected = min(
        four_call_examples[:256],
        key=lambda item: len(
            tokenizer(
                item["user_input"],
                add_special_tokens=False,
            )["input_ids"]
        ),
    )
    dataset = NativeFunctionCallingDataset(
        data_path=None,
        tokenizer=tokenizer,
        max_length=512,
        model=model,
        mode="train",
        use_eoc=True,
    )
    dataset.data = [selected]
    batch = collate_fn([dataset[0]], tokenizer)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    print(
        f"batch_shape={tuple(input_ids.shape)},"
        f"supervised_tokens={int((labels != -100).sum().item())},"
        f"function_calls={len(selected['function_calls'])}"
    )

    trainable_parameters = model.get_trainable_parameters()
    memory_parameters = [model.trainable_tool_embeddings]
    head_named_parameters = list(
        model.logit_bias_head.named_parameters()
    )
    head_parameters = [
        parameter for _, parameter in head_named_parameters
    ]
    assert {
        id(parameter) for parameter in trainable_parameters
    } == {
        id(parameter)
        for parameter in memory_parameters + head_parameters
    }, (
        "Optimizer parameter set differs from memory embeddings "
        "+ logit-bias head"
    )

    optimizer = AdamW(
        [
            {
                "params": trainable_parameters,
                "lr": MEMORY_LR,
                "weight_decay": 0.0,
                "name": "embeddings",
            }
        ]
    )
    memory_before = tensor_snapshot(memory_parameters)
    head_before = tensor_snapshot(head_parameters)

    logits, hidden_states = _forward_with_optional_hidden_states(
        model,
        input_ids,
        attention_mask,
        output_hidden_states=True,
        final_hidden_state_only=True,
    )
    assert torch.isfinite(logits).all(), "Non-finite LM logits"
    assert hidden_states is not None, "Final hidden states are missing"
    assert torch.isfinite(
        hidden_states
    ).all(), "Non-finite final hidden states"

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels != -100
    (
        boundary_hidden_states,
        tool_targets,
        batch_indices,
        time_indices,
        initial_site_count,
        eoc_site_count,
    ) = gather_logit_bias_examples(
        hidden_states,
        labels,
        model,
        return_indices=True,
    )
    assert (
        tool_targets.numel() > 0
    ), "No TapMem tool-head supervision sites"
    shift_logits = apply_logit_train_add(
        model,
        shift_logits,
        boundary_hidden_states,
        batch_indices,
        time_indices,
        detach=True,
        detach_head_from_ar_loss=False,
    )
    _, logit_bias_loss = compute_logit_bias_loss(
        model,
        boundary_hidden_states,
        tool_targets,
        detach=True,
    )
    ar_loss_values = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    ar_loss = ar_loss_values[valid_mask].mean()
    total_loss = ar_loss + 0.1 * logit_bias_loss
    assert torch.isfinite(total_loss), "Non-finite total loss"

    masks = build_shift_supervision_masks(
        shift_labels,
        model,
        use_eoc=True,
    )
    print(
        f"loss_total={float(total_loss.item()):.8f},"
        f"loss_ar={float(ar_loss.item()):.8f},"
        f"loss_logit_bias={float(logit_bias_loss.item()):.8f}"
    )
    print(
        f"sites_tool={int(masks['tool_mask'].sum().item())},"
        f"sites_eoc={int(masks['eoc_mask'].sum().item())},"
        f"sites_head={int(tool_targets.numel())},"
        f"sites_head_initial={initial_site_count},"
        f"sites_head_eoc={eoc_site_count}"
    )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.cuda.synchronize(device)

    base_grad_names = [
        name
        for name, parameter in base_named_parameters
        if parameter.grad is not None
    ]
    assert not base_grad_names, (
        "Base LM unexpectedly received gradients: "
        + ", ".join(base_grad_names[:10])
    )

    memory_missing, memory_nonfinite, memory_zero, memory_norms = (
        grad_audit(
            [
                (
                    "trainable_tool_embeddings",
                    model.trainable_tool_embeddings,
                )
            ]
        )
    )
    assert not memory_missing, (
        f"Missing memory gradients: {memory_missing}"
    )
    assert not memory_nonfinite, (
        f"Non-finite memory gradients: {memory_nonfinite}"
    )
    assert not memory_zero, f"Zero memory gradients: {memory_zero}"

    head_missing, head_nonfinite, head_zero, head_norms = (
        grad_audit(
            [
                (f"logit_bias_head.{name}", parameter)
                for name, parameter in head_named_parameters
            ]
        )
    )
    assert not head_missing, f"Missing head gradients: {head_missing}"
    assert not head_nonfinite, (
        f"Non-finite head gradients: {head_nonfinite}"
    )
    assert not head_zero, f"Zero head gradients: {head_zero}"

    optimizer.step()
    torch.cuda.synchronize(device)
    memory_max_delta, memory_changed_count = snapshot_delta(
        memory_before,
        memory_parameters,
    )
    head_max_delta, head_changed_count = snapshot_delta(
        head_before,
        head_parameters,
    )
    assert memory_changed_count == len(memory_parameters), (
        "Memory embeddings did not update"
    )
    assert head_changed_count == len(head_parameters), (
        "Some logit-bias parameters did not update"
    )
    assert math.isfinite(memory_max_delta) and memory_max_delta > 0.0
    assert math.isfinite(head_max_delta) and head_max_delta > 0.0

    peak_allocated_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )
    peak_reserved_gib = (
        torch.cuda.max_memory_reserved(device) / 1024**3
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    print("base_lm_requires_grad_count=0")
    print("base_lm_grad_count=0")
    print(
        "memory_grad_norm="
        f"{memory_norms['trainable_tool_embeddings']:.8f}"
    )
    for name, norm in sorted(head_norms.items()):
        print(f"{name}_grad_norm={norm:.8f}")
    print(f"memory_max_abs_step_delta={memory_max_delta:.8f}")
    print(f"head_max_abs_step_delta={head_max_delta:.8f}")
    print(f"peak_allocated_gib={peak_allocated_gib:.3f}")
    print(f"peak_reserved_gib={peak_reserved_gib:.3f}")
    print(f"device_total_gib={total_bytes / 1024**3:.3f}")
    print(
        f"device_free_after_step_gib={free_bytes / 1024**3:.3f}"
    )
    print(f"elapsed_seconds={time.time() - started:.3f}")
    print("SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
