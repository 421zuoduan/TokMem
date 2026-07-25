#!/usr/bin/env python3
"""Validate the local causal-conv1d CUDA extension against its reference."""

import torch

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from causal_conv1d.causal_conv1d_interface import (
    causal_conv1d_ref,
    causal_conv1d_update_ref,
)


torch.manual_seed(42)
device = torch.device("cuda")
dtype = torch.bfloat16

x = torch.randn(2, 64, 128, device=device, dtype=dtype, requires_grad=True)
weight = torch.randn(64, 4, device=device, dtype=dtype, requires_grad=True)
bias = torch.randn(64, device=device, dtype=dtype, requires_grad=True)

fast = causal_conv1d_fn(x, weight, bias, activation="silu")
reference = causal_conv1d_ref(x, weight, bias, activation="silu")
forward_max_abs = (fast.float() - reference.float()).abs().max().item()
fast.float().square().mean().backward()

state_fast = torch.randn(2, 64, 4, device=device, dtype=dtype)
state_reference = state_fast.clone()
token = torch.randn(2, 64, device=device, dtype=dtype)
update_fast = causal_conv1d_update(
    token,
    state_fast,
    weight.detach(),
    bias.detach(),
    activation="silu",
)
update_reference = causal_conv1d_update_ref(
    token,
    state_reference,
    weight.detach(),
    bias.detach(),
    activation="silu",
)
update_max_abs = (
    update_fast.float() - update_reference.float()
).abs().max().item()
state_max_abs = (
    state_fast.float() - state_reference.float()
).abs().max().item()

torch.cuda.synchronize()
assert torch.isfinite(fast).all()
assert torch.isfinite(update_fast).all()
assert all(
    gradient is not None and torch.isfinite(gradient).all()
    for gradient in (x.grad, weight.grad, bias.grad)
)
assert forward_max_abs <= 0.0625
assert update_max_abs <= 0.0625
assert state_max_abs == 0.0

print(
    {
        "forward_max_abs": forward_max_abs,
        "update_max_abs": update_max_abs,
        "state_max_abs": state_max_abs,
        "backward_finite": True,
    }
)
