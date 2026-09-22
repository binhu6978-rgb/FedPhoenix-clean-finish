#!/usr/bin/env python
"""Explicit response aggregation for InteractionDispatchV2."""

from collections import OrderedDict

import torch


def _legacy_weighted_tensor(tensors, weights, accumulation_device="cpu"):
    value = tensors[0].detach().to(accumulation_device).clone() * weights[0]
    for tensor, weight in zip(tensors[1:], weights[1:]):
        value += tensor.detach().to(accumulation_device) * weight
    return torch.div(value, sum(weights)).to("cpu")


def aggregate_buffers_legacy_compatible(
    returned_states, weights, buffer_names, accumulation_device="cpu"
):
    """Reproduce legacy Aggregation arithmetic for non-parameter state."""
    return OrderedDict(
        (
            name,
            _legacy_weighted_tensor(
                [state[name] for state in returned_states], weights, accumulation_device
            ),
        )
        for name in buffer_names
    )


def aggregate_client_responses(
    global_state, client_records, param_names, accumulation_device="cpu"
):
    if not client_records:
        raise ValueError("client_records must not be empty")
    weights = [record["weight"] for record in client_records]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("client weights must be non-negative with positive total")
    param_names = list(param_names)
    param_set = set(param_names)
    returned_states = [record["returned_state"] for record in client_records]
    result = OrderedDict()

    # Exact legacy arithmetic is used only for the mathematical no-action case.
    # This makes step_ratio=0 a strict FedAvg degeneration check while active
    # V2 rounds use the explicit response equation below.
    no_dispatch_action = all(
        torch.equal(record["dispatch_state"][name], global_state[name])
        for record in client_records
        for name in param_names
    )
    total_weight = float(sum(weights))
    for name, global_tensor in global_state.items():
        if name not in param_set:
            result[name] = _legacy_weighted_tensor(
                [state[name] for state in returned_states], weights, accumulation_device
            )
        elif no_dispatch_action:
            result[name] = _legacy_weighted_tensor(
                [state[name] for state in returned_states], weights, accumulation_device
            )
        else:
            accumulator = torch.zeros_like(
                global_tensor, device=accumulation_device, dtype=torch.float32
            )
            for record, weight in zip(client_records, weights):
                returned = record["returned_state"][name].to(
                    accumulation_device, dtype=torch.float32
                )
                dispatched = record["dispatch_state"][name].to(
                    accumulation_device, dtype=torch.float32
                )
                accumulator.add_(returned - dispatched, alpha=float(weight))
            updated = global_tensor.detach().to(
                accumulation_device, dtype=torch.float32
            ) + accumulator / total_weight
            result[name] = updated.to(device="cpu", dtype=global_tensor.dtype)
    return result
