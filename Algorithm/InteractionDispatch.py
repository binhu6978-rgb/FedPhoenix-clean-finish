#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Server-side repeated-interaction controller for InteractionDispatch.

Only trainable parameters participate in the response geometry.  Model
buffers are cloned and transported with state dicts, but are never flattened,
measured, or changed by a dispatch delta.
"""

from collections import OrderedDict
import math

import torch


def clone_state_to_cpu(state_dict):
    """Detach, move to CPU, and clone every tensor in a state dict."""
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in state_dict.items()
    )


def flatten_trainable_state(state_dict, param_names):
    """Flatten named trainable parameters into one CPU float32 tensor."""
    missing = [name for name in param_names if name not in state_dict]
    if missing:
        raise KeyError(f"Missing trainable parameter keys: {missing}")
    if not param_names:
        return torch.empty(0, dtype=torch.float32, device="cpu")
    return torch.cat(
        [
            state_dict[name]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
            for name in param_names
        ]
    )


def add_flat_delta_to_state(
    base_state,
    delta,
    param_names,
    param_shapes,
    param_offsets,
):
    """Return an independent state with ``delta`` added to parameters only."""
    delta = delta.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    expected = sum(int(math.prod(param_shapes[name])) for name in param_names)
    if delta.numel() != expected:
        raise ValueError(
            f"Dispatch delta has {delta.numel()} elements; expected {expected}"
        )

    result = clone_state_to_cpu(base_state)
    for name in param_names:
        if name not in result:
            raise KeyError(f"Missing trainable parameter key: {name}")
        start, end = param_offsets[name]
        chunk = delta[start:end].reshape(param_shapes[name])
        base_tensor = result[name]
        if not torch.is_floating_point(base_tensor):
            raise TypeError(f"Trainable parameter {name} is not floating point")
        result[name] = base_tensor + chunk.to(dtype=base_tensor.dtype)
    return result


def build_corrected_local_state(
    global_state,
    dispatch_state,
    returned_state,
    param_names,
):
    """Remove the direct server dispatch action from a returned local state.

    Returned buffers are preserved.  Trainable parameters become
    ``global + returned - dispatch``, so the existing model-averaging routine
    aggregates only client-generated updates relative to actual dispatches.
    """
    result = clone_state_to_cpu(returned_state)
    for name in param_names:
        if name not in global_state or name not in dispatch_state or name not in result:
            raise KeyError(f"Missing trainable parameter key: {name}")
        returned_tensor = result[name]
        corrected = (
            global_state[name].detach().to(device="cpu", dtype=torch.float32)
            + returned_state[name].detach().to(device="cpu", dtype=torch.float32)
            - dispatch_state[name].detach().to(device="cpu", dtype=torch.float32)
        )
        result[name] = corrected.to(dtype=returned_tensor.dtype)
    return result


class InteractionDispatchController:
    """Maintain each client's latest real input/update response observation."""

    def __init__(self, model, tau=0.0, max_step_ratio=0.1, eps=1e-12):
        if tau < 0:
            raise ValueError("tau must be non-negative")
        if max_step_ratio < 0:
            raise ValueError("max_step_ratio must be non-negative")
        self.tau = float(tau)
        self.max_step_ratio = float(max_step_ratio)
        self.eps = float(eps)
        self.client_states = {}

        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.param_names = [name for name, _ in named_parameters]
        self.param_shapes = {
            name: tuple(parameter.shape) for name, parameter in named_parameters
        }
        self.param_offsets = {}
        offset = 0
        for name, parameter in named_parameters:
            next_offset = offset + parameter.numel()
            self.param_offsets[name] = (offset, next_offset)
            offset = next_offset
        self.num_params = int(offset)

    def _base_info(self, client_id, state):
        return {
            "client_id": int(client_id),
            "num_visits_before": int(state["num_visits"]) if state else 0,
            "has_observation": bool(
                state
                and state.get("obs_direction") is not None
                and state.get("obs_response") is not None
            ),
            "conflict": None,
            "support": None,
            "alpha": 0.0,
            "obs_input_norm": float(state.get("obs_input_norm", 0.0)) if state else 0.0,
            "reason": "cold_start",
        }

    def make_dispatch_delta(self, client_id, global_direction_unit):
        """Return a bounded experienced-direction dispatch delta, if supported."""
        client_id = int(client_id)
        state = self.client_states.get(client_id)
        info = self._base_info(client_id, state)

        if state is None or state["num_visits"] < 2:
            return None, info
        if not info["has_observation"]:
            info["reason"] = "no_observation"
            return None, info
        if global_direction_unit is None:
            info["reason"] = "no_global_direction"
            return None, info

        global_direction_unit = global_direction_unit.detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(-1)
        if global_direction_unit.numel() != self.num_params:
            raise ValueError(
                "Global direction size does not match the controller parameter geometry"
            )
        if not bool(torch.isfinite(global_direction_unit).all()):
            info["reason"] = "no_global_direction"
            return None, info

        last_update = state["last_update"].to(dtype=torch.float32)
        conflict = -float(torch.dot(last_update, global_direction_unit).item())
        info["conflict"] = conflict
        if not math.isfinite(conflict) or conflict <= self.tau:
            info["reason"] = "no_need"
            return None, info

        response = state["obs_response"].to(dtype=torch.float32)
        support = float(torch.dot(response, global_direction_unit).item())
        info["support"] = support
        if not math.isfinite(support) or support <= self.eps:
            info["reason"] = "no_support"
            return None, info

        alpha_raw = (conflict - self.tau) / support
        alpha_max = self.max_step_ratio * float(state["obs_input_norm"])
        alpha = min(alpha_raw, alpha_max)
        if not math.isfinite(alpha) or alpha <= 0.0:
            info["reason"] = "no_support"
            return None, info

        direction = state["obs_direction"].to(dtype=torch.float32)
        delta = alpha * direction
        if not bool(torch.isfinite(delta).all()):
            info["reason"] = "no_support"
            return None, info

        info["alpha"] = float(alpha)
        info["reason"] = "active"
        return delta, info

    @staticmethod
    def _history_half(vector, label):
        stored = vector.detach().to(device="cpu", dtype=torch.float16).clone()
        if not bool(torch.isfinite(stored).all()):
            raise OverflowError(f"{label} cannot be represented in CPU float16")
        return stored

    def record_interaction(
        self,
        client_id,
        dispatch_vector,
        client_update_vector,
        round_idx,
    ):
        """Record an actual dispatch/update pair and refresh the secant response."""
        client_id = int(client_id)
        for label, vector in (
            ("dispatch_vector", dispatch_vector),
            ("client_update_vector", client_update_vector),
        ):
            if vector.device.type != "cpu" or vector.dtype != torch.float32:
                raise TypeError(f"{label} must be a CPU float32 tensor")
            if vector.ndim != 1 or vector.numel() != self.num_params:
                raise ValueError(
                    f"{label} must be flat with {self.num_params} elements"
                )
            if not bool(torch.isfinite(vector).all()):
                raise ValueError(f"{label} contains non-finite values")

        state = self.client_states.get(client_id)
        if state is None:
            self.client_states[client_id] = {
                "num_visits": 1,
                "last_dispatch": self._history_half(
                    dispatch_vector, "dispatch_vector"
                ),
                "last_update": self._history_half(
                    client_update_vector, "client_update_vector"
                ),
                "obs_direction": None,
                "obs_response": None,
                "obs_input_norm": 0.0,
                "last_round": int(round_idx),
            }
            return

        previous_dispatch = state["last_dispatch"].to(dtype=torch.float32)
        previous_update = state["last_update"].to(dtype=torch.float32)
        input_change = dispatch_vector - previous_dispatch
        response_change = client_update_vector - previous_update
        input_norm = float(torch.linalg.vector_norm(input_change).item())

        obs_direction = None
        obs_response = None
        obs_input_norm = 0.0
        if math.isfinite(input_norm) and input_norm > self.eps:
            direction_float = input_change / input_norm
            response_float = response_change / input_norm
            if bool(torch.isfinite(direction_float).all()) and bool(
                torch.isfinite(response_float).all()
            ):
                direction_half = direction_float.to(dtype=torch.float16)
                response_half = response_float.to(dtype=torch.float16)
                if bool(torch.isfinite(direction_half).all()) and bool(
                    torch.isfinite(response_half).all()
                ):
                    obs_direction = direction_half.cpu().clone()
                    obs_response = response_half.cpu().clone()
                    obs_input_norm = input_norm

        state.update(
            {
                "num_visits": int(state["num_visits"]) + 1,
                "last_dispatch": self._history_half(
                    dispatch_vector, "dispatch_vector"
                ),
                "last_update": self._history_half(
                    client_update_vector, "client_update_vector"
                ),
                "obs_direction": obs_direction,
                "obs_response": obs_response,
                "obs_input_norm": float(obs_input_norm),
                "last_round": int(round_idx),
            }
        )

    def num_clients_with_observation(self):
        return sum(
            state["obs_direction"] is not None
            and state["obs_response"] is not None
            for state in self.client_states.values()
        )

    def history_memory_bytes(self):
        """Return storage occupied by persistent history tensors."""
        total = 0
        for state in self.client_states.values():
            for key in (
                "last_dispatch",
                "last_update",
                "obs_direction",
                "obs_response",
            ):
                tensor = state.get(key)
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return int(total)
