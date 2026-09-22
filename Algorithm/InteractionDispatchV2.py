#!/usr/bin/env python
"""InteractionDispatchV2 parameter geometry and persistent controller.

The response observation is deliberately directional: it gates an action but
never predicts its magnitude.  All persistent tensors live on CPU.
"""

from collections import OrderedDict
import math

import torch


def clone_state_to_cpu(state):
    return OrderedDict(
        (name, tensor.detach().to("cpu").clone()) for name, tensor in state.items()
    )


class ParameterGeometry:
    """Fixed flat geometry over trainable named parameters (never buffers)."""

    def __init__(self, model):
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.param_names = [name for name, _ in named]
        self.param_shapes = OrderedDict(
            (name, tuple(parameter.shape)) for name, parameter in named
        )
        self.param_offsets = OrderedDict()
        self.param_numels = OrderedDict()
        self.group_names = []
        self.group_indices = OrderedDict()
        offset = 0
        group_members = OrderedDict()
        for name, parameter in named:
            numel = int(parameter.numel())
            self.param_offsets[name] = (offset, offset + numel)
            self.param_numels[name] = numel
            offset += numel
            prefix = name.split(".", 1)[0]
            group = prefix if prefix in {"features", "classifier", "fc"} else "all"
            group_members.setdefault(group, []).append(name)
        self.num_params = int(offset)
        self.group_names = list(group_members)
        for group, names in group_members.items():
            indices = []
            for name in names:
                start, end = self.param_offsets[name]
                indices.append((start, end))
            self.group_indices[group] = indices

    def flatten_state(self, state_or_model):
        state = (
            state_or_model.state_dict()
            if hasattr(state_or_model, "state_dict")
            else state_or_model
        )
        result = torch.empty(self.num_params, dtype=torch.float32, device="cpu")
        for name in self.param_names:
            start, end = self.param_offsets[name]
            source = state[name].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            result[start:end].copy_(source)
        return result

    def add_delta_to_state(self, base_state, delta):
        delta = delta.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if delta.numel() != self.num_params:
            raise ValueError("delta size does not match parameter geometry")
        result = clone_state_to_cpu(base_state)
        for name in self.param_names:
            start, end = self.param_offsets[name]
            base = result[name]
            changed = base.float() + delta[start:end].reshape(self.param_shapes[name])
            result[name] = changed.to(dtype=base.dtype)
        return result

    def split_vector_by_parameter(self, vector):
        vector = vector.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if vector.numel() != self.num_params:
            raise ValueError("vector size does not match parameter geometry")
        return OrderedDict(
            (
                name,
                vector[start:end].reshape(self.param_shapes[name]),
            )
            for name, (start, end) in self.param_offsets.items()
        )

    def group_projection_diagnostics(self, update, response_direction, global_direction):
        diagnostics = OrderedDict()
        for group, spans in self.group_indices.items():
            conflict = 0.0
            support = 0.0
            update_sq = 0.0
            numel = 0
            for start, end in spans:
                u = update[start:end]
                r = response_direction[start:end]
                g = global_direction[start:end]
                conflict -= float(torch.dot(u, g).item())
                support += float(torch.dot(r, g).item())
                update_sq += float(torch.dot(u, u).item())
                numel += end - start
            diagnostics[group] = {
                "parameter_numel": int(numel),
                "conflict": float(conflict),
                "support": float(support),
                "update_norm": float(math.sqrt(max(update_sq, 0.0))),
                "response_direction_projection": float(support),
            }
        return diagnostics


class InteractionDispatchV2Controller:
    """Store the latest real interaction and directional secant evidence."""

    def __init__(self, model, step_ratio=0.2, eps=1e-12):
        if step_ratio < 0:
            raise ValueError("step_ratio must be non-negative")
        self.step_ratio = float(step_ratio)
        self.eps = float(eps)
        self.geometry = ParameterGeometry(model)
        self.client_states = {}

    @property
    def param_names(self):
        return self.geometry.param_names

    @property
    def num_params(self):
        return self.geometry.num_params

    def _info(self, client_id, state):
        return {
            "client_id": int(client_id),
            "num_visits_before": int(state["num_visits"]) if state else 0,
            "has_observation": bool(
                state
                and state.get("obs_direction") is not None
                and state.get("obs_response_direction") is not None
            ),
            "conflict": None,
            "support": None,
            "total_conflict": None,
            "total_support": None,
            "alpha": 0.0,
            "effective_ratio": 0.0,
            "obs_input_norm": float(state.get("obs_input_norm", 0.0)) if state else 0.0,
            "last_round": int(state["last_round"]) if state else None,
            "reason": "cold_start",
            "group_diagnostics": {},
        }

    def make_dispatch_delta(self, client_id, global_direction_unit):
        client_id = int(client_id)
        state = self.client_states.get(client_id)
        info = self._info(client_id, state)
        if state is None or int(state["num_visits"]) < 2:
            return None, info
        if not info["has_observation"]:
            info["reason"] = "no_observation"
            return None, info
        if global_direction_unit is None:
            info["reason"] = "no_global_direction"
            return None, info

        g = global_direction_unit.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if g.numel() != self.num_params or not bool(torch.isfinite(g).all()):
            info["reason"] = "no_global_direction"
            return None, info
        g_norm = float(torch.linalg.vector_norm(g).item())
        if not math.isfinite(g_norm) or g_norm <= self.eps:
            info["reason"] = "no_global_direction"
            return None, info
        g = g / g_norm

        update = state["last_update"].float()
        response = state["obs_response_direction"].float()
        response_norm = float(torch.linalg.vector_norm(response).item())
        if not math.isfinite(response_norm) or response_norm <= self.eps:
            info["reason"] = "no_observation"
            return None, info
        response = response / response_norm
        conflict = -float(torch.dot(update, g).item())
        support = float(torch.dot(response, g).item())
        info.update(
            {
                "conflict": conflict,
                "support": support,
                "total_conflict": conflict,
                "total_support": support,
                "group_diagnostics": self.geometry.group_projection_diagnostics(
                    update, response, g
                ),
            }
        )
        if not math.isfinite(conflict) or conflict <= 0.0:
            info["reason"] = "no_need"
            return None, info
        if not math.isfinite(support) or support <= 0.0:
            info["reason"] = "no_support"
            return None, info

        direction = state["obs_direction"].float()
        direction_norm = float(torch.linalg.vector_norm(direction).item())
        if not math.isfinite(direction_norm) or direction_norm <= self.eps:
            info["reason"] = "no_observation"
            return None, info
        direction = direction / direction_norm
        alpha = self.step_ratio * float(state["obs_input_norm"])
        if not math.isfinite(alpha):
            info["reason"] = "no_observation"
            return None, info
        delta = direction * alpha
        info["alpha"] = float(alpha)
        info["effective_ratio"] = (
            float(alpha / state["obs_input_norm"])
            if float(state["obs_input_norm"]) > self.eps
            else 0.0
        )
        info["reason"] = "active"
        return delta, info

    def _validate_vector(self, vector, label):
        if vector.device.type != "cpu" or vector.dtype != torch.float32:
            raise TypeError(f"{label} must be CPU float32")
        if vector.ndim != 1 or vector.numel() != self.num_params:
            raise ValueError(f"{label} has invalid geometry")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError(f"{label} contains non-finite values")

    def record_interaction(self, client_id, dispatch_vector, client_update_vector, round_idx):
        self._validate_vector(dispatch_vector, "dispatch_vector")
        self._validate_vector(client_update_vector, "client_update_vector")
        client_id = int(client_id)
        state = self.client_states.get(client_id)
        if state is None:
            self.client_states[client_id] = {
                "num_visits": 1,
                "last_dispatch": dispatch_vector.detach().clone(),
                "last_update": client_update_vector.detach().clone(),
                "obs_direction": None,
                "obs_response_direction": None,
                "obs_input_norm": 0.0,
                "last_round": int(round_idx),
            }
            return

        d = dispatch_vector - state["last_dispatch"]
        r = client_update_vector - state["last_update"]
        d_norm = float(torch.linalg.vector_norm(d).item())
        r_norm = float(torch.linalg.vector_norm(r).item())
        obs_direction = None
        obs_response_direction = None
        obs_input_norm = 0.0
        if (
            math.isfinite(d_norm)
            and math.isfinite(r_norm)
            and d_norm > self.eps
            and r_norm > self.eps
        ):
            d_hat = d / d_norm
            r_hat = r / r_norm
            d_half = d_hat.to(dtype=torch.float16)
            r_half = r_hat.to(dtype=torch.float16)
            if bool(torch.isfinite(d_half).all()) and bool(torch.isfinite(r_half).all()):
                obs_direction = d_half.detach().clone()
                obs_response_direction = r_half.detach().clone()
                obs_input_norm = d_norm
        state.update(
            {
                "num_visits": int(state["num_visits"]) + 1,
                "last_dispatch": dispatch_vector.detach().clone(),
                "last_update": client_update_vector.detach().clone(),
                "obs_direction": obs_direction,
                "obs_response_direction": obs_response_direction,
                "obs_input_norm": float(obs_input_norm),
                "last_round": int(round_idx),
            }
        )

    def num_clients_seen(self):
        return len(self.client_states)

    def num_clients_with_observation(self):
        return sum(
            state.get("obs_direction") is not None
            and state.get("obs_response_direction") is not None
            for state in self.client_states.values()
        )

    def history_memory_bytes(self):
        total = 0
        for state in self.client_states.values():
            for name in (
                "last_dispatch",
                "last_update",
                "obs_direction",
                "obs_response_direction",
            ):
                tensor = state.get(name)
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return int(total)
