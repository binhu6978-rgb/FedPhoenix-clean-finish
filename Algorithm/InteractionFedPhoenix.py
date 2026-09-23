"""Deterministic repeated-interaction dispatch on FedPhoenix task models.

Only Conv2d.weight kernels, Linear parameters, and BatchNorm affine
parameters enter the VGG interaction geometry. Reset masks remove direct
reset-contaminated coordinates; they cannot remove indirect SGD effects.
"""

import math

import torch
from torch import nn


EPS = 1e-12


class InteractionGeometry:
    def __init__(self, model, space="reset_aware"):
        if space not in {"head", "reset_aware"}:
            raise ValueError("space must be head or reset_aware")
        self.space = space
        modules = dict(model.named_modules())
        parameters = dict(model.named_parameters())
        conv_followed_by_bn = set()
        for parent_name, parent in modules.items():
            if not isinstance(parent, nn.Sequential):
                continue
            children = list(parent.named_children())
            for (conv_name, conv), (_, following) in zip(children, children[1:]):
                if isinstance(conv, nn.Conv2d) and isinstance(
                    following, nn.modules.batchnorm._BatchNorm
                ):
                    conv_followed_by_bn.add(
                        f"{parent_name}.{conv_name}" if parent_name else conv_name
                    )
        for name, module in modules.items():
            if (isinstance(module, nn.Conv2d) and module.bias is not None
                    and name not in conv_followed_by_bn):
                raise ValueError(
                    f"Conv bias exclusion is only validated for Conv-BN pairs: {name}"
                )
        self.conv_layers = {
            name: int(module.weight.shape[0]) for name, module in modules.items()
            if isinstance(module, nn.Conv2d)
        }
        self.names = []
        self.conv_by_param = {}
        self.slices = {}
        offset = 0
        for name, parameter in parameters.items():
            if not parameter.requires_grad:
                continue
            module_name, _, local_name = name.rpartition(".")
            module = modules[module_name]
            is_conv_weight = isinstance(module, nn.Conv2d) and local_name == "weight"
            is_head = isinstance(module, nn.Linear) or (
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                and local_name in {"weight", "bias"}
            )
            if not (is_head or (space == "reset_aware" and is_conv_weight)):
                continue
            self.names.append(name)
            self.slices[name] = (offset, offset + parameter.numel(), tuple(parameter.shape))
            offset += parameter.numel()
            if is_conv_weight:
                self.conv_by_param[name] = module_name
        if not self.names:
            raise ValueError("interaction geometry has no trainable parameters")
        self.names = tuple(self.names)
        self.numel = offset
        self.conv_numel = sum(
            self.slices[name][1] - self.slices[name][0] for name in self.conv_by_param
        )

    def flatten(self, state):
        return torch.cat([
            state[name].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            for name in self.names
        ])

    def mask(self, excluded_kernels):
        """Expand kernel-level reset masks only for current geometry operations."""
        result = torch.ones(self.numel, dtype=torch.bool)
        for name, layer in self.conv_by_param.items():
            start, end, shape = self.slices[name]
            keep = ~excluded_kernels[layer]
            result[start:end] = keep.repeat_interleave((end - start) // shape[0])
        return result

    def materialize(self, model, base_state, delta):
        """Apply a proposed action and return its actually represented tensors."""
        if delta.shape != (self.numel,) or not bool(torch.isfinite(delta).all()):
            raise ValueError("invalid dispatch delta")
        state = model.state_dict()
        materialized = {}
        with torch.no_grad():
            for name in self.names:
                start, end, shape = self.slices[name]
                chunk = delta[start:end]
                if not bool(torch.any(chunk != 0)):
                    continue
                tensor = state[name]
                before = base_state[name].detach().to(tensor.device, dtype=tensor.dtype)
                tensor.add_(chunk.reshape(shape).to(tensor.device, dtype=tensor.dtype))
                difference = tensor.detach().clone() - before
                if not bool(torch.isfinite(difference).all()):
                    raise ValueError(f"non-finite materialized action in {name}")
                if bool(torch.any(difference != 0)):
                    materialized[name] = difference
        return materialized


class ResetLedger:
    def __init__(self, model):
        self.kernel_counts = {
            name: int(module.weight.shape[0])
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)
        }
        self.rounds = []

    def empty(self):
        return {name: torch.zeros(count, dtype=torch.bool)
                for name, count in self.kernel_counts.items()}

    def from_trace(self, trace):
        mask = self.empty()
        for layer in trace["layers"]:
            name = layer["name"]
            if name not in mask or int(layer["num_kernels"]) != mask[name].numel():
                raise ValueError("reset trace does not match model Conv geometry")
            mask[name][layer["reset_indices"]] = True
        return mask

    @staticmethod
    def union_into(target, source):
        for name in target:
            target[name] |= source[name]

    def record(self, round_idx, traces):
        if round_idx != len(self.rounds):
            raise ValueError("reset ledger rounds must be chronological")
        combined = self.empty()
        for trace in traces:
            self.union_into(combined, self.from_trace(trace))
        self.rounds.append(combined)
        return combined

    def interval_union(self, start, stop):
        """Union L[start], ..., L[stop-1]."""
        if not 0 <= start <= stop <= len(self.rounds):
            raise ValueError("invalid reset ledger interval")
        combined = self.empty()
        for round_mask in self.rounds[start:stop]:
            self.union_into(combined, round_mask)
        return combined

    def previous(self, round_idx):
        return self.rounds[round_idx - 1] if round_idx > 0 else self.empty()


class PackedObservation:
    """BF16 observation with invalid Conv kernels omitted entirely."""

    def __init__(self, geometry, vector, valid_kernels):
        self.parts = {}
        for name in geometry.names:
            start, end, shape = geometry.slices[name]
            piece = vector[start:end].reshape(shape)
            if name in geometry.conv_by_param:
                piece = piece[valid_kernels[geometry.conv_by_param[name]]]
            self.parts[name] = piece.to(torch.bfloat16).clone()

    def unpack(self, geometry, valid_kernels):
        result = torch.zeros(geometry.numel, dtype=torch.float32)
        for name in geometry.names:
            start, end, shape = geometry.slices[name]
            if name in geometry.conv_by_param:
                piece = torch.zeros(shape, dtype=torch.float32)
                piece[valid_kernels[geometry.conv_by_param[name]]] = (
                    self.parts[name].to(torch.float32)
                )
                result[start:end] = piece.reshape(-1)
            else:
                result[start:end] = self.parts[name].to(torch.float32).reshape(-1)
        return result

    def bytes(self):
        return sum(value.numel() * value.element_size() for value in self.parts.values())


class ClientHistory:
    def __init__(self, geometry, ledger):
        self.geometry = geometry
        self.ledger = ledger
        self.clients = {}

    def record(self, client_id, round_idx, actual_dispatch, client_update, own_reset):
        if (actual_dispatch.shape != (self.geometry.numel,) or
                client_update.shape != (self.geometry.numel,)):
            raise ValueError("interaction geometry mismatch")
        if not (bool(torch.isfinite(actual_dispatch).all()) and
                bool(torch.isfinite(client_update).all())):
            raise ValueError("non-finite interaction")
        client_id = int(client_id)
        previous = self.clients.get(client_id)
        update_float = client_update.detach().to("cpu", torch.float32)
        update_bf16 = update_float.to(torch.bfloat16)
        if not bool(torch.isfinite(update_bf16).all()):
            raise ValueError("client update cannot be stored as bf16")
        item = {
            "x": actual_dispatch.detach().to("cpu", torch.float32).clone(),
            "u": update_bf16.clone(),
            "d": None, "r": None, "hist_valid": None,
            "t_latest": int(round_idx), "t_prev": None,
            "R_latest": {k: v.clone() for k, v in own_reset.items()},
            "visits": 1 if previous is None else previous["visits"] + 1,
            "obs_valid": False,
        }
        if previous is not None:
            if round_idx <= previous["t_latest"]:
                raise ValueError("same-client interactions must be chronological")
            excluded = self.ledger.interval_union(previous["t_latest"], round_idx)
            self.ledger.union_into(excluded, own_reset)
            valid = {name: ~value for name, value in excluded.items()}
            hist_mask = self.geometry.mask(excluded)
            d = (item["x"] - previous["x"]) * hist_mask
            r = (update_float - previous["u"].to(torch.float32)) * hist_mask
            if bool(torch.isfinite(d).all()) and bool(torch.isfinite(r).all()):
                item["d"] = PackedObservation(self.geometry, d, valid)
                item["r"] = PackedObservation(self.geometry, r, valid)
                item["hist_valid"] = valid
                item["t_prev"] = previous["t_latest"]
                item["obs_valid"] = True
        self.clients[client_id] = item

    def bytes(self):
        total = 0
        for item in self.clients.values():
            for key in ("x", "u"):
                tensor = item[key]
                total += tensor.numel() * tensor.element_size()
            for key in ("d", "r"):
                if item[key] is not None:
                    total += item[key].bytes()
            for key in ("hist_valid", "R_latest"):
                if item[key] is not None:
                    total += sum(v.numel() * v.element_size() for v in item[key].values())
        return total


class InteractionController:
    def __init__(self, geometry, ledger, rho=0.05, max_gap=20, observe_only=False):
        if rho < 0 or max_gap < 1:
            raise ValueError("rho must be non-negative and max_gap positive")
        self.geometry = geometry
        self.ledger = ledger
        self.history = ClientHistory(geometry, ledger)
        self.rho = float(rho)
        self.max_gap = int(max_gap)
        self.observe_only = bool(observe_only)

    def decide(self, client_id, round_idx, current_reset, global_change):
        """Uses past completed interactions and the current client's own trace."""
        info = {"client_id": int(client_id), "reason": "cold_start", "gap": None,
                "c": None, "q": None, "s_raw": None, "s_cos": None,
                "lambda": 0.0, "d_norm": None, "r_norm": None,
                "u_norm": None, "g_norm": None, "delta_norm": 0.0,
                "head_delta_norm": 0.0, "backbone_delta_norm": 0.0,
                "hist_coverage": None, "decision_coverage": None,
                "ledger_excluded_ratio": None}
        item = self.history.clients.get(int(client_id))
        if item is None or item["visits"] < 2:
            return None, info
        if item["d"] is None or item["r"] is None:
            info["reason"] = "no_obs"
            return None, info
        gap = int(round_idx) - item["t_latest"]
        info["gap"] = gap
        if gap < 1 or gap > self.max_gap:
            info["reason"] = "stale"
            return None, info
        if global_change is None:
            info["reason"] = "no_g"
            return None, info
        excluded = {name: ~valid.clone() for name, valid in item["hist_valid"].items()}
        previous_reset = self.ledger.previous(round_idx)
        before = self.geometry.mask(excluded)
        self.ledger.union_into(excluded, previous_reset)
        after_ledger = self.geometry.mask(excluded)
        self.ledger.union_into(excluded, current_reset)
        mask = self.geometry.mask(excluded)
        info["hist_coverage"] = float(before.float().mean())
        info["decision_coverage"] = float(mask.float().mean())
        info["ledger_excluded_ratio"] = float((before & ~after_ledger).float().mean())
        d = item["d"].unpack(self.geometry, item["hist_valid"]) * mask
        r = item["r"].unpack(self.geometry, item["hist_valid"]) * mask
        u = item["u"] * mask
        g = global_change * mask
        d_norm = float(torch.linalg.vector_norm(d.to(torch.float64)))
        r_norm = float(torch.linalg.vector_norm(r.to(torch.float64)))
        u_norm = float(torch.linalg.vector_norm(u.to(torch.float64)))
        g_norm = float(torch.linalg.vector_norm(g.to(torch.float64)))
        info.update(d_norm=d_norm, r_norm=r_norm, u_norm=u_norm, g_norm=g_norm)
        if not all(math.isfinite(v) for v in (d_norm, r_norm, u_norm, g_norm)):
            info["reason"] = "no_obs"
            return None, info
        if d_norm <= EPS:
            info["reason"] = "zero_d"
            return None, info
        if g_norm <= EPS:
            info["reason"] = "no_g"
            return None, info
        g_hat = g.to(torch.float64) / (g_norm + EPS)
        c = -float(torch.dot(u.to(torch.float64), g_hat))
        q = float(torch.dot(r.to(torch.float64), g_hat))
        info.update(c=c, q=q, s_raw=q / (d_norm + EPS),
                    s_cos=q / (r_norm + EPS))
        if not math.isfinite(c) or not math.isfinite(q):
            info["reason"] = "no_obs"
            return None, info
        if c <= 0:
            info["reason"] = "no_need"
            return None, info
        if q <= 0:
            info["reason"] = "no_support"
            return None, info
        if self.observe_only or self.rho == 0:
            info["reason"] = "observe_only"
            return None, info
        lam = min(self.rho, c / max(q, EPS))
        if not math.isfinite(lam) or not 0 <= lam <= self.rho:
            info["reason"] = "no_obs"
            return None, info
        delta = (d.to(torch.float64) * lam).to(torch.float32)
        if not bool(torch.isfinite(delta).all()):
            info["reason"] = "no_obs"
            return None, info
        backbone_sq = 0.0
        head_sq = 0.0
        for name in self.geometry.names:
            start, end, _ = self.geometry.slices[name]
            squared = float(torch.sum(delta[start:end].to(torch.float64) ** 2))
            if name in self.geometry.conv_by_param:
                backbone_sq += squared
            else:
                head_sq += squared
        info.update(reason="active", **{"lambda": lam},
                    delta_norm=float(torch.linalg.vector_norm(delta.to(torch.float64))),
                    backbone_delta_norm=math.sqrt(backbone_sq),
                    head_delta_norm=math.sqrt(head_sq))
        return delta, info


def corrected_state(returned_state, materialized_action):
    """Return y-(x-b) for active parameters, preserving every other y tensor."""
    result = {name: value.detach().clone() for name, value in returned_state.items()}
    for name, action in materialized_action.items():
        result[name] = result[name] - action.to(result[name].device, result[name].dtype)
    return result
