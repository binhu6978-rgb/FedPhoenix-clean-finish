#!/usr/bin/env python
"""Observer-only diagnostics for longitudinal FedPhoenix client specialization.

Only ``nn.Conv2d.weight`` tensors are inspected.  Persistent history contains
one float32 score and one boolean validity value per output kernel; dispatched
models and local-update tensors never survive the current round.
"""

from collections import OrderedDict
import json
import math
import random

import numpy as np
import torch
from torch import nn


SCORE_TYPES = ("update_norm", "residual", "norm_residual", "angular")


def capture_global_rng_state(include_cuda=True):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": None,
    }
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def global_rng_state_equal(left, right):
    if left["python"] != right["python"]:
        return False
    for left_item, right_item in zip(left["numpy"], right["numpy"]):
        if isinstance(left_item, np.ndarray):
            if not np.array_equal(left_item, right_item):
                return False
        elif left_item != right_item:
            return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    if (left["cuda"] is None) != (right["cuda"] is None):
        return False
    if left["cuda"] is not None:
        if len(left["cuda"]) != len(right["cuda"]):
            return False
        if any(not torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])):
            return False
    return True


class ConvKernelGeometry:
    """Fixed output-kernel geometry over Conv2d weights only."""

    def __init__(self, model, reset_ratio):
        if not 0 <= reset_ratio <= 1:
            raise ValueError("reset_ratio must be between zero and one")
        layers = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                parameter_name = f"{module_name}.weight" if module_name else "weight"
                layers.append(
                    {
                        "layer_name": module_name,
                        "parameter_name": parameter_name,
                        "output_kernels": int(module.weight.shape[0]),
                        "kernel_shape": tuple(module.weight.shape[1:]),
                        "reset_budget": int(module.weight.shape[0] * reset_ratio),
                    }
                )
        if not layers:
            raise ValueError("FedPhoenixHistoryObserver requires at least one Conv2d layer")
        self.layers = tuple(layers)
        self.by_name = OrderedDict((item["layer_name"], item) for item in layers)

    def reset_masks_from_trace(self, task_trace):
        traced = {item["name"]: item for item in task_trace.get("layers", [])}
        masks = OrderedDict()
        eligible = OrderedDict()
        for layer_name, layer in self.by_name.items():
            mask = torch.zeros(layer["output_kernels"], dtype=torch.bool, device="cpu")
            trace_layer = traced.get(layer_name)
            if trace_layer is not None:
                if int(trace_layer["num_kernels"]) != layer["output_kernels"]:
                    raise ValueError("task trace kernel count does not match model geometry")
                indices = [int(value) for value in trace_layer.get("reset_indices", [])]
                if indices:
                    mask[torch.tensor(indices, dtype=torch.long)] = True
                eligible[layer_name] = True
            else:
                eligible[layer_name] = False
            masks[layer_name] = mask
        return masks, eligible

    def conv_update(self, dispatch_state, returned_state):
        updates = OrderedDict()
        for layer in self.layers:
            name = layer["parameter_name"]
            dispatched = dispatch_state[name].detach().to(device="cpu", dtype=torch.float32)
            returned = returned_state[name].detach().to(device="cpu", dtype=torch.float32)
            update = (returned - dispatched).clone()
            expected = (layer["output_kernels"], *layer["kernel_shape"])
            if tuple(update.shape) != expected:
                raise ValueError(f"unexpected convolution shape for {name}")
            updates[layer["layer_name"]] = update
        return updates


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = 0.5 * ((position + 1) + end)
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def spearman_correlation(left, right, eps=1e-12):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if not math.isfinite(denominator) or denominator <= eps:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def topk_comparison(previous, current, common_valid, reset_budget):
    common = torch.nonzero(common_valid, as_tuple=False).reshape(-1).tolist()
    count = len(common)
    k_eval = min(int(reset_budget), count)
    if k_eval <= 0:
        return None
    previous_topk = sorted(common, key=lambda idx: (-float(previous[idx]), idx))[:k_eval]
    current_topk = sorted(common, key=lambda idx: (-float(current[idx]), idx))[:k_eval]
    intersection = len(set(previous_topk).intersection(current_topk))
    overlap = intersection / k_eval
    random_expected = k_eval / count
    previous_values = [float(previous[idx]) for idx in common]
    current_values = [float(current[idx]) for idx in common]
    return {
        "common_valid_kernels": int(count),
        "k_eval": int(k_eval),
        "previous_topk": previous_topk,
        "current_topk": current_topk,
        "intersection_count": int(intersection),
        "overlap_rate": float(overlap),
        "random_overlap_rate": float(random_expected),
        "overlap_lift": float(overlap - random_expected),
        "spearman": spearman_correlation(previous_values, current_values),
    }


class FedPhoenixHistoryObserver:
    """Compute reset-aware scores and compare them with strictly older history."""

    def __init__(self, model, reset_ratio, eps=1e-12, max_cross_clients=10):
        self.geometry = ConvKernelGeometry(model, reset_ratio)
        self.eps = float(eps)
        self.max_cross_clients = int(max_cross_clients)
        self.history = {}

    def build_interaction(
        self,
        client_id,
        client_weight,
        dispatch_state,
        returned_state,
        task_trace,
        task_id=None,
    ):
        masks, eligible = self.geometry.reset_masks_from_trace(task_trace)
        return {
            "client_id": int(client_id),
            "client_weight": float(client_weight),
            "task_id": int(task_id) if task_id is not None else None,
            "task_seed": int(task_trace["seed"]),
            "updates": self.geometry.conv_update(dispatch_state, returned_state),
            "reset_masks": masks,
            "reset_eligible": eligible,
        }

    def _score_current_round(self, interactions):
        current = {int(item["client_id"]): OrderedDict() for item in interactions}
        for layer_name, geometry in self.geometry.by_name.items():
            first_update = interactions[0]["updates"][layer_name]
            weighted_sum = torch.zeros_like(first_update, dtype=torch.float32, device="cpu")
            clean_weight = torch.zeros(geometry["output_kernels"], dtype=torch.float32)
            for item in interactions:
                clean = ~item["reset_masks"][layer_name]
                weight = float(item["client_weight"])
                weighted_sum[clean] = (
                    weighted_sum[clean] + item["updates"][layer_name][clean] * weight
                )
                clean_weight[clean] = clean_weight[clean] + weight

            for item in interactions:
                client_id = int(item["client_id"])
                own_clean = ~item["reset_masks"][layer_name]
                weight = float(item["client_weight"])
                peer_weight = clean_weight - own_clean.float() * weight
                valid = own_clean & (peer_weight > 0)
                update = item["updates"][layer_name]
                numerator = weighted_sum.clone()
                numerator[own_clean] = numerator[own_clean] - update[own_clean] * weight
                consensus = torch.zeros_like(update)
                if bool(valid.any()):
                    view_shape = (geometry["output_kernels"],) + (1,) * (update.ndim - 1)
                    denominator_view = peer_weight.reshape(view_shape)
                    consensus[valid] = numerator[valid] / denominator_view[valid]

                update_flat = update.reshape(geometry["output_kernels"], -1)
                consensus_flat = consensus.reshape(geometry["output_kernels"], -1)
                residual_flat = update_flat - consensus_flat
                update_norm = torch.linalg.vector_norm(update_flat, dim=1)
                consensus_norm = torch.linalg.vector_norm(consensus_flat, dim=1)
                residual_norm = torch.linalg.vector_norm(residual_flat, dim=1)
                score_values = OrderedDict()
                score_validity = OrderedDict()

                def store(name, values, score_valid):
                    output = torch.full(
                        (geometry["output_kernels"],),
                        float("nan"),
                        dtype=torch.float32,
                    )
                    output[score_valid] = values[score_valid].float()
                    score_values[name] = output
                    score_validity[name] = score_valid.detach().clone()

                store("update_norm", update_norm, valid)
                store("residual", residual_norm, valid)
                denominator = update_norm + consensus_norm + self.eps
                store("norm_residual", residual_norm / denominator, valid)
                angular_valid = valid & (update_norm > self.eps) & (consensus_norm > self.eps)
                cosine = torch.zeros_like(update_norm)
                if bool(angular_valid.any()):
                    dot = (update_flat * consensus_flat).sum(dim=1)
                    cosine[angular_valid] = (
                        dot[angular_valid]
                        / (update_norm[angular_valid] * consensus_norm[angular_valid])
                    ).clamp(-1.0, 1.0)
                store("angular", 1.0 - cosine, angular_valid)
                current[client_id][layer_name] = {
                    "scores": score_values,
                    "validity": score_validity,
                    "own_clean": own_clean.detach().clone(),
                    "reset_eligible_now": bool(item["reset_eligible"][layer_name]),
                }
        return current

    def score_current_round(self, interactions):
        """Public, side-effect-free clean leave-one-out score computation."""
        return self._score_current_round(interactions)

    def _cross_comparisons(
        self, client_id, layer_name, score_type, current_score, current_valid, reset_budget, round_idx
    ):
        candidates = []
        for other_id in sorted(self.history):
            if int(other_id) == int(client_id):
                continue
            layer_history = self.history[other_id].get(layer_name)
            if layer_history is None or int(layer_history["round"]) >= int(round_idx):
                continue
            candidates.append((int(other_id), layer_history))
            if len(candidates) >= self.max_cross_clients:
                break
        overlap_values = []
        spearman_values = []
        used_ids = []
        used_rounds = []
        for other_id, layer_history in candidates:
            comparison = topk_comparison(
                layer_history["scores"][score_type],
                current_score,
                layer_history["validity"][score_type] & current_valid,
                reset_budget,
            )
            if comparison is None:
                continue
            overlap_values.append(comparison["overlap_rate"])
            if comparison["spearman"] is not None:
                spearman_values.append(comparison["spearman"])
            used_ids.append(other_id)
            used_rounds.append(int(layer_history["round"]))
        return {
            "cross_history_count": len(overlap_values),
            "cross_client_ids": used_ids,
            "cross_history_rounds": used_rounds,
            "cross_overlap_rate_mean": (
                float(np.mean(overlap_values)) if overlap_values else None
            ),
            "cross_spearman_mean": (
                float(np.mean(spearman_values)) if spearman_values else None
            ),
        }

    def process_round(self, interactions, round_idx):
        if not interactions:
            return []
        client_ids = [int(item["client_id"]) for item in interactions]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("a client may appear only once in a round")
        current = self._score_current_round(interactions)
        events = []

        # All comparisons read the untouched pre-round history.  Current scores
        # are committed only after every same- and cross-client comparison.
        for client_id in client_ids:
            previous_client = self.history.get(client_id, {})
            for layer_name, current_layer in current[client_id].items():
                previous_layer = previous_client.get(layer_name)
                if previous_layer is None or int(previous_layer["round"]) >= int(round_idx):
                    continue
                geometry = self.geometry.by_name[layer_name]
                for score_type in SCORE_TYPES:
                    previous_score = previous_layer["scores"][score_type]
                    current_score = current_layer["scores"][score_type]
                    common_valid = (
                        previous_layer["validity"][score_type]
                        & current_layer["validity"][score_type]
                    )
                    comparison = topk_comparison(
                        previous_score,
                        current_score,
                        common_valid,
                        geometry["reset_budget"],
                    )
                    if comparison is None:
                        continue
                    cross = self._cross_comparisons(
                        client_id,
                        layer_name,
                        score_type,
                        current_score,
                        current_layer["validity"][score_type],
                        geometry["reset_budget"],
                        round_idx,
                    )
                    cross_overlap = cross["cross_overlap_rate_mean"]
                    cross_spearman = cross["cross_spearman_mean"]
                    event = {
                        "round": int(round_idx),
                        "client_id": int(client_id),
                        "previous_round": int(previous_layer["round"]),
                        "gap": int(round_idx) - int(previous_layer["round"]),
                        "layer": layer_name,
                        "score_type": score_type,
                        "output_kernels": int(geometry["output_kernels"]),
                        "reset_budget": int(geometry["reset_budget"]),
                        "reset_eligible_now": bool(current_layer["reset_eligible_now"]),
                        **comparison,
                        **cross,
                        "identity_overlap_advantage": (
                            comparison["overlap_rate"] - cross_overlap
                            if cross_overlap is not None
                            else None
                        ),
                        "identity_rank_advantage": (
                            comparison["spearman"] - cross_spearman
                            if comparison["spearman"] is not None
                            and cross_spearman is not None
                            else None
                        ),
                    }
                    event["previous_topk"] = json.dumps(event["previous_topk"])
                    event["current_topk"] = json.dumps(event["current_topk"])
                    event["cross_client_ids"] = json.dumps(event["cross_client_ids"])
                    event["cross_history_rounds"] = json.dumps(event["cross_history_rounds"])
                    events.append(event)

        for client_id in client_ids:
            client_history = self.history.setdefault(client_id, {})
            for layer_name, current_layer in current[client_id].items():
                client_history[layer_name] = {
                    "round": int(round_idx),
                    "scores": OrderedDict(
                        (name, tensor.detach().to("cpu", dtype=torch.float32).clone())
                        for name, tensor in current_layer["scores"].items()
                    ),
                    "validity": OrderedDict(
                        (name, tensor.detach().to("cpu", dtype=torch.bool).clone())
                        for name, tensor in current_layer["validity"].items()
                    ),
                    "reset_eligible": bool(current_layer["reset_eligible_now"]),
                }
        return events

    def history_memory_bytes(self):
        total = 0
        for client_history in self.history.values():
            for layer_history in client_history.values():
                for mapping in (layer_history["scores"], layer_history["validity"]):
                    for tensor in mapping.values():
                        total += tensor.numel() * tensor.element_size()
        return int(total)

    def history_tensor_inventory(self):
        inventory = []
        for client_id, client_history in self.history.items():
            for layer_name, layer_history in client_history.items():
                output_kernels = self.geometry.by_name[layer_name]["output_kernels"]
                for category in ("scores", "validity"):
                    for score_type, tensor in layer_history[category].items():
                        inventory.append(
                            {
                                "client_id": int(client_id),
                                "layer": layer_name,
                                "category": category,
                                "score_type": score_type,
                                "shape": tuple(tensor.shape),
                                "device": tensor.device.type,
                                "numel": tensor.numel(),
                                "output_kernels": output_kernels,
                            }
                        )
        return inventory
