#!/usr/bin/env python
"""Deterministic history-informed retargeting for FedPhoenix tasks.

The baseline FedPhoenix task is always constructed first.  This module only
moves already-sampled reset kernels to client-history-selected output channels;
it never samples a value or changes the number of reset slots.
"""

import copy
import hashlib
import math
from collections import OrderedDict

import torch
from torch import nn


class ConvKernelGeometry:
    """The output-kernel geometry of ``nn.Conv2d.weight`` tensors."""

    def __init__(self, model):
        layers = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layers.append(
                    {
                        "layer_name": module_name,
                        "parameter_name": (
                            f"{module_name}.weight" if module_name else "weight"
                        ),
                        "output_kernels": int(module.weight.shape[0]),
                        "kernel_shape": tuple(module.weight.shape[1:]),
                    }
                )
        if not layers:
            raise ValueError("TargetedFedPhoenix requires at least one Conv2d layer")
        self.layers = tuple(layers)
        self.by_name = OrderedDict((item["layer_name"], item) for item in layers)


def _ordered_topk(scores, validity, count):
    """Return finite valid indices by score desc, index asc."""
    if count <= 0:
        return []
    candidates = [
        index
        for index in range(int(scores.numel()))
        if bool(validity[index]) and math.isfinite(float(scores[index]))
    ]
    candidates.sort(key=lambda index: (-float(scores[index]), index))
    return candidates[: int(count)]


def deterministic_reset_priority(task_seed, layer_name, kernel_index, namespace):
    """Return a process-independent private priority without using any RNG."""
    payload = (
        f"{int(task_seed)}|{str(layer_name)}|{int(kernel_index)}|{str(namespace)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def _priority_order(indices, task_seed, layer_name, namespace):
    return sorted(
        (int(index) for index in indices),
        key=lambda index: (
            deterministic_reset_priority(
                task_seed, layer_name, index, namespace
            ),
            index,
        ),
    )


class TargetedFedPhoenixController:
    """Latest-interaction client history plus deterministic task retargeting."""

    def __init__(self, model, target_ratio=0.25, max_history_gap=20):
        if not 0.0 <= float(target_ratio) <= 1.0:
            raise ValueError("target_ratio must be between zero and one")
        if int(max_history_gap) < 0:
            raise ValueError("max_history_gap must be non-negative")
        self.geometry = ConvKernelGeometry(model)
        self.target_ratio = float(target_ratio)
        self.max_history_gap = int(max_history_gap)
        self.history = {}

    def _history_status(self, client_id, current_round):
        entry = self.history.get(int(client_id))
        if entry is None:
            return None, None, "cold_start"
        gap = int(current_round) - int(entry["last_round"])
        if gap <= 0:
            raise ValueError("history must come from a strictly earlier round")
        if gap > self.max_history_gap:
            return entry, gap, "stale_history"
        return entry, gap, None

    def retarget_task(
        self,
        global_model,
        baseline_task_model,
        baseline_trace,
        client_id,
        current_round,
    ):
        """Retarget one already-built FedPhoenix task in place.

        ``baseline_task_model`` is returned so that every fallback path is
        bitwise identical to the original task.  The enriched trace records
        both baseline and final reset locations.
        """
        entry, history_gap, client_fallback = self._history_status(
            client_id, current_round
        )
        final_trace = copy.deepcopy(baseline_trace)
        final_trace["client_id"] = int(client_id)
        final_trace["history_available"] = entry is not None
        final_trace["history_gap"] = history_gap
        final_trace["target_ratio"] = self.target_ratio
        final_trace["max_history_gap"] = self.max_history_gap

        baseline_state = baseline_task_model.state_dict()
        global_state = global_model.state_dict()
        final_layers = []

        for layer_trace in baseline_trace.get("layers", []):
            layer_name = layer_trace["name"]
            if layer_name not in self.geometry.by_name:
                raise ValueError(f"unknown convolution layer in task trace: {layer_name}")
            geometry = self.geometry.by_name[layer_name]
            reset_indices = [int(value) for value in layer_trace["reset_indices"]]
            if len(reset_indices) != len(set(reset_indices)):
                raise ValueError("baseline reset indices contain duplicates")
            reset_count = len(reset_indices)
            requested = int(math.floor(self.target_ratio * reset_count))
            targeted = []

            if client_fallback is not None:
                fallback_reason = client_fallback
            elif requested <= 0:
                fallback_reason = "no_target_slots"
            else:
                history_layer = entry["layers"].get(layer_name)
                if history_layer is None:
                    fallback_reason = "no_valid_history"
                else:
                    targeted = _ordered_topk(
                        history_layer["score"],
                        history_layer["validity"],
                        requested,
                    )
                    fallback_reason = "targeted" if targeted else "no_valid_history"

            target_set = set(targeted)
            random_candidates = [
                index for index in reset_indices if index not in target_set
            ]
            retention_order = _priority_order(
                random_candidates,
                baseline_trace["seed"],
                layer_name,
                "random-retention",
            )
            random_kept = sorted(
                retention_order[: reset_count - len(targeted)]
            )
            final_indices = sorted(target_set.union(random_kept))
            if len(final_indices) != reset_count:
                raise AssertionError("retargeting changed the reset slot count")
            if len(final_indices) != len(set(final_indices)):
                raise AssertionError("retargeting produced duplicate reset indices")

            baseline_set = set(reset_indices)
            final_set = set(final_indices)
            added = _priority_order(
                target_set - baseline_set,
                baseline_trace["seed"],
                layer_name,
                "target-pairing",
            )
            displaced = _priority_order(
                baseline_set - final_set,
                baseline_trace["seed"],
                layer_name,
                "donor-pairing",
            )
            if len(added) != len(displaced):
                raise AssertionError("target and donor counts differ")

            donor_mapping = []
            parameter_name = geometry["parameter_name"]
            task_weight = baseline_state[parameter_name]
            source_weight = global_state[parameter_name]
            donor_values = {
                donor: task_weight[donor].detach().clone()
                for donor in displaced
            }
            with torch.no_grad():
                for target, donor in zip(added, displaced):
                    task_weight[target].copy_(
                        donor_values[donor].to(
                            device=task_weight.device, dtype=task_weight.dtype
                        )
                    )
                    task_weight[donor].copy_(
                        source_weight[donor].detach().to(
                            device=task_weight.device, dtype=task_weight.dtype
                        )
                    )
                    donor_mapping.append(
                        {"target_index": int(target), "donor_index": int(donor)}
                    )

            enriched = copy.deepcopy(layer_trace)
            enriched.update(
                {
                    "baseline_reset_indices": reset_indices,
                    "targeted_indices": targeted,
                    "retention_policy": "sha256_private_priority",
                    "retention_priority_namespace": "random-retention",
                    "donor_pairing_policy": "sha256_private_priority",
                    "target_pairing_priority_namespace": "target-pairing",
                    "donor_pairing_priority_namespace": "donor-pairing",
                    "random_candidates": random_candidates,
                    "random_kept_indices": random_kept,
                    "displaced_donor_indices": displaced,
                    "final_reset_indices": final_indices,
                    "reset_indices": final_indices,
                    "donor_mapping": donor_mapping,
                    "target_requested": int(requested),
                    "target_actual": int(len(targeted)),
                    "history_gap": history_gap,
                    "history_available": entry is not None,
                    "fallback_reason": fallback_reason,
                    "baseline_random_overlap": int(len(target_set & baseline_set)),
                    "replaced_reset_slots": int(len(added)),
                }
            )
            final_layers.append(enriched)

        final_trace["layers"] = final_layers
        final_trace["num_reset_layers"] = len(final_layers)
        final_trace["num_reset_kernels"] = sum(
            len(layer["final_reset_indices"]) for layer in final_layers
        )
        final_trace["targeted_reset_slots"] = sum(
            int(layer["target_actual"]) for layer in final_layers
        )
        final_trace["random_reset_slots"] = sum(
            len(layer["random_kept_indices"]) for layer in final_layers
        )
        final_trace["baseline_random_overlap"] = sum(
            int(layer["baseline_random_overlap"]) for layer in final_layers
        )
        final_trace["replaced_reset_slots"] = sum(
            int(layer["replaced_reset_slots"]) for layer in final_layers
        )
        return baseline_task_model, final_trace

    def observe(
        self,
        client_id,
        current_round,
        actual_dispatch_state,
        returned_state,
        final_trace,
    ):
        """Overwrite a client's latest CPU scalar update-norm history."""
        reset_by_layer = {
            layer["name"]: set(
                int(value)
                for value in layer.get(
                    "final_reset_indices", layer.get("reset_indices", [])
                )
            )
            for layer in final_trace.get("layers", [])
        }
        layers = OrderedDict()
        for geometry in self.geometry.layers:
            layer_name = geometry["layer_name"]
            parameter_name = geometry["parameter_name"]
            dispatched = actual_dispatch_state[parameter_name].detach().to(
                device="cpu", dtype=torch.float32
            )
            returned = returned_state[parameter_name].detach().to(
                device="cpu", dtype=torch.float32
            )
            update = returned - dispatched
            score = torch.linalg.vector_norm(
                update.reshape(geometry["output_kernels"], -1), dim=1
            ).to(device="cpu", dtype=torch.float32)
            validity = torch.ones(
                geometry["output_kernels"], dtype=torch.bool, device="cpu"
            )
            reset_indices = sorted(reset_by_layer.get(layer_name, set()))
            if reset_indices:
                reset_tensor = torch.tensor(reset_indices, dtype=torch.long)
                validity[reset_tensor] = False
                score[reset_tensor] = float("nan")
            layers[layer_name] = {
                "score": score.detach().clone(),
                "validity": validity.detach().clone(),
            }
        self.history[int(client_id)] = {
            "last_round": int(current_round),
            "layers": layers,
        }

    def history_memory_bytes(self):
        total = 0
        for entry in self.history.values():
            total += 8  # one scalar last-round value
            for layer in entry["layers"].values():
                total += layer["score"].numel() * layer["score"].element_size()
                total += (
                    layer["validity"].numel() * layer["validity"].element_size()
                )
        return int(total)

    def history_inventory(self):
        return {
            "clients": len(self.history),
            "memory_bytes": self.history_memory_bytes(),
            "all_cpu": all(
                layer["score"].device.type == "cpu"
                and layer["validity"].device.type == "cpu"
                for entry in self.history.values()
                for layer in entry["layers"].values()
            ),
            "scalar_kernel_arrays_only": all(
                layer["score"].ndim == 1 and layer["validity"].ndim == 1
                for entry in self.history.values()
                for layer in entry["layers"].values()
            ),
        }
