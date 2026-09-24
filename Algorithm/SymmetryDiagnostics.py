"""Small scalar diagnostics on FedPhoenix's actually reset Conv kernels."""

import math

import torch


RESET_METRICS = (
    "reset_action_norm",
    "reset_response_norm",
    "reset_recovery_coeff",
    "reset_recovery_cosine",
    "reset_residual_ratio",
)


def reset_response_scalars(global_state, task_state, canonical_state, task_trace):
    """Measure reset action and local response in canonical coordinates."""
    action_sq = None
    response_sq = None
    action_response_dot = None
    reset_kernel_count = 0
    with torch.no_grad():
        for layer in task_trace["layers"]:
            indices = layer["reset_indices"]
            if not indices:
                continue
            key = f"{layer['name']}.weight"
            global_weight = global_state[key]
            device = global_weight.device
            cpu_index = torch.as_tensor(indices, dtype=torch.long)
            device_index = cpu_index.to(device)
            task_kernels = task_state[key].index_select(0, cpu_index).to(
                device=device, dtype=torch.float64
            )
            global_kernels = global_weight.index_select(0, device_index).to(torch.float64)
            returned_kernels = canonical_state[key].index_select(
                0, device_index
            ).to(torch.float64)
            action = task_kernels - global_kernels
            response = returned_kernels - task_kernels
            layer_action_sq = action.square().sum()
            layer_response_sq = response.square().sum()
            layer_dot = (action * response).sum()
            action_sq = layer_action_sq if action_sq is None else action_sq + layer_action_sq
            response_sq = (
                layer_response_sq if response_sq is None
                else response_sq + layer_response_sq
            )
            action_response_dot = (
                layer_dot if action_response_dot is None
                else action_response_dot + layer_dot
            )
            reset_kernel_count += len(indices)

        if reset_kernel_count:
            action_sq, response_sq, dot = torch.stack(
                (action_sq, response_sq, action_response_dot)
            ).cpu().tolist()
        else:
            action_sq = response_sq = dot = 0.0

    eps = 1e-12
    action_norm = math.sqrt(max(action_sq, 0.0))
    response_norm = math.sqrt(max(response_sq, 0.0))
    residual_sq = max(action_sq + response_sq + 2.0 * dot, 0.0)
    return {
        "reset_kernel_count": reset_kernel_count,
        "reset_action_norm": action_norm,
        "reset_response_norm": response_norm,
        "reset_recovery_coeff": -dot / (action_sq + eps),
        "reset_recovery_cosine": -dot / ((action_norm + eps) * (response_norm + eps)),
        "reset_residual_ratio": math.sqrt(residual_sq) / (action_norm + eps),
    }
