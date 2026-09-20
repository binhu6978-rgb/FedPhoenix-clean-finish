"""Recovery-aware client--reset task matching for FedPhoenix.

The module deliberately separates three interfaces so a learned Agent can
replace the heuristic later without changing the FL training loop:

1. ``build_recovery_score_matrix`` obtains task-conditioned local signals;
2. ``joint_match`` enforces the participation and one-copy-per-client budget;
3. the caller trains the selected client from its assigned reset copy.

Only private training samples are used for probes.  The server receives scalar
features/scores in this simulation; no test-set metric enters the matching.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset


def _logits(output):
    return output["output"] if isinstance(output, dict) else output


def _load_probe_batch(dataset, indices):
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=max(1, len(indices)),
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader))


def build_client_probe_batches(
    dataset,
    dict_users,
    candidate_clients,
    probe_samples,
    seed,
    round_idx,
):
    """Create fixed support/query probes shared by every task in a round."""
    batches = {}
    for client_id in candidate_clients:
        indices = np.asarray(list(dict_users[int(client_id)]), dtype=np.int64)
        if len(indices) < 2:
            raise ValueError(f"Client {client_id} needs at least two probe samples")
        rng = np.random.default_rng(
            int(seed) + int(round_idx) * 100003 + int(client_id) * 997
        )
        count = min(int(probe_samples), len(indices))
        selected = rng.choice(indices, size=count, replace=False)
        split = max(1, count // 2)
        support_indices = selected[:split]
        query_indices = selected[split:]
        if len(query_indices) == 0:
            query_indices = support_indices
        batches[int(client_id)] = {
            "support": _load_probe_batch(dataset, support_indices),
            "query": _load_probe_batch(dataset, query_indices),
            "support_size": int(len(support_indices)),
            "query_size": int(len(query_indices)),
        }
    return batches


def _evaluate_batch_loss(model, batch, device):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    model.eval()
    with torch.no_grad():
        loss = F.cross_entropy(_logits(model(images)), labels)
    return float(loss.item())


def _reset_gradient_alignment(model, global_state, task_state, reset_trace):
    """Cosine between the local SGD direction and the pre-reset direction."""
    parameters = dict(model.named_parameters())
    dot = 0.0
    update_norm_sq = 0.0
    repair_norm_sq = 0.0

    for layer_trace in reset_trace.get("layers", []):
        parameter_name = f"{layer_trace['name']}.weight"
        parameter = parameters.get(parameter_name)
        if parameter is None or parameter.grad is None:
            continue
        indices = layer_trace.get("reset_indices", [])
        if not indices:
            continue
        grad = parameter.grad.detach()[indices]
        desired_repair = (
            global_state[parameter_name][indices]
            - task_state[parameter_name][indices]
        ).to(grad.device)
        local_update = -grad
        dot += float(torch.sum(local_update * desired_repair).item())
        update_norm_sq += float(torch.sum(local_update * local_update).item())
        repair_norm_sq += float(torch.sum(desired_repair * desired_repair).item())

    denominator = math.sqrt(update_norm_sq * repair_norm_sq)
    if denominator <= 1e-12:
        return 0.0, math.sqrt(update_norm_sq), math.sqrt(repair_norm_sq)
    return dot / denominator, math.sqrt(update_norm_sq), math.sqrt(repair_norm_sq)


def _standardize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    std = matrix.std()
    if std <= 1e-12:
        return np.zeros_like(matrix)
    return (matrix - matrix.mean()) / std


def _build_probe_optimizer(args, model):
    """Optionally match the virtual probe optimizer to local training.

    The historical probe uses a plain manual SGD step.  Keeping that as the
    default preserves existing runs, while the opt-in path lets the probe use
    the same optimizer family and regularization as ``LocalUpdate_FedAvg``.
    """
    if not bool(getattr(args, "recovery_probe_match_local_optimizer", False)):
        return None

    optimizer_name = str(getattr(args, "optimizer", "sgd")).lower()
    parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    learning_rate = float(args.recovery_probe_lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(getattr(args, "momentum", 0.0)),
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(
        "recovery_probe_match_local_optimizer currently supports "
        "the local SGD and Adam optimizers"
    )


def build_recovery_score_matrix(
    args,
    global_model,
    task_models,
    task_traces,
    dataset,
    dict_users,
    candidate_clients,
    round_idx,
):
    """Measure one-step functional recovery for every client--task pair.

    A client uses half of a small local probe as support for one virtual SGD
    step and the other half as a query set.  The score combines:

    - recovery gain: reset query loss minus post-step query loss;
    - reset advantage: global query loss minus post-step query loss;
    - residual damage: positive loss still above the unreset global model;
    - gradient alignment on reset kernels (diagnostic, low default weight).

    Every component is standardized over the current candidate x task matrix,
    making weights comparable across rounds and datasets.
    """
    if len(task_models) != len(task_traces):
        raise ValueError("task_models and task_traces must have equal length")
    if len(candidate_clients) < len(task_models):
        raise ValueError("candidate pool must be at least as large as task count")

    device = args.device
    probes = build_client_probe_batches(
        dataset,
        dict_users,
        candidate_clients,
        args.recovery_probe_samples,
        args.seed,
        round_idx,
    )
    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    global_probe_losses = {}
    global_probe_model = copy.deepcopy(global_model).to(device)
    for client_id in candidate_clients:
        global_probe_losses[int(client_id)] = _evaluate_batch_loss(
            global_probe_model,
            probes[int(client_id)]["query"],
            device,
        )
    global_probe_model.to("cpu")
    del global_probe_model

    shape = (len(candidate_clients), len(task_models))
    recovery_gain = np.zeros(shape, dtype=np.float64)
    reset_advantage = np.zeros(shape, dtype=np.float64)
    residual_damage = np.zeros(shape, dtype=np.float64)
    gradient_alignment = np.zeros(shape, dtype=np.float64)
    feature_rows = []

    for row_id, client_id in enumerate(candidate_clients):
        support = probes[int(client_id)]["support"]
        query = probes[int(client_id)]["query"]
        global_loss = global_probe_losses[int(client_id)]

        for task_id, (task_model, trace) in enumerate(zip(task_models, task_traces)):
            model = copy.deepcopy(task_model).to(device)
            probe_optimizer = _build_probe_optimizer(args, model)
            task_state = {
                key: value.detach().cpu().clone()
                for key, value in task_model.state_dict().items()
            }
            reset_loss = _evaluate_batch_loss(model, query, device)

            support_images, support_labels = support
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            probe_steps = int(getattr(args, "recovery_probe_steps", 1))
            if probe_steps < 1:
                raise ValueError("recovery_probe_steps must be at least one")
            alignment = 0.0
            reset_grad_norm = 0.0
            repair_norm = 0.0
            support_loss = None
            for probe_step in range(probe_steps):
                model.train()
                if probe_optimizer is None:
                    model.zero_grad(set_to_none=True)
                else:
                    probe_optimizer.zero_grad(set_to_none=True)
                support_loss = F.cross_entropy(
                    _logits(model(support_images)), support_labels
                )
                support_loss.backward()
                if probe_step == 0:
                    (
                        alignment,
                        reset_grad_norm,
                        repair_norm,
                    ) = _reset_gradient_alignment(
                        model, global_state, task_state, trace
                    )

                if probe_optimizer is None:
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.add_(
                                    parameter.grad,
                                    alpha=-float(args.recovery_probe_lr),
                                )
                else:
                    probe_optimizer.step()
            adapted_loss = _evaluate_batch_loss(model, query, device)

            gain = reset_loss - adapted_loss
            advantage = global_loss - adapted_loss
            residual = max(adapted_loss - global_loss, 0.0)
            recovery_gain[row_id, task_id] = gain
            reset_advantage[row_id, task_id] = advantage
            residual_damage[row_id, task_id] = residual
            gradient_alignment[row_id, task_id] = alignment
            feature_rows.append(
                {
                    "round": int(round_idx),
                    "client_id": int(client_id),
                    "task_id": int(task_id),
                    "support_size": probes[int(client_id)]["support_size"],
                    "query_size": probes[int(client_id)]["query_size"],
                    "probe_steps": probe_steps,
                    "global_query_loss": global_loss,
                    "reset_query_loss": reset_loss,
                    "adapted_query_loss": adapted_loss,
                    "support_loss": float(support_loss.item()),
                    "reset_damage": reset_loss - global_loss,
                    "one_step_recovery_gain": gain,
                    "reset_advantage": advantage,
                    "residual_damage": residual,
                    "gradient_alignment": alignment,
                    "reset_gradient_norm": reset_grad_norm,
                    "repair_direction_norm": repair_norm,
                    "score": None,
                    "selected": False,
                }
            )
            model.to("cpu")
            del model

    score_matrix = (
        float(args.recovery_gain_weight) * _standardize(recovery_gain)
        + float(args.recovery_advantage_weight) * _standardize(reset_advantage)
        - float(args.recovery_residual_weight) * _standardize(residual_damage)
        + float(args.recovery_alignment_weight) * _standardize(gradient_alignment)
    )
    for feature, score in zip(feature_rows, score_matrix.reshape(-1)):
        feature["score"] = float(score)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score_matrix, feature_rows


def joint_match(score_matrix, candidate_clients):
    """Maximum-weight one-to-one client--task assignment."""
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if score_matrix.shape[0] != len(candidate_clients):
        raise ValueError("candidate client count does not match score rows")
    if score_matrix.shape[0] < score_matrix.shape[1]:
        raise ValueError("not enough clients to cover all reset tasks")

    rows, tasks = linear_sum_assignment(-score_matrix)
    assignments = [
        {
            "client_row": int(row),
            "client_id": int(candidate_clients[int(row)]),
            "task_id": int(task),
            "score": float(score_matrix[int(row), int(task)]),
        }
        for row, task in zip(rows, tasks)
    ]
    return sorted(assignments, key=lambda item: item["task_id"])


def confidence_gate_assignment(
    score_matrix,
    candidate_clients,
    proposed_assignments,
    min_gain_per_task=0.0,
):
    """Fall back to FedPhoenix's original assignment when gain is too small.

    This gate is defined for assignment-only matching, where the candidate
    matrix is square and the original FedPhoenix assignment is its diagonal.
    The score gain is normalized by task count so one threshold transfers
    across participation budgets.
    """
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("confidence gating requires a square score matrix")
    if score_matrix.shape[0] != len(candidate_clients):
        raise ValueError("candidate client count does not match score rows")
    if min_gain_per_task < 0:
        raise ValueError("min_gain_per_task must be non-negative")

    task_count = score_matrix.shape[1]
    baseline_assignments = [
        {
            "client_row": int(task_id),
            "client_id": int(candidate_clients[task_id]),
            "task_id": int(task_id),
            "score": float(score_matrix[task_id, task_id]),
        }
        for task_id in range(task_count)
    ]
    baseline_score = float(np.trace(score_matrix))
    proposed_score = float(
        sum(
            score_matrix[item["client_row"], item["task_id"]]
            for item in proposed_assignments
        )
    )
    gain_per_task = (proposed_score - baseline_score) / max(task_count, 1)
    matching_applied = gain_per_task >= float(min_gain_per_task)
    return (
        proposed_assignments if matching_applied else baseline_assignments,
        float(gain_per_task),
        bool(matching_applied),
    )


def mark_selected_features(feature_rows, assignments):
    selected = {
        (int(item["client_id"]), int(item["task_id"]))
        for item in assignments
    }
    for row in feature_rows:
        row["selected"] = (row["client_id"], row["task_id"]) in selected
    return feature_rows
