#!/usr/bin/env python
"""Standalone InteractionDispatchV2 federated training loop."""

import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch

from Algorithm.InteractionDispatchV2 import (
    InteractionDispatchV2Controller,
    clone_state_to_cpu,
)
from models.Aggregation_InteractionDispatch import aggregate_client_responses
from models.Update_InteractionDispatch import LocalUpdate_InteractionDispatch


def _safe_suffix(value):
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(value))


def _output_paths(args):
    suffix = f"_{_safe_suffix(args.run_name)}" if args.run_name else ""
    stem = f"{args.dataset}_{args.model}_{args.algorithm}_seed{args.seed}{suffix}"
    directory = os.path.abspath(args.metrics_log_dir)
    os.makedirs(directory, exist_ok=True)
    return {
        "csv": os.path.join(directory, stem + ".csv"),
        "groups": os.path.join(directory, stem + "_groups.csv"),
        "summary": os.path.join(directory, stem + "_summary.json"),
        "config": os.path.join(directory, stem + "_config.json"),
    }


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().to("cpu").contiguous()
        digest.update(np.asarray(contiguous).tobytes())
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def train_interaction_dispatch_v2(
    args,
    net_glob,
    dataset_train,
    dataset_test,
    dict_users,
    evaluate_round_accuracy,
    print_peak_accuracy=None,
):
    """Train using actual client responses y_i-x_i and a V2 controller."""
    paths = _output_paths(args)
    config = {
        key: getattr(args, key)
        for key in (
            "algorithm", "dataset", "model", "epochs", "num_users", "frac",
            "local_ep", "local_bs", "bs", "optimizer", "lr", "momentum",
            "weight_decay", "iid", "noniid_case", "data_beta", "seed",
            "id2_step_ratio", "num_workers",
        )
    }
    _json_dump(paths["config"], config)

    net_glob.train()
    controller = InteractionDispatchV2Controller(
        net_glob, step_ratio=args.id2_step_ratio
    )
    geometry = controller.geometry
    previous_global_vector = None
    accuracies = []
    metric_rows = []
    group_rows = []
    all_reason_counts = Counter()
    first_active_checked = False
    total_active = 0
    failed_clients = 0

    for round_idx in range(args.epochs):
        round_start = time.perf_counter()
        print("*" * 80)
        print("Round {:3d}".format(round_idx))
        global_state = clone_state_to_cpu(net_glob.state_dict())
        global_vector = geometry.flatten_state(global_state)
        global_direction = None
        if previous_global_vector is not None:
            change = global_vector - previous_global_vector
            change_norm = float(torch.linalg.vector_norm(change).item())
            if math.isfinite(change_norm) and change_norm > controller.eps:
                global_direction = change / change_norm

        m = max(int(args.frac * args.num_users), 1)
        selected = np.random.choice(range(args.num_users), m, replace=False)
        records = []
        dispatch_events = []
        round_alphas = []
        round_losses = []

        for selected_id in selected:
            client_id = int(selected_id)
            delta, info = controller.make_dispatch_delta(client_id, global_direction)
            if delta is None:
                dispatch_state = clone_state_to_cpu(global_state)
            else:
                dispatch_state = geometry.add_delta_to_state(global_state, delta)
            actual_dispatch_vector = geometry.flatten_state(dispatch_state)

            if info["reason"] == "active" and float(info["alpha"]) > 0.0:
                actual_norm = float(
                    torch.linalg.vector_norm(actual_dispatch_vector - global_vector).item()
                )
                info["actual_delta_norm"] = actual_norm
                tolerance = max(1e-6, 5e-3 * abs(float(info["alpha"])))
                if not first_active_checked:
                    if not math.isfinite(actual_norm) or actual_norm <= 0:
                        raise AssertionError("active V2 dispatch did not change the model")
                    if abs(actual_norm - float(info["alpha"])) > tolerance:
                        raise AssertionError(
                            "V2 materialized dispatch norm differs from alpha: "
                            f"{actual_norm} vs {info['alpha']} (tol={tolerance})"
                        )
                    first_active_checked = True
                round_alphas.append(float(info["alpha"]))
                total_active += 1

            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(dispatch_state)
            net_local.to(args.device)
            local = LocalUpdate_InteractionDispatch(
                args=args,
                dataset_test=dataset_test,
                dataset=dataset_train,
                idxs=dict_users[client_id],
            )
            try:
                local_result = local.train_from_dispatch(net_local)
            except Exception:
                failed_clients += 1
                raise
            returned_state = local_result["returned_state"]
            round_losses.append(float(local_result["mean_loss"]))
            returned_vector = geometry.flatten_state(returned_state)
            client_update = returned_vector - actual_dispatch_vector
            records.append(
                {
                    "client_id": client_id,
                    "dispatch_state": dispatch_state,
                    "returned_state": returned_state,
                    "weight": len(dict_users[client_id]),
                }
            )
            controller.record_interaction(
                client_id, actual_dispatch_vector, client_update, round_idx
            )
            all_reason_counts[info["reason"]] += 1
            dispatch_events.append(info)
            for group, diagnostic in info.get("group_diagnostics", {}).items():
                group_rows.append(
                    {
                        "round": round_idx + 1,
                        "client_id": client_id,
                        "reason": info["reason"],
                        "group": group,
                        **diagnostic,
                    }
                )
            del net_local, local, local_result, returned_vector, client_update

        new_global_state = aggregate_client_responses(
            global_state,
            records,
            geometry.param_names,
            accumulation_device=args.device,
        )
        net_glob.load_state_dict(new_global_state)
        previous_global_vector = global_vector
        train_seconds = time.perf_counter() - round_start
        accuracy = float(
            evaluate_round_accuracy(net_glob, dataset_test, args, round_idx + 1)
        )
        accuracies.append(accuracy)
        reasons = Counter(event["reason"] for event in dispatch_events)
        metric_rows.append(
            {
                "round": round_idx + 1,
                "algorithm": args.algorithm,
                "seed": int(args.seed),
                "id2_step_ratio": float(args.id2_step_ratio),
                "test_accuracy": accuracy,
                "round_train_seconds": float(train_seconds),
                "selected_clients": json.dumps([int(value) for value in selected.tolist()]),
                "global_state_sha256": _state_sha256(new_global_state),
                "active_interventions": int(len(round_alphas)),
                "no_need_count": int(reasons["no_need"]),
                "no_support_count": int(reasons["no_support"]),
                "cold_start_count": int(reasons["cold_start"]),
                "no_observation_count": int(reasons["no_observation"]),
                "no_global_direction_count": int(reasons["no_global_direction"]),
                "mean_alpha": float(np.mean(round_alphas) if round_alphas else 0.0),
                "max_alpha": float(max(round_alphas) if round_alphas else 0.0),
                "mean_local_loss": float(np.mean(round_losses)),
                "num_clients_seen": int(controller.num_clients_seen()),
                "num_clients_with_observation": int(
                    controller.num_clients_with_observation()
                ),
                "history_memory_bytes": int(controller.history_memory_bytes()),
                "dispatch_events": json.dumps(dispatch_events, separators=(",", ":")),
            }
        )
        _write_csv(paths["csv"], metric_rows)
        _write_csv(paths["groups"], group_rows)
        print(
            "InteractionDispatchV2 diagnostics: "
            f"round={round_idx + 1} active={len(round_alphas)} "
            f"seen={controller.num_clients_seen()} "
            f"observed={controller.num_clients_with_observation()} "
            f"history_bytes={controller.history_memory_bytes()}"
        )
        del records, new_global_state, global_state
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    peak = max(accuracies)
    peak_index = int(np.argmax(accuracies))
    summary = {
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "id2_step_ratio": float(args.id2_step_ratio),
        "completed_rounds": len(accuracies),
        "peak_accuracy": float(peak),
        "peak_round": peak_index + 1,
        "final_accuracy": float(accuracies[-1]),
        "last20_mean": float(np.mean(accuracies[-20:])),
        "active_count": int(total_active),
        "active_rate": float(total_active / (len(accuracies) * m)),
        "reason_counts": dict(all_reason_counts),
        "maximum_history_memory_bytes": int(
            max(row["history_memory_bytes"] for row in metric_rows)
        ),
        "final_num_clients_seen": int(controller.num_clients_seen()),
        "final_num_clients_with_observation": int(
            controller.num_clients_with_observation()
        ),
        "failed_client_updates": int(failed_clients),
        "metrics_csv": paths["csv"],
        "group_metrics_csv": paths["groups"],
    }
    _json_dump(paths["summary"], summary)
    if print_peak_accuracy is not None:
        print_peak_accuracy(accuracies, args.algorithm)
    return summary
