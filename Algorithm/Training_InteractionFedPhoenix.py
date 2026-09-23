"""FedPhoenix training loop with a bounded, corrected client dispatch."""

import copy
import json
import math
import time
from collections import Counter

import numpy as np
import torch

from Algorithm.InteractionFedPhoenix import (
    InteractionController, InteractionGeometry, ResetLedger, corrected_state,
)
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg


def _weighted_responses(responses, weights):
    result = torch.zeros_like(responses[0])
    total = sum(weights)
    for response, weight in zip(responses, weights):
        result += response * (float(weight) / total)
    return result


def train_interaction_fedphoenix(
    args, net_glob, dataset_train, dataset_test, dict_users,
    build_tasks, evaluate_round_accuracy, print_peak_accuracy,
    write_training_metrics,
):
    """Keep FedPhoenix sampling, tasks, local SGD, and aggregation unchanged."""
    net_glob.train()
    geometry = InteractionGeometry(net_glob, args.ifp_space)
    ledger = ResetLedger(net_glob)
    controller = InteractionController(
        geometry, ledger, args.ifp_rho, args.ifp_max_gap,
        bool(args.ifp_observe_only),
    )
    previous_global_vector = None
    accuracies = []
    rows = []
    args.density_local = 0.01
    print(f"InteractionFedPhoenix {args.ifp_space}: {geometry.numel} parameters")
    print(f"Estimated dense fp32 history for {args.num_users} clients: "
          f"{4 * geometry.numel * 4 * args.num_users / 2**30:.2f} GiB; "
          "observations store only reliable Conv kernels in bf16")

    for round_idx in range(args.epochs):
        round_start = time.perf_counter()
        if args.density_local > 1 or args.density_local < 0:
            args.density_local = 0
        print("*" * 80)
        print("Round {:3d}".format(round_idx))

        global_vector = geometry.flatten(net_glob.state_dict())
        global_change = (
            global_vector - previous_global_vector
            if previous_global_vector is not None else None
        )
        w_locals = []
        lens = []
        responses = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        task_models, task_traces = build_tasks(net_glob, round_idx, m, args)
        current_union = ledger.record(round_idx, task_traces)
        assignments = []
        events = []

        for task_id, idx in enumerate(idxs_users):
            client_id = int(idx)
            base_model = task_models[task_id]
            base_state = base_model.state_dict()
            own_reset = ledger.from_trace(task_traces[task_id])
            proposed, event = controller.decide(
                client_id, round_idx, own_reset, global_change
            )

            # Exact original FedPhoenix construction on the inactive path.
            net_local = copy.deepcopy(base_model).to(args.device)
            materialized = {}
            if proposed is not None:
                materialized = geometry.materialize(net_local, base_state, proposed)
                if not materialized:
                    event["reason"] = "rounded_to_zero"
                    event["lambda"] = 0.0
                    event["delta_norm"] = 0.0
            actual_dispatch = geometry.flatten(net_local.state_dict())
            if materialized:
                backbone_sq = sum(
                    float(torch.sum(value.to(torch.float64) ** 2))
                    for name, value in materialized.items()
                    if name in geometry.conv_by_param
                )
                head_sq = sum(
                    float(torch.sum(value.to(torch.float64) ** 2))
                    for name, value in materialized.items()
                    if name not in geometry.conv_by_param
                )
                materialized_norm = math.sqrt(backbone_sq + head_sq)
                event["materialized_delta_norm"] = materialized_norm
                if materialized_norm > args.ifp_rho * event["d_norm"] + max(
                    1e-6, 1e-4 * event["delta_norm"]
                ):
                    raise AssertionError("materialized action exceeds rho cap")
                event["proposed_delta_norm"] = event["delta_norm"]
                event["delta_norm"] = materialized_norm
                event["backbone_delta_norm"] = math.sqrt(backbone_sq)
                event["head_delta_norm"] = math.sqrt(head_sq)

            local = LocalUpdate_FedAvg(
                args=args, dataset=dataset_train, idxs=dict_users[idx],
                dataset_test=dataset_test,
            )
            w = local.train(net=net_local)
            response = geometry.flatten(w) - actual_dispatch
            if materialized:
                corrected = corrected_state(w, materialized)
                w_locals.append(corrected)
                # z=b+y-x on every actually changed coordinate.
                for name, action in materialized.items():
                    expected = w[name] - action
                    if not torch.equal(corrected[name], expected):
                        raise AssertionError("corrected state did not remove dispatch")
            else:
                # Do not perform state arithmetic on the zero-action path.
                w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            responses.append(response)
            controller.history.record(
                client_id, round_idx, actual_dispatch, response, own_reset
            )
            assignments.append({"client_id": client_id, "task_id": int(task_id)})
            event["materialized_delta_norm"] = event.get("materialized_delta_norm", 0.0)
            event["delta_over_d"] = (
                event["materialized_delta_norm"] / event["d_norm"]
                if event["d_norm"] and event["d_norm"] > 0 else 0.0
            )
            event["delta_over_u"] = (
                event["materialized_delta_norm"] / event["u_norm"]
                if event["u_norm"] and event["u_norm"] > 0 else 0.0
            )
            events.append(event)

        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)
        if (round_idx + 1) % args.ifp_check_every == 0 or round_idx == 0:
            clean_mask = geometry.mask(current_union)
            actual_progress = geometry.flatten(net_glob.state_dict()) - global_vector
            predicted_progress = _weighted_responses(responses, lens)
            error = float(torch.max(torch.abs(
                (actual_progress - predicted_progress) * clean_mask
            )))
            if error > 5e-5:
                raise AssertionError(
                    f"aggregation-clean invariant failed at round {round_idx}: {error}"
                )
        else:
            error = None
        previous_global_vector = global_vector
        round_train_seconds = time.perf_counter() - round_start

        accuracy = evaluate_round_accuracy(
            net_glob, dataset_test, args, round_idx + 1
        )
        accuracies.append(accuracy)
        reasons = Counter(event["reason"] for event in events)
        reset_count = sum(int(mask.sum()) for mask in current_union.values())
        kernel_count = sum(mask.numel() for mask in current_union.values())
        rows.append({
            "round": round_idx + 1,
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "test_accuracy": accuracy,
            "peak_accuracy_so_far": max(accuracies),
            "round_train_seconds": float(round_train_seconds),
            "selected_clients": json.dumps([int(v) for v in idxs_users.tolist()]),
            "assignments": json.dumps(assignments),
            "task_seeds": json.dumps([int(trace["seed"]) for trace in task_traces]),
            "history_eligible_rate": sum(
                event["reason"] not in {"cold_start", "stale", "no_obs"}
                for event in events
            ) / m,
            "history_gaps": json.dumps([event["gap"] for event in events]),
            "reset_kernel_coverage": reset_count / kernel_count,
            "reset_coverage_per_layer": json.dumps({
                name: float(mask.float().mean())
                for name, mask in current_union.items()
            }),
            "historical_mask_coverage": json.dumps(
                [event["hist_coverage"] for event in events]
            ),
            "decision_mask_coverage": json.dumps(
                [event["decision_coverage"] for event in events]
            ),
            "ledger_excluded_ratio": json.dumps(
                [event["ledger_excluded_ratio"] for event in events]
            ),
            "active_rate": reasons["active"] / m,
            "lambda_rho_saturation_rate": (
                sum(event["reason"] == "active" and
                    abs(event["lambda"] - args.ifp_rho) <= 1e-12
                    for event in events) / max(1, reasons["active"])
            ),
            "no_need_rate": reasons["no_need"] / m,
            "no_support_rate": reasons["no_support"] / m,
            "reason_counts": json.dumps(reasons),
            "history_memory_bytes": controller.history.bytes(),
            "aggregation_clean_max_error": error,
            "dispatch_events": json.dumps(events, allow_nan=False),
        })
        del task_models
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    print_peak_accuracy(accuracies, args.algorithm)
    write_training_metrics(args, rows)
    return rows
