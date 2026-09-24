"""FedPhoenix training with a reversible horizontal Conv model view."""

import copy
import csv
import json
import os
import statistics
import time

import numpy as np
import torch

from Algorithm.SymmetryView import (
    ViewScheduler, apply_horizontal_view_, map_back_state,
    verify_flip_equivariance,
)
from Algorithm.SymmetryDiagnostics import RESET_METRICS, reset_response_scalars
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg


def train_symmetry_fedphoenix(
    args, net_glob, dataset_train, dataset_test, dict_users,
    build_tasks, evaluate_round_accuracy, print_peak_accuracy,
    write_training_metrics,
):
    net_glob.train()
    equivariance_error = verify_flip_equivariance(
        net_glob, input_shape=(2, args.num_channels, 32, 32)
    )
    print(f"Horizontal-view equivariance max error: {equivariance_error:.9g}")
    print(f"CIFAR10 train transform: {dataset_train.transform}")
    scheduler = ViewScheduler(args.sym_view, args.seed)
    accuracies = []
    rows = []
    args.density_local = 0.01
    os.makedirs(args.metrics_log_dir, exist_ok=True)
    event_path = os.path.abspath(os.path.join(
        args.metrics_log_dir,
        f"{args.run_name or 'symmetry'}_client_events.csv",
    ))
    event_fields = [
        "round", "client_id", "task_id", "task_seed", "visit_index",
        "is_returning", "previous_view", "current_view", "reset_kernel_count",
        *RESET_METRICS,
    ]
    event_file = open(event_path, "w", newline="", encoding="utf-8")
    event_writer = csv.DictWriter(event_file, fieldnames=event_fields)
    event_writer.writeheader()

    for round_idx in range(args.epochs):
        round_start = time.perf_counter()
        if args.density_local > 1 or args.density_local < 0:
            args.density_local = 0
        print("*" * 80)
        print("Round {:3d}".format(round_idx))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        task_models, task_traces = build_tasks(net_glob, round_idx, m, args)
        global_state = net_glob.state_dict()
        assignments = []
        views = []
        round_events = []
        violations = 0

        for task_id, idx in enumerate(idxs_users):
            previous = scheduler.last.get(int(idx))
            previous_view = previous[0] if previous is not None else None
            view, violation = scheduler.choose(idx, round_idx)
            visit_index = scheduler.participation_count[int(idx)]
            violations += int(violation)
            net_local = copy.deepcopy(task_models[task_id]).to(args.device)
            if view == "F":
                apply_horizontal_view_(net_local)
            local = LocalUpdate_FedAvg(
                args=args, dataset=dataset_train, idxs=dict_users[idx],
                dataset_test=dataset_test,
            )
            w = local.train(net=net_local)
            if view == "F":
                w_locals.append(map_back_state(w, net_local))
            else:
                # Preserve the original FedPhoenix path for the identity view.
                w_locals.append(copy.deepcopy(w))
            response = reset_response_scalars(
                global_state, task_models[task_id].state_dict(),
                w_locals[-1], task_traces[task_id],
            )
            round_events.append({
                "round": round_idx + 1,
                "client_id": int(idx),
                "task_id": task_id,
                "task_seed": int(task_traces[task_id]["seed"]),
                "visit_index": visit_index,
                "is_returning": visit_index > 1,
                "previous_view": previous_view,
                "current_view": view,
                **response,
            })
            lens.append(len(dict_users[idx]))
            assignments.append({"client_id": int(idx), "task_id": int(task_id)})
            views.append(view)

        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)
        round_train_seconds = time.perf_counter() - round_start
        accuracy = evaluate_round_accuracy(
            net_glob, dataset_test, args, round_idx + 1
        )
        accuracies.append(accuracy)
        first_time = [event for event in round_events if not event["is_returning"]]
        returning = [event for event in round_events if event["is_returning"]]
        view_stats = {}
        for label in ("I", "F"):
            view_events = [event for event in round_events if event["current_view"] == label]
            for metric in RESET_METRICS:
                values = [event[metric] for event in view_events]
                view_stats[f"{metric}_mean_{label}"] = (
                    statistics.fmean(values) if values else None
                )
                view_stats[f"{metric}_median_{label}"] = (
                    statistics.median(values) if values else None
                )
        rows.append({
            "round": round_idx + 1,
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "test_accuracy": accuracy,
            "peak_accuracy_so_far": max(accuracies),
            "round_train_seconds": float(round_train_seconds),
            "selected_clients": json.dumps([int(v) for v in idxs_users.tolist()]),
            "assignments": json.dumps(assignments),
            "task_ids": json.dumps(list(range(m))),
            "task_seeds": json.dumps([int(trace["seed"]) for trace in task_traces]),
            "views": json.dumps(views),
            "flip_fraction": views.count("F") / m,
            "alt_violations": violations,
            "returning_client_rate": len(returning) / m,
            "mean_visit_index": statistics.fmean(
                event["visit_index"] for event in round_events
            ),
            "max_visit_index": max(event["visit_index"] for event in round_events),
            "first_time_client_count": len(first_time),
            "returning_client_count": len(returning),
            "first_time_F_fraction": (
                sum(event["current_view"] == "F" for event in first_time) / len(first_time)
                if first_time else None
            ),
            "returning_F_fraction": (
                sum(event["current_view"] == "F" for event in returning) / len(returning)
                if returning else None
            ),
            "I_client_count": views.count("I"),
            "F_client_count": views.count("F"),
            **view_stats,
        })
        event_writer.writerows(round_events)
        event_file.flush()
        del task_models
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    print_peak_accuracy(accuracies, args.algorithm)
    event_file.close()
    print(f"Symmetry client events saved to {event_path}")
    write_training_metrics(args, rows)
    return rows
