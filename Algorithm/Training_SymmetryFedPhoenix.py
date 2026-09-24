"""FedPhoenix training with a reversible horizontal Conv model view."""

import copy
import json
import time

import numpy as np
import torch

from Algorithm.SymmetryView import (
    ViewScheduler, apply_horizontal_view_, map_back_state,
    verify_flip_equivariance,
)
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
        assignments = []
        views = []
        violations = 0

        for task_id, idx in enumerate(idxs_users):
            view, violation = scheduler.choose(idx, round_idx)
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
        })
        del task_models
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    print_peak_accuracy(accuracies, args.algorithm)
    write_training_metrics(args, rows)
    return rows
