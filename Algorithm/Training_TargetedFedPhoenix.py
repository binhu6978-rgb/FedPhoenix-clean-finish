#!/usr/bin/env python
"""Training path for TargetedFedPhoenix V1."""

import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from Algorithm.FedPhoenixHistoryObserver import (
    FedPhoenixHistoryObserver,
    capture_global_rng_state,
    global_rng_state_equal,
)
from Algorithm.TargetedFedPhoenix import TargetedFedPhoenixController
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg


ROUND_FIELDS = [
    "round", "algorithm", "seed", "tfp_target_ratio", "tfp_max_history_gap",
    "tfp_score_type", "tfp_target_layers", "tfp_start_round",
    "test_accuracy", "round_train_seconds", "selected_clients", "task_seeds",
    "global_state_sha256", "clients_with_history", "clients_history_eligible",
    "clients_targeted", "clients_dispatch_changed", "clients_cold_start",
    "clients_stale", "clients_no_valid_history", "clients_no_target_slots",
    "clients_before_start_round",
    "total_reset_slots", "targeted_reset_slots", "random_reset_slots",
    "actual_target_fraction", "baseline_random_overlap_slots",
    "baseline_random_overlap_rate", "replaced_reset_slots",
    "mean_history_gap_for_targeted", "history_memory_bytes",
    "controller_rng_unchanged", "failed_client_updates",
    "residual_total_kernels", "residual_valid_kernels",
    "residual_valid_fraction", "residual_own_reset_invalid",
    "residual_no_clean_peer_invalid",
]

LAYER_FIELDS = [
    "round", "layer", "baseline_reset_slots", "targeted_slots",
    "random_slots", "targeted_clients", "baseline_random_overlap_slots",
    "replaced_reset_slots", "cold_start_count", "stale_history_count",
    "no_target_slots_count", "no_valid_history_count", "targeted_count",
    "before_start_round_count", "non_target_layer_count",
]


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _json_dump(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _paths(args):
    directory = os.path.abspath(args.metrics_log_dir)
    os.makedirs(directory, exist_ok=True)
    return {
        "directory": directory,
        "config": os.path.join(directory, "config.json"),
        "rounds": os.path.join(directory, "round_metrics.csv"),
        "layers": os.path.join(directory, "layer_metrics.csv"),
        "traces": os.path.join(directory, "task_traces.jsonl"),
        "summary": os.path.join(directory, "summary.json"),
    }


def _layer_round_rows(round_number, traces):
    aggregate = defaultdict(Counter)
    for trace in traces:
        for layer in trace.get("layers", []):
            values = aggregate[layer["name"]]
            values["baseline_reset_slots"] += len(layer["baseline_reset_indices"])
            values["targeted_slots"] += int(layer["target_actual"])
            values["random_slots"] += len(layer["random_kept_indices"])
            values["targeted_clients"] += int(layer["target_actual"] > 0)
            values["baseline_random_overlap_slots"] += int(
                layer["baseline_random_overlap"]
            )
            values["replaced_reset_slots"] += int(layer["replaced_reset_slots"])
            values[layer["fallback_reason"] + "_count"] += 1
    rows = []
    for layer_name in sorted(aggregate):
        values = aggregate[layer_name]
        rows.append(
            {
                "round": int(round_number),
                "layer": layer_name,
                **{
                    field: int(values[field])
                    for field in LAYER_FIELDS
                    if field not in {"round", "layer"}
                },
            }
        )
    return rows


def _client_round_counts(traces):
    clients_with_history = sum(bool(trace["history_available"]) for trace in traces)
    eligible = sum(
        bool(trace["history_available"])
        and trace["history_gap"] is not None
        and int(trace["history_gap"]) <= int(trace["max_history_gap"])
        for trace in traces
    )
    targeted = sum(int(trace["targeted_reset_slots"]) > 0 for trace in traces)
    changed = sum(int(trace["replaced_reset_slots"]) > 0 for trace in traces)
    cold = sum(
        any(layer["fallback_reason"] == "cold_start" for layer in trace["layers"])
        for trace in traces
    )
    stale = sum(
        any(layer["fallback_reason"] == "stale_history" for layer in trace["layers"])
        for trace in traces
    )
    no_valid = sum(
        any(layer["fallback_reason"] == "no_valid_history" for layer in trace["layers"])
        and int(trace["targeted_reset_slots"]) == 0
        for trace in traces
    )
    no_slots = sum(
        bool(trace["layers"])
        and all(layer["fallback_reason"] == "no_target_slots" for layer in trace["layers"])
        for trace in traces
    )
    before_start = sum(
        any(layer["fallback_reason"] == "before_start_round" for layer in trace["layers"])
        for trace in traces
    )
    return {
        "clients_with_history": int(clients_with_history),
        "clients_history_eligible": int(eligible),
        "clients_targeted": int(targeted),
        "clients_dispatch_changed": int(changed),
        "clients_cold_start": int(cold),
        "clients_stale": int(stale),
        "clients_no_valid_history": int(no_valid),
        "clients_no_target_slots": int(no_slots),
        "clients_before_start_round": int(before_start),
    }


def _summarize(paths, args, accuracies, controller, runtime_seconds, failed_clients):
    with open(paths["rounds"], newline="", encoding="utf-8") as handle:
        rounds = list(csv.DictReader(handle))
    with open(paths["layers"], newline="", encoding="utf-8") as handle:
        layer_rows = list(csv.DictReader(handle))

    totals = Counter()
    for row in rounds:
        for key in (
            "clients_with_history", "clients_history_eligible", "clients_targeted",
            "clients_dispatch_changed", "clients_cold_start", "clients_stale",
            "clients_no_valid_history", "clients_no_target_slots", "total_reset_slots",
            "targeted_reset_slots", "random_reset_slots",
            "baseline_random_overlap_slots", "replaced_reset_slots",
            "clients_before_start_round", "residual_total_kernels",
            "residual_valid_kernels", "residual_own_reset_invalid",
            "residual_no_clean_peer_invalid",
        ):
            totals[key] += int(row[key])
    layer_totals = defaultdict(Counter)
    for row in layer_rows:
        values = layer_totals[row["layer"]]
        for key in LAYER_FIELDS:
            if key not in {"round", "layer"}:
                values[key] += int(row[key])

    targeted_slots = totals["targeted_reset_slots"]
    total_slots = totals["total_reset_slots"]
    gaps = [
        float(row["mean_history_gap_for_targeted"])
        for row in rounds
        if row["mean_history_gap_for_targeted"] not in {"", "None"}
        and int(row["clients_targeted"]) > 0
    ]
    gap_weights = [
        int(row["clients_targeted"])
        for row in rounds
        if row["mean_history_gap_for_targeted"] not in {"", "None"}
        and int(row["clients_targeted"]) > 0
    ]
    summary = {
        "algorithm": args.algorithm,
        "tfp_target_ratio": float(args.tfp_target_ratio),
        "tfp_max_history_gap": int(args.tfp_max_history_gap),
        "tfp_score_type": str(args.tfp_score_type),
        "tfp_target_layers": str(args.tfp_target_layers),
        "tfp_start_round": int(args.tfp_start_round),
        "seed": int(args.seed),
        "completed_rounds": len(accuracies),
        "peak_accuracy": float(max(accuracies)),
        "peak_round": int(np.argmax(accuracies)) + 1,
        "final_accuracy": float(accuracies[-1]),
        "last20_mean": float(np.mean(accuracies[-20:])),
        "last50_mean": float(np.mean(accuracies[-50:])),
        "rounds201_300_mean": (
            float(np.mean(accuracies[200:300])) if len(accuracies) >= 300 else None
        ),
        "best_rolling5_mean": float(
            max(np.mean(accuracies[index - 4:index + 1]) for index in range(4, len(accuracies)))
        ),
        "best_rolling5_end_round": int(
            max(
                range(4, len(accuracies)),
                key=lambda index: np.mean(accuracies[index - 4:index + 1]),
            )
            + 1
        ),
        "runtime_seconds": float(runtime_seconds),
        "client_statistics": dict(totals),
        "returning_history_target_rate": (
            float(totals["clients_targeted"] / totals["clients_with_history"])
            if totals["clients_with_history"] else 0.0
        ),
        "eligible_history_target_rate": (
            float(totals["clients_targeted"] / totals["clients_history_eligible"])
            if totals["clients_history_eligible"] else 0.0
        ),
        "average_actual_target_fraction": (
            float(targeted_slots / total_slots) if total_slots else 0.0
        ),
        "baseline_random_overlap_rate": (
            float(totals["baseline_random_overlap_slots"] / targeted_slots)
            if targeted_slots else 0.0
        ),
        "actual_replaced_reset_slots": int(totals["replaced_reset_slots"]),
        "mean_history_gap_for_targeted": (
            float(np.average(gaps, weights=gap_weights)) if gaps else None
        ),
        "layer_statistics": [
            {"layer": layer, **dict(values)}
            for layer, values in sorted(layer_totals.items())
        ],
        "maximum_history_memory_bytes": max(
            int(row["history_memory_bytes"]) for row in rounds
        ),
        "history_inventory": controller.history_inventory(),
        "controller_rng_unchanged": all(
            row["controller_rng_unchanged"].lower() == "true" for row in rounds
        ),
        "failed_client_updates": int(failed_clients),
        "valid_residual_history_fraction": (
            float(totals["residual_valid_kernels"] / totals["residual_total_kernels"])
            if totals["residual_total_kernels"]
            else None
        ),
        "residual_no_clean_peer_invalid_count": int(
            totals["residual_no_clean_peer_invalid"]
        ),
        "output_files": paths,
    }
    _json_dump(paths["summary"], summary)
    return summary


def train_targeted_fedphoenix(
    args,
    net_glob,
    dataset_train,
    dataset_test,
    dict_users,
    build_fedphoenix_tasks,
    evaluate_round_accuracy,
    print_peak_accuracy,
):
    """Run legacy FedPhoenix with reset locations deterministically retargeted."""
    paths = _paths(args)
    _json_dump(paths["config"], vars(args))
    controller = TargetedFedPhoenixController(
        net_glob,
        target_ratio=args.tfp_target_ratio,
        max_history_gap=args.tfp_max_history_gap,
        score_type=args.tfp_score_type,
        target_layers=args.tfp_target_layers,
        start_round=args.tfp_start_round,
    )
    residual_observer = (
        FedPhoenixHistoryObserver(net_glob, reset_ratio=args.reset)
        if args.tfp_score_type == "residual"
        else None
    )
    accuracies = []
    failed_clients = 0
    rng_unchanged = True
    started = time.perf_counter()
    net_glob.train()
    args.density_local = 0.01

    with open(paths["rounds"], "w", newline="", encoding="utf-8") as round_handle, open(
        paths["layers"], "w", newline="", encoding="utf-8"
    ) as layer_handle, open(paths["traces"], "w", encoding="utf-8") as trace_handle:
        round_writer = csv.DictWriter(round_handle, fieldnames=ROUND_FIELDS)
        layer_writer = csv.DictWriter(layer_handle, fieldnames=LAYER_FIELDS)
        round_writer.writeheader()
        layer_writer.writeheader()

        for round_idx in range(args.epochs):
            round_number = round_idx + 1
            round_start = time.perf_counter()
            if args.density_local > 1 or args.density_local < 0:
                args.density_local = 0
            print("*" * 80)
            print("Round {:3d}".format(round_idx))
            m = max(int(args.frac * args.num_users), 1)
            selected = np.random.choice(range(args.num_users), m, replace=False)
            task_models, baseline_traces = build_fedphoenix_tasks(
                net_glob, round_idx, m, args
            )
            w_locals = []
            lens = []
            final_traces = []
            residual_interactions = []

            for task_id, selected_id in enumerate(selected):
                client_id = int(selected_id)
                rng_before = capture_global_rng_state()
                task_model, final_trace = controller.retarget_task(
                    global_model=net_glob,
                    baseline_task_model=task_models[task_id],
                    baseline_trace=baseline_traces[task_id],
                    client_id=client_id,
                    current_round=round_number,
                )
                rng_after = capture_global_rng_state()
                unchanged = global_rng_state_equal(rng_before, rng_after)
                rng_unchanged = rng_unchanged and unchanged
                if not unchanged:
                    raise AssertionError("retargeting changed global RNG state")

                net_local = copy.deepcopy(task_model).to(args.device)
                local = LocalUpdate_FedAvg(
                    args=args,
                    dataset=dataset_train,
                    idxs=dict_users[client_id],
                    dataset_test=dataset_test,
                )
                try:
                    returned_state = local.train(net=net_local)
                except Exception:
                    failed_clients += 1
                    raise
                w_locals.append(copy.deepcopy(returned_state))
                lens.append(len(dict_users[client_id]))

                rng_before = capture_global_rng_state()
                if residual_observer is None:
                    controller.observe(
                        client_id=client_id,
                        current_round=round_number,
                        actual_dispatch_state=task_model.state_dict(),
                        returned_state=returned_state,
                        final_trace=final_trace,
                    )
                else:
                    residual_interactions.append(
                        residual_observer.build_interaction(
                            client_id=client_id,
                            client_weight=len(dict_users[client_id]),
                            dispatch_state=task_model.state_dict(),
                            returned_state=returned_state,
                            task_trace=final_trace,
                            task_id=task_id,
                        )
                    )
                rng_after = capture_global_rng_state()
                unchanged = global_rng_state_equal(rng_before, rng_after)
                rng_unchanged = rng_unchanged and unchanged
                if not unchanged:
                    raise AssertionError("history observation changed global RNG state")
                final_trace["task_id"] = int(task_id)
                final_traces.append(final_trace)
                trace_handle.write(
                    json.dumps(
                        {
                            "round": round_number,
                            "task_id": int(task_id),
                            "client_id": client_id,
                            "trace": final_trace,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                del net_local, local, returned_state

            residual_statistics = {
                "total_kernels": 0,
                "valid_kernels": 0,
                "valid_fraction": None,
                "own_reset_invalid": 0,
                "no_clean_peer_invalid": 0,
            }
            if residual_observer is not None:
                rng_before = capture_global_rng_state()
                current_scores = residual_observer.score_current_round(
                    residual_interactions
                )
                residual_statistics = controller.observe_residual_round(
                    current_scores, round_number
                )
                rng_after = capture_global_rng_state()
                unchanged = global_rng_state_equal(rng_before, rng_after)
                rng_unchanged = rng_unchanged and unchanged
                if not unchanged:
                    raise AssertionError("residual history observation changed global RNG state")

            trace_handle.flush()
            w_glob = Aggregation(w_locals, lens)
            net_glob.load_state_dict(w_glob)
            train_seconds = time.perf_counter() - round_start
            accuracy = float(
                evaluate_round_accuracy(net_glob, dataset_test, args, round_number)
            )
            accuracies.append(accuracy)

            client_counts = _client_round_counts(final_traces)
            total_slots = sum(int(trace["num_reset_kernels"]) for trace in final_traces)
            targeted_slots = sum(
                int(trace["targeted_reset_slots"]) for trace in final_traces
            )
            overlap_slots = sum(
                int(trace["baseline_random_overlap"]) for trace in final_traces
            )
            replaced_slots = sum(
                int(trace["replaced_reset_slots"]) for trace in final_traces
            )
            targeted_gaps = [
                int(trace["history_gap"])
                for trace in final_traces
                if int(trace["targeted_reset_slots"]) > 0
            ]
            round_writer.writerow(
                {
                    "round": round_number,
                    "algorithm": args.algorithm,
                    "seed": int(args.seed),
                    "tfp_target_ratio": float(args.tfp_target_ratio),
                    "tfp_max_history_gap": int(args.tfp_max_history_gap),
                    "tfp_score_type": str(args.tfp_score_type),
                    "tfp_target_layers": str(args.tfp_target_layers),
                    "tfp_start_round": int(args.tfp_start_round),
                    "test_accuracy": accuracy,
                    "round_train_seconds": float(train_seconds),
                    "selected_clients": json.dumps(
                        [int(value) for value in selected.tolist()]
                    ),
                    "task_seeds": json.dumps(
                        [int(trace["seed"]) for trace in baseline_traces]
                    ),
                    "global_state_sha256": _state_sha256(net_glob.state_dict()),
                    **client_counts,
                    "total_reset_slots": int(total_slots),
                    "targeted_reset_slots": int(targeted_slots),
                    "random_reset_slots": int(total_slots - targeted_slots),
                    "actual_target_fraction": (
                        float(targeted_slots / total_slots) if total_slots else 0.0
                    ),
                    "baseline_random_overlap_slots": int(overlap_slots),
                    "baseline_random_overlap_rate": (
                        float(overlap_slots / targeted_slots) if targeted_slots else 0.0
                    ),
                    "replaced_reset_slots": int(replaced_slots),
                    "mean_history_gap_for_targeted": (
                        float(np.mean(targeted_gaps)) if targeted_gaps else None
                    ),
                    "history_memory_bytes": controller.history_memory_bytes(),
                    "controller_rng_unchanged": bool(rng_unchanged),
                    "failed_client_updates": int(failed_clients),
                    "residual_total_kernels": int(residual_statistics["total_kernels"]),
                    "residual_valid_kernels": int(residual_statistics["valid_kernels"]),
                    "residual_valid_fraction": residual_statistics["valid_fraction"],
                    "residual_own_reset_invalid": int(
                        residual_statistics["own_reset_invalid"]
                    ),
                    "residual_no_clean_peer_invalid": int(
                        residual_statistics["no_clean_peer_invalid"]
                    ),
                }
            )
            layer_writer.writerows(_layer_round_rows(round_number, final_traces))
            round_handle.flush()
            layer_handle.flush()
            print(
                "TargetedFedPhoenix: "
                f"round={round_number} clients_targeted={client_counts['clients_targeted']} "
                f"targeted_slots={targeted_slots}/{total_slots} "
                f"replaced_slots={replaced_slots} history_bytes={controller.history_memory_bytes()}"
            )
            del task_models, baseline_traces, final_traces, w_locals, w_glob
            del residual_interactions
            if args.device.type == "cuda":
                torch.cuda.empty_cache()

    summary = _summarize(
        paths,
        args,
        accuracies,
        controller,
        time.perf_counter() - started,
        failed_clients,
    )
    print_peak_accuracy(accuracies, args.algorithm)
    return summary
