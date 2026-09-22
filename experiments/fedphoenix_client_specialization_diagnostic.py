#!/usr/bin/env python
"""Trajectory-preserving FedPhoenix client-specialization diagnostic."""

import copy
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict

import numpy as np
import torch

from Algorithm.FedPhoenixHistoryObserver import (
    SCORE_TYPES,
    FedPhoenixHistoryObserver,
    capture_global_rng_state,
    global_rng_state_equal,
)
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg


EVENT_FIELDS = [
    "round", "client_id", "previous_round", "gap", "layer", "score_type",
    "output_kernels", "reset_budget", "reset_eligible_now",
    "common_valid_kernels", "k_eval", "previous_topk", "current_topk",
    "intersection_count", "overlap_rate", "random_overlap_rate", "overlap_lift",
    "spearman", "cross_history_count", "cross_client_ids", "cross_history_rounds",
    "cross_overlap_rate_mean", "cross_spearman_mean",
    "identity_overlap_advantage", "identity_rank_advantage",
]

ROUND_FIELDS = [
    "round", "algorithm", "seed", "test_accuracy", "round_train_seconds",
    "selected_clients", "assignments", "task_seeds", "reset_indices",
    "global_state_sha256", "persistence_events", "num_clients_seen",
    "history_memory_bytes", "observer_rng_unchanged",
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


def _trace_reset_indices(task_traces):
    return [
        {
            "seed": int(trace["seed"]),
            "layers": [
                {
                    "name": layer["name"],
                    "reset_indices": [int(value) for value in layer["reset_indices"]],
                }
                for layer in trace.get("layers", [])
            ],
        }
        for trace in task_traces
    ]


def _optional_float(row, name):
    value = row.get(name, "")
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _gap_bucket(gap):
    gap = int(gap)
    if gap <= 10:
        return "1-10"
    if gap <= 20:
        return "11-20"
    if gap <= 40:
        return "21-40"
    return ">40"


def _metric_summary(rows):
    fields = {
        "same_overlap": "overlap_rate",
        "random_expected_overlap": "random_overlap_rate",
        "cross_overlap": "cross_overlap_rate_mean",
        "identity_overlap_advantage": "identity_overlap_advantage",
        "same_spearman": "spearman",
        "cross_spearman": "cross_spearman_mean",
        "identity_rank_advantage": "identity_rank_advantage",
    }
    values = {
        output: [value for row in rows if (value := _optional_float(row, source)) is not None]
        for output, source in fields.items()
    }
    return {
        "number_valid_events": len(rows),
        "mean_same_overlap": float(np.mean(values["same_overlap"])) if values["same_overlap"] else None,
        "median_same_overlap": float(np.median(values["same_overlap"])) if values["same_overlap"] else None,
        "mean_random_expected_overlap": (
            float(np.mean(values["random_expected_overlap"]))
            if values["random_expected_overlap"] else None
        ),
        "mean_cross_overlap": float(np.mean(values["cross_overlap"])) if values["cross_overlap"] else None,
        "mean_identity_overlap_advantage": (
            float(np.mean(values["identity_overlap_advantage"]))
            if values["identity_overlap_advantage"] else None
        ),
        "mean_same_spearman": float(np.mean(values["same_spearman"])) if values["same_spearman"] else None,
        "median_same_spearman": float(np.median(values["same_spearman"])) if values["same_spearman"] else None,
        "mean_cross_spearman": float(np.mean(values["cross_spearman"])) if values["cross_spearman"] else None,
        "mean_identity_rank_advantage": (
            float(np.mean(values["identity_rank_advantage"]))
            if values["identity_rank_advantage"] else None
        ),
    }


def _write_table(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_persistence(events_path, observer, output_dir):
    with open(events_path, newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))

    persistence_rows = []
    for score_type in SCORE_TYPES:
        score_events = [row for row in events if row["score_type"] == score_type]
        for scope, scoped in (
            ("all_layers", score_events),
            (
                "reset_eligible_layers",
                [row for row in score_events if row["reset_eligible_now"].lower() == "true"],
            ),
        ):
            persistence_rows.append(
                {"score_type": score_type, "scope": scope, **_metric_summary(scoped)}
            )

    layer_rows = []
    for layer_name, geometry in observer.geometry.by_name.items():
        for score_type in SCORE_TYPES:
            selected = [
                row for row in events
                if row["layer"] == layer_name and row["score_type"] == score_type
            ]
            layer_rows.append(
                {
                    "layer": layer_name,
                    "score_type": score_type,
                    "output_kernels": geometry["output_kernels"],
                    "reset_budget": geometry["reset_budget"],
                    **_metric_summary(selected),
                }
            )

    gap_rows = []
    for score_type in SCORE_TYPES:
        for bucket in ("1-10", "11-20", "21-40", ">40"):
            selected = [
                row for row in events
                if row["score_type"] == score_type and _gap_bucket(row["gap"]) == bucket
            ]
            if selected:
                gap_rows.append(
                    {"score_type": score_type, "gap_bucket": bucket, **_metric_summary(selected)}
                )

    persistence_path = os.path.join(output_dir, "persistence_summary.csv")
    layer_path = os.path.join(output_dir, "layer_summary.csv")
    gap_path = os.path.join(output_dir, "gap_summary.csv")
    _write_table(persistence_path, persistence_rows)
    _write_table(layer_path, layer_rows)
    _write_table(gap_path, gap_rows)
    return persistence_rows, layer_rows, gap_rows


def _format_optional(value):
    return "NA" if value is None else f"{value:.6f}"


def _write_results_summary(path, run_summary, persistence_rows):
    lines = [
        "# FedPhoenix client-specialization diagnostic", "",
        "This observer-only run preserves the original FedPhoenix training trajectory.", "",
        "## Accuracy", "",
        f"- Peak: {run_summary['peak_accuracy']:.6f}% at round {run_summary['peak_round']}",
        f"- Final: {run_summary['final_accuracy']:.6f}%",
        f"- Last-20 mean: {run_summary['last20_mean']:.6f}%", "",
        "## Persistence", "",
        "| Score | Scope | Events | Same overlap | Cross overlap | Random | Identity advantage | Same Spearman | Cross Spearman | Rank advantage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in persistence_rows:
        lines.append(
            f"| {row['score_type']} | {row['scope']} | {row['number_valid_events']} | "
            f"{_format_optional(row['mean_same_overlap'])} | "
            f"{_format_optional(row['mean_cross_overlap'])} | "
            f"{_format_optional(row['mean_random_expected_overlap'])} | "
            f"{_format_optional(row['mean_identity_overlap_advantage'])} | "
            f"{_format_optional(row['mean_same_spearman'])} | "
            f"{_format_optional(row['mean_cross_spearman'])} | "
            f"{_format_optional(row['mean_identity_rank_advantage'])} |"
        )
    lines.extend(
        [
            "", "## Scope", "",
            "The diagnostic measures longitudinal predictiveness of historical kernel rankings. "
            "It does not establish that resetting high-scoring kernels improves utility.", "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def train_fedphoenix_history_diagnostic(
    args,
    net_glob,
    dataset_train,
    dataset_test,
    dict_users,
    build_fedphoenix_tasks,
    evaluate_round_accuracy,
    print_peak_accuracy,
):
    output_dir = os.path.abspath(args.metrics_log_dir)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    round_path = os.path.join(output_dir, "round_metrics.csv")
    events_path = os.path.join(output_dir, "persistence_events.csv")
    _json_dump(config_path, vars(args))

    observer = FedPhoenixHistoryObserver(net_glob, reset_ratio=args.reset)
    accuracies = []
    max_history_memory = 0
    observer_rng_unchanged = True
    failed_clients = 0
    args.density_local = 0.01
    net_glob.train()

    with open(round_path, "w", newline="", encoding="utf-8") as round_handle, open(
        events_path, "w", newline="", encoding="utf-8"
    ) as event_handle:
        round_writer = csv.DictWriter(round_handle, fieldnames=ROUND_FIELDS)
        event_writer = csv.DictWriter(event_handle, fieldnames=EVENT_FIELDS)
        round_writer.writeheader()
        event_writer.writeheader()

        for round_idx in range(args.epochs):
            round_start = time.perf_counter()
            if args.density_local > 1 or args.density_local < 0:
                args.density_local = 0
            print("*" * 80)
            print("Round {:3d}".format(round_idx))

            w_locals = []
            lens = []
            interactions = []
            assignments = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            task_models, task_traces = build_fedphoenix_tasks(
                net_glob, round_idx, m, args
            )

            for task_id, selected_id in enumerate(idxs_users):
                client_id = int(selected_id)
                net_local = copy.deepcopy(task_models[task_id]).to(args.device)
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
                client_size = len(dict_users[client_id])
                lens.append(client_size)
                assignments.append({"client_id": client_id, "task_id": int(task_id)})

                rng_before = capture_global_rng_state()
                interaction = observer.build_interaction(
                    client_id=client_id,
                    client_weight=client_size,
                    dispatch_state=task_models[task_id].state_dict(),
                    returned_state=returned_state,
                    task_trace=task_traces[task_id],
                    task_id=task_id,
                )
                rng_after = capture_global_rng_state()
                unchanged = global_rng_state_equal(rng_before, rng_after)
                observer_rng_unchanged = observer_rng_unchanged and unchanged
                if not unchanged:
                    raise AssertionError("observer interaction capture changed global RNG state")
                interactions.append(interaction)

            rng_before = capture_global_rng_state()
            round_events = observer.process_round(interactions, round_idx + 1)
            rng_after = capture_global_rng_state()
            unchanged = global_rng_state_equal(rng_before, rng_after)
            observer_rng_unchanged = observer_rng_unchanged and unchanged
            if not unchanged:
                raise AssertionError("observer round processing changed global RNG state")
            event_writer.writerows(round_events)
            event_handle.flush()

            w_glob = Aggregation(w_locals, lens)
            net_glob.load_state_dict(w_glob)
            train_seconds = time.perf_counter() - round_start
            accuracy = float(
                evaluate_round_accuracy(net_glob, dataset_test, args, round_idx + 1)
            )
            accuracies.append(accuracy)
            history_bytes = observer.history_memory_bytes()
            max_history_memory = max(max_history_memory, history_bytes)
            round_writer.writerow(
                {
                    "round": round_idx + 1,
                    "algorithm": args.algorithm,
                    "seed": int(args.seed),
                    "test_accuracy": accuracy,
                    "round_train_seconds": train_seconds,
                    "selected_clients": json.dumps([int(value) for value in idxs_users.tolist()]),
                    "assignments": json.dumps(assignments),
                    "task_seeds": json.dumps([int(trace["seed"]) for trace in task_traces]),
                    "reset_indices": json.dumps(_trace_reset_indices(task_traces), separators=(",", ":")),
                    "global_state_sha256": _state_sha256(net_glob.state_dict()),
                    "persistence_events": len(round_events),
                    "num_clients_seen": len(observer.history),
                    "history_memory_bytes": history_bytes,
                    "observer_rng_unchanged": observer_rng_unchanged,
                }
            )
            round_handle.flush()
            print(
                "FedPhoenixHistoryObserver: "
                f"round={round_idx + 1} events={len(round_events)} "
                f"clients_seen={len(observer.history)} history_bytes={history_bytes}"
            )
            del interactions, task_models, task_traces, w_locals, w_glob
            if args.device.type == "cuda":
                torch.cuda.empty_cache()

    persistence_rows, layer_rows, gap_rows = summarize_persistence(
        events_path, observer, output_dir
    )
    peak_accuracy = max(accuracies)
    summary = {
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "completed_rounds": len(accuracies),
        "peak_accuracy": float(peak_accuracy),
        "peak_round": int(np.argmax(accuracies)) + 1,
        "final_accuracy": float(accuracies[-1]),
        "last20_mean": float(np.mean(accuracies[-20:])),
        "maximum_history_memory_bytes": int(max_history_memory),
        "final_clients_seen": len(observer.history),
        "observer_rng_unchanged": bool(observer_rng_unchanged),
        "failed_client_updates": int(failed_clients),
        "persistence_summary": persistence_rows,
        "output_files": {
            "round_metrics": round_path,
            "persistence_events": events_path,
            "persistence_summary": os.path.join(output_dir, "persistence_summary.csv"),
            "gap_summary": os.path.join(output_dir, "gap_summary.csv"),
            "layer_summary": os.path.join(output_dir, "layer_summary.csv"),
        },
    }
    _json_dump(os.path.join(output_dir, "summary.json"), summary)
    _write_results_summary(
        os.path.join(output_dir, "RESULTS_SUMMARY.md"), summary, persistence_rows
    )
    print_peak_accuracy(accuracies, args.algorithm)
    return summary
