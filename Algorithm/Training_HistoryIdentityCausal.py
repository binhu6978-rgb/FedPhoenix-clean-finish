"""Four-arm, 1000-round FedPhoenix identity-source causal ablation.

B0 is the current FedPhoenix local-training/aggregation path with observer-only
score-bank logging. H/S/P use the same frozen B0 bank and frozen matching map.
"""

import copy
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from Algorithm.FedPhoenixHistoryObserver import (
    capture_global_rng_state,
    global_rng_state_equal,
)
from Algorithm.HistoryIdentityCausal import (
    FrozenHistoryController,
    FrozenScoreBankReader,
    FrozenScoreBankWriter,
)
from Algorithm.TargetedFedPhoenix import TargetedFedPhoenixController
from Algorithm.Training_TargetedFedPhoenix import _state_sha256
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg


RUN_NAMES = {
    "HistoryIdentityB0": "B0",
    "HistoryIdentitySame": "H_same",
    "HistoryIdentityShuffled": "S_shuffled",
    "HistoryIdentityPopulation": "P_population",
}
ROUND_FIELDS = (
    "round", "algorithm", "seed", "test_accuracy", "round_train_seconds",
    "selected_clients", "task_seeds", "global_state_sha256",
    "total_reset_slots", "targeted_reset_slots", "replaced_reset_slots",
    "targeted_clients", "fallback_clients", "fallback_client_layers",
    "history_memory_bytes", "source_support_sha256", "rng_unchanged",
    "failed_client_updates",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        + "\n", encoding="utf-8",
    )


def _support_sha256(controller, selected):
    digest = hashlib.sha256()
    for client in selected:
        digest.update(str(client).encode("ascii"))
        entry = controller.source_entries.get(client)
        if entry is None:
            digest.update(b"fallback")
            continue
        for name, layer in entry["layers"].items():
            digest.update(name.encode("ascii"))
            digest.update(layer["validity"].numpy().tobytes())
    return digest.hexdigest()


def _history_bytes(bank_history):
    return sum(
        layer["score"].numel() * layer["score"].element_size()
        + layer["validity"].numel() * layer["validity"].element_size()
        for entry in bank_history.values() for layer in entry["layers"].values()
    )


def train_history_identity_causal(
    args, net_glob, dataset_train, dataset_test, dict_users,
    build_fedphoenix_tasks, evaluate_round_accuracy, print_peak_accuracy,
):
    if args.algorithm not in RUN_NAMES:
        raise ValueError("unrecognized history identity arm")
    if not (
        args.dataset == "cifar10" and args.model == "vgg"
        and args.num_users == 100 and args.frac == 0.1
        and args.epochs in {5, 1000} and args.seed == 1
        and args.local_ep == 5 and args.local_bs == 50 and args.bs == 256
        and args.optimizer == "sgd" and args.lr == 0.01
        and args.momentum == 0.5 and args.weight_decay == 0
        and args.iid == 0 and args.noniid_case == 5
        and args.data_beta == 0.3 and args.generate_data == 0
        and args.num_classes == 10 and args.num_channels == 3
        and args.num_workers == 0
        and args.FP_conv == 1000 and args.reset == 0.015625
        and args.remethod == "ori_normal"
    ):
        raise ValueError("history identity arm protocol differs from frozen specification")
    run_name = RUN_NAMES[args.algorithm]
    is_baseline = run_name == "B0"
    if not is_baseline and not (
        args.tfp_score_type == "update_norm"
        and args.tfp_target_ratio == 0.25
        and args.tfp_max_history_gap == 10
        and args.tfp_target_layers == "features.20,features.24,features.27,features.30"
        and args.tfp_start_round == 151
    ):
        raise ValueError("targeted arm differs from frozen E5 specification")

    run_dir = Path(args.metrics_log_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    root_dir = run_dir.parent
    schedule_path = root_dir / "matching_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))["rounds"]
    if len(schedule) < args.epochs:
        raise ValueError("frozen matching schedule is shorter than the requested run")
    partition_path = Path("data/cifar10_100_noniidCase5_beta0.3.json").resolve()
    config = {
        "args": vars(args),
        "run_name": run_name,
        "history_bank_policy": (
            "B0 observer-only producer" if is_baseline else
            "frozen B0 score/validity bank, never on-policy arm history"
        ),
        "partition_sha256": _sha256_file(partition_path),
        "matching_schedule_sha256": _sha256_file(schedule_path),
        "implementation_sha256": {
            path: _sha256_file(Path(path).resolve())
            for path in (
                "main_fed.py",
                "Algorithm/Phoenix_util.py",
                "Algorithm/TargetedFedPhoenix.py",
                "Algorithm/HistoryIdentityCausal.py",
                "Algorithm/Training_HistoryIdentityCausal.py",
            )
        },
    }
    _write_json(run_dir / "config.json", config)

    if is_baseline:
        controller = TargetedFedPhoenixController(net_glob, target_ratio=0.25,
            max_history_gap=10, score_type="update_norm",
            target_layers=args.tfp_target_layers, start_round=151)
        bank_writer = FrozenScoreBankWriter(run_dir / "score_bank", controller.geometry)
        bank_reader = None
    else:
        controller = FrozenHistoryController(
            net_glob, mode=run_name, reset_ratio=args.reset,
            target_ratio=args.tfp_target_ratio,
            max_history_gap=args.tfp_max_history_gap,
            score_type=args.tfp_score_type,
            target_layers=args.tfp_target_layers,
            start_round=args.tfp_start_round,
        )
        bank_writer = None
        bank_reader = FrozenScoreBankReader(root_dir / "B0/score_bank")

    accuracies = []
    failed_clients = 0
    started = time.perf_counter()
    net_glob.train()
    args.density_local = 0.01
    round_path = run_dir / "round_metrics.csv"
    trace_path = run_dir / "task_traces.jsonl"
    with round_path.open("w", newline="", encoding="utf-8") as round_handle, \
            trace_path.open("w", encoding="utf-8") as trace_handle:
        writer = csv.DictWriter(round_handle, fieldnames=ROUND_FIELDS)
        writer.writeheader()
        for round_idx in range(args.epochs):
            round_number = round_idx + 1
            round_started = time.perf_counter()
            if args.density_local > 1 or args.density_local < 0:
                args.density_local = 0
            print("*" * 80)
            print("Round {:3d}".format(round_idx), flush=True)
            count = max(int(args.frac * args.num_users), 1)
            selected = [int(value) for value in np.random.choice(
                range(args.num_users), count, replace=False
            )]
            frozen = schedule[round_idx]
            if frozen["round"] != round_number or selected != frozen["selected_clients"]:
                raise AssertionError(f"round {round_number}: selected clients differ from frozen schedule")
            task_models, baseline_traces = build_fedphoenix_tasks(
                net_glob, round_idx, count, args
            )
            task_seeds = [int(trace["seed"]) for trace in baseline_traces]
            if task_seeds != frozen["task_seeds"]:
                raise AssertionError(f"round {round_number}: task seeds differ from frozen schedule")
            if bank_reader is not None:
                bank_history = bank_reader.through(round_idx)
                rng_before = capture_global_rng_state()
                controller.prepare_round(frozen, bank_history)
                rng_unchanged = global_rng_state_equal(
                    rng_before, capture_global_rng_state()
                )
                support_hash = _support_sha256(controller, selected)
            else:
                bank_history = controller.history
                rng_unchanged = True
                support_hash = ""
            w_locals = []
            lens = []
            traces = []
            for task_id, client in enumerate(selected):
                task_model = task_models[task_id]
                if not is_baseline:
                    rng_before = capture_global_rng_state()
                    task_model, trace = controller.retarget_task(
                        global_model=net_glob,
                        baseline_task_model=task_model,
                        baseline_trace=baseline_traces[task_id],
                        client_id=client,
                        current_round=round_number,
                    )
                    rng_unchanged &= global_rng_state_equal(
                        rng_before, capture_global_rng_state()
                    )
                else:
                    trace = baseline_traces[task_id]
                net_local = copy.deepcopy(task_model).to(args.device)
                local = LocalUpdate_FedAvg(
                    args=args, dataset=dataset_train,
                    idxs=dict_users[client], dataset_test=dataset_test,
                )
                try:
                    returned = local.train(net=net_local)
                except Exception:
                    failed_clients += 1
                    raise
                w_locals.append(copy.deepcopy(returned))
                lens.append(len(dict_users[client]))
                if is_baseline:
                    rng_before = capture_global_rng_state()
                    controller.observe(
                        client_id=client, current_round=round_number,
                        actual_dispatch_state=task_model.state_dict(),
                        returned_state=returned, final_trace=trace,
                    )
                    rng_unchanged &= global_rng_state_equal(
                        rng_before, capture_global_rng_state()
                    )
                traces.append(trace)
                trace_handle.write(json.dumps({
                    "round": round_number, "task_id": task_id,
                    "client_id": client, "trace": trace,
                }, separators=(",", ":")) + "\n")
                del local, net_local, returned
            if not rng_unchanged:
                raise AssertionError("history observer/controller changed global RNG")
            if is_baseline:
                bank_writer.write_round(round_number, selected, controller.history)
            w_glob = Aggregation(w_locals, lens)
            net_glob.load_state_dict(w_glob)
            round_train_seconds = time.perf_counter() - round_started
            accuracy = evaluate_round_accuracy(
                net_glob, dataset_test, args, round_number
            )
            accuracies.append(float(accuracy))
            total_slots = sum(int(trace["num_reset_kernels"]) for trace in traces)
            targeted_slots = sum(int(trace.get("targeted_reset_slots", 0)) for trace in traces)
            replaced_slots = sum(int(trace.get("replaced_reset_slots", 0)) for trace in traces)
            fallback_clients = sum(
                bool(trace.get("matching_fallback"))
                and trace["matching_fallback"] != "before_start_round"
                for trace in traces
            ) if not is_baseline else 0
            fallback_layers = sum(
                layer["fallback_reason"] in {"no_valid_history", "no_exact_age_peer"}
                for trace in traces for layer in trace.get("layers", [])
            ) if not is_baseline else 0
            writer.writerow({
                "round": round_number,
                "algorithm": args.algorithm,
                "seed": args.seed,
                "test_accuracy": accuracy,
                "round_train_seconds": round_train_seconds,
                "selected_clients": json.dumps(selected),
                "task_seeds": json.dumps(task_seeds),
                "global_state_sha256": _state_sha256(net_glob.state_dict()),
                "total_reset_slots": total_slots,
                "targeted_reset_slots": targeted_slots,
                "replaced_reset_slots": replaced_slots,
                "targeted_clients": sum(int(trace.get("targeted_reset_slots", 0) > 0) for trace in traces),
                "fallback_clients": fallback_clients,
                "fallback_client_layers": fallback_layers,
                "history_memory_bytes": _history_bytes(bank_history),
                "source_support_sha256": support_hash,
                "rng_unchanged": rng_unchanged,
                "failed_client_updates": failed_clients,
            })
            round_handle.flush()
            trace_handle.flush()
            del task_models, w_locals, w_glob
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
    print_peak_accuracy(accuracies, args.algorithm)
    totals = Counter()
    with round_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("total_reset_slots", "targeted_reset_slots", "replaced_reset_slots",
                        "targeted_clients", "fallback_clients", "fallback_client_layers"):
                totals[key] += int(row[key])
    _write_json(run_dir / "summary.json", {
        "algorithm": args.algorithm,
        "run_name": run_name,
        "completed_rounds": args.epochs,
        "peak_accuracy": max(accuracies),
        "peak_round": accuracies.index(max(accuracies)) + 1,
        "final_accuracy": accuracies[-1],
        "runtime_seconds": time.perf_counter() - started,
        "failed_client_updates": failed_clients,
        "totals": dict(totals),
        "history_bank_policy": config["history_bank_policy"],
        "partition_sha256": config["partition_sha256"],
        "matching_schedule_sha256": config["matching_schedule_sha256"],
    })
