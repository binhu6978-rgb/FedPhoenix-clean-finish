#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end FedAvg degeneration regression for InteractionDispatch."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main_fed  # noqa: E402
from models.Nets import CNNCifar  # noqa: E402
from utils.get_dataset import get_dataset  # noqa: E402
from utils.set_seed import set_random_seed  # noqa: E402


ROUND_PATTERN = re.compile(
    r"ROUND_ACCURACY\s+method=\S+\s+round=(\d+)\s+accuracy=([0-9.eE+-]+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="results/interaction_dispatch_degeneration_regression",
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def build_training_args(output_dir, gpu):
    return SimpleNamespace(
        dataset="cifar10",
        model="cnn",
        num_users=100,
        frac=0.1,
        local_ep=5,
        local_bs=50,
        bs=256,
        optimizer="sgd",
        lr=0.01,
        momentum=0.5,
        weight_decay=0.0,
        iid=0,
        noniid_case=5,
        data_beta=0.3,
        generate_data=0,
        num_classes=10,
        num_channels=3,
        seed=1,
        gpu=gpu,
        num_workers=0,
        verbose=False,
        epochs=5,
        id_tau=0.0,
        id_max_step_ratio=0.0,
        metrics_log_dir=str(output_dir / "metrics"),
        run_name="degeneration",
        algorithm="FedAvg",
        device=torch.device(f"cuda:{gpu}"),
    )


@contextlib.contextmanager
def record_client_selections():
    records = []
    original_choice = main_fed.np.random.choice

    def recording_choice(*args, **kwargs):
        selected = original_choice(*args, **kwargs)
        records.append([int(item) for item in selected.tolist()])
        return selected

    main_fed.np.random.choice = recording_choice
    try:
        yield records
    finally:
        main_fed.np.random.choice = original_choice


def run_algorithm(name, model, args, dataset_train, dataset_test, dict_users):
    args.algorithm = name
    main_fed.args = args
    set_random_seed(args.seed)
    stream = io.StringIO()
    with record_client_selections() as selections:
        with contextlib.redirect_stdout(stream):
            if name == "FedAvg":
                main_fed.FedAvg(model, dataset_train, dataset_test, dict_users)
            else:
                main_fed.InteractionDispatch(
                    model,
                    dataset_train,
                    None,
                    dataset_test,
                    dict_users,
                )
    text = stream.getvalue()
    accuracies = [float(match.group(2)) for match in ROUND_PATTERN.finditer(text)]
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    return state, accuracies, selections, text


def compare_states(left, right):
    if left.keys() != right.keys():
        raise AssertionError("Final state_dict keys differ")
    max_absolute_difference = 0.0
    failing_keys = []
    for name in left:
        if torch.is_floating_point(left[name]):
            difference = float((left[name] - right[name]).abs().max().item())
            max_absolute_difference = max(max_absolute_difference, difference)
            if not torch.allclose(left[name], right[name], rtol=1e-6, atol=1e-7):
                failing_keys.append(name)
        elif not torch.equal(left[name], right[name]):
            failing_keys.append(name)
    return max_absolute_difference, failing_keys


def read_active_counts(metrics_dir):
    csv_paths = sorted(metrics_dir.glob("*InteractionDispatch*.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(
            f"Expected one InteractionDispatch metrics CSV, found {csv_paths}"
        )
    with csv_paths[0].open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["active_interventions"]) for row in rows], csv_paths[0]


def repository_relative(path):
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def main():
    cli = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_training_args(output_dir, cli.gpu)
    torch.cuda.set_device(cli.gpu)

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    initial_model = CNNCifar(args)
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model

    fedavg_model = CNNCifar(args).to(args.device)
    fedavg_model.load_state_dict(initial_state)
    fedavg_state, fedavg_accuracies, fedavg_selections, fedavg_stdout = run_algorithm(
        "FedAvg", fedavg_model, args, dataset_train, dataset_test, dict_users
    )
    (output_dir / "fedavg.stdout.log").write_text(
        fedavg_stdout, encoding="utf-8"
    )
    del fedavg_model
    torch.cuda.empty_cache()

    interaction_model = CNNCifar(args).to(args.device)
    interaction_model.load_state_dict(initial_state)
    (
        interaction_state,
        interaction_accuracies,
        interaction_selections,
        interaction_stdout,
    ) = run_algorithm(
        "InteractionDispatch",
        interaction_model,
        args,
        dataset_train,
        dataset_test,
        dict_users,
    )
    (output_dir / "interaction_dispatch.stdout.log").write_text(
        interaction_stdout, encoding="utf-8"
    )
    del interaction_model
    torch.cuda.empty_cache()

    max_difference, failing_keys = compare_states(
        fedavg_state, interaction_state
    )
    active_counts, metrics_path = read_active_counts(
        Path(args.metrics_log_dir)
    )
    accuracy_differences = [
        abs(left - right)
        for left, right in zip(fedavg_accuracies, interaction_accuracies)
    ]
    checks = {
        "five_rounds_recorded": (
            len(fedavg_accuracies) == len(interaction_accuracies) == 5
        ),
        "client_selections_identical": (
            fedavg_selections == interaction_selections
            and len(fedavg_selections) == 5
        ),
        "active_interventions_all_zero": (
            len(active_counts) == 5 and all(value == 0 for value in active_counts)
        ),
        "accuracies_identical": (
            len(accuracy_differences) == 5
            and max(accuracy_differences, default=float("inf")) <= 1e-9
        ),
        "final_states_allclose": not failing_keys,
    }
    protocol = dict(vars(args))
    protocol["device"] = str(args.device)
    protocol["metrics_log_dir"] = repository_relative(args.metrics_log_dir)
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol": protocol,
        "checks": checks,
        "fedavg_accuracies": fedavg_accuracies,
        "interaction_dispatch_accuracies": interaction_accuracies,
        "maximum_accuracy_difference": max(
            accuracy_differences, default=None
        ),
        "maximum_final_state_absolute_difference": max_difference,
        "non_allclose_state_keys": failing_keys,
        "fedavg_client_selections": fedavg_selections,
        "interaction_dispatch_client_selections": interaction_selections,
        "active_interventions_per_round": active_counts,
        "interaction_metrics_csv": repository_relative(metrics_path),
    }
    report_path = output_dir / "degeneration_regression.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
