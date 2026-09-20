#!/usr/bin/env python
"""Validation-only proxy search before reporting fixed-budget test peaks.

The controller never reads test accuracy.  It compares confidence thresholds
and score-component variants on validation metrics, then launches one 1200
round run for the strongest new candidate when it beats the existing pilot.
"""

import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "results", "peak_proxy_logs")
METRICS_DIR = os.path.join(ROOT, "results", "peak_proxy_metrics")
MATCHING_DIR = os.path.join(ROOT, "results", "peak_proxy_matching")
REPORT = os.path.join(ROOT, "results", "autonomous_peak_proxy_report.json")
EXISTING_SUMMARY = os.path.join(
    ROOT, "results", "gated_beta03_metrics",
    "cifar10_resnet18_FedPhoenixRecovery_seed1_"
    "gated_warmup50_gain06_seed1_200_summary.json",
)


CANDIDATES = [
    {"tag": "gain04", "min_gain": 0.4, "gain": 1.0, "adv": 1.0, "res": 0.5, "align": 0.1},
    {"tag": "gain05", "min_gain": 0.5, "gain": 1.0, "adv": 1.0, "res": 0.5, "align": 0.1},
    {"tag": "gain07", "min_gain": 0.7, "gain": 1.0, "adv": 1.0, "res": 0.5, "align": 0.1},
    {"tag": "noalign", "min_gain": 0.6, "gain": 1.0, "adv": 1.0, "res": 0.5, "align": 0.0},
    {"tag": "gainonly", "min_gain": 0.6, "gain": 1.0, "adv": 0.0, "res": 0.5, "align": 0.0},
    {"tag": "noresidual", "min_gain": 0.6, "gain": 1.0, "adv": 1.0, "res": 0.0, "align": 0.1},
]


def run_name(candidate, epochs):
    return f"peakproxy_{candidate['tag']}_warmup50_seed1_{epochs}"


def summary_path(candidate, epochs):
    return os.path.join(
        METRICS_DIR,
        f"cifar10_resnet18_FedPhoenixRecovery_seed1_"
        f"{run_name(candidate, epochs)}_summary.json",
    )


def command(candidate, epochs):
    name = run_name(candidate, epochs)
    return [
        sys.executable, "-u", "-X", "utf8", "main_fed.py",
        "--dataset", "cifar10", "--model", "resnet18",
        "--algorithm", "FedPhoenixRecovery", "--recovery_mode", "assignment_only",
        "--epochs", str(epochs), "--num_users", "100", "--frac", "0.1",
        "--local_ep", "5", "--local_bs", "50", "--bs", "256",
        "--lr", "0.01", "--momentum", "0.5", "--weight_decay", "0",
        "--iid", "0", "--noniid_case", "5", "--data_beta", "0.3",
        "--FP_conv", "1000", "--FP_fc", "0", "--reset", "0.015625",
        "--remethod", "ori_normal", "--seed", "1", "--generate_data", "0",
        "--validation_samples", "1000", "--eval_every", "5",
        "--recovery_probe_samples", "64", "--recovery_probe_steps", "1",
        "--recovery_start_round", "50", "--recovery_end_round", "-1",
        "--recovery_min_assignment_gain", str(candidate["min_gain"]),
        "--recovery_gain_weight", str(candidate["gain"]),
        "--recovery_advantage_weight", str(candidate["adv"]),
        "--recovery_residual_weight", str(candidate["res"]),
        "--recovery_alignment_weight", str(candidate["align"]),
        "--recovery_assignment_weighting", "uniform",
        "--recovery_assignment_inertia", "0",
        "--metrics_log_dir", os.path.relpath(METRICS_DIR, ROOT),
        "--recovery_log_dir", os.path.relpath(MATCHING_DIR, ROOT),
        "--run_name", name, "--gpu", "0",
    ]


def validation_metrics(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "best_validation_accuracy": data["best_validation_accuracy"],
        "best_validation_loss": data["best_validation_loss"],
        "best_validation_round": data["best_validation_round"],
    }


def train(candidate, epochs):
    name = run_name(candidate, epochs)
    summary = summary_path(candidate, epochs)
    if not os.path.exists(summary):
        with open(os.path.join(LOG_DIR, f"{name}.stdout.log"), "w", encoding="utf-8") as out, open(
            os.path.join(LOG_DIR, f"{name}.stderr.log"), "w", encoding="utf-8"
        ) as err:
            result = subprocess.run(command(candidate, epochs), cwd=ROOT, stdout=out, stderr=err)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {err.name}")
    return {
        "name": name,
        "config": candidate,
        **validation_metrics(summary),
        "summary": summary,
    }


def rank_key(item):
    return item["best_validation_accuracy"], -item["best_validation_loss"]


def write(status, pilots, selected=None, full=None):
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": status,
                "selection_rule": "validation accuracy, then validation loss; test metrics are not read",
                "primary_reporting_metric_after_selection": "peak test accuracy within 1200 fixed rounds",
                "pilots": pilots,
                "selected": selected,
                "full": full,
            }, handle, ensure_ascii=False, indent=2,
        )


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MATCHING_DIR, exist_ok=True)
    existing = {
        "name": "existing_gain06",
        "config": None,
        **validation_metrics(EXISTING_SUMMARY),
        "summary": EXISTING_SUMMARY,
    }
    pilots = [existing]
    write("pilot_search", pilots)
    for candidate in CANDIDATES:
        pilots.append(train(candidate, 200))
        write("pilot_search", pilots)
    selected = max(pilots, key=rank_key)
    if selected["config"] is None:
        write("complete_existing_remains_best", pilots, selected=selected)
        return
    write("full_run", pilots, selected=selected)
    full = train(selected["config"], 1200)
    write("complete", pilots, selected=selected, full=full)


if __name__ == "__main__":
    main()
