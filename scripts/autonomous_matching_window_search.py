#!/usr/bin/env python
"""Compare recovery matching windows using validation metrics only."""

import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "results", "window_beta03_logs")
METRICS_DIR = os.path.join(ROOT, "results", "window_beta03_metrics")
MATCHING_DIR = os.path.join(ROOT, "results", "window_beta03_matching")
REPORT = os.path.join(ROOT, "results", "autonomous_matching_window_report.json")
EXISTING = os.path.join(
    ROOT,
    "results",
    "inertia_beta03_metrics",
    "cifar10_resnet18_FedPhoenixRecovery_seed1_"
    "inertia100_warmup50_gain02_seed1_1200_summary.json",
)


def name(end_round):
    return f"window_end{end_round}_inertia100_gain02_seed1_1200"


def summary_path(end_round):
    return os.path.join(
        METRICS_DIR,
        f"cifar10_resnet18_FedPhoenixRecovery_seed1_{name(end_round)}_summary.json",
    )


def command(end_round):
    return [
        sys.executable, "-u", "-X", "utf8", "main_fed.py",
        "--dataset", "cifar10", "--model", "resnet18",
        "--algorithm", "FedPhoenixRecovery", "--recovery_mode", "assignment_only",
        "--epochs", "1200", "--num_users", "100", "--frac", "0.1",
        "--local_ep", "5", "--local_bs", "50", "--bs", "256",
        "--lr", "0.01", "--momentum", "0.5", "--weight_decay", "0",
        "--iid", "0", "--noniid_case", "5", "--data_beta", "0.3",
        "--FP_conv", "1000", "--FP_fc", "0", "--reset", "0.015625",
        "--remethod", "ori_normal", "--seed", "1", "--generate_data", "0",
        "--validation_samples", "1000", "--eval_every", "5",
        "--recovery_probe_samples", "64", "--recovery_probe_steps", "1",
        "--recovery_probe_match_local_optimizer", "0",
        "--recovery_start_round", "50", "--recovery_end_round", str(end_round),
        "--recovery_min_assignment_gain", "0.2",
        "--recovery_assignment_weighting", "uniform",
        "--recovery_assignment_inertia", "1.0",
        "--metrics_log_dir", os.path.relpath(METRICS_DIR, ROOT),
        "--recovery_log_dir", os.path.relpath(MATCHING_DIR, ROOT),
        "--run_name", name(end_round), "--gpu", "0",
    ]


def validation_record(label, path):
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {
        "name": label,
        "best_validation_accuracy": metrics["best_validation_accuracy"],
        "best_validation_loss": metrics["best_validation_loss"],
        "best_validation_round": metrics["best_validation_round"],
        "summary": os.path.abspath(path),
    }


def write(status, runs):
    selected = max(
        runs,
        key=lambda item: (item["best_validation_accuracy"], -item["best_validation_loss"]),
    )
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": status,
                "selection_rule": "maximum validation accuracy, then minimum validation loss",
                "test_metrics_used_for_selection": False,
                "runs": runs,
                "selected": selected,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MATCHING_DIR, exist_ok=True)
    runs = [validation_record("existing_end1000", EXISTING)]
    write("running", runs)
    for end_round in (600, 800):
        summary = summary_path(end_round)
        if not os.path.exists(summary):
            with open(os.path.join(LOG_DIR, f"{name(end_round)}.stdout.log"), "w", encoding="utf-8") as out, open(
                os.path.join(LOG_DIR, f"{name(end_round)}.stderr.log"), "w", encoding="utf-8"
            ) as err:
                completed = subprocess.run(command(end_round), cwd=ROOT, stdout=out, stderr=err)
            if completed.returncode != 0:
                raise RuntimeError(f"window end {end_round} failed; inspect {err.name}")
        runs.append(validation_record(name(end_round), summary))
        write("running", runs)
    write("complete", runs)


if __name__ == "__main__":
    main()
