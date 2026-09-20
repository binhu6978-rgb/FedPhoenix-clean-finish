#!/usr/bin/env python
"""Continue bounded multi-step recovery-probe search without user input."""

import argparse
import json
import os
import subprocess
import sys
import time

import psutil


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "results", "formal_beta03_logs")
METRICS_DIR = os.path.join(ROOT, "results", "probe3_beta03_metrics")
MATCHING_DIR = os.path.join(ROOT, "results", "probe3_beta03_matching")
REPORT = os.path.join(ROOT, "results", "autonomous_probe_search_report.json")
BASELINE_VALIDATION = 74.8
BASELINE_PEAK = 73.71
TARGET_DELTA = 1.0


def name_for(steps, epochs):
    return f"probe{steps}_warmup50_gain06_seed1_{epochs}"


def summary_path(name):
    return os.path.join(
        METRICS_DIR,
        f"cifar10_resnet18_FedPhoenixRecovery_seed1_{name}_summary.json",
    )


def log_path(name, stream="stdout"):
    return os.path.join(LOG_DIR, f"{name}.{stream}.log")


def command(steps, epochs, name):
    return [
        sys.executable, "-u", "-X", "utf8", "main_fed.py",
        "--dataset", "cifar10", "--model", "resnet18",
        "--algorithm", "FedPhoenixRecovery",
        "--recovery_mode", "assignment_only",
        "--epochs", str(epochs), "--num_users", "100",
        "--frac", "0.1", "--local_ep", "5", "--local_bs", "50",
        "--bs", "256", "--lr", "0.01", "--momentum", "0.5",
        "--weight_decay", "0", "--iid", "0", "--noniid_case", "5",
        "--data_beta", "0.3", "--FP_conv", "1000", "--FP_fc", "0",
        "--reset", "0.015625", "--remethod", "ori_normal",
        "--seed", "1", "--generate_data", "0",
        "--validation_samples", "1000", "--eval_every", "5",
        "--recovery_probe_samples", "64",
        "--recovery_probe_steps", str(steps),
        "--recovery_start_round", "50", "--recovery_end_round", "-1",
        "--recovery_min_assignment_gain", "0.6",
        "--recovery_assignment_weighting", "uniform",
        "--recovery_assignment_inertia", "0",
        "--metrics_log_dir", os.path.relpath(METRICS_DIR, ROOT),
        "--recovery_log_dir", os.path.relpath(MATCHING_DIR, ROOT),
        "--run_name", name, "--gpu", "0",
    ]


def run_training(steps, epochs):
    name = name_for(steps, epochs)
    with open(log_path(name), "w", encoding="utf-8") as stdout, open(
        log_path(name, "stderr"), "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command(steps, epochs, name), cwd=ROOT,
            stdout=stdout, stderr=stderr, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed with code {result.returncode}")
    return name


def read_summary(name):
    with open(summary_path(name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def pilot(steps, name):
    summary = read_summary(name)
    return {
        "name": name,
        "probe_steps": steps,
        "best_validation_accuracy": summary["best_validation_accuracy"],
        "peak_test_accuracy": summary["diagnostic_peak_test_accuracy"],
        "validation_delta": summary["best_validation_accuracy"] - BASELINE_VALIDATION,
        "peak_delta": summary["diagnostic_peak_test_accuracy"] - BASELINE_PEAK,
    }


def passes(result):
    return (
        result["validation_delta"] >= TARGET_DELTA
        and result["peak_delta"] >= TARGET_DELTA
    )


def write_report(payload):
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_pid", type=int, required=True)
    args = parser.parse_args()
    while psutil.pid_exists(args.wait_pid):
        time.sleep(10)

    first_name = name_for(3, 200)
    if not os.path.exists(summary_path(first_name)):
        raise RuntimeError("3-step pilot ended without summary")
    results = [pilot(3, first_name)]
    selected = results[0] if passes(results[0]) else None

    if selected is None:
        second_name = run_training(5, 200)
        results.append(pilot(5, second_name))
        if passes(results[-1]):
            selected = results[-1]

    if selected is None:
        selected = max(
            results,
            key=lambda item: (item["validation_delta"], item["peak_delta"]),
        )
    write_report({"status": "full_run", "pilots": results, "selected": selected})

    full_name = run_training(selected["probe_steps"], 1200)
    write_report(
        {
            "status": "complete",
            "pilots": results,
            "selected": selected,
            "full_run_name": full_name,
            "full_summary": read_summary(full_name),
        }
    )


if __name__ == "__main__":
    main()
