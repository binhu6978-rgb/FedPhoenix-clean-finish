#!/usr/bin/env python
"""Run a bounded validation-first FedPhoenix recovery gate search."""

import argparse
import json
import os
import subprocess
import sys
import time

import psutil


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "results", "formal_beta03_logs")
METRICS_DIR = os.path.join(ROOT, "results", "gated_beta03_metrics")
MATCHING_DIR = os.path.join(ROOT, "results", "gated_beta03_matching")


def run_name(start_round, min_gain, epochs):
    gain_tag = f"{int(round(min_gain * 10)):02d}"
    return f"gated_warmup{start_round}_gain{gain_tag}_seed1_{epochs}"


def summary_path(name):
    return os.path.join(
        METRICS_DIR,
        f"cifar10_resnet18_FedPhoenixRecovery_seed1_{name}_summary.json",
    )


def log_path(name, stream="stdout"):
    return os.path.join(LOG_DIR, f"{name}.{stream}.log")


def training_command(start_round, min_gain, epochs, name):
    return [
        sys.executable,
        "-u",
        "-X",
        "utf8",
        "main_fed.py",
        "--dataset",
        "cifar10",
        "--model",
        "resnet18",
        "--algorithm",
        "FedPhoenixRecovery",
        "--recovery_mode",
        "assignment_only",
        "--epochs",
        str(epochs),
        "--num_users",
        "100",
        "--frac",
        "0.1",
        "--local_ep",
        "5",
        "--local_bs",
        "50",
        "--bs",
        "256",
        "--lr",
        "0.01",
        "--momentum",
        "0.5",
        "--weight_decay",
        "0",
        "--iid",
        "0",
        "--noniid_case",
        "5",
        "--data_beta",
        "0.3",
        "--FP_conv",
        "1000",
        "--FP_fc",
        "0",
        "--reset",
        "0.015625",
        "--remethod",
        "ori_normal",
        "--seed",
        "1",
        "--generate_data",
        "0",
        "--validation_samples",
        "1000",
        "--eval_every",
        "5",
        "--recovery_probe_samples",
        "64",
        "--recovery_start_round",
        str(start_round),
        "--recovery_end_round",
        "-1",
        "--recovery_min_assignment_gain",
        str(min_gain),
        "--metrics_log_dir",
        os.path.relpath(METRICS_DIR, ROOT),
        "--recovery_log_dir",
        os.path.relpath(MATCHING_DIR, ROOT),
        "--run_name",
        name,
        "--gpu",
        "0",
    ]


def run_training(start_round, min_gain, epochs):
    name = run_name(start_round, min_gain, epochs)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path(name), "w", encoding="utf-8") as stdout_handle, open(
        log_path(name, "stderr"), "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            training_command(start_round, min_gain, epochs, name),
            cwd=ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}; "
            f"see {log_path(name, 'stderr')}"
        )
    return name


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def wait_for_existing(pid, expected_summary, timeout_hours=3):
    deadline = time.time() + timeout_hours * 3600
    while psutil.pid_exists(pid) and time.time() < deadline:
        time.sleep(10)
    if psutil.pid_exists(pid):
        raise TimeoutError(f"process {pid} exceeded {timeout_hours} hours")
    if not os.path.exists(expected_summary):
        raise RuntimeError(
            f"process {pid} ended without summary {expected_summary}"
        )


def pilot_result(start_round, min_gain, name, baseline):
    summary = read_json(summary_path(name))
    return {
        "name": name,
        "start_round": start_round,
        "min_gain": min_gain,
        "best_validation_accuracy": summary["best_validation_accuracy"],
        "peak_test_accuracy": summary["diagnostic_peak_test_accuracy"],
        "validation_delta": (
            summary["best_validation_accuracy"]
            - baseline["best_validation_accuracy"]
        ),
        "peak_delta": (
            summary["diagnostic_peak_test_accuracy"]
            - baseline["peak_test_accuracy"]
        ),
        "summary": summary_path(name),
        "stdout": log_path(name),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_pid", type=int, required=True)
    parser.add_argument("--target_delta", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MATCHING_DIR, exist_ok=True)
    report_path = os.path.join(
        ROOT, "results", "autonomous_gate_search_report.json"
    )
    baseline_summary = read_json(
        os.path.join(
            ROOT,
            "results",
            "formal_beta03_first200",
            "first200_summary.json",
        )
    )["runs"]["FedPhoenix"]

    current = (100, 0.6)
    current_name = run_name(*current, 200)
    wait_for_existing(args.wait_pid, summary_path(current_name))

    candidates = [current, (50, 0.6), (100, 0.9), (50, 0.9)]
    results = []
    selected = None
    for candidate_index, (start_round, min_gain) in enumerate(candidates):
        name = run_name(start_round, min_gain, 200)
        if candidate_index > 0:
            run_training(start_round, min_gain, 200)
        result = pilot_result(
            start_round, min_gain, name, baseline_summary
        )
        results.append(result)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "pilot_search",
                    "target_delta": args.target_delta,
                    "baseline": baseline_summary,
                    "pilots": results,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        if (
            result["validation_delta"] >= args.target_delta
            and result["peak_delta"] >= args.target_delta
        ):
            selected = result
            break

    if selected is None:
        selected = max(
            results,
            key=lambda result: (
                result["validation_delta"],
                result["peak_delta"],
            ),
        )

    full_name = run_training(
        selected["start_round"], selected["min_gain"], 1200
    )
    full_summary = read_json(summary_path(full_name))
    comparison_dir = os.path.join(
        ROOT, "results", "autonomous_gated_1200"
    )
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            os.path.join("scripts", "extract_first200_metrics.py"),
            "--run",
            "FedPhoenix=results\\formal_beta03_logs\\fedphoenix_seed1.stdout.log",
            "--run",
            f"RecoveryGated={os.path.relpath(log_path(full_name), ROOT)}",
            "--reference",
            "FedPhoenix",
            "--max_round",
            "1200",
            "--eval_every",
            "5",
            "--output_dir",
            os.path.relpath(comparison_dir, ROOT),
        ],
        cwd=ROOT,
        check=True,
    )
    comparison = read_json(
        os.path.join(comparison_dir, "round1200_summary.json")
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "target_delta": args.target_delta,
                "baseline_200": baseline_summary,
                "pilots": results,
                "selected": selected,
                "full_run_name": full_name,
                "full_summary": full_summary,
                "comparison": comparison,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
