#!/usr/bin/env python
"""Validation-first search for optimizer-aligned recovery probes.

Only validation accuracy/loss chooses a candidate.  Test metrics are generated
by the training loop for protocol consistency but are never read here.
"""

import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "results", "optimizer_matched_logs")
METRICS_DIR = os.path.join(ROOT, "results", "optimizer_matched_metrics")
MATCHING_DIR = os.path.join(ROOT, "results", "optimizer_matched_matching")
REPORT_PATH = os.path.join(ROOT, "results", "autonomous_optimizer_matched_report.json")


def run_name(steps, probe_lr, epochs):
    lr_tag = str(probe_lr).replace(".", "p")
    return f"matchedopt_steps{steps}_plr{lr_tag}_warmup50_gain06_seed1_{epochs}"


def summary_path(name):
    return os.path.join(
        METRICS_DIR,
        f"cifar10_resnet18_FedPhoenixRecovery_seed1_{name}_summary.json",
    )


def command(steps, probe_lr, epochs, name):
    return [
        sys.executable, "-u", "-X", "utf8", "main_fed.py",
        "--dataset", "cifar10", "--model", "resnet18",
        "--algorithm", "FedPhoenixRecovery",
        "--recovery_mode", "assignment_only",
        "--epochs", str(epochs), "--num_users", "100", "--frac", "0.1",
        "--local_ep", "5", "--local_bs", "50", "--bs", "256",
        "--lr", "0.01", "--momentum", "0.5", "--weight_decay", "0",
        "--iid", "0", "--noniid_case", "5", "--data_beta", "0.3",
        "--FP_conv", "1000", "--FP_fc", "0", "--reset", "0.015625",
        "--remethod", "ori_normal", "--seed", "1", "--generate_data", "0",
        "--validation_samples", "1000", "--eval_every", "5",
        "--recovery_probe_samples", "64", "--recovery_probe_lr", str(probe_lr),
        "--recovery_probe_steps", str(steps),
        "--recovery_probe_match_local_optimizer", "1",
        "--recovery_start_round", "50", "--recovery_end_round", "-1",
        "--recovery_min_assignment_gain", "0.6",
        "--recovery_assignment_weighting", "uniform",
        "--recovery_assignment_inertia", "0",
        "--metrics_log_dir", os.path.relpath(METRICS_DIR, ROOT),
        "--recovery_log_dir", os.path.relpath(MATCHING_DIR, ROOT),
        "--run_name", name, "--gpu", "0",
    ]


def train(steps, probe_lr, epochs):
    name = run_name(steps, probe_lr, epochs)
    summary = summary_path(name)
    if not os.path.exists(summary):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, f"{name}.stdout.log"), "w", encoding="utf-8") as out, open(
            os.path.join(LOG_DIR, f"{name}.stderr.log"), "w", encoding="utf-8"
        ) as err:
            completed = subprocess.run(command(steps, probe_lr, epochs, name), cwd=ROOT, stdout=out, stderr=err)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed; inspect {err.name}")
    with open(summary, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {
        "name": name,
        "probe_steps": steps,
        "probe_lr": probe_lr,
        "best_validation_accuracy": metrics["best_validation_accuracy"],
        "best_validation_loss": metrics["best_validation_loss"],
        "best_validation_round": metrics["best_validation_round"],
        "summary": summary,
    }


def write_report(status, pilots, selected=None, full=None):
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": status,
                "selection_rule": "maximum validation accuracy, then minimum validation loss",
                "pilots": pilots,
                "selected": selected,
                "full_run": full,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MATCHING_DIR, exist_ok=True)
    candidates = [(1, 0.01), (2, 0.01), (3, 0.005)]
    pilots = []
    for steps, probe_lr in candidates:
        pilot = train(steps, probe_lr, 200)
        pilots.append(pilot)
        write_report("pilot_search", pilots)

    selected = max(
        pilots,
        key=lambda item: (item["best_validation_accuracy"], -item["best_validation_loss"]),
    )
    write_report("full_run", pilots, selected=selected)
    full = train(selected["probe_steps"], selected["probe_lr"], 1200)
    write_report("complete", pilots, selected=selected, full=full)


if __name__ == "__main__":
    main()
