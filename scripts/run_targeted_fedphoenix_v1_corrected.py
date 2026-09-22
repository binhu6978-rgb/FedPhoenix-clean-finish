#!/usr/bin/env python
"""Correctness-gated corrected rho=0.25 TargetedFedPhoenix run."""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_FEDPHOENIX = {
    "name": "FedPhoenix",
    "peak_accuracy": 78.35,
    "peak_round": 218,
    "final_accuracy": 73.93,
    "last20_mean": 69.665,
}
BIASED_V1 = {
    "name": "biased_TargetedFedPhoenix_rho0.25",
    "peak_accuracy": 78.27,
    "peak_round": 236,
    "final_accuracy": 72.30,
    "last20_mean": 72.243,
}
RUN_ID = "target0p25_r300_seed1"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_args():
    return [
        "--dataset", "cifar10", "--model", "vgg", "--epochs", "300",
        "--num_users", "100", "--frac", "0.1", "--local_ep", "5",
        "--local_bs", "50", "--bs", "256", "--optimizer", "sgd",
        "--lr", "0.01", "--momentum", "0.5", "--weight_decay", "0",
        "--iid", "0", "--noniid_case", "5", "--data_beta", "0.3",
        "--generate_data", "0", "--num_classes", "10", "--num_channels", "3",
        "--seed", "1", "--num_workers", "0", "--FP_conv", "1000",
        "--reset", "0.015625", "--remethod", "ori_normal",
        "--tfp_target_ratio", "0.25", "--tfp_max_history_gap", "20",
    ]


def run_logged(command, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        return subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)


def correctness_gates(python, output_dir, gpu):
    logs = output_dir / "logs"
    regression = run_logged(
        [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"],
        logs / "regression.stdout.log",
        logs / "regression.stderr.log",
    )
    if regression.returncode != 0:
        raise RuntimeError("full regression tests failed")

    parity = run_logged(
        [
            str(python),
            str(ROOT / "experiments" / "check_targeted_fedphoenix_parity.py"),
            "--output_dir", str(output_dir),
            "--python", str(python),
            "--gpu", str(gpu),
        ],
        logs / "parity.stdout.log",
        logs / "parity.stderr.log",
    )
    if parity.returncode != 0:
        raise RuntimeError("rho=0 strict parity failed")
    parity_report = read_json(output_dir / "parity_report.json")
    if not parity_report["passed"]:
        raise RuntimeError("rho=0 strict parity report did not pass")

    active_smoke = run_logged(
        [
            str(python),
            str(
                ROOT
                / "experiments"
                / "check_targeted_fedphoenix_active_determinism.py"
            ),
            "--output", str(output_dir / "active_smoke_report.json"),
        ],
        logs / "active_smoke.stdout.log",
        logs / "active_smoke.stderr.log",
    )
    if active_smoke.returncode != 0:
        raise RuntimeError("active deterministic smoke failed")
    smoke_report = read_json(output_dir / "active_smoke_report.json")
    if not smoke_report["passed"]:
        raise RuntimeError("active deterministic smoke report did not pass")
    return {
        "full_regression_tests_passed": True,
        "parity": parity_report,
        "active_deterministic_smoke": smoke_report,
    }


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def build_index_bias_report(trace_path):
    by_layer = defaultdict(
        lambda: {"donor": [], "kept": [], "events": 0, "prefix_events": 0}
    )
    total_events = 0
    total_prefix = 0
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)["trace"]
            for layer in trace.get("layers", []):
                if int(layer["replaced_reset_slots"]) <= 0:
                    continue
                output_kernels = int(layer["num_kernels"])
                denominator = max(output_kernels - 1, 1)
                record = by_layer[layer["name"]]
                record["events"] += 1
                total_events += 1
                record["donor"].extend(
                    int(index) / denominator
                    for index in layer["displaced_donor_indices"]
                )
                record["kept"].extend(
                    int(index) / denominator
                    for index in layer["random_kept_indices"]
                )
                smallest = sorted(int(index) for index in layer["random_candidates"])[
                    : len(layer["random_kept_indices"])
                ]
                is_prefix = sorted(layer["random_kept_indices"]) == smallest
                record["prefix_events"] += int(is_prefix)
                total_prefix += int(is_prefix)
    layers = []
    all_donors = []
    all_kept = []
    for layer_name, values in sorted(by_layer.items()):
        all_donors.extend(values["donor"])
        all_kept.extend(values["kept"])
        layers.append(
            {
                "layer": layer_name,
                "replacement_events": int(values["events"]),
                "displaced_donor_normalized_index": distribution(values["donor"]),
                "random_kept_normalized_index": distribution(values["kept"]),
                "prefix_retention_events": int(values["prefix_events"]),
                "prefix_retention_fraction": float(
                    values["prefix_events"] / values["events"]
                ),
            }
        )
    return {
        "retention_policy": "sha256_private_priority",
        "replacement_events": int(total_events),
        "prefix_retention_events": int(total_prefix),
        "prefix_retention_fraction": (
            float(total_prefix / total_events) if total_events else None
        ),
        "overall_displaced_donor_normalized_index": distribution(all_donors),
        "overall_random_kept_normalized_index": distribution(all_kept),
        "layers": layers,
    }


def accuracy_comparison(summary, reference):
    return {
        "reference": reference,
        "peak_delta": float(summary["peak_accuracy"] - reference["peak_accuracy"]),
        "peak_round_delta": int(summary["peak_round"] - reference["peak_round"]),
        "final_delta": float(summary["final_accuracy"] - reference["final_accuracy"]),
        "last20_delta": float(summary["last20_mean"] - reference["last20_mean"]),
    }


def write_outputs(output_dir, summary, correctness, bias_report, launcher_runtime, attempts):
    vs_fedphoenix = accuracy_comparison(summary, ORIGINAL_FEDPHOENIX)
    vs_biased = accuracy_comparison(summary, BIASED_V1)
    conclusion = (
        "late stability improvement persists after removing index-selection bias"
        if summary["last20_mean"] > ORIGINAL_FEDPHOENIX["last20_mean"]
        else "previous late stability improvement was materially confounded by the bias"
    )
    comparison = {
        "correctness": correctness,
        "original_fedphoenix": ORIGINAL_FEDPHOENIX,
        "biased_targeted_v1_rho0.25": BIASED_V1,
        "corrected_targeted_v1_rho0.25": summary,
        "delta_vs_original_fedphoenix": vs_fedphoenix,
        "delta_vs_biased_v1": vs_biased,
        "index_bias_report": bias_report,
        "launcher_runtime_seconds": float(launcher_runtime),
        "attempts": attempts,
        "strict_conclusion": conclusion,
    }
    write_json(output_dir / "comparison.json", comparison)
    write_json(output_dir / "index_bias_report.json", bias_report)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        row = {
            "ratio": 0.25,
            "peak_accuracy": summary["peak_accuracy"],
            "peak_round": summary["peak_round"],
            "final_accuracy": summary["final_accuracy"],
            "last20_mean": summary["last20_mean"],
            "peak_delta_vs_fedphoenix": vs_fedphoenix["peak_delta"],
            "last20_delta_vs_fedphoenix": vs_fedphoenix["last20_delta"],
            "peak_delta_vs_biased_v1": vs_biased["peak_delta"],
            "last20_delta_vs_biased_v1": vs_biased["last20_delta"],
            "actual_target_fraction": summary["average_actual_target_fraction"],
            "baseline_random_overlap_rate": summary["baseline_random_overlap_rate"],
            "replaced_reset_slots": summary["actual_replaced_reset_slots"],
            "prefix_retention_fraction": bias_report["prefix_retention_fraction"],
            "runtime_seconds": summary["runtime_seconds"],
        }
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    stats = summary["client_statistics"]
    donor = bias_report["overall_displaced_donor_normalized_index"]
    kept = bias_report["overall_random_kept_normalized_index"]
    lines = [
        "# Corrected TargetedFedPhoenix V1 rho=0.25", "",
        f"- Full regression tests passed: {correctness['full_regression_tests_passed']}",
        f"- rho=0 exact parity passed: {correctness['parity']['passed']}",
        f"- Active deterministic smoke passed: {correctness['active_deterministic_smoke']['passed']}",
        "", "## Accuracy", "",
        "| Method | Peak | Peak round | Final | Last20 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| FedPhoenix | {ORIGINAL_FEDPHOENIX['peak_accuracy']:.4f} | 218 | {ORIGINAL_FEDPHOENIX['final_accuracy']:.4f} | {ORIGINAL_FEDPHOENIX['last20_mean']:.4f} |",
        f"| Biased V1 rho=.25 | {BIASED_V1['peak_accuracy']:.4f} | 236 | {BIASED_V1['final_accuracy']:.4f} | {BIASED_V1['last20_mean']:.4f} |",
        f"| Corrected V1 rho=.25 | {summary['peak_accuracy']:.4f} | {summary['peak_round']} | {summary['final_accuracy']:.4f} | {summary['last20_mean']:.4f} |",
        "", "## Mechanism", "",
        f"- Targeted/random/replaced slots: {stats['targeted_reset_slots']}/{stats['random_reset_slots']}/{stats['replaced_reset_slots']}",
        f"- Actual target fraction: {summary['average_actual_target_fraction']:.6f}",
        f"- Cold/stale/no-valid: {stats['clients_cold_start']}/{stats['clients_stale']}/{stats['clients_no_valid_history']}",
        f"- Baseline-random overlap: {summary['baseline_random_overlap_rate']:.6f}",
        f"- Maximum history memory: {summary['maximum_history_memory_bytes']} bytes", "",
        "## Index-bias sanity", "",
        f"- Prefix-retention fraction: {bias_report['prefix_retention_fraction']:.6f}",
        f"- Donor normalized index mean/median/q25/q75: {donor['mean']:.6f}/{donor['median']:.6f}/{donor['q25']:.6f}/{donor['q75']:.6f}",
        f"- Kept normalized index mean/median/q25/q75: {kept['mean']:.6f}/{kept['median']:.6f}/{kept['q25']:.6f}/{kept['q75']:.6f}",
        "", "## Strict conclusion", "", conclusion + ".", "",
    ]
    (output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="results/targeted_fedphoenix_v1_corrected"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    correctness = correctness_gates(Path(args.python), output_dir, args.gpu)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": 1,
        "created_at": now(),
        "updated_at": now(),
        "status": "running",
        "protocol": "corrected TargetedFedPhoenix rho=0.25 VGG CIFAR10 seed=1 300 rounds",
        "correctness": correctness,
        "attempts": [],
    }
    write_json(manifest_path, manifest)

    run_dir = output_dir / "run" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(args.python)), str(ROOT / "main_fed.py"),
        "--algorithm", "TargetedFedPhoenix", "--gpu", str(args.gpu),
        "--run_name", "corrected_" + RUN_ID,
        "--metrics_log_dir", str(run_dir),
        *protocol_args(),
    ]
    completed = False
    for attempt in range(1, int(args.max_retries) + 2):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        attempt_started = time.perf_counter()
        stdout_path = output_dir / "logs" / f"{RUN_ID}.attempt{attempt}.stdout.log"
        stderr_path = output_dir / "logs" / f"{RUN_ID}.attempt{attempt}.stderr.log"
        manifest["active_attempt"] = attempt
        manifest["active_command"] = command
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        result = run_logged(command, stdout_path, stderr_path)
        attempt_row = {
            "attempt": attempt,
            "return_code": result.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "runtime_seconds": time.perf_counter() - attempt_started,
            "finished_at": now(),
        }
        manifest["attempts"].append(attempt_row)
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        if result.returncode == 0 and summary_path.exists():
            completed = True
            break
    if not completed:
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise RuntimeError("corrected rho=0.25 failed after retries")

    summary = read_json(run_dir / "summary.json")
    bias_report = build_index_bias_report(run_dir / "task_traces.jsonl")
    comparison = write_outputs(
        output_dir,
        summary,
        correctness,
        bias_report,
        time.perf_counter() - started,
        manifest["attempts"],
    )
    manifest["status"] = "completed"
    manifest["comparison"] = comparison
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
