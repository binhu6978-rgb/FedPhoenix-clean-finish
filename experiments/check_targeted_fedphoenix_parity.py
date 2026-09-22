#!/usr/bin/env python
"""Strict ten-round rho=0 parity gate for TargetedFedPhoenix."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCURACY_PATTERN = re.compile(r"ROUND_ACCURACY.*accuracy=([0-9.]+)")


def protocol_args():
    return [
        "--dataset", "cifar10", "--model", "vgg", "--epochs", "10",
        "--num_users", "100", "--frac", "0.1", "--local_ep", "5",
        "--local_bs", "50", "--bs", "256", "--optimizer", "sgd",
        "--lr", "0.01", "--momentum", "0.5", "--weight_decay", "0",
        "--iid", "0", "--noniid_case", "5", "--data_beta", "0.3",
        "--generate_data", "0", "--num_classes", "10", "--num_channels", "3",
        "--seed", "1", "--num_workers", "0", "--FP_conv", "1000",
        "--reset", "0.015625", "--remethod", "ori_normal",
    ]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def reset_masks(task_rows, final=False):
    result = []
    for item in task_rows:
        trace = item["trace"]
        result.append(
            [
                (
                    layer["name"],
                    layer.get("final_reset_indices", layer["reset_indices"])
                    if final
                    else layer["reset_indices"],
                )
                for layer in trace.get("layers", [])
            ]
        )
    return result


def run_one(python, parity_dir, algorithm, gpu):
    run_dir = parity_dir / algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    selection = run_dir / "selection.jsonl"
    hashes = run_dir / "hashes.jsonl"
    tasks = run_dir / "baseline_tasks.jsonl"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    metrics = run_dir / "metrics"
    command = [
        str(python), str(ROOT / "scripts" / "run_main_with_selection_trace.py"),
        "--selection_trace", str(selection),
        "--global_hash_trace", str(hashes),
        "--fedphoenix_task_trace", str(tasks),
        "--", "--algorithm", algorithm, "--gpu", str(gpu),
        "--run_name", f"targeted_parity_{algorithm}",
        "--metrics_log_dir", str(metrics),
        *protocol_args(),
    ]
    if algorithm == "TargetedFedPhoenix":
        command.extend(
            ["--tfp_target_ratio", "0", "--tfp_max_history_gap", "20"]
        )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{algorithm} parity failed; inspect {stderr_path}")
    output = stdout_path.read_text(encoding="utf-8")
    values = [float(match.group(1)) for match in ACCURACY_PATTERN.finditer(output)]
    return {
        "selection": read_jsonl(selection),
        "hashes": read_jsonl(hashes),
        "baseline_tasks": read_jsonl(tasks),
        "accuracies": values,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/targeted_fedphoenix_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    parity_dir = output_dir / "parity"
    if parity_dir.exists():
        shutil.rmtree(parity_dir)
    parity_dir.mkdir(parents=True)

    baseline = run_one(Path(args.python), parity_dir, "FedPhoenix", args.gpu)
    targeted = run_one(
        Path(args.python), parity_dir, "TargetedFedPhoenix", args.gpu
    )
    enriched_tasks = read_jsonl(targeted["metrics"] / "task_traces.jsonl")
    targeted_rounds = list(
        csv.DictReader(
            (targeted["metrics"] / "round_metrics.csv").open(encoding="utf-8")
        )
    )

    selection_exact = baseline["selection"] == targeted["selection"]
    baseline_tasks_exact = baseline["baseline_tasks"] == targeted["baseline_tasks"]
    task_seeds_exact = [
        item["trace"]["seed"] for item in baseline["baseline_tasks"]
    ] == [item["trace"]["seed"] for item in targeted["baseline_tasks"]]
    baseline_masks_exact = reset_masks(baseline["baseline_tasks"]) == reset_masks(
        targeted["baseline_tasks"]
    )
    final_masks_exact = reset_masks(baseline["baseline_tasks"]) == reset_masks(
        enriched_tasks, final=True
    )
    accuracy_differences = [
        abs(left - right)
        for left, right in zip(baseline["accuracies"], targeted["accuracies"])
    ]
    accuracy_exact = (
        len(baseline["accuracies"]) == len(targeted["accuracies"]) == 10
        and max(accuracy_differences, default=float("inf")) < 1e-6
    )
    hashes_exact = baseline["hashes"] == targeted["hashes"]
    own_hashes = [row["global_state_sha256"] for row in targeted_rounds]
    baseline_hashes = [row["global_state_sha256"] for row in baseline["hashes"]]
    own_hashes_exact = own_hashes == baseline_hashes
    final_hash_exact = bool(hashes_exact and baseline_hashes)
    report = {
        "rounds": 10,
        "target_ratio": 0.0,
        "selected_clients_exact": selection_exact,
        "task_count": len(baseline["baseline_tasks"]),
        "task_seeds_exact": task_seeds_exact,
        "baseline_reset_masks_exact": baseline_masks_exact,
        "final_reset_masks_exact": final_masks_exact,
        "baseline_task_traces_exact": baseline_tasks_exact,
        "accuracy_exact_within_1e-6": accuracy_exact,
        "maximum_accuracy_difference": max(accuracy_differences, default=None),
        "all_round_hashes_exact": hashes_exact,
        "targeted_metrics_hashes_exact": own_hashes_exact,
        "final_global_sha256_exact": final_hash_exact,
        "baseline_final_sha256": baseline_hashes[-1] if baseline_hashes else None,
        "targeted_final_sha256": own_hashes[-1] if own_hashes else None,
    }
    report["passed"] = all(
        value
        for key, value in report.items()
        if key.endswith("_exact") or key == "accuracy_exact_within_1e-6"
    )
    (output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("TargetedFedPhoenix rho=0 parity failed")


if __name__ == "__main__":
    main()
