#!/usr/bin/env python
"""Ten-round trajectory parity gate for FedPhoenixHistoryDiagnostic."""

import argparse
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


def run_one(python, output_dir, algorithm, gpu):
    run_dir = output_dir / algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    selection = run_dir / "selection.jsonl"
    hashes = run_dir / "hashes.jsonl"
    tasks = run_dir / "tasks.jsonl"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = [
        str(python), str(ROOT / "scripts" / "run_main_with_selection_trace.py"),
        "--selection_trace", str(selection),
        "--global_hash_trace", str(hashes),
        "--fedphoenix_task_trace", str(tasks),
        "--", "--algorithm", algorithm, "--gpu", str(gpu),
        "--run_name", f"history_parity_{algorithm}",
        "--metrics_log_dir", str(run_dir / "metrics"),
        *protocol_args(),
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"{algorithm} parity run failed; inspect {stderr_path}"
        )
    output = stdout_path.read_text(encoding="utf-8")
    accuracies = [float(match.group(1)) for match in ACCURACY_PATTERN.finditer(output)]
    return {
        "selection": read_jsonl(selection),
        "hashes": read_jsonl(hashes),
        "tasks": read_jsonl(tasks),
        "accuracies": accuracies,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="results/fedphoenix_specialization_diagnostic/parity",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    baseline = run_one(Path(args.python), output_dir, "FedPhoenix", args.gpu)
    diagnostic = run_one(
        Path(args.python), output_dir, "FedPhoenixHistoryDiagnostic", args.gpu
    )
    selection_exact = baseline["selection"] == diagnostic["selection"]
    task_trace_exact = baseline["tasks"] == diagnostic["tasks"]
    task_seeds_exact = [item["trace"]["seed"] for item in baseline["tasks"]] == [
        item["trace"]["seed"] for item in diagnostic["tasks"]
    ]
    reset_masks_exact = [
        [(layer["name"], layer["reset_indices"]) for layer in item["trace"]["layers"]]
        for item in baseline["tasks"]
    ] == [
        [(layer["name"], layer["reset_indices"]) for layer in item["trace"]["layers"]]
        for item in diagnostic["tasks"]
    ]
    accuracy_differences = [
        abs(left - right)
        for left, right in zip(baseline["accuracies"], diagnostic["accuracies"])
    ]
    accuracy_exact = (
        len(baseline["accuracies"]) == len(diagnostic["accuracies"]) == 10
        and max(accuracy_differences, default=float("inf")) < 1e-6
    )
    hashes_exact = baseline["hashes"] == diagnostic["hashes"]
    final_hash_exact = bool(
        baseline["hashes"]
        and diagnostic["hashes"]
        and baseline["hashes"][-1]["global_state_sha256"]
        == diagnostic["hashes"][-1]["global_state_sha256"]
    )
    report = {
        "rounds": 10,
        "selected_clients_exact": selection_exact,
        "task_count": len(baseline["tasks"]),
        "task_seeds_exact": task_seeds_exact,
        "reset_masks_exact": reset_masks_exact,
        "full_task_traces_exact": task_trace_exact,
        "accuracy_exact_within_1e-6": accuracy_exact,
        "maximum_accuracy_difference": max(accuracy_differences, default=None),
        "all_round_hashes_exact": hashes_exact,
        "final_global_sha256_exact": final_hash_exact,
        "baseline_final_sha256": (
            baseline["hashes"][-1]["global_state_sha256"] if baseline["hashes"] else None
        ),
        "diagnostic_final_sha256": (
            diagnostic["hashes"][-1]["global_state_sha256"]
            if diagnostic["hashes"] else None
        ),
        "passed": all(
            [
                selection_exact,
                task_seeds_exact,
                reset_masks_exact,
                task_trace_exact,
                accuracy_exact,
                hashes_exact,
                final_hash_exact,
            ]
        ),
    }
    report_path = output_dir.parent / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("FedPhoenixHistoryDiagnostic parity failed")


if __name__ == "__main__":
    main()
