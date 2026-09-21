#!/usr/bin/env python
"""10-round trajectory parity gate for the longitudinal diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
ACCURACY_PATTERN = re.compile(r"ROUND_ACCURACY .*?round=(\d+) accuracy=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="results/fedphoenix_longitudinal_diagnostic/parity_seed1",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def fixed_args(algorithm: str, gpu: int, epochs: int = 10) -> List[str]:
    return [
        "--dataset", "cifar10", "--model", "vgg", "--num_users", "100",
        "--frac", "0.1", "--local_ep", "5", "--local_bs", "50",
        "--bs", "256", "--optimizer", "sgd", "--lr", "0.01",
        "--momentum", "0.5", "--weight_decay", "0", "--iid", "0",
        "--noniid_case", "5", "--data_beta", "0.3", "--generate_data", "0",
        "--num_classes", "10", "--num_channels", "3", "--seed", "1",
        "--gpu", str(gpu), "--num_workers", "0", "--epochs", str(epochs),
        "--FP_conv", "1000", "--FP_fc", "0", "--reset", "0.015625",
        "--remethod", "ori_normal", "--algorithm", algorithm,
    ]


def diagnostic_args(gpu: int, epochs: int = 10) -> List[str]:
    baseline_args = fixed_args("FedAvg", gpu, epochs)
    algorithm_position = baseline_args.index("--algorithm")
    return baseline_args[:algorithm_position] + baseline_args[algorithm_position + 2 :]


def _launch_pair(commands: Dict[str, Sequence[str]], log_dir: Path) -> None:
    processes = {}
    handles = {}
    try:
        for name, command in commands.items():
            stdout = (log_dir / f"{name}.stdout.log").open("w", encoding="utf-8")
            stderr = (log_dir / f"{name}.stderr.log").open("w", encoding="utf-8")
            handles[name] = (stdout, stderr)
            processes[name] = subprocess.Popen(
                list(command), cwd=ROOT, stdout=stdout, stderr=stderr
            )
        failures = {}
        for name, process in processes.items():
            code = process.wait()
            if code != 0:
                failures[name] = code
        if failures:
            raise RuntimeError(f"Parity subprocess failures: {failures}")
    finally:
        for pair in handles.values():
            for handle in pair:
                handle.close()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _accuracies_from_log(path: Path) -> List[float]:
    return [float(match.group(2)) for match in ACCURACY_PATTERN.finditer(path.read_text(encoding="utf-8"))]


def _compare_algorithm(
    algorithm: str, output: Path, tolerance: float
) -> Dict[str, Any]:
    lower = algorithm.lower()
    original_selection = _read_jsonl(output / "traces" / f"original_{lower}.jsonl")
    diagnostic_rows = _read_csv(output / "diagnostic" / f"{lower}_trajectory.csv")
    original_clients = [row["selected_clients"] for row in original_selection]
    diagnostic_clients = [json.loads(row["selected_clients"]) for row in diagnostic_rows]
    client_match = original_clients == diagnostic_clients

    original_accuracy = _accuracies_from_log(output / "logs" / f"original_{lower}.stdout.log")
    diagnostic_accuracy = [float(row["test_accuracy"]) for row in diagnostic_rows]
    accuracy_deltas = [
        abs(left - right) for left, right in zip(original_accuracy, diagnostic_accuracy)
    ]
    accuracy_match = (
        len(original_accuracy) == len(diagnostic_accuracy) == 10
        and max(accuracy_deltas, default=float("inf")) <= tolerance
    )

    task_seed_match = True
    assignment_match = True
    if algorithm == "FedPhoenix":
        baseline_metrics = _read_csv(
            output
            / "baseline_metrics"
            / "cifar10_vgg_FedPhoenix_seed1_parity_original_fedphoenix.csv"
        )
        baseline_task_seeds = [json.loads(row["task_seeds"]) for row in baseline_metrics]
        diagnostic_task_seeds = [json.loads(row["task_seeds"]) for row in diagnostic_rows]
        task_seed_match = baseline_task_seeds == diagnostic_task_seeds
        baseline_assignments = [json.loads(row["assignments"]) for row in baseline_metrics]
        diagnostic_assignments = [json.loads(row["assignments"]) for row in diagnostic_rows]
        assignment_match = baseline_assignments == diagnostic_assignments

    passed = client_match and accuracy_match and task_seed_match and assignment_match
    return {
        "algorithm": algorithm,
        "passed": passed,
        "selected_clients_match": client_match,
        "round_accuracy_match": accuracy_match,
        "max_accuracy_abs_delta": max(accuracy_deltas, default=None),
        "task_seeds_match": task_seed_match,
        "task_assignments_match": assignment_match,
        "original_rounds": len(original_accuracy),
        "diagnostic_rounds": len(diagnostic_accuracy),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    for directory in ("logs", "traces", "diagnostic", "baseline_metrics"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for trace in (output / "traces").glob("*.jsonl"):
        trace.unlink()

    original_commands = {}
    diagnostic_commands = {}
    for algorithm in ("FedAvg", "FedPhoenix"):
        lower = algorithm.lower()
        original_commands[f"original_{lower}"] = [
            args.python,
            str(ROOT / "scripts" / "run_main_with_selection_trace.py"),
            "--selection_trace",
            str(output / "traces" / f"original_{lower}.jsonl"),
            "--",
            *fixed_args(algorithm, args.gpu),
            "--metrics_log_dir",
            str(output / "baseline_metrics"),
            "--run_name",
            f"parity_original_{lower}",
        ]
        diagnostic_commands[f"diagnostic_{lower}"] = [
            args.python,
            str(ROOT / "experiments" / "fedphoenix_longitudinal_interaction_diagnostic.py"),
            "--source_algorithm",
            algorithm,
            *diagnostic_args(args.gpu),
            "--output_dir",
            str(output / "diagnostic"),
        ]

    # User-authorized algorithm-level parallelism: two baselines, then two diagnostics.
    _launch_pair(original_commands, output / "logs")
    _launch_pair(diagnostic_commands, output / "logs")

    results = [
        _compare_algorithm(algorithm, output, args.tolerance)
        for algorithm in ("FedAvg", "FedPhoenix")
    ]
    report = {
        "passed": all(result["passed"] for result in results),
        "tolerance": args.tolerance,
        "results": results,
    }
    (output / "parity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Longitudinal diagnostic parity check", ""]
    for result in results:
        lines.extend(
            [
                f"## {result['algorithm']}",
                "",
                f"- Passed: **{result['passed']}**",
                f"- Selected clients match: {result['selected_clients_match']}",
                f"- Round accuracies match: {result['round_accuracy_match']}",
                f"- Max accuracy absolute delta: {result['max_accuracy_abs_delta']}",
                f"- Task seeds match: {result['task_seeds_match']}",
                f"- Task assignments match: {result['task_assignments_match']}",
                "",
            ]
        )
    (output / "PARITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Parity failed; refusing to authorize the 300-round run")


if __name__ == "__main__":
    main()
