#!/usr/bin/env python
"""Ten-round parity gate for same-direction shadow transfer diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
ACCURACY_PATTERN = re.compile(r"ROUND_ACCURACY .*?round=(\d+) accuracy=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="results/same_direction_shadow_transfer/parity_seed1"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def protocol_args(gpu: int, epochs: int = 10) -> List[str]:
    return [
        "--dataset", "cifar10", "--model", "vgg", "--num_users", "100",
        "--frac", "0.1", "--local_ep", "5", "--local_bs", "50",
        "--bs", "256", "--optimizer", "sgd", "--lr", "0.01",
        "--momentum", "0.5", "--weight_decay", "0", "--iid", "0",
        "--noniid_case", "5", "--data_beta", "0.3", "--generate_data", "0",
        "--num_classes", "10", "--num_channels", "3", "--seed", "1",
        "--gpu", str(gpu), "--num_workers", "0", "--epochs", str(epochs),
    ]


def _run(command: List[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        code = subprocess.call(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if code != 0:
        raise RuntimeError(f"Parity subprocess failed ({code}): {' '.join(command[:3])}")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    logs = output / "logs"
    diagnostic = output / "diagnostic"
    baseline_metrics = output / "baseline_metrics"
    for directory in (output, logs, diagnostic, baseline_metrics):
        directory.mkdir(parents=True, exist_ok=True)
    selection_trace = output / "original_selected_clients.jsonl"
    hash_trace = output / "original_global_hashes.jsonl"
    for path in (selection_trace, hash_trace):
        if path.exists():
            path.unlink()

    baseline_command = [
        args.python,
        str(ROOT / "scripts" / "run_main_with_selection_trace.py"),
        "--selection_trace", str(selection_trace),
        "--global_hash_trace", str(hash_trace),
        "--",
        *protocol_args(args.gpu),
        "--algorithm", "FedAvg",
        "--metrics_log_dir", str(baseline_metrics),
        "--run_name", "same_direction_shadow_parity_original",
    ]
    diagnostic_command = [
        args.python,
        str(ROOT / "experiments" / "same_direction_shadow_transfer.py"),
        *protocol_args(args.gpu),
        "--num_monitored_clients", "10",
        "--probe_ratios", "0.02,0.05,0.10,0.20,0.40",
        "--output_dir", str(diagnostic),
        "--parity_mode",
    ]
    _run(
        baseline_command,
        logs / "original.stdout.log",
        logs / "original.stderr.log",
    )
    _run(
        diagnostic_command,
        logs / "diagnostic.stdout.log",
        logs / "diagnostic.stderr.log",
    )

    trajectory = _csv(diagnostic / "trajectory.csv")
    baseline_clients = [row["selected_clients"] for row in _jsonl(selection_trace)]
    diagnostic_clients = [json.loads(row["selected_clients"]) for row in trajectory]
    client_match = baseline_clients == diagnostic_clients

    baseline_text = (logs / "original.stdout.log").read_text(encoding="utf-8")
    baseline_accuracy = [
        float(match.group(2)) for match in ACCURACY_PATTERN.finditer(baseline_text)
    ]
    diagnostic_accuracy = [float(row["test_accuracy"]) for row in trajectory]
    deltas = [
        abs(left - right) for left, right in zip(baseline_accuracy, diagnostic_accuracy)
    ]
    accuracy_match = (
        len(baseline_accuracy) == len(diagnostic_accuracy) == 10
        and max(deltas, default=float("inf")) <= args.tolerance
    )

    baseline_hashes = [row["global_state_sha256"] for row in _jsonl(hash_trace)]
    diagnostic_hashes = [row["global_state_sha256"] for row in trajectory]
    hash_match = baseline_hashes == diagnostic_hashes and len(baseline_hashes) == 10
    shadow_event_count = sum(int(row["round_shadow_event_count"]) for row in trajectory)
    passed = client_match and accuracy_match and hash_match
    report = {
        "passed": passed,
        "seed": 1,
        "rounds": 10,
        "selected_clients_match": client_match,
        "round_accuracy_match": accuracy_match,
        "max_accuracy_abs_delta": max(deltas, default=None),
        "global_state_sha256_match": hash_match,
        "shadow_event_count": shadow_event_count,
        "tolerance": args.tolerance,
    }
    (output / "parity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "PARITY_REPORT.md").write_text(
        "\n".join(
            [
                "# Same-direction shadow parity",
                "",
                f"- Passed: **{passed}**",
                f"- Selected clients match: {client_match}",
                f"- Round accuracy match: {accuracy_match}",
                f"- Max accuracy absolute delta: {report['max_accuracy_abs_delta']}",
                f"- Global state SHA-256 match: {hash_match}",
                f"- Shadow scalar events exercised: {shadow_event_count}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("Parity failed; refusing the 300-round run")


if __name__ == "__main__":
    main()
