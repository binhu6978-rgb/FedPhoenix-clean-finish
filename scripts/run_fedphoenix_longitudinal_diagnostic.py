#!/usr/bin/env python
"""Parity-gated seed-1 launcher with FedAvg/FedPhoenix parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="results/fedphoenix_longitudinal_diagnostic"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--skip_parity", action="store_true")
    return parser.parse_args()


def diagnostic_command(python: str, algorithm: str, gpu: int, output: Path) -> List[str]:
    return [
        python,
        str(ROOT / "experiments" / "fedphoenix_longitudinal_interaction_diagnostic.py"),
        "--source_algorithm", algorithm,
        "--dataset", "cifar10", "--model", "vgg", "--num_users", "100",
        "--frac", "0.1", "--local_ep", "5", "--local_bs", "50",
        "--bs", "256", "--optimizer", "sgd", "--lr", "0.01",
        "--momentum", "0.5", "--weight_decay", "0", "--iid", "0",
        "--noniid_case", "5", "--data_beta", "0.3", "--generate_data", "0",
        "--num_classes", "10", "--num_channels", "3", "--seed", "1",
        "--gpu", str(gpu), "--num_workers", "0", "--epochs", "300",
        "--num_monitored_clients", "10", "--FP_conv", "1000", "--FP_fc", "0",
        "--reset", "0.015625", "--remethod", "ori_normal",
        "--output_dir", str(output),
    ]


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at_unix"] = time.time()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_output = Path(args.output_dir).resolve()
    seed_output = root_output / "seed1"
    logs = root_output / "logs"
    parity_output = root_output / "parity_seed1"
    for directory in (root_output, seed_output, logs):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = root_output / "manifest.json"
    manifest: Dict[str, Any] = {
        "status": "starting",
        "seed": 1,
        "epochs": 300,
        "parallel_algorithms": True,
        "runs": {},
    }
    write_manifest(manifest_path, manifest)

    if not args.skip_parity:
        manifest["status"] = "parity_running"
        write_manifest(manifest_path, manifest)
        parity_log = (logs / "parity.stdout.log").open("w", encoding="utf-8")
        parity_err = (logs / "parity.stderr.log").open("w", encoding="utf-8")
        try:
            code = subprocess.call(
                [
                    args.python,
                    str(ROOT / "scripts" / "check_fedphoenix_longitudinal_parity.py"),
                    "--output_dir", str(parity_output), "--gpu", str(args.gpu),
                    "--python", args.python,
                ],
                cwd=ROOT,
                stdout=parity_log,
                stderr=parity_err,
            )
        finally:
            parity_log.close()
            parity_err.close()
        manifest["parity_returncode"] = code
        if code != 0:
            manifest["status"] = "parity_failed"
            write_manifest(manifest_path, manifest)
            raise SystemExit("Parity failed; 300-round runs were not started")
    else:
        parity_report = parity_output / "parity_report.json"
        if not parity_report.exists() or not json.loads(parity_report.read_text(encoding="utf-8"))["passed"]:
            raise SystemExit("--skip_parity requires an existing passing parity report")

    manifest["status"] = "training_parallel"
    write_manifest(manifest_path, manifest)
    pending = {"FedAvg", "FedPhoenix"}
    attempts = {algorithm: 0 for algorithm in pending}
    while pending:
        processes = {}
        handles = {}
        for algorithm in sorted(pending):
            attempts[algorithm] += 1
            stem = algorithm.lower()
            stdout = (logs / f"{stem}.attempt{attempts[algorithm]}.stdout.log").open("w", encoding="utf-8")
            stderr = (logs / f"{stem}.attempt{attempts[algorithm]}.stderr.log").open("w", encoding="utf-8")
            handles[algorithm] = (stdout, stderr)
            process = subprocess.Popen(
                diagnostic_command(args.python, algorithm, args.gpu, seed_output),
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
            )
            processes[algorithm] = process
            manifest["runs"][algorithm] = {
                "status": "running",
                "attempt": attempts[algorithm],
                "pid": process.pid,
            }
        write_manifest(manifest_path, manifest)
        failed = set()
        for algorithm, process in processes.items():
            code = process.wait()
            for handle in handles[algorithm]:
                handle.close()
            if code == 0:
                manifest["runs"][algorithm].update({"status": "completed", "returncode": 0})
            else:
                failed.add(algorithm)
                manifest["runs"][algorithm].update({"status": "failed", "returncode": code})
        write_manifest(manifest_path, manifest)
        exhausted = [algorithm for algorithm in failed if attempts[algorithm] > args.max_retries]
        if exhausted:
            manifest["status"] = "failed"
            manifest["exhausted_retries"] = exhausted
            write_manifest(manifest_path, manifest)
            raise SystemExit(f"Diagnostic failed after retries: {exhausted}")
        pending = failed

    manifest["status"] = "analyzing"
    write_manifest(manifest_path, manifest)
    code = subprocess.call(
        [
            args.python,
            str(ROOT / "scripts" / "analyze_fedphoenix_longitudinal_diagnostic.py"),
            "--seed_dir", str(seed_output), "--root_output", str(root_output),
        ],
        cwd=ROOT,
    )
    if code != 0:
        manifest["status"] = "analysis_failed"
        write_manifest(manifest_path, manifest)
        raise SystemExit(code)
    manifest["status"] = "completed"
    write_manifest(manifest_path, manifest)
    print(f"Completed: {root_output}")


if __name__ == "__main__":
    main()
