#!/usr/bin/env python
"""Parity-gated, resumable seed-1 launcher for shadow transfer diagnostic."""

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
        "--output_dir", default="results/same_direction_shadow_transfer"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--skip_parity", action="store_true")
    return parser.parse_args()


def training_command(python: str, gpu: int, seed_dir: Path) -> List[str]:
    return [
        python,
        str(ROOT / "experiments" / "same_direction_shadow_transfer.py"),
        "--dataset", "cifar10", "--model", "vgg", "--num_users", "100",
        "--frac", "0.1", "--local_ep", "5", "--local_bs", "50",
        "--bs", "256", "--optimizer", "sgd", "--lr", "0.01",
        "--momentum", "0.5", "--weight_decay", "0", "--iid", "0",
        "--noniid_case", "5", "--data_beta", "0.3", "--generate_data", "0",
        "--num_classes", "10", "--num_channels", "3", "--seed", "1",
        "--gpu", str(gpu), "--num_workers", "0", "--epochs", "300",
        "--num_monitored_clients", "10",
        "--probe_ratios", "0.02,0.05,0.10,0.20,0.40",
        "--output_dir", str(seed_dir),
    ]


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at_unix"] = time.time()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    seed_dir = output / "seed1"
    logs = output / "logs"
    parity_dir = output / "parity_seed1"
    for directory in (output, seed_dir, logs):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"seed": 1, "epochs": 300, "runs": {}}
    manifest["status"] = "starting"
    write_manifest(manifest_path, manifest)

    parity_report = parity_dir / "parity_report.json"
    parity_passed = (
        parity_report.exists()
        and json.loads(parity_report.read_text(encoding="utf-8")).get("passed") is True
    )
    if not args.skip_parity and not parity_passed:
        manifest["status"] = "parity_running"
        write_manifest(manifest_path, manifest)
        with (logs / "parity.stdout.log").open("w", encoding="utf-8") as stdout, (
            logs / "parity.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            code = subprocess.call(
                [
                    args.python,
                    str(ROOT / "scripts" / "check_same_direction_shadow_parity.py"),
                    "--output_dir", str(parity_dir), "--gpu", str(args.gpu),
                    "--python", args.python,
                ],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
            )
        if code != 0:
            manifest.update({"status": "parity_failed", "parity_returncode": code})
            write_manifest(manifest_path, manifest)
            raise SystemExit("Parity failed; 300-round run was not started")
        parity_passed = True
    if not parity_passed:
        raise SystemExit("A passing parity report is required")

    existing = manifest.get("runs", {}).get("seed1", {})
    if existing.get("status") == "completed" and (seed_dir / "summary.json").exists():
        manifest["status"] = "analyzing"
        write_manifest(manifest_path, manifest)
    else:
        completed = False
        for attempt in range(1, args.max_retries + 2):
            manifest["status"] = "training"
            manifest.setdefault("runs", {})["seed1"] = {
                "status": "running", "attempt": attempt
            }
            write_manifest(manifest_path, manifest)
            with (logs / f"seed1.attempt{attempt}.stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout, (logs / f"seed1.attempt{attempt}.stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    training_command(args.python, args.gpu, seed_dir),
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                )
                manifest["runs"]["seed1"]["pid"] = process.pid
                write_manifest(manifest_path, manifest)
                code = process.wait()
            if code == 0:
                manifest["runs"]["seed1"].update(
                    {"status": "completed", "returncode": 0}
                )
                completed = True
                break
            manifest["runs"]["seed1"].update(
                {"status": "failed", "returncode": code}
            )
            write_manifest(manifest_path, manifest)
        if not completed:
            manifest["status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise SystemExit("Seed 1 failed after all retries")

    manifest["status"] = "analyzing"
    write_manifest(manifest_path, manifest)
    code = subprocess.call(
        [
            args.python,
            str(ROOT / "scripts" / "analyze_same_direction_shadow_transfer.py"),
            "--seed_dir", str(seed_dir), "--output_dir", str(output),
        ],
        cwd=ROOT,
    )
    if code != 0:
        manifest["status"] = "analysis_failed"
        write_manifest(manifest_path, manifest)
        raise SystemExit(code)
    manifest["status"] = "completed"
    write_manifest(manifest_path, manifest)
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
