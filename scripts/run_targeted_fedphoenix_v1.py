#!/usr/bin/env python
"""Correctness-gated launcher for the two TargetedFedPhoenix V1 screens."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RATIOS = (0.25, 0.50)
BASELINE = {
    "peak_accuracy": 78.35,
    "peak_round": 218,
    "final_accuracy": 73.93,
    "last20_mean": 69.665,
}


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
        "--tfp_max_history_gap", "20",
    ]


def run_id(ratio):
    return f"target{ratio:.2f}".replace(".", "p") + "_r300_seed1"


def command_for(python, output_dir, gpu, ratio):
    rid = run_id(ratio)
    return [
        str(python), str(ROOT / "main_fed.py"),
        "--algorithm", "TargetedFedPhoenix",
        "--tfp_target_ratio", str(ratio),
        "--gpu", str(gpu),
        "--run_name", rid,
        "--metrics_log_dir", str(output_dir / "runs" / rid),
        *protocol_args(),
    ]


def run_correctness(python, output_dir, gpu):
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    unit_stdout = logs / "unit_tests.stdout.log"
    unit_stderr = logs / "unit_tests.stderr.log"
    with unit_stdout.open("w", encoding="utf-8") as stdout, unit_stderr.open(
        "w", encoding="utf-8"
    ) as stderr:
        unit = subprocess.run(
            [
                str(python), "-m", "unittest",
                "tests.test_targeted_fedphoenix", "-v",
            ],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
        )
    if unit.returncode != 0:
        raise RuntimeError(f"unit tests failed; inspect {unit_stderr}")

    parity_stdout = logs / "parity.stdout.log"
    parity_stderr = logs / "parity.stderr.log"
    with parity_stdout.open("w", encoding="utf-8") as stdout, parity_stderr.open(
        "w", encoding="utf-8"
    ) as stderr:
        parity = subprocess.run(
            [
                str(python),
                str(ROOT / "experiments" / "check_targeted_fedphoenix_parity.py"),
                "--output_dir", str(output_dir),
                "--python", str(python),
                "--gpu", str(gpu),
            ],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
        )
    if parity.returncode != 0:
        raise RuntimeError(f"rho=0 parity failed; inspect {parity_stderr}")
    report = read_json(output_dir / "parity_report.json")
    if not report.get("passed"):
        raise RuntimeError("rho=0 parity report is not passing")
    return {"unit_tests_passed": True, "parity": report}


def terminate_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def start_attempt(python, output_dir, gpu, ratio, attempt, mode):
    rid = run_id(ratio)
    run_dir = output_dir / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    stdout_path = output_dir / "logs" / f"{rid}.attempt{attempt}.{mode}.stdout.log"
    stderr_path = output_dir / "logs" / f"{rid}.attempt{attempt}.{mode}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    command = command_for(python, output_dir, gpu, ratio)
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    return {
        "ratio": ratio,
        "run_id": rid,
        "attempt": attempt,
        "mode": mode,
        "command": command,
        "process": process,
        "stdout_handle": stdout,
        "stderr_handle": stderr,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "started_at": now(),
        "started_clock": time.perf_counter(),
    }


def close_attempt(item):
    item["stdout_handle"].close()
    item["stderr_handle"].close()
    return_code = item["process"].returncode
    stderr_text = Path(item["stderr_log"]).read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "attempt": item["attempt"],
        "mode": item["mode"],
        "return_code": return_code,
        "stdout_log": item["stdout_log"],
        "stderr_log": item["stderr_log"],
        "started_at": item["started_at"],
        "finished_at": now(),
        "wall_seconds": time.perf_counter() - item["started_clock"],
        "cuda_oom": "out of memory" in stderr_text.lower(),
    }


def update_manifest(path, manifest):
    manifest["updated_at"] = now()
    write_json(path, manifest)


def run_parallel(python, output_dir, gpu, manifest, manifest_path):
    active = []
    for ratio in RATIOS:
        entry = manifest["runs"][run_id(ratio)]
        entry["attempt"] += 1
        item = start_attempt(
            python, output_dir, gpu, ratio, entry["attempt"], "parallel"
        )
        entry.update(
            {
                "status": "running",
                "pid": item["process"].pid,
                "stdout_log": item["stdout_log"],
                "stderr_log": item["stderr_log"],
            }
        )
        active.append(item)
    update_manifest(manifest_path, manifest)

    finished = []
    oom = False
    while active:
        for item in list(active):
            if item["process"].poll() is None:
                continue
            active.remove(item)
            result = close_attempt(item)
            finished.append((item, result))
            entry = manifest["runs"][item["run_id"]]
            entry["attempt_history"].append(result)
            entry["status"] = "completed" if result["return_code"] == 0 else "failed"
            oom = oom or result["cuda_oom"]
            update_manifest(manifest_path, manifest)
        if oom:
            for item in active:
                terminate_process(item["process"])
                result = close_attempt(item)
                result["stopped_due_to_peer_oom"] = True
                manifest["runs"][item["run_id"]]["attempt_history"].append(result)
                manifest["runs"][item["run_id"]]["status"] = "stopped_for_oom_fallback"
            active.clear()
            return False, True
        if active:
            time.sleep(2)

    success = all(result["return_code"] == 0 for _, result in finished)
    return success, False


def run_sequential(python, output_dir, gpu, manifest, manifest_path):
    for ratio in RATIOS:
        rid = run_id(ratio)
        entry = manifest["runs"][rid]
        entry["attempt"] += 1
        item = start_attempt(
            python, output_dir, gpu, ratio, entry["attempt"], "sequential"
        )
        entry.update(
            {
                "status": "running",
                "pid": item["process"].pid,
                "stdout_log": item["stdout_log"],
                "stderr_log": item["stderr_log"],
            }
        )
        update_manifest(manifest_path, manifest)
        item["process"].wait()
        result = close_attempt(item)
        entry["attempt_history"].append(result)
        if result["return_code"] != 0:
            entry["status"] = "failed"
            update_manifest(manifest_path, manifest)
            raise RuntimeError(f"{rid} failed during sequential fallback")
        entry["status"] = "completed"
        update_manifest(manifest_path, manifest)


def collect_outputs(output_dir, manifest, correctness, wall_seconds):
    summaries = []
    for ratio in RATIOS:
        rid = run_id(ratio)
        path = output_dir / "runs" / rid / "summary.json"
        if not path.exists():
            raise RuntimeError(f"missing run summary: {path}")
        summary = read_json(path)
        summary["run_id"] = rid
        summaries.append(summary)
        manifest["runs"][rid]["summary"] = summary

    rows = []
    comparisons = []
    for summary in summaries:
        row = {
            "run_id": summary["run_id"],
            "ratio": summary["tfp_target_ratio"],
            "peak_accuracy": summary["peak_accuracy"],
            "peak_round": summary["peak_round"],
            "final_accuracy": summary["final_accuracy"],
            "last20_mean": summary["last20_mean"],
            "peak_delta_vs_fedphoenix": summary["peak_accuracy"] - BASELINE["peak_accuracy"],
            "final_delta_vs_fedphoenix": summary["final_accuracy"] - BASELINE["final_accuracy"],
            "last20_delta_vs_fedphoenix": summary["last20_mean"] - BASELINE["last20_mean"],
            "average_actual_target_fraction": summary["average_actual_target_fraction"],
            "baseline_random_overlap_rate": summary["baseline_random_overlap_rate"],
            "actual_replaced_reset_slots": summary["actual_replaced_reset_slots"],
            "maximum_history_memory_bytes": summary["maximum_history_memory_bytes"],
            "failed_client_updates": summary["failed_client_updates"],
            "runtime_seconds": summary["runtime_seconds"],
        }
        rows.append(row)
        comparisons.append({**row, "client_statistics": summary["client_statistics"], "layer_statistics": summary["layer_statistics"]})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison = {
        "baseline_fedphoenix": BASELINE,
        "correctness": correctness,
        "parallel_requested": int(manifest["parallel_requested"]),
        "cuda_oom_detected": bool(manifest["cuda_oom_detected"]),
        "sequential_fallback_used": bool(manifest["sequential_fallback_used"]),
        "launcher_wall_seconds": float(wall_seconds),
        "runs": comparisons,
    }
    write_json(output_dir / "comparison.json", comparison)

    report = [
        "# TargetedFedPhoenix V1 screening", "",
        f"- Unit tests passed: {correctness['unit_tests_passed']}",
        f"- rho=0 exact parity passed: {correctness['parity']['passed']}",
        f"- CUDA OOM: {manifest['cuda_oom_detected']}",
        f"- Sequential fallback: {manifest['sequential_fallback_used']}",
        f"- Launcher wall time: {wall_seconds:.1f} seconds", "",
        "## Accuracy versus fixed FedPhoenix reference", "",
        "| Ratio | Peak | Peak round | Final | Last20 | Peak delta | Final delta | Last20 delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {float(row['ratio']):.2f} | {float(row['peak_accuracy']):.4f} | "
            f"{int(row['peak_round'])} | {float(row['final_accuracy']):.4f} | "
            f"{float(row['last20_mean']):.4f} | "
            f"{float(row['peak_delta_vs_fedphoenix']):+.4f} | "
            f"{float(row['final_delta_vs_fedphoenix']):+.4f} | "
            f"{float(row['last20_delta_vs_fedphoenix']):+.4f} |"
        )
    report.extend(["", "## Mechanism statistics", ""])
    for summary in summaries:
        stats = summary["client_statistics"]
        report.extend(
            [
                f"### ratio={summary['tfp_target_ratio']:.2f}", "",
                f"- Returning clients with history: {stats['clients_with_history']}",
                f"- Clients targeted: {stats['clients_targeted']}",
                f"- Clients with changed dispatch: {stats['clients_dispatch_changed']}",
                f"- Actual target fraction: {summary['average_actual_target_fraction']:.6f}",
                f"- Baseline-random overlap: {summary['baseline_random_overlap_rate']:.6f}",
                f"- Replaced slots: {summary['actual_replaced_reset_slots']}",
                f"- Cold/stale/no-valid/Q=0 client counts: {stats['clients_cold_start']}/"
                f"{stats['clients_stale']}/{stats['clients_no_valid_history']}/"
                f"{stats['clients_no_target_slots']}",
                f"- Maximum history memory: {summary['maximum_history_memory_bytes']} bytes", "",
            ]
        )
    improved = any(row["peak_delta_vs_fedphoenix"] > 0 for row in rows)
    report.extend(
        [
            "## Strict conclusion", "",
            (
                "At least one history-informed targeted-reset configuration improved "
                "peak accuracy over the fixed FedPhoenix reference in this 300-round screen."
                if improved
                else "History-informed targeted reset did not improve peak accuracy over "
                "the fixed FedPhoenix reference in this 300-round screen."
            ), "",
            "This result evaluates reset targeting; it does not by itself establish that "
            "high update norm is a harmful-kernel score.", "",
        ]
    )
    (output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/targeted_fedphoenix_v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--parallel", type=int, choices=[1, 2], default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    correctness = run_correctness(Path(args.python), output_dir, args.gpu)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": 1,
        "created_at": now(),
        "updated_at": now(),
        "status": "running",
        "protocol": "VGG/CIFAR10 alpha=0.3 seed=1 300 rounds",
        "parallel_requested": int(args.parallel),
        "cuda_oom_detected": False,
        "sequential_fallback_used": False,
        "correctness": correctness,
        "runs": {
            run_id(ratio): {
                "ratio": ratio,
                "status": "pending",
                "attempt": 0,
                "attempt_history": [],
            }
            for ratio in RATIOS
        },
    }
    write_json(manifest_path, manifest)
    if args.parallel == 2:
        success, oom = run_parallel(
            Path(args.python), output_dir, args.gpu, manifest, manifest_path
        )
        manifest["cuda_oom_detected"] = bool(oom)
        if oom:
            manifest["sequential_fallback_used"] = True
            update_manifest(manifest_path, manifest)
            run_sequential(
                Path(args.python), output_dir, args.gpu, manifest, manifest_path
            )
        elif not success:
            manifest["status"] = "failed"
            update_manifest(manifest_path, manifest)
            raise RuntimeError("parallel targeted experiments failed without CUDA OOM")
    else:
        run_sequential(Path(args.python), output_dir, args.gpu, manifest, manifest_path)

    comparison = collect_outputs(
        output_dir, manifest, correctness, time.perf_counter() - started
    )
    manifest["status"] = "completed"
    manifest["comparison"] = comparison
    update_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
