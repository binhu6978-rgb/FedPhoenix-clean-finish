#!/usr/bin/env python
"""Correctness-gated, predefined six-run TargetedFedPhoenix focused screen."""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRONG_LAYERS = "features.20,features.24,features.27,features.30"
FEDPHOENIX = {
    "name": "FedPhoenix",
    "peak_accuracy": 78.35,
    "peak_round": 218,
    "final_accuracy": 73.93,
    "last20_mean": 69.665,
}
CORRECTED_V1 = {
    "name": "corrected_TargetedFedPhoenix_V1",
    "peak_accuracy": 78.33000183105469,
    "peak_round": 284,
    "final_accuracy": 73.54000091552734,
    "last20_mean": 70.75949974060059,
}
RUNS = [
    {
        "batch": 1,
        "id": "E1_fresh_gap10_ratio025_all",
        "score": "update_norm", "ratio": 0.25, "gap": 10,
        "layers": "all", "start": 1,
    },
    {
        "batch": 1,
        "id": "E2_gap10_ratio0125_all",
        "score": "update_norm", "ratio": 0.125, "gap": 10,
        "layers": "all", "start": 1,
    },
    {
        "batch": 2,
        "id": "E3_gap20_ratio025_stronglayers",
        "score": "update_norm", "ratio": 0.25, "gap": 20,
        "layers": STRONG_LAYERS, "start": 1,
    },
    {
        "batch": 2,
        "id": "E4_gap10_ratio025_stronglayers",
        "score": "update_norm", "ratio": 0.25, "gap": 10,
        "layers": STRONG_LAYERS, "start": 1,
    },
    {
        "batch": 3,
        "id": "E5_start151_gap10_ratio025_stronglayers",
        "score": "update_norm", "ratio": 0.25, "gap": 10,
        "layers": STRONG_LAYERS, "start": 151,
    },
    {
        "batch": 3,
        "id": "E6_residual_gap10_ratio025_stronglayers",
        "score": "residual", "ratio": 0.25, "gap": 10,
        "layers": STRONG_LAYERS, "start": 1,
    },
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_args(run):
    return [
        "--dataset", "cifar10", "--model", "vgg", "--epochs", "300",
        "--num_users", "100", "--frac", "0.1", "--local_ep", "5",
        "--local_bs", "50", "--bs", "256", "--optimizer", "sgd",
        "--lr", "0.01", "--momentum", "0.5", "--weight_decay", "0",
        "--iid", "0", "--noniid_case", "5", "--data_beta", "0.3",
        "--generate_data", "0", "--num_classes", "10", "--num_channels", "3",
        "--seed", "1", "--num_workers", "0", "--FP_conv", "1000",
        "--FP_fc", "0", "--reset", "0.015625", "--remethod", "ori_normal",
        "--tfp_score_type", run["score"],
        "--tfp_target_ratio", str(run["ratio"]),
        "--tfp_max_history_gap", str(run["gap"]),
        "--tfp_target_layers", run["layers"],
        "--tfp_start_round", str(run["start"]),
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
        logs / "regression.stdout.log", logs / "regression.stderr.log",
    )
    if regression.returncode:
        raise RuntimeError("full regression tests failed")
    parity = run_logged(
        [
            str(python), str(ROOT / "experiments" / "check_targeted_fedphoenix_parity.py"),
            "--output_dir", str(output_dir), "--python", str(python),
            "--gpu", str(gpu),
        ],
        logs / "parity.stdout.log", logs / "parity.stderr.log",
    )
    parity_report = read_json(output_dir / "parity_report.json")
    if parity.returncode or not parity_report["passed"]:
        raise RuntimeError("rho=0 ten-round strict parity failed")
    active = run_logged(
        [
            str(python),
            str(ROOT / "experiments" / "check_targeted_fedphoenix_active_determinism.py"),
            "--output", str(output_dir / "active_smoke_report.json"),
        ],
        logs / "active_smoke.stdout.log", logs / "active_smoke.stderr.log",
    )
    active_report = read_json(output_dir / "active_smoke_report.json")
    if active.returncode or not active_report["passed"]:
        raise RuntimeError("corrected active deterministic smoke failed")
    return {
        "full_regression_tests_passed": True,
        "targeted_unit_test_count": 43,
        "rho0_parity": parity_report,
        "active_deterministic_smoke": active_report,
    }


def command_for(python, gpu, output_dir, run):
    run_dir = output_dir / f"batch{run['batch']}" / run["id"]
    return run_dir, [
        str(python), str(ROOT / "main_fed.py"),
        "--algorithm", "TargetedFedPhoenix", "--gpu", str(gpu),
        "--run_name", run["id"], "--metrics_log_dir", str(run_dir),
        *protocol_args(run),
    ]


def contains_oom(*paths):
    needles = ("cuda out of memory", "outofmemoryerror", "out of memory")
    for path in paths:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if any(needle in text for needle in needles):
                return True
    return False


def remove_stale_summary(run_dir):
    summary = run_dir / "summary.json"
    if summary.exists():
        summary.unlink()


def launch_parallel_batch(python, gpu, output_dir, batch_runs, attempt):
    processes = []
    for run in batch_runs:
        run_dir, command = command_for(python, gpu, output_dir, run)
        run_dir.mkdir(parents=True, exist_ok=True)
        remove_stale_summary(run_dir)
        stdout_path = output_dir / "logs" / f"{run['id']}.parallel{attempt}.stdout.log"
        stderr_path = output_dir / "logs" / f"{run['id']}.parallel{attempt}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        processes.append(
            {
                "run": run, "run_dir": run_dir, "command": command,
                "process": process, "stdout": stdout, "stderr": stderr,
                "stdout_path": stdout_path, "stderr_path": stderr_path,
                "started": time.perf_counter(),
            }
        )
    oom_seen = False
    while any(item["process"].poll() is None for item in processes):
        for item in processes:
            if item["process"].poll() is not None and contains_oom(
                item["stdout_path"], item["stderr_path"]
            ):
                oom_seen = True
        if oom_seen:
            for item in processes:
                if item["process"].poll() is None:
                    item["process"].terminate()
            break
        time.sleep(5)
    rows = []
    for item in processes:
        try:
            return_code = item["process"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            item["process"].kill()
            return_code = item["process"].wait()
        item["stdout"].close()
        item["stderr"].close()
        oom_seen = oom_seen or contains_oom(item["stdout_path"], item["stderr_path"])
        rows.append(
            {
                "run_id": item["run"]["id"], "mode": "parallel",
                "attempt": int(attempt), "return_code": int(return_code),
                "runtime_seconds": time.perf_counter() - item["started"],
                "stdout_log": str(item["stdout_path"]),
                "stderr_log": str(item["stderr_path"]),
                "oom": contains_oom(item["stdout_path"], item["stderr_path"]),
                "finished_at": now(),
            }
        )
    return rows, oom_seen


def launch_sequential(python, gpu, output_dir, run, attempt, reason):
    run_dir, command = command_for(python, gpu, output_dir, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_summary(run_dir)
    stdout_path = output_dir / "logs" / f"{run['id']}.sequential{attempt}.stdout.log"
    stderr_path = output_dir / "logs" / f"{run['id']}.sequential{attempt}.stderr.log"
    started = time.perf_counter()
    result = run_logged(command, stdout_path, stderr_path)
    return {
        "run_id": run["id"], "mode": "sequential", "reason": reason,
        "attempt": int(attempt), "return_code": int(result.returncode),
        "runtime_seconds": time.perf_counter() - started,
        "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
        "oom": contains_oom(stdout_path, stderr_path), "finished_at": now(),
    }


def completed_summary(output_dir, run):
    path = output_dir / f"batch{run['batch']}" / run["id"] / "summary.json"
    if not path.exists():
        return None
    summary = read_json(path)
    return summary if int(summary.get("completed_rounds", 0)) == 300 else None


def run_batch(python, gpu, output_dir, batch_runs, manifest, manifest_path, max_retries):
    batch_number = batch_runs[0]["batch"]
    rows, oom = launch_parallel_batch(python, gpu, output_dir, batch_runs, 1)
    manifest["attempts"].extend(rows)
    manifest["batches"][str(batch_number)] = {
        "parallel_attempted": True, "oom_detected": bool(oom),
        "sequential_fallback": False, "status": "running",
    }
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    if oom:
        manifest["batches"][str(batch_number)]["sequential_fallback"] = True
        for position, run in enumerate(batch_runs, start=1):
            row = launch_sequential(
                python, gpu, output_dir, run, position, "batch_cuda_oom"
            )
            manifest["attempts"].append(row)
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)
            if row["return_code"] != 0:
                raise RuntimeError(f"{run['id']} failed during OOM sequential fallback")
    else:
        for run, row in zip(batch_runs, rows):
            if row["return_code"] == 0 and completed_summary(output_dir, run):
                continue
            succeeded = False
            for retry in range(1, int(max_retries) + 1):
                retry_row = launch_sequential(
                    python, gpu, output_dir, run, retry, "non_oom_runtime_retry"
                )
                manifest["attempts"].append(retry_row)
                manifest["updated_at"] = now()
                write_json(manifest_path, manifest)
                if retry_row["return_code"] == 0 and completed_summary(output_dir, run):
                    succeeded = True
                    break
            if not succeeded:
                raise RuntimeError(f"{run['id']} failed after retries")
    for run in batch_runs:
        if completed_summary(output_dir, run) is None:
            raise RuntimeError(f"{run['id']} did not produce a 300-round summary")
    manifest["batches"][str(batch_number)]["status"] = "completed"
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)


def add_deltas(summary):
    summary = dict(summary)
    for label, reference in (("fedphoenix", FEDPHOENIX), ("corrected_v1", CORRECTED_V1)):
        summary[f"peak_delta_vs_{label}"] = (
            float(summary["peak_accuracy"]) - reference["peak_accuracy"]
        )
        summary[f"last20_delta_vs_{label}"] = (
            float(summary["last20_mean"]) - reference["last20_mean"]
        )
    return summary


def verify_e5_hash_parity(output_dir):
    reference_path = ROOT / "results" / "fedphoenix_specialization_diagnostic" / "round_metrics.csv"
    e5_path = output_dir / "batch3" / RUNS[4]["id"] / "round_metrics.csv"
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))[:150]
    with e5_path.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))[:150]
    exact = len(reference) == len(observed) == 150 and all(
        left["global_state_sha256"] == right["global_state_sha256"]
        for left, right in zip(reference, observed)
    )
    report = {
        "reference": str(reference_path), "observed": str(e5_path),
        "rounds_compared": min(len(reference), len(observed)),
        "global_state_hashes_exact": bool(exact),
        "first_mismatch_round": next(
            (
                index + 1 for index, (left, right) in enumerate(zip(reference, observed))
                if left["global_state_sha256"] != right["global_state_sha256"]
            ),
            None,
        ),
    }
    if not exact:
        raise RuntimeError("E5 rounds 1-150 do not exactly match FedPhoenix hashes")
    return report


def comparison_conclusions(by_id):
    e1, e2, e3, e4, e5, e6 = (by_id[run["id"]] for run in RUNS)
    exceeded = [run_id for run_id, value in by_id.items() if value["peak_accuracy"] > 78.35]
    dimension_changes = {
        "targeting strength": e2["peak_accuracy"] - e1["peak_accuracy"],
        "history freshness": max(
            e1["peak_accuracy"] - CORRECTED_V1["peak_accuracy"],
            e4["peak_accuracy"] - e3["peak_accuracy"],
        ),
        "layer scope": max(
            e3["peak_accuracy"] - CORRECTED_V1["peak_accuracy"],
            e4["peak_accuracy"] - e1["peak_accuracy"],
        ),
        "intervention timing": e5["peak_accuracy"] - e4["peak_accuracy"],
        "score semantics": e6["peak_accuracy"] - e4["peak_accuracy"],
    }
    return {
        "A_gap10_vs_gap20": {
            "all_layers_delta_E1_vs_correctedV1": e1["peak_accuracy"] - CORRECTED_V1["peak_accuracy"],
            "strong_layers_delta_E4_vs_E3": e4["peak_accuracy"] - e3["peak_accuracy"],
        },
        "B_ratio0125_vs_ratio025_peak_delta_E2_vs_E1": e2["peak_accuracy"] - e1["peak_accuracy"],
        "C_strong_layers_vs_all_peak_delta_E3_vs_correctedV1": e3["peak_accuracy"] - CORRECTED_V1["peak_accuracy"],
        "D_fresh_strong_combination": {
            "E4_vs_E1": e4["peak_accuracy"] - e1["peak_accuracy"],
            "E4_vs_E3": e4["peak_accuracy"] - e3["peak_accuracy"],
        },
        "E_delayed_vs_immediate_peak_delta_E5_vs_E4": e5["peak_accuracy"] - e4["peak_accuracy"],
        "F_residual_vs_update_norm_peak_delta_E6_vs_E4": e6["peak_accuracy"] - e4["peak_accuracy"],
        "G_peak_above_78_35": bool(exceeded),
        "configs_above_78_35": exceeded,
        "H_next_dimension_if_none_exceeded": (
            None if exceeded else max(dimension_changes, key=dimension_changes.get)
        ),
        "dimension_peak_deltas": dimension_changes,
    }


def write_outputs(output_dir, summaries, correctness, manifest):
    by_id = {run["id"]: add_deltas(summaries[run["id"]]) for run in RUNS}
    e5_parity = verify_e5_hash_parity(output_dir)
    conclusions = comparison_conclusions(by_id)
    comparison = {
        "references": {"fedphoenix": FEDPHOENIX, "corrected_v1": CORRECTED_V1},
        "correctness": correctness, "runs": by_id,
        "e5_rounds1_150_hash_parity": e5_parity,
        "conclusions": conclusions,
        "oom_or_sequential_fallback": {
            key: value for key, value in manifest["batches"].items()
        },
    }
    write_json(output_dir / "comparison.json", comparison)
    fields = [
        "run_id", "score_type", "ratio", "gap", "target_layers", "start_round",
        "peak_accuracy", "peak_round", "final_accuracy", "last20_mean", "last50_mean",
        "rounds201_300_mean", "best_rolling5_mean", "best_rolling5_end_round",
        "peak_delta_vs_fedphoenix", "peak_delta_vs_corrected_v1",
        "last20_delta_vs_fedphoenix", "last20_delta_vs_corrected_v1",
        "clients_with_history", "clients_history_eligible", "clients_targeted",
        "clients_stale", "clients_cold_start", "total_reset_slots",
        "targeted_reset_slots", "average_actual_target_fraction",
        "actual_replaced_reset_slots", "maximum_history_memory_bytes",
        "valid_residual_history_fraction", "residual_no_clean_peer_invalid_count",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in RUNS:
            summary = by_id[run["id"]]
            stats = summary["client_statistics"]
            writer.writerow(
                {
                    "run_id": run["id"], "score_type": run["score"],
                    "ratio": run["ratio"], "gap": run["gap"],
                    "target_layers": run["layers"], "start_round": run["start"],
                    **{key: summary.get(key) for key in fields if key in summary},
                    **{key: stats.get(key) for key in fields if key in stats},
                }
            )
    ranked = sorted(
        by_id.items(), key=lambda item: (-item[1]["peak_accuracy"], item[0])
    )
    lines = [
        "# TargetedFedPhoenix overnight focused screening", "",
        f"- Full regression: {correctness['full_regression_tests_passed']}",
        f"- rho=0 strict parity: {correctness['rho0_parity']['passed']}",
        f"- E5 rounds 1-150 FedPhoenix hash parity: {e5_parity['global_state_hashes_exact']}",
        "", "## Complete results", "",
        "| Rank | Run | Peak | Round | Final | Last20 | Last50 | R201-300 | Best roll5 | Roll5 end |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, (run_id, value) in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {run_id} | {value['peak_accuracy']:.4f} | {value['peak_round']} | "
            f"{value['final_accuracy']:.4f} | {value['last20_mean']:.4f} | "
            f"{value['last50_mean']:.4f} | {value['rounds201_300_mean']:.4f} | "
            f"{value['best_rolling5_mean']:.4f} | {value['best_rolling5_end_round']} |"
        )
    lines.extend(["", "## Strict mechanism conclusions", ""])
    lines.append(f"- A gap<=10 vs gap<=20: {conclusions['A_gap10_vs_gap20']}")
    lines.append(f"- B rho=.125 vs .25 peak delta: {conclusions['B_ratio0125_vs_ratio025_peak_delta_E2_vs_E1']:.4f} pp")
    lines.append(f"- C strong layers vs all peak delta: {conclusions['C_strong_layers_vs_all_peak_delta_E3_vs_correctedV1']:.4f} pp")
    lines.append(f"- D fresh+strong combination: {conclusions['D_fresh_strong_combination']}")
    lines.append(f"- E delayed vs immediate peak delta: {conclusions['E_delayed_vs_immediate_peak_delta_E5_vs_E4']:.4f} pp")
    lines.append(f"- F residual vs update_norm peak delta: {conclusions['F_residual_vs_update_norm_peak_delta_E6_vs_E4']:.4f} pp")
    lines.append(f"- G any peak > 78.35: {conclusions['G_peak_above_78_35']} {conclusions['configs_above_78_35']}")
    lines.append(f"- H next dimension if none exceeded: {conclusions['H_next_dimension_if_none_exceeded']}")
    lines.extend(["", "No additional ratios, gaps, seeds, or long runs were launched.", ""])
    (output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/targeted_fedphoenix_overnight6")
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
    python = Path(args.python)
    correctness = correctness_gates(python, output_dir, args.gpu)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": 1, "created_at": now(), "updated_at": now(),
        "status": "running", "protocol": "predefined TargetedFedPhoenix overnight6",
        "fixed_runs": RUNS, "correctness": correctness,
        "batch_order": [[RUNS[0]["id"], RUNS[1]["id"]], [RUNS[2]["id"], RUNS[3]["id"]], [RUNS[4]["id"], RUNS[5]["id"]]],
        "batches": {}, "attempts": [],
    }
    write_json(manifest_path, manifest)
    try:
        for batch_number in (1, 2, 3):
            batch_runs = [run for run in RUNS if run["batch"] == batch_number]
            if all(completed_summary(output_dir, run) is not None for run in batch_runs):
                manifest["batches"][str(batch_number)] = {
                    "parallel_attempted": True,
                    "oom_detected": False,
                    "sequential_fallback": False,
                    "status": "completed",
                    "resumed_from_existing_summaries": True,
                }
                manifest["updated_at"] = now()
                write_json(manifest_path, manifest)
                continue
            run_batch(
                python, args.gpu, output_dir, batch_runs,
                manifest, manifest_path, args.max_retries,
            )
        summaries = {run["id"]: completed_summary(output_dir, run) for run in RUNS}
        comparison = write_outputs(output_dir, summaries, correctness, manifest)
        manifest["status"] = "completed"
        manifest["comparison"] = comparison
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
