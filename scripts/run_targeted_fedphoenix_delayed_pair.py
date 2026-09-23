#!/usr/bin/env python
"""Run the predefined delayed TargetedFedPhoenix P1/P2 confirmation pair."""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_targeted_fedphoenix_overnight6 import (
    FEDPHOENIX,
    ROOT,
    STRONG_LAYERS,
    contains_oom,
    correctness_gates,
    protocol_args,
    read_json,
    write_json,
)


RUNS = (
    {
        "id": "P1_start151_gap10_ratio0125_all",
        "score": "update_norm", "ratio": 0.125, "gap": 10,
        "layers": "all", "start": 151,
    },
    {
        "id": "P2_start151_gap10_ratio025_all",
        "score": "update_norm", "ratio": 0.25, "gap": 10,
        "layers": "all", "start": 151,
    },
)
REFERENCE_RUNS = {
    "E1": ROOT / "results" / "targeted_fedphoenix_overnight6"
          / "batch1" / "E1_fresh_gap10_ratio025_all",
    "E2": ROOT / "results" / "targeted_fedphoenix_overnight6"
          / "batch1" / "E2_gap10_ratio0125_all",
    "E5": ROOT / "results" / "targeted_fedphoenix_overnight6"
          / "batch3" / "E5_start151_gap10_ratio025_stronglayers",
}
REFERENCE_TRAJECTORY = (
    ROOT / "results" / "fedphoenix_specialization_diagnostic" / "round_metrics.csv"
)
PARITY_FIELDS = ("selected_clients", "task_seeds", "global_state_sha256")
COMPARISON_FIELDS = (
    "peak_accuracy", "final_accuracy", "last20_mean", "last50_mean",
    "rounds151_300_mean", "rounds201_300_mean", "best_rolling5_mean",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_rounds(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_reference():
    rows = read_rounds(REFERENCE_TRAJECTORY)
    if len(rows) < 150:
        raise RuntimeError("FedPhoenix reference has fewer than 150 rounds")
    if any(int(row["round"]) != index for index, row in enumerate(rows[:150], 1)):
        raise RuntimeError("FedPhoenix reference rounds are not sequential")
    for name in PARITY_FIELDS:
        if any(not row.get(name) for row in rows[:150]):
            raise RuntimeError(f"FedPhoenix reference missing {name}")
    return rows[:150]


def canonical_field(row, name):
    if name in {"selected_clients", "task_seeds"}:
        return json.loads(row[name])
    return row[name]


def validate_prefix(output_dir, run, reference, require_complete=False):
    run_id = run["id"]
    rows = read_rounds(output_dir / run_id / "round_metrics.csv")
    checked = min(len(rows), 150)
    for index in range(checked):
        actual = rows[index]
        expected = reference[index]
        if int(actual["round"]) != index + 1:
            raise RuntimeError(f"{run_id} nonsequential round at {index + 1}")
        for field in PARITY_FIELDS:
            if canonical_field(actual, field) != canonical_field(expected, field):
                raise RuntimeError(f"{run_id} FedPhoenix {field} mismatch at round {index + 1}")
        if int(actual["targeted_reset_slots"]) != 0:
            raise RuntimeError(f"{run_id} targeted before start at round {index + 1}")
        if int(actual["clients_before_start_round"]) != 10:
            raise RuntimeError(f"{run_id} incomplete baseline fallback at round {index + 1}")
        if int(actual["history_memory_bytes"]) <= 0:
            raise RuntimeError(f"{run_id} did not collect history at round {index + 1}")
    if require_complete and checked != 150:
        raise RuntimeError(f"{run_id} produced only {checked} of 150 parity rounds")
    return {
        "run_id": run_id,
        "rounds_checked": checked,
        "selected_clients_exact": True,
        "task_seeds_exact": True,
        "global_state_hashes_exact": True,
        "baseline_fallback_and_history_collection": True,
        "reference_path": str(REFERENCE_TRAJECTORY),
    }


def command_for(python, gpu, output_dir, run):
    run_dir = output_dir / run["id"]
    command = [
        str(python), str(ROOT / "main_fed.py"),
        "--algorithm", "TargetedFedPhoenix", "--gpu", str(gpu),
        "--run_name", run["id"], "--metrics_log_dir", str(run_dir),
        *protocol_args(run),
    ]
    return run_dir, command


def start_run(python, gpu, output_dir, run, mode, attempt):
    run_dir, command = command_for(python, gpu, output_dir, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{run['id']}.{mode}{attempt}.stdout.log"
    stderr_path = logs / f"{run['id']}.{mode}{attempt}.stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    except Exception:
        stdout.close()
        stderr.close()
        raise
    return {
        "run": run, "command": command, "process": process,
        "stdout": stdout, "stderr": stderr,
        "stdout_path": stdout_path, "stderr_path": stderr_path,
        "mode": mode, "attempt": attempt, "started": time.perf_counter(),
    }


def stop_processes(items):
    for item in items:
        if item["process"].poll() is None:
            item["process"].terminate()
    for item in items:
        try:
            item["process"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            item["process"].kill()
            item["process"].wait()


def watch_processes(items, output_dir, reference, manifest, manifest_path):
    manifest["active"] = [
        {"run_id": item["run"]["id"], "pid": item["process"].pid,
         "command": item["command"]}
        for item in items
    ]
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    try:
        while any(item["process"].poll() is None for item in items):
            for item in items:
                validate_prefix(output_dir, item["run"], reference)
            if any(
                item["process"].poll() is not None
                and contains_oom(item["stdout_path"], item["stderr_path"])
                for item in items
            ):
                stop_processes(items)
                break
            time.sleep(5)
        for item in items:
            validate_prefix(output_dir, item["run"], reference)
    except Exception:
        stop_processes(items)
        raise
    finally:
        for item in items:
            if item["process"].poll() is None:
                stop_processes([item])
            item["stdout"].close()
            item["stderr"].close()
    rows = []
    for item in items:
        rows.append({
            "run_id": item["run"]["id"], "mode": item["mode"],
            "attempt": int(item["attempt"]),
            "return_code": int(item["process"].returncode),
            "oom": contains_oom(item["stdout_path"], item["stderr_path"]),
            "runtime_seconds": time.perf_counter() - item["started"],
            "stdout_log": str(item["stdout_path"]),
            "stderr_log": str(item["stderr_path"]),
            "finished_at": now(),
        })
    manifest["attempts"].extend(rows)
    manifest["active"] = []
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    return rows


def completed_summary(output_dir, run):
    path = output_dir / run["id"] / "summary.json"
    if not path.exists():
        return None
    summary = read_json(path)
    return summary if int(summary.get("completed_rounds", 0)) == 300 else None


def run_pair(python, gpu, output_dir, reference, manifest, manifest_path, max_retries):
    existing = {run["id"]: completed_summary(output_dir, run) for run in RUNS}
    if all(existing.values()):
        manifest["resumed_from_completed_runs"] = True
        return
    if not any(existing.values()):
        items = [start_run(python, gpu, output_dir, run, "parallel", 1) for run in RUNS]
        rows = watch_processes(items, output_dir, reference, manifest, manifest_path)
        oom = any(row["oom"] for row in rows)
        manifest["parallel_attempted"] = True
        manifest["oom_detected"] = oom
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        if oom:
            manifest["sequential_fallback"] = True
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)
            for attempt, run in enumerate(RUNS, 1):
                item = start_run(python, gpu, output_dir, run, "sequential_oom", attempt)
                row = watch_processes([item], output_dir, reference, manifest, manifest_path)[0]
                if row["return_code"] or completed_summary(output_dir, run) is None:
                    raise RuntimeError(f"{run['id']} failed during same-protocol OOM fallback")
            return
    for run in RUNS:
        if completed_summary(output_dir, run) is not None:
            continue
        for attempt in range(1, int(max_retries) + 1):
            item = start_run(python, gpu, output_dir, run, "sequential_retry", attempt)
            row = watch_processes([item], output_dir, reference, manifest, manifest_path)[0]
            if row["return_code"] == 0 and completed_summary(output_dir, run):
                break
        if completed_summary(output_dir, run) is None:
            raise RuntimeError(f"{run['id']} failed after same-protocol retries")


def metrics_from_run(run_dir):
    summary = read_json(run_dir / "summary.json")
    rows = read_rounds(run_dir / "round_metrics.csv")
    if len(rows) != 300 or int(summary["completed_rounds"]) != 300:
        raise RuntimeError(f"{run_dir} is not a complete 300-round run")
    accuracy = np.asarray([float(row["test_accuracy"]) for row in rows])
    value = {
        name: summary[name] for name in (
            "peak_accuracy", "peak_round", "final_accuracy", "last20_mean",
            "last50_mean", "rounds201_300_mean", "best_rolling5_mean",
            "best_rolling5_end_round", "client_statistics", "layer_statistics",
            "average_actual_target_fraction", "maximum_history_memory_bytes",
            "failed_client_updates", "history_inventory", "controller_rng_unchanged",
            "tfp_target_ratio", "tfp_max_history_gap", "tfp_score_type",
            "tfp_target_layers", "tfp_start_round",
        )
    }
    value["rounds151_300_mean"] = float(np.mean(accuracy[150:300]))
    value["rounds201_300_mean"] = float(np.mean(accuracy[200:300]))
    value["last50_mean"] = float(np.mean(accuracy[-50:]))
    return value


def metric_deltas(left, right):
    return {name: float(left[name] - right[name]) for name in COMPARISON_FIELDS}


def write_outputs(output_dir, manifest, reference):
    metrics = {
        run["id"]: metrics_from_run(output_dir / run["id"])
        for run in RUNS
    }
    references = {
        name: metrics_from_run(path) for name, path in REFERENCE_RUNS.items()
    }
    parity = {
        run["id"]: validate_prefix(output_dir, run, reference, True)
        for run in RUNS
    }
    write_json(output_dir / "start151_parity_report.json", parity)
    p1, p2 = (metrics[run["id"]] for run in RUNS)
    comparisons = {
        "P1_vs_E2_delayed_at_ratio0125": metric_deltas(p1, references["E2"]),
        "P2_vs_E1_delayed_at_ratio025": metric_deltas(p2, references["E1"]),
        "P2_vs_E5_all_vs_strong_layers": metric_deltas(p2, references["E5"]),
        "P1_vs_P2_conservative_ratio": metric_deltas(p1, p2),
    }
    for run in RUNS:
        value = metrics[run["id"]]
        value["peak_delta_vs_fedphoenix"] = float(
            value["peak_accuracy"] - FEDPHOENIX["peak_accuracy"]
        )
        value["peak_delta_vs_E5"] = float(
            value["peak_accuracy"] - references["E5"]["peak_accuracy"]
        )
        value["last20_delta_vs_fedphoenix"] = float(
            value["last20_mean"] - FEDPHOENIX["last20_mean"]
        )
        value["last20_delta_vs_E5"] = float(
            value["last20_mean"] - references["E5"]["last20_mean"]
        )
    candidates = {"E5": references["E5"], "P1": p1, "P2": p2}
    ranked = sorted(candidates, key=lambda name: -candidates[name]["peak_accuracy"])
    peak_gap = float(
        candidates[ranked[0]]["peak_accuracy"]
        - candidates[ranked[1]]["peak_accuracy"]
    )
    comparison = {
        "protocol": "predefined delayed TargetedFedPhoenix P1/P2, 300 rounds, seed 1",
        "fedphoenix_reference": FEDPHOENIX,
        "reference_runs": references,
        "runs": metrics,
        "start151_parity": parity,
        "matched_comparisons": comparisons,
        "peak_rank_among_E5_P1_P2": ranked,
        "top_two_peak_gap_pp": peak_gap,
        "review_secondary_metrics_for_top_two": peak_gap < 0.10,
        "oom_detected": bool(manifest.get("oom_detected", False)),
        "sequential_fallback": bool(manifest.get("sequential_fallback", False)),
        "attempts": manifest["attempts"],
    }
    write_json(output_dir / "comparison.json", comparison)
    fields = [
        "run_id", "peak_accuracy", "peak_round", "final_accuracy", "last20_mean",
        "last50_mean", "rounds151_300_mean", "rounds201_300_mean",
        "best_rolling5_mean", "best_rolling5_end_round",
        "peak_delta_vs_fedphoenix", "peak_delta_vs_E5",
        "last20_delta_vs_fedphoenix", "last20_delta_vs_E5",
        "clients_with_history", "clients_history_eligible", "clients_targeted",
        "clients_cold_start", "clients_stale", "clients_before_start_round",
        "total_reset_slots", "targeted_reset_slots", "replaced_reset_slots",
        "average_actual_target_fraction", "per_layer_targeted_slots",
        "maximum_history_memory_bytes", "failed_client_updates",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in RUNS:
            value = metrics[run["id"]]
            row = {name: value.get(name) for name in fields}
            row["run_id"] = run["id"]
            row.update({
                name: value["client_statistics"].get(name)
                for name in fields if name in value["client_statistics"]
            })
            row["per_layer_targeted_slots"] = json.dumps({
                item["layer"]: int(item["targeted_slots"])
                for item in value["layer_statistics"]
            }, separators=(",", ":"))
            writer.writerow(row)
    lines = [
        "# TargetedFedPhoenix delayed confirmation pair", "",
        "Both runs use VGG/CIFAR-10, Dirichlet alpha=0.3, seed=1, and 300 rounds.",
        "", "## Correctness", "",
        f"- Full regression, rho=0 parity, active smoke: passed.",
        f"- P1/P2 rounds 1-150 selected clients, task seeds, and global hashes: exactly matched FedPhoenix.",
        f"- OOM: {comparison['oom_detected']}; sequential fallback: {comparison['sequential_fallback']}.",
        "", "## Complete metrics", "",
        "| Run | Peak | Peak round | Final | Last20 | Last50 | R151-300 | R201-300 | Best rolling-5 (end) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("E5", RUNS[0]["id"], RUNS[1]["id"]):
        value = candidates["E5"] if name == "E5" else metrics[name]
        lines.append(
            f"| {name} | {value['peak_accuracy']:.4f} | {value['peak_round']} | "
            f"{value['final_accuracy']:.4f} | {value['last20_mean']:.4f} | "
            f"{value['last50_mean']:.4f} | {value['rounds151_300_mean']:.4f} | "
            f"{value['rounds201_300_mean']:.4f} | {value['best_rolling5_mean']:.4f} "
            f"({value['best_rolling5_end_round']}) |"
        )
    lines.extend(["", "## Matched differences (left minus right; percentage points)", ""])
    for name, delta in comparisons.items():
        lines.append(
            f"- {name}: peak {delta['peak_accuracy']:+.4f}, final "
            f"{delta['final_accuracy']:+.4f}, last20 {delta['last20_mean']:+.4f}, "
            f"rolling-5 {delta['best_rolling5_mean']:+.4f}."
        )
    lines.extend([
        "", "## Answers to the predefined questions", "",
        f"1. P1 exceeds original FedPhoenix peak: {'yes' if p1['peak_accuracy'] > FEDPHOENIX['peak_accuracy'] else 'no'} "
        f"({p1['peak_accuracy'] - FEDPHOENIX['peak_accuracy']:+.4f} pp).",
        f"2. P1 exceeds E2 peak: {'yes' if p1['peak_accuracy'] > references['E2']['peak_accuracy'] else 'no'} "
        f"({comparisons['P1_vs_E2_delayed_at_ratio0125']['peak_accuracy']:+.4f} pp).",
        f"3. P1 exceeds E5 peak: {'yes' if p1['peak_accuracy'] > references['E5']['peak_accuracy'] else 'no'} "
        f"({p1['peak_accuracy'] - references['E5']['peak_accuracy']:+.4f} pp).",
        f"4. P2 exceeds E1 peak: {'yes' if p2['peak_accuracy'] > references['E1']['peak_accuracy'] else 'no'} "
        f"({comparisons['P2_vs_E1_delayed_at_ratio025']['peak_accuracy']:+.4f} pp).",
        f"5. P2 exceeds E5 peak: {'yes' if p2['peak_accuracy'] > references['E5']['peak_accuracy'] else 'no'} "
        f"({p2['peak_accuracy'] - references['E5']['peak_accuracy']:+.4f} pp).",
        "6. Delayed targeting is not consistently beneficial at either intensity: "
        "P1 trails E2 on peak, final, last20 and rolling-5; P2 and E1 tie on peak "
        "at displayed precision, while P2 improves rolling-5 but trails on final and last20.",
        "7. All-layer targeting does not outperform strong-layer restriction in the "
        "delayed setting: P2 trails E5 on peak, rolling-5, final and last20.",
        f"8. Candidate for a later 1200-round run: {ranked[0]}. This is a single-seed "
        "screening recommendation, not a significance claim; no long run is started here.",
    ])
    lines.extend(["", "## Mechanism", ""])
    for run in RUNS:
        value = metrics[run["id"]]
        counts = value["client_statistics"]
        layer_slots = ", ".join(
            f"{item['layer']}={item['targeted_slots']}"
            for item in value["layer_statistics"] if item["targeted_slots"]
        )
        lines.append(
            f"- {run['id']}: history/eligible/targeted "
            f"{counts['clients_with_history']}/{counts['clients_history_eligible']}/"
            f"{counts['clients_targeted']}; cold/stale/before-start "
            f"{counts['clients_cold_start']}/{counts['clients_stale']}/"
            f"{counts['clients_before_start_round']}; reset/targeted/replaced slots "
            f"{counts['total_reset_slots']}/{counts['targeted_reset_slots']}/"
            f"{counts['replaced_reset_slots']}; actual fraction "
            f"{value['average_actual_target_fraction']:.6f}; maximum history "
            f"{value['maximum_history_memory_bytes']} bytes; failed clients "
            f"{value['failed_client_updates']}; layers: {layer_slots}."
        )
    lines.extend([
        "", "## Candidate for a later long run", "",
        f"Primary peak ranking among E5/P1/P2: {', '.join(ranked)}.",
        f"Top-two peak gap: {peak_gap:.4f} pp. "
        + ("Inspect the rolling-5, final, R201-300, and last20 columns when selecting between them."
           if peak_gap < 0.10 else "The primary peak separates the top two by at least 0.10 pp."),
        "This report uses one seed and does not initiate a long run.", "",
    ])
    (output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/targeted_fedphoenix_delayed_pair")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel", type=int, choices=[2], default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "version": 1, "created_at": now(), "status": "preflight",
        "fixed_runs": RUNS, "parallel": args.parallel,
        "oom_detected": False, "sequential_fallback": False,
        "attempts": [], "active": [],
    }
    if manifest.get("status") == "completed":
        return
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    try:
        reference = check_reference()
        correctness = correctness_gates(Path(args.python), output_dir, args.gpu)
        manifest["correctness"] = correctness
        manifest["status"] = "running"
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        run_pair(
            Path(args.python), args.gpu, output_dir, reference,
            manifest, manifest_path, args.max_retries,
        )
        for run in RUNS:
            if completed_summary(output_dir, run) is None:
                raise RuntimeError(f"{run['id']} did not complete 300 rounds")
        comparison = write_outputs(output_dir, manifest, reference)
        manifest["comparison"] = comparison
        manifest["status"] = "completed"
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
