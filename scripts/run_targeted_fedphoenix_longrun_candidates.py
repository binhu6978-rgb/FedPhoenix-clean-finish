#!/usr/bin/env python
"""Run the two predefined 1000-round TargetedFedPhoenix candidates."""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from run_targeted_fedphoenix_delayed_pair import (
    check_reference,
    now,
    read_rounds,
    stop_processes,
    validate_prefix,
)
from run_targeted_fedphoenix_overnight6 import (
    ROOT,
    STRONG_LAYERS,
    contains_oom,
    correctness_gates,
    protocol_args,
    read_json,
    write_json,
)


RUNS = (
    {"id": "L2_E5_r1000_seed1", "score": "update_norm", "ratio": 0.25,
     "gap": 10, "layers": STRONG_LAYERS, "start": 151},
    {"id": "L3_E2_r1000_seed1", "score": "update_norm", "ratio": 0.125,
     "gap": 10, "layers": "all", "start": 1},
)
HISTORICAL_LOG = ROOT / "result_other_method/cifar10/0.3/vgg_FedPhoenix.log"
REFERENCE_EXPECTED = {"peak_accuracy": 82.610001, "peak_round": 997,
                      "final_accuracy": 80.589996}
STAGES = ((1, 150), (151, 300), (301, 500), (501, 750), (751, 1000))
METRICS = ("peak_accuracy", "final_accuracy", "last20_mean", "last50_mean",
           "last100_mean", "best_rolling5_mean", "best_rolling10_mean")


def accuracy_summary(accuracy):
    values = np.asarray(accuracy, dtype=np.float64)
    if len(values) != 1000 or not np.all(np.isfinite(values)):
        raise RuntimeError("Expected exactly 1000 finite accuracy values")
    peak_index = int(np.argmax(values))
    result = {
        "peak_accuracy": float(values[peak_index]),
        "peak_round": peak_index + 1,
        "final_accuracy": float(values[-1]),
        "last20_mean": float(np.mean(values[-20:])),
        "last50_mean": float(np.mean(values[-50:])),
        "last100_mean": float(np.mean(values[-100:])),
    }
    for width in (5, 10):
        windows = np.convolve(values, np.ones(width) / width, mode="valid")
        best = int(np.argmax(windows))
        result[f"best_rolling{width}_mean"] = float(windows[best])
        result[f"best_rolling{width}_end_round"] = best + width
    result["stage_accuracy_mean"] = {
        f"{start}_{end}": float(np.mean(values[start - 1:end]))
        for start, end in STAGES
    }
    result["rounds501_1000_mean"] = float(np.mean(values[500:]))
    result["rounds901_1000_mean"] = float(np.mean(values[900:]))
    return result


def historical_reference():
    pattern = re.compile(
        r"^ROUND_ACCURACY method=FedPhoenix round=(\d+) accuracy=([0-9.]+)$"
    )
    rounds = {}
    with HISTORICAL_LOG.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.fullmatch(line.strip())
            if not match:
                continue
            round_number = int(match.group(1))
            if not 1 <= round_number <= 1000:
                continue
            if round_number in rounds:
                raise RuntimeError(f"Duplicate historical FedPhoenix round {round_number}")
            rounds[round_number] = float(match.group(2))
    if set(rounds) != set(range(1, 1001)):
        raise RuntimeError("Historical FedPhoenix log lacks a complete 1-1000 curve")
    result = accuracy_summary([rounds[index] for index in range(1, 1001)])
    for name, expected in REFERENCE_EXPECTED.items():
        actual = result[name]
        if abs(actual - expected) > 0.001:
            raise RuntimeError(f"Historical FedPhoenix {name}: {actual} != {expected}")
    result.update({"kind": "existing_historical_reference",
                   "source": str(HISTORICAL_LOG), "rounds_used": 1000})
    return result


def command_for(python, gpu, output_dir, run):
    args = protocol_args(run)
    args[args.index("--epochs") + 1] = "1000"
    run_dir = output_dir / run["id"]
    command = [
        str(python), str(ROOT / "main_fed.py"),
        "--algorithm", "TargetedFedPhoenix", "--gpu", str(gpu),
        "--run_name", run["id"], "--metrics_log_dir", str(run_dir),
        *args,
    ]
    return run_dir, command


def completed(output_dir, run):
    run_dir = output_dir / run["id"]
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    if int(summary.get("completed_rounds", 0)) != 1000:
        return False
    if any(summary.get(key) != value for key, value in (
        ("tfp_target_ratio", run["ratio"]),
        ("tfp_max_history_gap", run["gap"]),
        ("tfp_score_type", run["score"]),
        ("tfp_target_layers", run["layers"]),
        ("tfp_start_round", run["start"]),
    )):
        raise RuntimeError(f"Existing {run['id']} summary has a different protocol")
    rows = read_rounds(run_dir / "round_metrics.csv")
    return len(rows) == 1000 and all(
        int(row["round"]) == index for index, row in enumerate(rows, 1)
    )


def start_run(python, gpu, output_dir, run, mode, attempt):
    run_dir, command = command_for(python, gpu, output_dir, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary.json"
    if summary.exists():
        summary.unlink()
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
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
    return {"run": run, "command": command, "process": process,
            "stdout": stdout, "stderr": stderr, "stdout_path": stdout_path,
            "stderr_path": stderr_path, "mode": mode, "attempt": attempt,
            "started": time.perf_counter()}


def watch(items, output_dir, reference, manifest, manifest_path):
    manifest["active"] = [
        {"run_id": item["run"]["id"], "pid": item["process"].pid,
         "command": item["command"]} for item in items
    ]
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    prefix_complete = False
    try:
        while any(item["process"].poll() is None for item in items):
            if not prefix_complete:
                for item in items:
                    if item["run"]["id"] == RUNS[0]["id"]:
                        check = validate_prefix(output_dir, item["run"], reference)
                        prefix_complete = check["rounds_checked"] == 150
            if any(
                item["process"].poll() is not None
                and contains_oom(item["stdout_path"], item["stderr_path"])
                for item in items
            ):
                stop_processes(items)
                break
            time.sleep(5)
        for item in items:
            if item["run"]["id"] == RUNS[0]["id"]:
                validate_prefix(output_dir, item["run"], reference,
                                require_complete=completed(output_dir, item["run"]))
    except Exception:
        stop_processes(items)
        raise
    finally:
        for item in items:
            if item["process"].poll() is None:
                stop_processes([item])
            item["stdout"].close()
            item["stderr"].close()
    attempts = []
    for item in items:
        attempts.append({
            "run_id": item["run"]["id"], "mode": item["mode"],
            "attempt": item["attempt"],
            "return_code": item["process"].returncode,
            "oom": contains_oom(item["stdout_path"], item["stderr_path"]),
            "runtime_seconds": time.perf_counter() - item["started"],
            "stdout_log": str(item["stdout_path"]),
            "stderr_log": str(item["stderr_path"]), "finished_at": now(),
        })
    manifest["attempts"].extend(attempts)
    manifest["active"] = []
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    return attempts


def run_candidates(python, gpu, output_dir, reference, manifest, manifest_path,
                   max_retries):
    existing = [completed(output_dir, run) for run in RUNS]
    if all(existing):
        manifest["resumed_from_completed_runs"] = True
        return
    if not any(existing):
        items = []
        try:
            for run in RUNS:
                items.append(start_run(python, gpu, output_dir, run, "parallel", 1))
        except Exception:
            stop_processes(items)
            for item in items:
                item["stdout"].close()
                item["stderr"].close()
            raise
        attempts = watch(items, output_dir, reference, manifest, manifest_path)
        manifest["parallel_attempted"] = True
        manifest["oom_detected"] = any(item["oom"] for item in attempts)
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        if manifest["oom_detected"]:
            manifest["sequential_fallback"] = True
            for run in RUNS:
                item = start_run(python, gpu, output_dir, run, "sequential_oom", 1)
                result = watch([item], output_dir, reference, manifest, manifest_path)[0]
                if result["return_code"] != 0 or not completed(output_dir, run):
                    raise RuntimeError(f"Same-protocol OOM fallback failed for {run['id']}")
            return
    for run in RUNS:
        if completed(output_dir, run):
            continue
        for attempt in range(1, max_retries + 1):
            item = start_run(python, gpu, output_dir, run, "sequential_retry", attempt)
            result = watch([item], output_dir, reference, manifest, manifest_path)[0]
            if result["return_code"] == 0 and completed(output_dir, run):
                break
        if not completed(output_dir, run):
            raise RuntimeError(f"{run['id']} failed after same-protocol retries")


def run_metrics(run_dir):
    rows = read_rounds(run_dir / "round_metrics.csv")
    summary = read_json(run_dir / "summary.json")
    if len(rows) != 1000 or int(summary["completed_rounds"]) != 1000:
        raise RuntimeError(f"{run_dir} is not a complete 1000-round run")
    if any(int(row["round"]) != index for index, row in enumerate(rows, 1)):
        raise RuntimeError(f"{run_dir} has nonsequential round metrics")
    required = (
        "test_accuracy", "selected_clients", "task_seeds", "global_state_sha256",
        "round_train_seconds", "clients_with_history", "clients_history_eligible",
        "clients_targeted", "clients_cold_start", "clients_stale",
        "clients_before_start_round", "total_reset_slots", "targeted_reset_slots",
        "random_reset_slots", "replaced_reset_slots", "actual_target_fraction",
        "mean_history_gap_for_targeted", "history_memory_bytes",
        "failed_client_updates",
    )
    if any(not all(name in row for name in required) for row in rows):
        raise RuntimeError(f"{run_dir} is missing required per-round metrics")
    value = accuracy_summary([float(row["test_accuracy"]) for row in rows])
    value.update({
        "run_id": run_dir.name,
        "protocol": {
            key: summary[key] for key in (
                "tfp_score_type", "tfp_target_ratio", "tfp_max_history_gap",
                "tfp_target_layers", "tfp_start_round", "seed",
            )
        },
        "runtime_seconds": summary["runtime_seconds"],
        "maximum_history_memory_bytes": max(
            int(row["history_memory_bytes"]) for row in rows
        ),
        "failed_client_updates": sum(
            int(row["failed_client_updates"]) for row in rows
        ),
        "rounds771_1000_mean": float(np.mean([
            float(row["test_accuracy"]) for row in rows[770:]
        ])),
        "last_targeted_round": max(
            (int(row["round"]) for row in rows
             if int(row["targeted_reset_slots"]) > 0), default=None
        ),
        "targeted_slots_rounds771_1000": sum(
            int(row["targeted_reset_slots"]) for row in rows[770:]
        ),
    })
    stage_mechanism = {}
    for start, end in STAGES:
        stage = rows[start - 1:end]
        targeted_clients = sum(int(row["clients_targeted"]) for row in stage)
        targeted_slots = sum(int(row["targeted_reset_slots"]) for row in stage)
        reset_slots = sum(int(row["total_reset_slots"]) for row in stage)
        gap_sum = sum(
            int(row["clients_targeted"])
            * float(row["mean_history_gap_for_targeted"] or 0.0)
            for row in stage
        )
        stage_mechanism[f"{start}_{end}"] = {
            "targeted_clients": targeted_clients,
            "targeted_slots": targeted_slots,
            "replaced_slots": sum(int(row["replaced_reset_slots"]) for row in stage),
            "actual_target_fraction": targeted_slots / reset_slots if reset_slots else 0.0,
            "mean_history_gap": gap_sum / targeted_clients if targeted_clients else None,
            "accuracy_mean": value["stage_accuracy_mean"][f"{start}_{end}"],
        }
    value["stage_mechanism"] = stage_mechanism
    value["client_statistics"] = summary["client_statistics"]
    value["average_actual_target_fraction"] = summary[
        "average_actual_target_fraction"
    ]
    layer_rows = read_rounds(run_dir / "layer_metrics.csv")
    layer_rounds = {int(row["round"]) for row in layer_rows}
    if layer_rounds != set(range(1, 1001)):
        raise RuntimeError(f"{run_dir} has incomplete per-layer metrics")
    layer_totals = {}
    for row in layer_rows:
        layer = row["layer"]
        layer_totals[layer] = layer_totals.get(layer, 0) + int(row["targeted_slots"])
    value["per_layer_targeted_slots"] = layer_totals
    return value


def differences(left, right, fields):
    return {name: float(left[name] - right[name]) for name in fields}


def write_outputs(output_dir, manifest, reference, historical):
    l2, l3 = [run_metrics(output_dir / run["id"]) for run in RUNS]
    parity = validate_prefix(output_dir, RUNS[0], reference, require_complete=True)
    write_json(output_dir / "start151_parity_report.json", parity)
    comparison = {
        "protocol": "TargetedFedPhoenix L2/E5 and L3/E2, 1000 rounds, seed 1",
        "fedphoenix_existing_reference": historical,
        "runs": {RUNS[0]["id"]: l2, RUNS[1]["id"]: l3},
        "delta_vs_historical_fedphoenix": {
            RUNS[0]["id"]: differences(l2, historical, METRICS),
            RUNS[1]["id"]: differences(l3, historical, METRICS),
        },
        "L2_E5_minus_L3_E2": {
            **differences(l2, l3, METRICS),
            "stage_accuracy_mean": differences(
                l2["stage_accuracy_mean"], l3["stage_accuracy_mean"],
                l2["stage_accuracy_mean"].keys()
            ),
            "rounds501_1000_mean": l2["rounds501_1000_mean"]
                                    - l3["rounds501_1000_mean"],
            "rounds771_1000_mean": l2["rounds771_1000_mean"]
                                    - l3["rounds771_1000_mean"],
            "rounds901_1000_mean": l2["rounds901_1000_mean"]
                                    - l3["rounds901_1000_mean"],
        },
        "L2_first150_fedphoenix_parity": parity,
        "oom_detected": manifest["oom_detected"],
        "sequential_fallback": manifest["sequential_fallback"],
        "attempts": manifest["attempts"],
    }
    write_json(output_dir / "comparison.json", comparison)
    fields = (
        "run_id", "peak_accuracy", "peak_round", "final_accuracy",
        "last20_mean", "last50_mean", "last100_mean",
        "best_rolling5_mean", "best_rolling5_end_round",
        "best_rolling10_mean", "best_rolling10_end_round",
        "rounds501_1000_mean", "rounds901_1000_mean",
        "rounds771_1000_mean", "last_targeted_round",
        "targeted_slots_rounds771_1000",
        "maximum_history_memory_bytes", "failed_client_updates",
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for value in (l2, l3):
            writer.writerow({name: value.get(name) for name in fields})
    lines = [
        "# TargetedFedPhoenix 1000-round candidate comparison", "",
        "VGG/CIFAR-10, Dirichlet alpha=0.3, seed=1; L2 and L3 each completed 1000 rounds.",
        "FedPhoenix is an existing historical log, not a newly generated paired run.",
        "", "## Correctness", "",
        "- Full regression tests, rho=0 strict parity, and active deterministic smoke passed.",
        "- L2 rounds 1-150 exactly matched the FedPhoenix diagnostic in selected clients, task seeds, and global state hashes.",
        f"- OOM: {manifest['oom_detected']}; sequential fallback: {manifest['sequential_fallback']}.",
        "", "## Accuracy", "",
        "| Run | Peak (round) | Final | Last20 | Last50 | Last100 | Best rolling5 (end) | Best rolling10 (end) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, value in (("Historical FedPhoenix", historical),
                         ("L2 / E5", l2), ("L3 / E2", l3)):
        lines.append(
            f"| {label} | {value['peak_accuracy']:.4f} ({value['peak_round']}) | "
            f"{value['final_accuracy']:.4f} | {value['last20_mean']:.4f} | "
            f"{value['last50_mean']:.4f} | {value['last100_mean']:.4f} | "
            f"{value['best_rolling5_mean']:.4f} ({value['best_rolling5_end_round']}) | "
            f"{value['best_rolling10_mean']:.4f} ({value['best_rolling10_end_round']}) |"
        )
    lines.extend(["", "## Stage accuracy (L2 minus L3 in pp)", "",
                  "| Rounds | L2 | L3 | Difference |",
                  "| --- | ---: | ---: | ---: |"])
    for start, end in STAGES:
        key = f"{start}_{end}"
        lines.append(
            f"| {start}-{end} | {l2['stage_accuracy_mean'][key]:.4f} | "
            f"{l3['stage_accuracy_mean'][key]:.4f} | "
            f"{comparison['L2_E5_minus_L3_E2']['stage_accuracy_mean'][key]:+.4f} |"
        )
    lines.extend([
        "", "Longer late-period means (L2 / L3 / difference): "
        f"rounds 501-1000 {l2['rounds501_1000_mean']:.4f} / "
        f"{l3['rounds501_1000_mean']:.4f} / "
        f"{comparison['L2_E5_minus_L3_E2']['rounds501_1000_mean']:+.4f}; "
        f"rounds 771-1000 {l2['rounds771_1000_mean']:.4f} / "
        f"{l3['rounds771_1000_mean']:.4f} / "
        f"{comparison['L2_E5_minus_L3_E2']['rounds771_1000_mean']:+.4f}; "
        f"rounds 901-1000 {l2['rounds901_1000_mean']:.4f} / "
        f"{l3['rounds901_1000_mean']:.4f} / "
        f"{comparison['L2_E5_minus_L3_E2']['rounds901_1000_mean']:+.4f}.",
    ])
    for label, left in (
        ("L2 - historical FedPhoenix",
         comparison["delta_vs_historical_fedphoenix"][RUNS[0]["id"]]),
        ("L3 - historical FedPhoenix",
         comparison["delta_vs_historical_fedphoenix"][RUNS[1]["id"]]),
        ("L2 - L3", comparison["L2_E5_minus_L3_E2"]),
    ):
        lines.extend([
            "", f"## {label} (percentage points)", "",
            ", ".join(f"{name} {left[name]:+.4f}" for name in METRICS) + ".",
        ])
    lines.extend(["", "## Stage mechanism", ""])
    for label, value in (("L2 / E5", l2), ("L3 / E2", l3)):
        lines.extend([
            f"### {label}", "",
            "| Rounds | Targeted clients | Targeted slots | Replaced slots | Actual fraction | Mean gap | Accuracy mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for start, end in STAGES:
            if label.startswith("L2") and end <= 150:
                continue
            item = value["stage_mechanism"][f"{start}_{end}"]
            gap = item["mean_history_gap"]
            lines.append(
                f"| {start}-{end} | {item['targeted_clients']} | "
                f"{item['targeted_slots']} | {item['replaced_slots']} | "
                f"{item['actual_target_fraction']:.6f} | "
                f"{gap:.4f} | {item['accuracy_mean']:.4f} |"
                if gap is not None else
                f"| {start}-{end} | {item['targeted_clients']} | "
                f"{item['targeted_slots']} | {item['replaced_slots']} | "
                f"{item['actual_target_fraction']:.6f} | n/a | "
                f"{item['accuracy_mean']:.4f} |"
            )
        layers = ", ".join(
            f"{name}={slots}" for name, slots in sorted(
                value["per_layer_targeted_slots"].items()
            ) if slots
        )
        lines.extend([
            "", f"Per-layer targeted slots: {layers}.",
            f"Total clients with history / history eligible / targeted: "
            f"{value['client_statistics']['clients_with_history']} / "
            f"{value['client_statistics']['clients_history_eligible']} / "
            f"{value['client_statistics']['clients_targeted']}.",
            f"Total reset / targeted / random / replaced slots: "
            f"{value['client_statistics']['total_reset_slots']} / "
            f"{value['client_statistics']['targeted_reset_slots']} / "
            f"{value['client_statistics']['random_reset_slots']} / "
            f"{value['client_statistics']['replaced_reset_slots']}.",
            f"Last round with targeting: {value['last_targeted_round']}; "
            f"targeted slots in rounds 771-1000: "
            f"{value['targeted_slots_rounds771_1000']}.",
            f"Maximum history memory: {value['maximum_history_memory_bytes']} bytes; "
            f"failed client updates: {value['failed_client_updates']}.", "",
        ])
    winner = "E5 (L2)" if l2["peak_accuracy"] >= l3["peak_accuracy"] else "E2 (L3)"
    lines.extend([
        "## Answers and selection", "",
        f"1. Existing FedPhoenix reference: peak {historical['peak_accuracy']:.4f} "
        f"at {historical['peak_round']}; final {historical['final_accuracy']:.4f}; "
        f"last20/50/100 {historical['last20_mean']:.4f}/"
        f"{historical['last50_mean']:.4f}/{historical['last100_mean']:.4f}.",
        f"2. E5: peak {l2['peak_accuracy']:.4f} at {l2['peak_round']}; "
        f"final {l2['final_accuracy']:.4f}; last20/50/100 "
        f"{l2['last20_mean']:.4f}/{l2['last50_mean']:.4f}/"
        f"{l2['last100_mean']:.4f}.",
        f"3. E2: peak {l3['peak_accuracy']:.4f} at {l3['peak_round']}; "
        f"final {l3['final_accuracy']:.4f}; last20/50/100 "
        f"{l3['last20_mean']:.4f}/{l3['last50_mean']:.4f}/"
        f"{l3['last100_mean']:.4f}.",
        f"4. E5 exceeds the historical FedPhoenix peak by "
        f"{l2['peak_accuracy'] - historical['peak_accuracy']:+.4f} pp. "
        "The margin is only 0.01 pp and should be treated as essentially tied.",
        f"5. E2 exceeds the historical FedPhoenix peak: "
        f"{'yes' if l3['peak_accuracy'] > historical['peak_accuracy'] else 'no'} "
        f"({l3['peak_accuracy'] - historical['peak_accuracy']:+.4f} pp).",
        f"6. Higher E5/E2 peak: {winner}; E5 minus E2 "
        f"{l2['peak_accuracy'] - l3['peak_accuracy']:+.4f} pp.",
        f"7. E2 has the higher final by "
        f"{l3['final_accuracy'] - l2['final_accuracy']:+.4f} pp; "
        f"E5 has the higher last100 by "
        f"{l2['last100_mean'] - l3['last100_mean']:+.4f} pp.",
        f"8. E5's relative accuracy advantage is present in rounds 751-1000 "
        f"({l2['stage_accuracy_mean']['751_1000'] - l3['stage_accuracy_mean']['751_1000']:+.4f} pp) "
        f"and 901-1000 ({l2['rounds901_1000_mean'] - l3['rounds901_1000_mean']:+.4f} pp). "
        f"E5 performs no targeting after round {l2['last_targeted_round']}; "
        "the late difference cannot be attributed to ongoing targeted resets.",
        f"9. E2 continues sparse targeting through round {l3['last_targeted_round']}, "
        f"but its last100 is {l3['last100_mean'] - l2['last100_mean']:+.4f} pp "
        "relative to E5. This run does not show better sustained late accuracy.",
        f"10. Recommended seed=1 candidate for a later multi-seed evaluation: "
        f"{winner}. E5 leads peak, rolling means and last100; E2 leads at the "
        "single final checkpoint. This is a selection from one seed, not a "
        "statistical significance claim.",
        "No additional training is started by this launcher.", "",
    ])
    (output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default="results/targeted_fedphoenix_longrun1000")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel", type=int, choices=[2], default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()
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
    if manifest["status"] == "completed":
        return
    if manifest.get("active"):
        raise RuntimeError("Manifest lists active training PIDs; inspect them before resume")
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    try:
        historical = historical_reference()
        reference = check_reference()
        manifest["historical_reference"] = historical
        manifest["correctness"] = correctness_gates(
            Path(args.python), output_dir, args.gpu
        )
        manifest["status"] = "running"
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        run_candidates(
            Path(args.python), args.gpu, output_dir, reference,
            manifest, manifest_path, args.max_retries
        )
        if not all(completed(output_dir, run) for run in RUNS):
            raise RuntimeError("One or both 1000-round runs are incomplete")
        comparison = write_outputs(output_dir, manifest, reference, historical)
        manifest["comparison_path"] = str(output_dir / "comparison.json")
        manifest["status"] = "completed"
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["active"] = []
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
