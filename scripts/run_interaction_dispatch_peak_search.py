#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resumable, serial peak-accuracy search for InteractionDispatch.

The launcher varies only ``id_tau`` and ``id_max_step_ratio``.  Every child
process uses the fixed FL protocol declared in ``COMMON_ARGUMENTS``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
REASONS = (
    "cold_start",
    "no_observation",
    "no_global_direction",
    "no_need",
    "no_support",
    "active",
)
SUCCESS_STATUSES = {"completed", "reused"}
PEAK_PATTERN = re.compile(
    r"PEAK_ACCURACY\s+method=(\S+)\s+round=(\d+)\s+accuracy=([0-9.eE+-]+)"
)
ROUND_PATTERN = re.compile(
    r"ROUND_ACCURACY\s+method=(\S+)\s+round=(\d+)\s+accuracy=([0-9.eE+-]+)"
)

COMMON_ARGUMENTS = {
    "dataset": "cifar10",
    "model": "resnet18",
    "num_users": 100,
    "frac": 0.1,
    "local_ep": 5,
    "local_bs": 50,
    "bs": 256,
    "optimizer": "sgd",
    "lr": 0.01,
    "momentum": 0.5,
    "weight_decay": 0,
    "iid": 0,
    "noniid_case": 5,
    "data_beta": 0.3,
    "num_classes": 10,
    "num_channels": 3,
    "generate_data": 0,
    "num_workers": 0,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="results/interaction_dispatch_peak_search",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--stop_after_phase",
        choices=("diagnostic", "screening", "promotion", "full", "multiseed"),
        default="multiseed",
    )
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--skip_fedphoenix_full", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "common_protocol": COMMON_ARGUMENTS,
        "runs": {},
    }


def process_exists(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def sanitize_number(value):
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def run_id_for(config):
    pieces = [
        config["phase"],
        config["algorithm"].lower(),
        f"e{config['epochs']}",
        f"s{config['seed']}",
    ]
    if config["algorithm"] == "InteractionDispatch":
        pieces.extend(
            [
                f"tau{sanitize_number(config['id_tau'])}",
                f"ratio{sanitize_number(config['id_max_step_ratio'])}",
            ]
        )
    return "_".join(pieces)


def make_config(
    phase,
    algorithm,
    epochs,
    seed=1,
    id_tau=0.0,
    id_max_step_ratio=0.0,
):
    config = {
        "phase": phase,
        "algorithm": algorithm,
        "epochs": int(epochs),
        "seed": int(seed),
        "id_tau": float(id_tau),
        "id_max_step_ratio": float(id_max_step_ratio),
    }
    config["run_id"] = run_id_for(config)
    return config


def build_command(config, cli, run_directory):
    command = [
        str(Path(cli.python).resolve()),
        str(REPO_ROOT / "main_fed.py"),
        "--algorithm",
        config["algorithm"],
        "--epochs",
        str(config["epochs"]),
        "--seed",
        str(config["seed"]),
        "--gpu",
        str(cli.gpu),
        "--run_name",
        config["run_id"],
        "--metrics_log_dir",
        str(run_directory / "metrics"),
    ]
    for name, value in COMMON_ARGUMENTS.items():
        command.extend([f"--{name}", str(value)])
    if config["algorithm"] == "InteractionDispatch":
        command.extend(
            [
                "--id_tau",
                repr(config["id_tau"]),
                "--id_max_step_ratio",
                repr(config["id_max_step_ratio"]),
            ]
        )
    return command


def parse_stdout(path):
    peak_accuracy = None
    peak_round = None
    round_accuracies = []
    if not path.exists():
        return peak_accuracy, peak_round, round_accuracies
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            peak_match = PEAK_PATTERN.search(line)
            if peak_match:
                peak_round = int(peak_match.group(2))
                peak_accuracy = float(peak_match.group(3))
            round_match = ROUND_PATTERN.search(line)
            if round_match:
                round_accuracies.append(float(round_match.group(3)))
    return peak_accuracy, peak_round, round_accuracies


def find_metrics_csv(run_directory, algorithm):
    candidates = sorted((run_directory / "metrics").glob("*.csv"))
    matching = [path for path in candidates if algorithm in path.name]
    return matching[0] if len(matching) == 1 else None


def parse_interaction_metrics(metrics_path):
    csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
    reason_counts = Counter({reason: 0 for reason in REASONS})
    active_alphas = []
    positive_conflicts = []
    finite_support_count = 0
    support_positive_count = 0
    event_count = 0
    maximum_history_memory = 0
    final_history_memory = 0
    maximum_clients_with_observation = 0
    final_clients_with_observation = 0
    accuracies = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            accuracies.append(float(row["test_accuracy"]))
            final_history_memory = int(row.get("history_memory_bytes", 0) or 0)
            maximum_history_memory = max(
                maximum_history_memory, final_history_memory
            )
            final_clients_with_observation = int(
                row.get("num_clients_with_observation", 0) or 0
            )
            maximum_clients_with_observation = max(
                maximum_clients_with_observation,
                final_clients_with_observation,
            )
            events = json.loads(row.get("dispatch_events", "[]"))
            for event in events:
                event_count += 1
                reason = event.get("reason")
                reason_counts[reason] += 1
                alpha = float(event.get("alpha", 0.0) or 0.0)
                if reason == "active":
                    active_alphas.append(alpha)
                conflict = event.get("conflict")
                if (
                    event.get("has_observation")
                    and conflict is not None
                    and math.isfinite(float(conflict))
                    and float(conflict) > 0.0
                ):
                    positive_conflicts.append(float(conflict))
                support = event.get("support")
                if support is not None and math.isfinite(float(support)):
                    finite_support_count += 1
                    if float(support) > 0.0:
                        support_positive_count += 1
    peak_accuracy = max(accuracies) if accuracies else None
    peak_round = accuracies.index(peak_accuracy) + 1 if accuracies else None
    return {
        "peak_accuracy": peak_accuracy,
        "peak_round": peak_round,
        "final_accuracy": accuracies[-1] if accuracies else None,
        "completed_rounds": len(accuracies),
        "dispatch_event_count": event_count,
        "active_count": int(reason_counts["active"]),
        "active_rate": (
            float(reason_counts["active"] / event_count) if event_count else 0.0
        ),
        "mean_alpha": (
            float(statistics.fmean(active_alphas)) if active_alphas else 0.0
        ),
        "max_alpha": max(active_alphas, default=0.0),
        "reason_counts": dict(reason_counts),
        "clients_with_observation": final_clients_with_observation,
        "maximum_clients_with_observation": maximum_clients_with_observation,
        "final_history_memory_bytes": final_history_memory,
        "maximum_history_memory_bytes": maximum_history_memory,
        "positive_conflicts": positive_conflicts,
        "finite_support_count": finite_support_count,
        "support_positive_count": support_positive_count,
    }


def is_memory_failure(stdout_path, stderr_path):
    patterns = (
        "out of memory",
        "memoryerror",
        "cannot allocate memory",
        "not enough memory",
        "cudnn_status_alloc_failed",
    )
    for path in (stdout_path, stderr_path):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lowered = line.lower()
                if any(pattern in lowered for pattern in patterns):
                    return True
    return False


def compact_result(entry):
    result = {
        "run_id": entry["run_id"],
        "phase": entry["phase"],
        "algorithm": entry["algorithm"],
        "epochs": entry["epochs"],
        "seed": entry["seed"],
        "id_tau": entry.get("id_tau"),
        "id_max_step_ratio": entry.get("id_max_step_ratio"),
        "status": entry.get("status"),
        "return_code": entry.get("return_code"),
        "runtime_seconds": entry.get("runtime_seconds"),
        "peak_accuracy": entry.get("peak_accuracy"),
        "peak_round": entry.get("peak_round"),
        "final_accuracy": entry.get("final_accuracy"),
        "completed_rounds": entry.get("completed_rounds"),
        "active_count": entry.get("active_count", 0),
        "active_rate": entry.get("active_rate", 0.0),
        "mean_alpha": entry.get("mean_alpha", 0.0),
        "max_alpha": entry.get("max_alpha", 0.0),
        "degenerated_to_passive": entry.get("active_count", 0) == 0,
        "clients_with_observation": entry.get("clients_with_observation", 0),
        "maximum_history_memory_bytes": entry.get(
            "maximum_history_memory_bytes", 0
        ),
        "reason_counts": entry.get("reason_counts", {}),
        "stdout_log": entry.get("stdout_log"),
        "stderr_log": entry.get("stderr_log"),
    }
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = []
    for row in rows:
        item = dict(row)
        item["reason_counts"] = json.dumps(
            item.get("reason_counts", {}), sort_keys=True
        )
        flattened.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)


def update_global_summary(output_dir, manifest):
    rows = [compact_result(entry) for entry in manifest["runs"].values()]
    rows.sort(key=lambda row: (row["phase"], row["run_id"]))
    write_csv(output_dir / "summary.csv", rows)


def run_experiment(config, cli, output_dir, manifest, manifest_path):
    run_id = config["run_id"]
    existing = manifest["runs"].get(run_id)
    if existing and existing.get("status") in SUCCESS_STATUSES:
        return existing
    if (
        existing
        and existing.get("status") in {"failed", "failed_memory"}
        and not cli.retry_failed
    ):
        return existing
    if cli.dry_run:
        print("DRY_RUN", run_id)
        return {**config, "status": "planned"}

    run_directory = output_dir / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "metrics").mkdir(parents=True, exist_ok=True)
    logs_directory = output_dir / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_directory / f"{run_id}.stdout.log"
    stderr_path = logs_directory / f"{run_id}.stderr.log"
    command = build_command(config, cli, run_directory)
    entry = {
        **config,
        "status": "running",
        "command": command,
        "started_at": utc_now(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    manifest["runs"][run_id] = entry
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    update_global_summary(output_dir, manifest)

    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.perf_counter()
    process = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            entry["pid"] = process.pid
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            return_code = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        entry["status"] = "interrupted"
        entry["runtime_seconds"] = time.perf_counter() - started
        entry["finished_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        update_global_summary(output_dir, manifest)
        raise

    entry["return_code"] = int(return_code)
    entry["runtime_seconds"] = time.perf_counter() - started
    entry["finished_at"] = utc_now()
    peak_accuracy, peak_round, stdout_accuracies = parse_stdout(stdout_path)
    entry["peak_accuracy"] = peak_accuracy
    entry["peak_round"] = peak_round
    entry["final_accuracy"] = (
        stdout_accuracies[-1] if stdout_accuracies else None
    )
    entry["completed_rounds"] = len(stdout_accuracies)

    metrics_path = find_metrics_csv(run_directory, config["algorithm"])
    if metrics_path is not None:
        entry["metrics_csv"] = str(metrics_path)
    if config["algorithm"] == "InteractionDispatch" and metrics_path is not None:
        metrics = parse_interaction_metrics(metrics_path)
        positive_conflicts = metrics.pop("positive_conflicts")
        entry.update(metrics)
        if config["phase"] == "diagnostic":
            entry["positive_conflicts"] = positive_conflicts

    expected_complete = entry.get("completed_rounds") == config["epochs"]
    if return_code == 0 and expected_complete:
        entry["status"] = "completed"
    elif is_memory_failure(stdout_path, stderr_path):
        entry["status"] = "failed_memory"
    else:
        entry["status"] = "failed"
    entry.pop("pid", None)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    update_global_summary(output_dir, manifest)
    return entry


def quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stage_rows(manifest, phase):
    return [
        compact_result(entry)
        for entry in manifest["runs"].values()
        if entry["phase"] == phase
    ]


def rank_rows(rows):
    valid = [
        row
        for row in rows
        if row["status"] in SUCCESS_STATUSES
        and row["peak_accuracy"] is not None
    ]
    return sorted(
        valid,
        key=lambda row: (row["peak_accuracy"], row["final_accuracy"]),
        reverse=True,
    )


def select_with_active_priority(rows, count):
    ranked = rank_rows(rows)
    active = [row for row in ranked if row["active_count"] > 0]
    passive = [row for row in ranked if row["active_count"] == 0]
    selected = active[:count]
    selected.extend(passive[: max(0, count - len(selected))])
    return selected


def write_stage_summary(output_dir, manifest, phase, filename):
    rows = stage_rows(manifest, phase)
    rows = rank_rows(rows) + [
        row for row in rows if row["status"] not in SUCCESS_STATUSES
    ]
    write_csv(output_dir / filename, rows)


def write_best_config(output_dir, payload):
    atomic_json(output_dir / "best_config.json", payload)


def comparable_existing_metrics(algorithm, epochs, seed):
    results_root = REPO_ROOT / "results"
    for config_path in results_root.rglob("*_config.json"):
        if "interaction_dispatch_peak_search" in config_path.parts:
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected = {
            **COMMON_ARGUMENTS,
            "algorithm": algorithm,
            "epochs": epochs,
            "seed": seed,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            continue
        metrics_path = config_path.with_name(
            config_path.name.replace("_config.json", ".csv")
        )
        if metrics_path.exists():
            return metrics_path
    return None


def main():
    cli = parse_args()
    output_dir = (REPO_ROOT / cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    for entry in manifest["runs"].values():
        if entry.get("status") == "running":
            if process_exists(entry.get("pid")):
                raise RuntimeError(
                    f"Run {entry['run_id']} is still active as PID {entry['pid']}"
                )
            entry["status"] = "interrupted"
            entry.pop("pid", None)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    update_global_summary(output_dir, manifest)

    diagnostic_config = make_config(
        "diagnostic", "InteractionDispatch", 120, 1, 0.0, 0.1
    )
    diagnostic_entry = run_experiment(
        diagnostic_config, cli, output_dir, manifest, manifest_path
    )
    if cli.dry_run:
        return
    if diagnostic_entry.get("status") not in SUCCESS_STATUSES:
        raise RuntimeError("Phase 0 diagnostic failed; screening was not started")
    positive_conflicts = diagnostic_entry.get("positive_conflicts", [])
    q25 = quantile(positive_conflicts, 0.25)
    q50 = quantile(positive_conflicts, 0.50)
    q75 = quantile(positive_conflicts, 0.75)
    if len(positive_conflicts) >= 20:
        tau_candidates = sorted(
            {
                float(value)
                for value in (0.0, q25, q50, q75)
                if value is not None and math.isfinite(value) and value >= 0.0
            }
        )
        tau_source = "positive_conflict_quantiles"
    else:
        tau_candidates = [0.0, 0.0005, 0.001, 0.002]
        tau_source = "fallback_insufficient_positive_conflicts"
    finite_support_count = int(diagnostic_entry.get("finite_support_count", 0))
    support_positive_count = int(
        diagnostic_entry.get("support_positive_count", 0)
    )
    diagnostic_summary = {
        "run_id": diagnostic_entry["run_id"],
        "status": diagnostic_entry["status"],
        "positive_conflict_count": len(positive_conflicts),
        "positive_conflict_q25": q25,
        "positive_conflict_q50": q50,
        "positive_conflict_q75": q75,
        "finite_support_count": finite_support_count,
        "support_positive_count": support_positive_count,
        "support_positive_rate": (
            support_positive_count / finite_support_count
            if finite_support_count
            else 0.0
        ),
        "active_count": diagnostic_entry.get("active_count", 0),
        "reason_counts": diagnostic_entry.get("reason_counts", {}),
        "tau_source": tau_source,
        "tau_candidates": tau_candidates,
        "maximum_history_memory_bytes": diagnostic_entry.get(
            "maximum_history_memory_bytes", 0
        ),
    }
    atomic_json(output_dir / "diagnostic_summary.json", diagnostic_summary)
    if cli.stop_after_phase == "diagnostic":
        return

    screening_configs = [
        make_config("screening", "FedAvg", 300, 1)
    ]
    for tau in tau_candidates:
        for ratio in (0.02, 0.05, 0.10, 0.20, 0.40):
            screening_configs.append(
                make_config(
                    "screening", "InteractionDispatch", 300, 1, tau, ratio
                )
            )
    for config in screening_configs:
        run_experiment(config, cli, output_dir, manifest, manifest_path)
    write_stage_summary(
        output_dir, manifest, "screening", "screening_summary.csv"
    )
    screening_rows = [
        row
        for row in stage_rows(manifest, "screening")
        if row["algorithm"] == "InteractionDispatch"
    ]
    if not any(
        row["status"] in SUCCESS_STATUSES and row["active_count"] > 0
        for row in screening_rows
    ):
        write_best_config(
            output_dir,
            {
                "status": "stopped_all_screening_configs_passive",
                "message": (
                    "All completed screening configurations had zero active "
                    "interventions; gates were not modified."
                ),
                "best_screening_run": (
                    rank_rows(screening_rows)[0]
                    if rank_rows(screening_rows)
                    else None
                ),
            },
        )
        return
    if cli.stop_after_phase == "screening":
        return

    promoted = select_with_active_priority(screening_rows, 5)
    promotion_configs = [make_config("promotion", "FedAvg", 600, 1)]
    promotion_configs.extend(
        make_config(
            "promotion",
            "InteractionDispatch",
            600,
            1,
            row["id_tau"],
            row["id_max_step_ratio"],
        )
        for row in promoted
    )
    for config in promotion_configs:
        run_experiment(config, cli, output_dir, manifest, manifest_path)
    write_stage_summary(
        output_dir, manifest, "promotion", "promotion_summary.csv"
    )
    if cli.stop_after_phase == "promotion":
        return

    promotion_rows = [
        row
        for row in stage_rows(manifest, "promotion")
        if row["algorithm"] == "InteractionDispatch" and row["active_count"] > 0
    ]
    full_candidates = rank_rows(promotion_rows)[:3]
    if not full_candidates:
        write_best_config(
            output_dir,
            {
                "status": "stopped_no_active_promotion_config",
                "message": "No active promotion configuration completed successfully.",
            },
        )
        return

    full_configs = [make_config("full", "FedAvg", 1200, 1)]
    full_configs.extend(
        make_config(
            "full",
            "InteractionDispatch",
            1200,
            1,
            row["id_tau"],
            row["id_max_step_ratio"],
        )
        for row in full_candidates
    )
    existing_fedphoenix = comparable_existing_metrics("FedPhoenix", 1200, 1)
    if existing_fedphoenix is None and not cli.skip_fedphoenix_full:
        full_configs.append(make_config("full", "FedPhoenix", 1200, 1))
    for config in full_configs:
        run_experiment(config, cli, output_dir, manifest, manifest_path)
    write_stage_summary(output_dir, manifest, "full", "full_summary.csv")
    if cli.stop_after_phase == "full":
        return

    full_rows = [
        row
        for row in stage_rows(manifest, "full")
        if row["algorithm"] == "InteractionDispatch"
        and row["status"] in SUCCESS_STATUSES
        and row["active_count"] > 0
    ]
    best_two = rank_rows(full_rows)[:2]
    for row in best_two:
        for seed in (2, 3):
            config = make_config(
                "multiseed",
                "InteractionDispatch",
                1200,
                seed,
                row["id_tau"],
                row["id_max_step_ratio"],
            )
            run_experiment(config, cli, output_dir, manifest, manifest_path)

    multi_rows = [
        row
        for row in stage_rows(manifest, "multiseed")
        if row["algorithm"] == "InteractionDispatch"
        and row["status"] in SUCCESS_STATUSES
    ]
    grouped = {}
    for row in full_rows + multi_rows:
        key = (row["id_tau"], row["id_max_step_ratio"])
        grouped.setdefault(key, {})[int(row["seed"])] = row
    aggregate_rows = []
    for (tau, ratio), by_seed in grouped.items():
        if not all(seed in by_seed for seed in (1, 2, 3)):
            continue
        peak_values = [by_seed[seed]["peak_accuracy"] for seed in (1, 2, 3)]
        final_values = [by_seed[seed]["final_accuracy"] for seed in (1, 2, 3)]
        aggregate_rows.append(
            {
                "id_tau": tau,
                "id_max_step_ratio": ratio,
                "peak_accuracy_mean": statistics.fmean(peak_values),
                "peak_accuracy_std": statistics.pstdev(peak_values),
                "peak_accuracy_each_seed": json.dumps(
                    {seed: by_seed[seed]["peak_accuracy"] for seed in (1, 2, 3)}
                ),
                "peak_round_each_seed": json.dumps(
                    {seed: by_seed[seed]["peak_round"] for seed in (1, 2, 3)}
                ),
                "final_accuracy_mean": statistics.fmean(final_values),
                "active_count_each_seed": json.dumps(
                    {seed: by_seed[seed]["active_count"] for seed in (1, 2, 3)}
                ),
            }
        )
    aggregate_rows.sort(
        key=lambda row: (row["peak_accuracy_mean"], row["final_accuracy_mean"]),
        reverse=True,
    )
    write_csv(output_dir / "multiseed_summary.csv", aggregate_rows)

    all_full_id_rows = rank_rows(full_rows + multi_rows)
    best_single = all_full_id_rows[0] if all_full_id_rows else None
    best_mean = aggregate_rows[0] if aggregate_rows else None
    write_best_config(
        output_dir,
        {
            "status": "completed",
            "ranking_rule": ["peak_accuracy", "final_accuracy"],
            "best_single_run_peak": best_single,
            "best_mean_peak_across_seeds": best_mean,
            "comparable_existing_fedphoenix_metrics": (
                str(existing_fedphoenix) if existing_fedphoenix else None
            ),
        },
    )


if __name__ == "__main__":
    main()
