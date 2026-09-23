#!/usr/bin/env python
"""Freeze, gate, run and summarize the four predefined 1000-round arms.

No hyperparameter search. B0/H run first (H consumes B0's frozen score bank),
then S/P. Failed, incomplete run directories are archived before a safe retry.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from run_targeted_fedphoenix_overnight6 import (
    ROOT, STRONG_LAYERS, contains_oom, correctness_gates, protocol_args,
)

sys.path.insert(0, str(ROOT))
from Algorithm.HistoryIdentityCausal import build_matching_schedule


RUNS = (
    ("B0", "HistoryIdentityB0"),
    ("H_same", "HistoryIdentitySame"),
    ("S_shuffled", "HistoryIdentityShuffled"),
    ("P_population", "HistoryIdentityPopulation"),
)
SOURCE = ROOT / "results/targeted_fedphoenix_longrun1000/L2_E5_r1000_seed1/round_metrics.csv"
ORIGINAL_E5_TRACES = ROOT / "results/targeted_fedphoenix_longrun1000/L2_E5_r1000_seed1/task_traces.jsonl"
CURRENT_REFERENCE = ROOT / "results/fedphoenix_specialization_diagnostic/round_metrics.csv"
STAGES = ((1, 150), (151, 300), (301, 500), (501, 750),
          (751, 770), (771, 900), (901, 1000))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_schedule(output_dir):
    path = output_dir / "matching_schedule.json"
    source_rows = read_csv(SOURCE)
    if len(source_rows) != 1000:
        raise RuntimeError("frozen selected-client source lacks 1000 rounds")
    expected = {
        "version": 1,
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
        "seed": 1,
        "max_history_gap": 10,
        "start_round": 151,
        "rule": "exact-age cohort, SHA256 nonzero cyclic derangement; all peers for common validity",
        "rounds": build_matching_schedule(source_rows, seed=1,
                                           max_gap=10, start_round=151),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError("preexisting frozen matching schedule differs")
    else:
        write_json(path, expected)
    eligible = [match for row in expected["rounds"]
                for match in row["matches"].values()
                if 151 <= row["round"] <= 770
                and match["age"] is not None and match["age"] <= 10]
    report = {
        "schedule_sha256": sha256(path),
        "source_sha256": expected["source_sha256"],
        "eligible_client_events_151_770": len(eligible),
        "exact_age_peer_events": sum(match["donor_id"] is not None for match in eligible),
        "no_exact_age_peer_events": sum(match["donor_id"] is None for match in eligible),
        "donor_never_target": all(
            match["donor_id"] is None or match["donor_id"] != int(client)
            for row in expected["rounds"] for client, match in row["matches"].items()
        ),
    }
    write_json(output_dir / "matching_preflight.json", report)
    if not report["donor_never_target"]:
        raise AssertionError("frozen matching contains a self donor")
    return report


def command_for(python, gpu, output_dir, run_name, algorithm, epochs=1000):
    fixed = {"score": "update_norm", "ratio": 0.25, "gap": 10,
             "layers": STRONG_LAYERS, "start": 151}
    args = protocol_args(fixed)
    args[args.index("--epochs") + 1] = str(epochs)
    return [str(python), str(ROOT / "main_fed.py"),
            "--algorithm", algorithm, "--gpu", str(gpu),
            "--run_name", run_name,
            "--metrics_log_dir", str(output_dir / run_name), *args]


def start_run(command, output_dir, run_name, label):
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{run_name}.{label}.stdout.log"
    stderr_path = logs / f"{run_name}.{label}.stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    except Exception:
        stdout.close()
        stderr.close()
        raise
    return {"run_name": run_name, "process": process, "command": command,
            "stdout": stdout, "stderr": stderr,
            "stdout_path": stdout_path, "stderr_path": stderr_path,
            "started": time.monotonic(), "label": label}


def is_complete(output_dir, run_name, epochs=1000):
    run_dir = output_dir / run_name
    summary = run_dir / "summary.json"
    rounds = run_dir / "round_metrics.csv"
    if not summary.exists() or not rounds.exists():
        return False
    value = json.loads(summary.read_text(encoding="utf-8"))
    if value.get("completed_rounds") != epochs or value.get("run_name") != run_name:
        return False
    data = read_csv(rounds)
    if len(data) != epochs or any(int(row["round"]) != index
                                  for index, row in enumerate(data, 1)):
        return False
    if run_name == "B0" and not (run_dir / "score_bank" / f"round{epochs:04d}.npz").exists():
        return False
    return True


def archive_incomplete(output_dir, run_name, force=False):
    path = output_dir / run_name
    if not path.exists() or (not force and is_complete(output_dir, run_name)):
        return
    archive_root = output_dir / "failed_attempts"
    archive_root.mkdir(exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    destination = archive_root / f"{run_name}_{suffix}"
    path.rename(destination)


def watch(items, manifest, manifest_path, output_dir):
    manifest["active"] = [{"run_name": item["run_name"],
                           "pid": item["process"].pid,
                           "command": item["command"]} for item in items]
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    try:
        while any(item["process"].poll() is None for item in items):
            crashed_b0 = next((item for item in items
                               if item["run_name"] == "B0"
                               and item["process"].poll() not in (None, 0)), None)
            oom = any(item["process"].poll() is not None and
                      contains_oom(item["stdout_path"], item["stderr_path"])
                      for item in items)
            if crashed_b0 or oom:
                for item in items:
                    if item["process"].poll() is None:
                        item["process"].terminate()
                for item in items:
                    if item["process"].poll() is None:
                        try:
                            item["process"].wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            item["process"].kill()
                break
            time.sleep(5)
    finally:
        for item in items:
            if item["process"].poll() is None:
                item["process"].terminate()
            item["process"].wait()
            item["stdout"].close()
            item["stderr"].close()
            manifest["attempts"].append({
                "run_name": item["run_name"], "label": item["label"],
                "pid": item["process"].pid,
                "return_code": item["process"].returncode,
                "oom": contains_oom(item["stdout_path"], item["stderr_path"]),
                "runtime_seconds": time.monotonic() - item["started"],
                "stdout_log": str(item["stdout_path"]),
                "stderr_log": str(item["stderr_path"]),
                "finished_at": now(),
            })
        manifest["active"] = []
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)


def run_batch(python, gpu, output_dir, batch, manifest, manifest_path, max_retries):
    if batch[0][0] == "B0" and not is_complete(output_dir, "B0"):
        # A completed H tied to an incomplete B0 cannot be trusted.
        archive_incomplete(output_dir, "H_same", force=True)
    pending = [(name, algorithm) for name, algorithm in batch
               if not is_complete(output_dir, name)]
    if not pending:
        return
    for name, _ in pending:
        archive_incomplete(output_dir, name)
    items = []
    try:
        for name, algorithm in pending:
            items.append(start_run(
                command_for(python, gpu, output_dir, name, algorithm),
                output_dir, name, "parallel1",
            ))
    except Exception:
        for item in items:
            item["process"].terminate()
            item["process"].wait()
            item["stdout"].close()
            item["stderr"].close()
        raise
    watch(items, manifest, manifest_path, output_dir)
    if batch[0][0] == "B0" and not is_complete(output_dir, "B0"):
        archive_incomplete(output_dir, "H_same", force=True)
    if all(is_complete(output_dir, name) for name, _ in batch):
        return
    for attempt in range(1, max_retries + 1):
        for name, algorithm in batch:
            if is_complete(output_dir, name):
                continue
            archive_incomplete(output_dir, name)
            item = start_run(command_for(python, gpu, output_dir, name, algorithm),
                             output_dir, name, f"sequential_retry{attempt}")
            watch([item], manifest, manifest_path, output_dir)
            if not is_complete(output_dir, name):
                if name == "B0":
                    # H may have consumed the failed B0 bank; never reuse it.
                    archive_incomplete(output_dir, "H_same", force=True)
                    break
                continue
        if all(is_complete(output_dir, name) for name, _ in batch):
            return
    missing = [name for name, _ in batch if not is_complete(output_dir, name)]
    raise RuntimeError(f"same-protocol retries exhausted: {missing}")


def correctness_smoke(python, gpu, output_dir):
    smoke_root = output_dir / "smoke"
    smoke_root.mkdir(exist_ok=True)
    previous = output_dir / "identity_smoke_report.json"
    if previous.exists():
        value = json.loads(previous.read_text(encoding="utf-8"))
        if not all(bool(item) for key, item in value.items() if key != "rounds"):
            for run_name in ("B0", "H_same"):
                archive_incomplete(smoke_root, run_name, force=True)
    shutil.copyfile(output_dir / "matching_schedule.json",
                    smoke_root / "matching_schedule.json")
    logs = output_dir / "logs"
    for name, algorithm in RUNS[:2]:
        if is_complete(smoke_root, name, epochs=5):
            continue
        archive_incomplete(smoke_root, name)
        command = command_for(python, gpu, smoke_root, name, algorithm, epochs=5)
        item = start_run(command, smoke_root, name, "smoke")
        item["process"].wait()
        item["stdout"].close()
        item["stderr"].close()
        if item["process"].returncode != 0 or not is_complete(smoke_root, name, 5):
            raise RuntimeError(f"five-round {name} smoke failed: {item['stderr_path']}")
    baseline = read_csv(smoke_root / "B0/round_metrics.csv")
    same = read_csv(smoke_root / "H_same/round_metrics.csv")
    reference = read_csv(CURRENT_REFERENCE)[:5]
    parity_fields = ("selected_clients", "task_seeds", "global_state_sha256")
    parity = {
        "rounds": 5,
        "B0_vs_existing_observer": all(
            baseline[index][field] == reference[index][field]
            for index in range(5) for field in parity_fields
        ),
        "H_vs_B0_prefix": all(
            same[index][field] == baseline[index][field]
            for index in range(5) for field in parity_fields
        ),
        "H_zero_targeted_slots": all(
            int(row["targeted_reset_slots"]) == 0 for row in same
        ),
    }
    write_json(output_dir / "identity_smoke_report.json", parity)
    if not all(value for key, value in parity.items() if key != "rounds"):
        raise RuntimeError("identity five-round parity smoke failed")
    return parity


def accuracy_summary(rows):
    values = [float(row["test_accuracy"]) for row in rows]
    if len(values) != 1000:
        raise ValueError("expected 1000 accuracy values")
    peak_round = max(range(1000), key=values.__getitem__) + 1
    result = {
        "peak": values[peak_round - 1], "peak_round": peak_round,
        "final": values[-1], "last20": mean(values[-20:]),
        "last50": mean(values[-50:]), "last100": mean(values[-100:]),
        "mechanism_151_770_mean": mean(values[150:770]),
        "longterm_901_1000_mean": mean(values[900:1000]),
        "stage_means": {f"{a}_{b}": mean(values[a - 1:b]) for a, b in STAGES},
    }
    for window in (5, 10):
        rolling = [mean(values[index:index + window])
                   for index in range(1001 - window)]
        best = max(range(len(rolling)), key=rolling.__getitem__)
        result[f"best_rolling{window}"] = rolling[best]
        result[f"best_rolling{window}_end_round"] = best + window
    for field in ("total_reset_slots", "targeted_reset_slots", "replaced_reset_slots",
                  "targeted_clients", "fallback_clients", "fallback_client_layers"):
        result[field] = sum(int(row[field]) for row in rows)
    result["max_history_memory_bytes"] = max(int(row["history_memory_bytes"])
                                             for row in rows)
    result["failed_client_updates"] = max(int(row["failed_client_updates"])
                                          for row in rows)
    return result


def original_e5_delta(controlled_trace_path):
    def read_traces(path):
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                yield (value["round"], value["task_id"], value["client_id"]), value["trace"]
    original = dict(read_traces(ORIGINAL_E5_TRACES))
    controlled = dict(read_traces(controlled_trace_path))
    if set(original) != set(controlled):
        raise RuntimeError("controlled H client/task schedule differs from original E5")
    changed_eligibility = 0
    changed_target_slot_events = 0
    absolute_target_slot_delta = 0
    changed_reset_client_layers = 0
    changed_reset_indices = 0
    layer_comparisons = 0
    original_targeted_slots = 0
    controlled_targeted_slots = 0
    for key, trace in controlled.items():
        old = original[key]
        old_targeted = int(old["targeted_reset_slots"])
        new_targeted = int(trace["targeted_reset_slots"])
        original_targeted_slots += old_targeted
        controlled_targeted_slots += new_targeted
        changed_eligibility += (old_targeted > 0) != (new_targeted > 0)
        changed_target_slot_events += old_targeted != new_targeted
        absolute_target_slot_delta += abs(old_targeted - new_targeted)
        old_layers = {layer["name"]: layer for layer in old["layers"]}
        for layer in trace["layers"]:
            old_set = set(old_layers[layer["name"]]["final_reset_indices"])
            new_set = set(layer["final_reset_indices"])
            changed_reset_client_layers += old_set != new_set
            changed_reset_indices += len(old_set.symmetric_difference(new_set)) // 2
            layer_comparisons += 1
    return {
        "compared_client_events": len(controlled),
        "eligibility_changed_client_events": changed_eligibility,
        "targeted_slot_changed_client_events": changed_target_slot_events,
        "absolute_targeted_slot_difference": absolute_target_slot_delta,
        "original_targeted_slots": original_targeted_slots,
        "controlled_targeted_slots": controlled_targeted_slots,
        "net_targeted_slot_difference": controlled_targeted_slots - original_targeted_slots,
        "final_reset_location_changed_client_layers": changed_reset_client_layers,
        "final_reset_location_changed_fraction": changed_reset_client_layers / layer_comparisons,
        "sum_per_layer_reset_replacements": changed_reset_indices,
        "compared_client_layers": layer_comparisons,
    }


def summarize(output_dir, manifest):
    round_rows = {name: read_csv(output_dir / name / "round_metrics.csv")
                  for name, _ in RUNS}
    for index in range(1000):
        chosen = {(round_rows[name][index]["selected_clients"],
                   round_rows[name][index]["task_seeds"])
                  for name, _ in RUNS}
        if len(chosen) != 1:
            raise RuntimeError(f"selected clients/task seeds differ at round {index + 1}")
        targeted = [round_rows[name][index] for name in
                    ("H_same", "S_shuffled", "P_population")]
        for field in ("total_reset_slots", "targeted_reset_slots",
                      "targeted_clients", "source_support_sha256"):
            if len({row[field] for row in targeted}) != 1:
                raise RuntimeError(f"controlled arms differ in {field} at round {index + 1}")
        if index < 150 and len({round_rows[name][index]["global_state_sha256"]
                                for name, _ in RUNS}) != 1:
            raise RuntimeError(f"first-150 global hash parity failed at round {index + 1}")
    trace_paths = [output_dir / name / "task_traces.jsonl" for name in
                   ("H_same", "S_shuffled", "P_population")]
    with trace_paths[0].open(encoding="utf-8") as h_handle, \
            trace_paths[1].open(encoding="utf-8") as s_handle, \
            trace_paths[2].open(encoding="utf-8") as p_handle:
        compared_traces = 0
        for lines in itertools.zip_longest(h_handle, s_handle, p_handle):
            if any(line is None for line in lines):
                raise RuntimeError("H/S/P task traces have different lengths")
            events = [json.loads(line) for line in lines]
            keys = {(event["round"], event["task_id"], event["client_id"])
                    for event in events}
            if len(keys) != 1:
                raise RuntimeError("H/S/P task assignments differ")
            traces = [event["trace"] for event in events]
            for field in ("num_reset_kernels", "targeted_reset_slots",
                          "matched_donor_id", "matched_age", "matched_peer_count",
                          "matching_fallback", "common_valid_by_layer"):
                if any(trace[field] != traces[0][field] for trace in traces[1:]):
                    raise RuntimeError(f"H/S/P client event differs in {field}: {keys}")
            for layers in zip(*(trace["layers"] for trace in traces)):
                for field in ("name", "target_requested", "target_actual",
                              "baseline_reset_indices", "fallback_reason"):
                    if any(layer[field] != layers[0][field] for layer in layers[1:]):
                        raise RuntimeError(f"H/S/P client-layer differs in {field}: {keys}")
                if len({len(layer["final_reset_indices"]) for layer in layers}) != 1:
                    raise RuntimeError(f"H/S/P reset budget differs: {keys}")
            compared_traces += 1
    if compared_traces != 10000:
        raise RuntimeError(f"expected 10000 client task traces, got {compared_traces}")
    results = {name: accuracy_summary(rows) for name, rows in round_rows.items()}
    delta = original_e5_delta(output_dir / "H_same/task_traces.jsonl")
    comparison = {
        "protocol": "VGG/CIFAR-10/alpha0.3/seed1/1000 rounds; B0 frozen score bank",
        "primary_mechanism_window": "rounds151-770 mean accuracy",
        "primary_longterm_window": "rounds901-1000 mean accuracy",
        "runs": results,
        "H_minus_S": {key: results["H_same"][key] - results["S_shuffled"][key]
                      for key in ("mechanism_151_770_mean", "longterm_901_1000_mean")},
        "H_minus_P": {key: results["H_same"][key] - results["P_population"][key]
                      for key in ("mechanism_151_770_mean", "longterm_901_1000_mean")},
        "H_minus_B0": {key: results["H_same"][key] - results["B0"][key]
                       for key in ("mechanism_151_770_mean", "longterm_901_1000_mean")},
        "controlled_H_vs_original_E5": delta,
        "first150_global_hash_parity": True,
        "same_selection_task_seed_all1000": True,
        "same_HSP_budget_support_all1000": True,
        "same_HSP_client_layer_target_slots_all1000": True,
        "matching_preflight": json.loads((output_dir / "matching_preflight.json").read_text(encoding="utf-8")),
        "correctness": manifest["correctness"],
        "single_seed_no_significance_claim": True,
    }
    write_json(output_dir / "comparison.json", comparison)
    columns = ("run", "mechanism_151_770_mean", "longterm_901_1000_mean",
               "peak", "peak_round", "final", "last20", "last50", "last100",
               "best_rolling5", "best_rolling10", "targeted_clients",
               "targeted_reset_slots", "replaced_reset_slots", "fallback_clients",
               "fallback_client_layers", "max_history_memory_bytes", "failed_client_updates")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, _ in RUNS:
            writer.writerow({"run": name, **{key: results[name][key] for key in columns[1:]}})
    lines = [
        "# History identity causal ablation (seed 1)", "",
        "B0 is current-code FedPhoenix. H/S/P share a frozen B0 score/validity bank,",
        "exact-age source matching, common score support, FedPhoenix task seeds,",
        "reset budget and SHA256 pairing. This is a controlled off-policy history",
        "source ablation; H is not the original on-policy E5.", "",
        "## Accuracy (%)", "",
        "| Run | R151-770 | R901-1000 | Peak (round) | Final | Last20 | Last50 | Last100 | Roll5 | Roll10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, _ in RUNS:
        value = results[name]
        lines.append(
            f"| {name} | {value['mechanism_151_770_mean']:.4f} | "
            f"{value['longterm_901_1000_mean']:.4f} | "
            f"{value['peak']:.4f} ({value['peak_round']}) | "
            f"{value['final']:.4f} | {value['last20']:.4f} | "
            f"{value['last50']:.4f} | {value['last100']:.4f} | "
            f"{value['best_rolling5']:.4f} | {value['best_rolling10']:.4f} |"
        )
    lines.extend(["", "## Mechanism and controlled-original E5 comparison", "",
                  "| Run | Targeted clients | Slots | Replaced | Fallback clients | Fallback client-layers | Max history MB |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, _ in RUNS:
        value = results[name]
        lines.append(f"| {name} | {value['targeted_clients']} | "
                     f"{value['targeted_reset_slots']} | {value['replaced_reset_slots']} | "
                     f"{value['fallback_clients']} | {value['fallback_client_layers']} | "
                     f"{value['max_history_memory_bytes'] / 1e6:.3f} |")
    lines.extend(["", f"Controlled H vs original E5: {json.dumps(delta, ensure_ascii=False)}", "",
                  "## Stage accuracy (%)", "",
                  "| Stage | B0 | H | S | P |", "| --- | ---: | ---: | ---: | ---: |"])
    for a, b in STAGES:
        key = f"{a}_{b}"
        lines.append(f"| {a}-{b} | " + " | ".join(
            f"{results[name]['stage_means'][key]:.4f}"
            for name, _ in RUNS) + " |")
    lines.extend(["", "## Explicit questions", ""])
    for label, other in (("S", "S_shuffled"), ("P", "P_population"), ("B0", "B0")):
        h = results["H_same"]["longterm_901_1000_mean"]
        comparison_value = results[other]["longterm_901_1000_mean"]
        lines.append(f"- H vs {label}, R901-1000: {h - comparison_value:+.4f} pp "
                     f"({'higher' if h > comparison_value else 'lower' if h < comparison_value else 'equal'}).")
    lines.extend(["- Identity necessity requires H to outperform *both* source controls;",
                  "  one seed cannot establish equivalence or statistical significance.",
                  "- Main decision also requires H not below current-code B0 on the fixed late window.",
                  "- First 150 model hashes, all 1000 selected clients/task seeds, and",
                  "  H/S/P per-round budget/support parity were checked exactly.",
                  "- No other ratio, gap, layer, start, score or seed was tried.", ""])
    (output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/history_identity_causal_ablation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--gates_only", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "version": 1, "created_at": now(), "status": "preflight",
        "fixed_runs": RUNS, "batch_order": [["B0", "H_same"],
                                     ["S_shuffled", "P_population"]],
        "attempts": [], "active": [],
    }
    if manifest["status"] == "completed":
        return
    if manifest.get("active"):
        raise RuntimeError("manifest has active PIDs; inspect processes before resume")
    prepare_schedule(output_dir)
    if args.prepare_only:
        print(output_dir / "matching_preflight.json")
        return
    try:
        if "correctness" not in manifest:
            manifest["correctness"] = correctness_gates(
                Path(args.python), output_dir, args.gpu
            )
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)
        if "identity_smoke" not in manifest["correctness"]:
            manifest["correctness"]["identity_smoke"] = correctness_smoke(
                Path(args.python), args.gpu, output_dir
            )
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)
        if args.gates_only:
            manifest["status"] = "preflight_passed"
            manifest.pop("error", None)
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)
            print(output_dir / "identity_smoke_report.json")
            return
        manifest["status"] = "running"
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        run_batch(Path(args.python), args.gpu, output_dir, RUNS[:2],
                  manifest, manifest_path, args.max_retries)
        run_batch(Path(args.python), args.gpu, output_dir, RUNS[2:],
                  manifest, manifest_path, args.max_retries)
        if not all(is_complete(output_dir, name) for name, _ in RUNS):
            raise RuntimeError("not all four arms completed")
        summarize(output_dir, manifest)
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
