#!/usr/bin/env python
"""Run the fixed seed-1 InteractionDispatchV2 screen and promotion rule."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results" / "interaction_dispatch_peak_search_vgg"
RATIOS = (0.10, 0.20, 0.40)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def ratio_tag(value):
    return f"{value:.2f}".replace(".", "p")


def parity_result(output_dir):
    parity = output_dir / "parity"
    fed_selection = (parity / "fedavg" / "selection.jsonl").read_text(encoding="utf-8")
    v2_selection = (parity / "v2" / "selection.jsonl").read_text(encoding="utf-8")
    fed_hashes = [json.loads(line) for line in (parity / "fedavg" / "hashes.jsonl").read_text().splitlines()]
    v2_hashes = [json.loads(line) for line in (parity / "v2" / "hashes.jsonl").read_text().splitlines()]
    hash_matches = sum(
        left["global_state_sha256"] == right["global_state_sha256"]
        for left, right in zip(fed_hashes, v2_hashes)
    )
    result = {
        "rounds": len(fed_hashes),
        "selection_exact": fed_selection == v2_selection,
        "hash_exact_rounds": hash_matches,
        "hash_all_exact": len(fed_hashes) == len(v2_hashes) == 10 and hash_matches == 10,
    }
    if not result["selection_exact"] or not result["hash_all_exact"]:
        raise RuntimeError(f"InteractionDispatchV2 parity gate failed: {result}")
    write_json(parity / "parity_report.json", result)
    return result


def protocol_args():
    return [
        "--dataset", "cifar10", "--model", "vgg", "--num_users", "100",
        "--frac", "0.1", "--local_ep", "5", "--local_bs", "50",
        "--bs", "256", "--optimizer", "sgd", "--lr", "0.01",
        "--momentum", "0.5", "--weight_decay", "0", "--iid", "0",
        "--noniid_case", "5", "--data_beta", "0.3", "--generate_data", "0",
        "--num_classes", "10", "--num_channels", "3", "--seed", "1",
        "--num_workers", "0",
    ]


def run_id(ratio, epochs):
    return f"id2_ratio{ratio_tag(ratio)}_r{epochs}_seed1"


def expected_summary(run_dir, rid):
    return run_dir / "metrics" / f"cifar10_vgg_InteractionDispatchV2_seed1_{rid}_summary.json"


def launch_one(python, output_dir, gpu, ratio, epochs, max_retries, manifest):
    rid = run_id(ratio, epochs)
    run_dir = output_dir / "runs" / rid
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = expected_summary(run_dir, rid)
    entry = manifest["runs"].setdefault(
        rid,
        {
            "run_id": rid,
            "ratio": ratio,
            "epochs": epochs,
            "seed": 1,
            "status": "pending",
            "attempt": 0,
            "attempt_history": [],
        },
    )
    if entry.get("status") == "completed" and summary_path.exists():
        return read_json(summary_path)
    command = [
        str(python), str(ROOT / "main_fed.py"),
        "--algorithm", "InteractionDispatchV2",
        "--id2_step_ratio", str(ratio),
        "--epochs", str(epochs), "--gpu", str(gpu),
        "--run_name", rid, "--metrics_log_dir", str(metrics_dir),
        *protocol_args(),
    ]
    entry["command"] = command
    for _ in range(max_retries + 1):
        entry["attempt"] = int(entry.get("attempt", 0)) + 1
        attempt = entry["attempt"]
        stdout_path = output_dir / "logs" / f"{rid}.attempt{attempt}.stdout.log"
        stderr_path = output_dir / "logs" / f"{rid}.attempt{attempt}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        entry.update(
            {
                "status": "running", "started_at": now(),
                "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
            }
        )
        manifest["updated_at"] = now()
        write_json(output_dir / "manifest.json", manifest)
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        entry.update(
            {
                "return_code": process.returncode,
                "runtime_seconds": time.perf_counter() - started,
                "finished_at": now(),
            }
        )
        if process.returncode == 0 and summary_path.exists():
            summary = read_json(summary_path)
            entry.update(summary)
            entry["status"] = "completed"
            manifest["updated_at"] = now()
            write_json(output_dir / "manifest.json", manifest)
            return summary
        entry["attempt_history"].append(
            {
                "attempt": attempt,
                "return_code": process.returncode,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "finished_at": now(),
            }
        )
        entry["status"] = "failed"
        manifest["updated_at"] = now()
        write_json(output_dir / "manifest.json", manifest)
    raise RuntimeError(f"{rid} failed after {max_retries + 1} attempts")


def load_old_benchmarks():
    summary_rows = list(csv.DictReader((OLD_ROOT / "summary.csv").open(encoding="utf-8")))
    completed = [row for row in summary_rows if row["status"] == "completed" and row["completed_rounds"] == "300"]
    fedavg = next(row for row in completed if row["algorithm"] == "FedAvg")
    old_rows = [row for row in completed if row["algorithm"] == "InteractionDispatch"]
    old_best = max(old_rows, key=lambda row: float(row["peak_accuracy"]))

    def add_last20(row):
        rid = row["run_id"]
        metrics_dir = OLD_ROOT / "runs" / rid / "metrics"
        metrics_path = next(path for path in metrics_dir.glob("*.csv") if "config" not in path.name)
        metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        values = [float(item["test_accuracy"]) for item in metrics]
        return {
            "run_id": rid,
            "peak_accuracy": float(row["peak_accuracy"]),
            "peak_round": int(row["peak_round"]),
            "final_accuracy": float(row["final_accuracy"]),
            "last20_mean": sum(values[-20:]) / 20.0,
        }

    return add_last20(fedavg), add_last20(old_best)


def summarize_groups(summaries):
    aggregate = defaultdict(lambda: {"support": [], "conflict": [], "dominant": 0})
    event_groups = defaultdict(dict)
    for summary in summaries:
        with Path(summary["group_metrics_csv"]).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                group = row["group"]
                support = float(row["support"])
                aggregate[group]["support"].append(support)
                aggregate[group]["conflict"].append(float(row["conflict"]))
                key = (summary["id2_step_ratio"], row["round"], row["client_id"])
                event_groups[key][group] = support
    for groups in event_groups.values():
        if groups:
            dominant = max(groups, key=lambda name: abs(groups[name]))
            aggregate[dominant]["dominant"] += 1
    result = []
    for group, values in aggregate.items():
        supports = values["support"]
        conflicts = values["conflict"]
        count = len(supports)
        result.append(
            {
                "group": group,
                "count": count,
                "support_mean": sum(supports) / count,
                "support_abs_mean": sum(abs(value) for value in supports) / count,
                "support_positive_rate": sum(value > 0 for value in supports) / count,
                "support_negative_rate": sum(value < 0 for value in supports) / count,
                "conflict_mean": sum(conflicts) / count,
                "dominant_abs_support_count": values["dominant"],
                "dominant_abs_support_rate": values["dominant"] / len(event_groups),
            }
        )
    return result


def write_outputs(output_dir, manifest, screen, promoted, parity):
    fedavg, old_best = load_old_benchmarks()
    rows = []
    for summary in screen + ([promoted] if promoted else []):
        rows.append(
            {
                key: summary[key]
                for key in (
                    "id2_step_ratio", "epochs", "peak_accuracy", "peak_round",
                    "final_accuracy", "last20_mean", "active_count", "active_rate",
                    "maximum_history_memory_bytes", "failed_client_updates",
                )
            }
        )
        rows[-1]["no_need_count"] = summary["reason_counts"].get("no_need", 0)
        rows[-1]["no_support_count"] = summary["reason_counts"].get("no_support", 0)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group_rows = summarize_groups(screen)
    with (output_dir / "group_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    best = max(screen, key=lambda item: item["peak_accuracy"])
    comparison = {
        "parity": parity,
        "fedavg_300": fedavg,
        "old_interaction_dispatch_best_300": old_best,
        "v2_300": screen,
        "best_v2_300": best,
        "promotion_threshold": {
            "peak": old_best["peak_accuracy"] + 0.3,
            "last20": old_best["last20_mean"] + 0.5,
        },
        "promotion_triggered": promoted is not None,
        "promoted_600": promoted,
        "group_diagnostics": group_rows,
        "failed_runs": [name for name, entry in manifest["runs"].items() if entry.get("status") != "completed"],
    }
    write_json(output_dir / "comparison.json", comparison)
    report = [
        "# InteractionDispatchV2 result report", "",
        f"- Correctness: {parity['hash_exact_rounds']}/10 exact global hashes; selection exact={parity['selection_exact']}.",
        f"- FedAvg: peak {fedavg['peak_accuracy']:.4f}, last20 {fedavg['last20_mean']:.4f}.",
        f"- Old InteractionDispatch best: {old_best['run_id']}, peak {old_best['peak_accuracy']:.4f}, last20 {old_best['last20_mean']:.4f}.",
        "", "## V2 300-round screen", "",
    ]
    for item in screen:
        report.append(
            f"- ratio={item['id2_step_ratio']:.2f}: peak={item['peak_accuracy']:.4f} "
            f"(round {item['peak_round']}), final={item['final_accuracy']:.4f}, "
            f"last20={item['last20_mean']:.4f}, active_rate={item['active_rate']:.6f}."
        )
    report.extend(["", f"Promotion triggered: **{promoted is not None}**.", "", "## Coarse groups", ""])
    for item in group_rows:
        report.append(
            f"- {item['group']}: mean|support|={item['support_abs_mean']:.6g}, "
            f"positive={item['support_positive_rate']:.3f}, dominant={item['dominant_abs_support_rate']:.3f}."
        )
    (output_dir / "decision_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/interaction_dispatch_v2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_retries", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parity = parity_result(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "version": 1, "created_at": now(), "updated_at": now(),
        "protocol": "VGG/CIFAR10 alpha=0.3 seed=1", "runs": {},
    }
    screen = [
        launch_one(Path(args.python), output_dir, args.gpu, ratio, 300, args.max_retries, manifest)
        for ratio in RATIOS
    ]
    fedavg, old_best = load_old_benchmarks()
    best = max(screen, key=lambda item: item["peak_accuracy"])
    promote = (
        best["peak_accuracy"] >= old_best["peak_accuracy"] + 0.3
        or best["last20_mean"] >= old_best["last20_mean"] + 0.5
    )
    promoted = None
    if promote:
        promoted = launch_one(
            Path(args.python), output_dir, args.gpu,
            float(best["id2_step_ratio"]), 600, args.max_retries, manifest,
        )
    manifest["status"] = "completed"
    manifest["updated_at"] = now()
    manifest["promotion_triggered"] = promote
    write_json(manifest_path, manifest)
    write_outputs(output_dir, manifest, screen, promoted, parity)


if __name__ == "__main__":
    main()
