"""Analyze the fixed seed-1 SymmetryFedPhoenix-alt 1000-round run."""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "model_alt_1000"
BASELINE = ROOT / "results/history_identity_causal_ablation/B0/round_metrics.csv"
ALT = RUN / "cifar10_vgg_SymmetryFedPhoenix_seed1_model_alt_1000.csv"
EVENTS = RUN / "model_alt_1000_client_events.csv"
METRICS = (
    "reset_action_norm",
    "reset_response_norm",
    "reset_recovery_coeff",
    "reset_recovery_cosine",
    "reset_residual_ratio",
)
STAGES = ((1, 300), (301, 500), (501, 900), (901, 1000))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accuracy_metrics(rows):
    values = [float(row["test_accuracy"]) for row in rows]
    peak = max(values)
    result = {
        "peak_accuracy": peak,
        "peak_round": values.index(peak) + 1,
        "top10_mean": statistics.mean(sorted(values, reverse=True)[:10]),
        "round_1000_accuracy": values[-1],
    }
    for start, end in STAGES:
        result[f"rounds_{start}_{end}_mean"] = statistics.mean(values[start - 1:end])
    return result


def response_stats(events):
    result = {"client_events": len(events)}
    for metric in METRICS:
        values = [float(event[metric]) for event in events]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_median"] = statistics.median(values) if values else None
    return result


def main():
    b0 = read_csv(BASELINE)[:1000]
    alt = read_csv(ALT)
    events = read_csv(EVENTS)
    expected_rounds = list(range(1, 1001))
    if len(b0) != 1000 or len(alt) != 1000:
        raise ValueError("Expected 1000 complete B0 and alt rounds")
    if [int(row["round"]) for row in b0] != expected_rounds:
        raise ValueError("B0 rounds are incomplete or out of order")
    if [int(row["round"]) for row in alt] != expected_rounds:
        raise ValueError("Alt rounds are incomplete or out of order")
    if len(events) != 10000:
        raise ValueError(f"Expected 10000 client events, got {len(events)}")
    if not all(math.isfinite(float(row["test_accuracy"])) for row in b0 + alt):
        raise ValueError("Non-finite test accuracy")

    protocol = {
        "selected_clients_equal_B0": all(
            a["selected_clients"] == b["selected_clients"] for a, b in zip(alt, b0)
        ),
        "task_seeds_equal_B0": all(
            a["task_seeds"] == b["task_seeds"] for a, b in zip(alt, b0)
        ),
        "reported_alt_violations": sum(int(row["alt_violations"]) for row in alt),
    }
    if not all((protocol["selected_clients_equal_B0"],
                protocol["task_seeds_equal_B0"],
                protocol["reported_alt_violations"] == 0)):
        raise ValueError(f"Protocol check failed: {protocol}")

    visits = Counter()
    last_view = {}
    view_counts = Counter()
    visit_counts = Counter()
    returning_count = 0
    for row_index, event in enumerate(events):
        round_number = row_index // 10 + 1
        if int(event["round"]) != round_number:
            raise ValueError("Client events are incomplete or out of order")
        client = int(event["client_id"])
        view = event["current_view"]
        visit = int(event["visit_index"])
        visits[client] += 1
        if visit != visits[client] or view not in {"I", "F"}:
            raise ValueError("Visit index or view is invalid")
        previous = last_view.get(client)
        if event["previous_view"] != (previous or ""):
            raise ValueError("Previous-view log mismatch")
        if previous == view:
            raise ValueError("Independent alternation violation")
        if (event["is_returning"] == "True") != (visit > 1):
            raise ValueError("Returning-client log mismatch")
        if not all(math.isfinite(float(event[metric])) for metric in METRICS):
            raise ValueError("Non-finite reset response diagnostic")
        if int(event["reset_kernel_count"]) < 1:
            raise ValueError("No reset kernels in a client event")
        last_view[client] = view
        view_counts[view] += 1
        visit_counts[visit] += 1
        returning_count += visit > 1
    for round_index, row in enumerate(alt):
        block = events[round_index * 10:(round_index + 1) * 10]
        clients = json.loads(row["selected_clients"])
        views = json.loads(row["views"])
        if [int(event["client_id"]) for event in block] != clients:
            raise ValueError("Round/event client log mismatch")
        if [event["current_view"] for event in block] != views:
            raise ValueError("Round/event view log mismatch")
        if int(row["I_client_count"]) != views.count("I") or int(row["F_client_count"]) != views.count("F"):
            raise ValueError("Round view count mismatch")

    b0_metrics = accuracy_metrics(b0)
    alt_metrics = accuracy_metrics(alt)
    comparison = {
        "B0": b0_metrics,
        "SymmetryFedPhoenix_alt": alt_metrics,
        "alt_minus_B0_pp": {
            key: alt_metrics[key] - b0_metrics[key]
            for key in b0_metrics if key != "peak_round"
        },
    }
    response = {
        view: response_stats([event for event in events if event["current_view"] == view])
        for view in ("I", "F")
    }
    stages = {}
    for start, end in STAGES:
        stage_events = [event for event in events
                        if start <= int(event["round"]) <= end]
        stages[f"{start}-{end}"] = {
            view: response_stats([event for event in stage_events
                                  if event["current_view"] == view])
            for view in ("I", "F")
        }

    summary = {
        "sources": {
            "B0": str(BASELINE.relative_to(ROOT)),
            "alt_rounds": str(ALT.relative_to(ROOT)),
            "alt_events": str(EVENTS.relative_to(ROOT)),
        },
        "accuracy": comparison,
        "protocol": protocol,
        "interaction": {
            "client_events": len(events),
            "F_fraction": view_counts["F"] / len(events),
            "I_fraction": view_counts["I"] / len(events),
            "returning_client_rate": returning_count / len(events),
            "visit_index_counts": {
                "1": visit_counts[1],
                "2": visit_counts[2],
                "3": visit_counts[3],
                "4": visit_counts[4],
                "5+": sum(count for visit, count in visit_counts.items() if visit >= 5),
                "10+": sum(count for visit, count in visit_counts.items() if visit >= 10),
            },
            "visit_index_rates": {
                "1": visit_counts[1] / len(events),
                "2": visit_counts[2] / len(events),
                "3": visit_counts[3] / len(events),
                "4": visit_counts[4] / len(events),
                "5+": sum(count for visit, count in visit_counts.items() if visit >= 5) / len(events),
                "10+": sum(count for visit, count in visit_counts.items() if visit >= 10) / len(events),
            },
        },
        "reset_response_by_view": response,
        "reset_response_by_stage_and_view": stages,
    }
    (HERE / "model_alt_1000_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (HERE / "model_alt_1000_round_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "B0_accuracy", "alt_accuracy", "alt_minus_B0_pp",
            "returning_client_rate", "flip_fraction",
            "reset_recovery_coeff_mean_I", "reset_recovery_coeff_mean_F",
            "reset_residual_ratio_mean_I", "reset_residual_ratio_mean_F",
        ])
        for b0_row, alt_row in zip(b0, alt):
            b0_accuracy = float(b0_row["test_accuracy"])
            alt_accuracy = float(alt_row["test_accuracy"])
            writer.writerow([
                alt_row["round"], b0_accuracy, alt_accuracy,
                alt_accuracy - b0_accuracy,
                alt_row["returning_client_rate"], alt_row["flip_fraction"],
                alt_row["reset_recovery_coeff_mean_I"],
                alt_row["reset_recovery_coeff_mean_F"],
                alt_row["reset_residual_ratio_mean_I"],
                alt_row["reset_residual_ratio_mean_F"],
            ])

    with (HERE / "model_alt_1000_visit_response.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["visit_index", "view", "client_events", *(
            f"{metric}_mean" for metric in METRICS
        )])
        grouped = defaultdict(list)
        for event in events:
            grouped[(int(event["visit_index"]), event["current_view"])].append(event)
        for (visit, view), group in sorted(grouped.items()):
            stats = response_stats(group)
            writer.writerow([visit, view, len(group)] + [
                stats[f"{metric}_mean"] for metric in METRICS
            ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
