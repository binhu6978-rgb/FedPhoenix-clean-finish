"""Read-only audit of existing FedPhoenix and TargetedFedPhoenix artifacts.

This script never imports a training module or launches a model run. It reports
descriptive quantities only; accuracy differences are not causal estimates.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def round_map(path):
    return {int(row["round"]): row for row in rows(path)}


def accuracy_curve(path):
    pattern = re.compile(r"ROUND_ACCURACY\s+method=FedPhoenix\s+round=(\d+)\s+accuracy=([\d.]+)")
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                result[int(match.group(1))] = float(match.group(2))
    return result


def round_window(first, last, *maps):
    rounds = [r for r in range(first, last + 1) if all(r in mapping for mapping in maps)]
    return rounds


def summarize_curve(curve):
    values = list(curve.values())
    peak_round = max(curve, key=curve.get)
    return {
        "rounds": len(curve),
        "peak": curve[peak_round],
        "peak_round": peak_round,
        "final": curve[max(curve)],
        "last20": mean(values[-20:]),
        "last50": mean(values[-50:]),
        "last100": mean(values[-100:]),
    }


def phase_a(events):
    main = [row for row in events if row["score_type"] == "update_norm"]
    all_ages = []
    own_ages = []
    exact_events = 0
    cross_counts = []
    row_gaps = defaultdict(list)
    by_layer = defaultdict(list)
    for row in main:
        own_gap = int(row["gap"])
        current = int(row["round"])
        cross_rounds = json.loads(row["cross_history_rounds"])
        cross_ages = [current - int(r) for r in cross_rounds]
        own_ages.append(own_gap)
        all_ages.extend(cross_ages)
        exact_events += own_gap in cross_ages
        cross_counts.append(len(cross_ages))
        row_gaps[row["layer"]].append({
            "own_gap": own_gap,
            "mean_cross_gap": mean(cross_ages) if cross_ages else None,
            "exact_gap_available": own_gap in cross_ages,
        })
        by_layer[row["layer"]].append(row)
    layer_results = {}
    for layer, group in by_layer.items():
        layer_results[layer] = {
            "events": len(group),
            "width": int(group[0]["output_kernels"]),
            "reset_budget": int(group[0]["reset_budget"]),
            "mean_k_eval": mean(int(r["k_eval"]) for r in group),
            "mean_random_overlap": mean(float(r["random_overlap_rate"]) for r in group),
            "same_overlap": mean(float(r["overlap_rate"]) for r in group),
            "cross_overlap": mean(float(r["cross_overlap_rate_mean"]) for r in group),
            "identity_advantage": mean(float(r["identity_overlap_advantage"]) for r in group),
        }
    paired = [r for group in row_gaps.values() for r in group if r["mean_cross_gap"] is not None]
    return {
        "update_norm_events": len(main),
        "same_overlap": mean(float(r["overlap_rate"]) for r in main),
        "cross_overlap": mean(float(r["cross_overlap_rate_mean"]) for r in main),
        "random_overlap": mean(float(r["random_overlap_rate"]) for r in main),
        "same_spearman": mean(float(r["spearman"]) for r in main),
        "cross_spearman": mean(float(r["cross_spearman_mean"]) for r in main),
        "own_gap_mean": mean(own_ages),
        "own_gap_median": median(own_ages),
        "cross_gap_mean_per_comparison": mean(all_ages),
        "cross_gap_median_per_comparison": median(all_ages),
        "cross_gap_mean_per_event": mean(r["mean_cross_gap"] for r in paired),
        "fraction_events_any_exact_gap_cross": exact_events / len(main),
        "mean_cross_history_count": mean(cross_counts),
        "layers": layer_results,
    }


def donor_match_feasibility(rounds, max_gap=10, start_round=151):
    """Availability only: no alternative-score experiment is performed."""
    last_seen = {}
    eligible = 0
    exact_available = 0
    candidate_counts = []
    for current_round in sorted(rounds):
        selected = [int(client) for client in json.loads(rounds[current_round]["selected_clients"])]
        if start_round <= current_round <= 770:
            for client in selected:
                previous = last_seen.get(client)
                if previous is None or current_round - previous > max_gap:
                    continue
                eligible += 1
                peers = [
                    other for other, prior in last_seen.items()
                    if other != client and current_round - prior == current_round - previous
                ]
                candidate_counts.append(len(peers))
                exact_available += bool(peers)
        last_seen.update({client: current_round for client in selected})
    return {
        "eligible_selected_client_events_in_active_window": eligible,
        "events_with_any_exact_age_other_client": exact_available,
        "fraction_with_exact_age_other_client": exact_available / eligible if eligible else None,
        "median_exact_age_peer_count": median(candidate_counts) if candidate_counts else None,
        "minimum_exact_age_peer_count": min(candidate_counts) if candidate_counts else None,
        "note": "Matches only history age, not label profile, client sample count, or valid score mask.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / "results/research_decision_audit/offline_metrics.json"
    diagnostic = round_map(repo / "results/fedphoenix_specialization_diagnostic/round_metrics.csv")
    l2 = round_map(repo / "results/targeted_fedphoenix_longrun1000/L2_E5_r1000_seed1/round_metrics.csv")
    l3 = round_map(repo / "results/targeted_fedphoenix_longrun1000/L3_E2_r1000_seed1/round_metrics.csv")
    historical = accuracy_curve(repo / "result_other_method/cifar10/0.3/vgg_FedPhoenix.log")
    historical_first1000 = {r: historical[r] for r in range(1, 1001) if r in historical}
    diagnostic_acc = {r: float(row["test_accuracy"]) for r, row in diagnostic.items()}
    l2_acc = {r: float(row["test_accuracy"]) for r, row in l2.items()}
    l3_acc = {r: float(row["test_accuracy"]) for r, row in l3.items()}
    prefix = round_window(1, 150, diagnostic, l2)
    parity = {
        field: sum(diagnostic[r][field] == l2[r][field] for r in prefix)
        for field in ("selected_clients", "task_seeds", "global_state_sha256")
    }
    compared = round_window(1, 300, historical, diagnostic_acc)
    mismatches = [r for r in compared if abs(historical[r] - diagnostic_acc[r]) > 1e-5]
    abs_differences = sorted(abs(historical[r] - diagnostic_acc[r]) for r in compared)
    maximum_difference_round = max(compared, key=lambda r: abs(historical[r] - diagnostic_acc[r]))
    periods = [(1, 150), (151, 300), (301, 500), (501, 750), (751, 770), (771, 1000), (901, 1000)]
    stages = {}
    for first, last in periods:
        window = round_window(first, last, l2_acc, l3_acc, historical)
        stages[f"{first}-{last}"] = {
            "n": len(window),
            "l2_accuracy_mean": mean(l2_acc[r] for r in window),
            "l3_accuracy_mean": mean(l3_acc[r] for r in window),
            "historical_accuracy_mean": mean(historical[r] for r in window),
            "l2_minus_l3_mean": mean(l2_acc[r] - l3_acc[r] for r in window),
            "l2_minus_historical_mean": mean(l2_acc[r] - historical[r] for r in window),
            "l2_targeted_slots": sum(int(l2[r]["targeted_reset_slots"]) for r in window),
            "l3_targeted_slots": sum(int(l3[r]["targeted_reset_slots"]) for r in window),
        }
    last_targeted = max(r for r in l2 if int(l2[r]["targeted_reset_slots"]) > 0)
    # Active reset windows are deterministic for 13 convolutional layers:
    # current_iter = round - 1, active iff current_iter < (depth+1)*1000/13.
    target_depths = {"features.20": 6, "features.24": 7, "features.27": 8, "features.30": 9}
    stop_rounds = {
        layer: max(r for r in range(1, 1001) if r - 1 < (depth + 1) * 1000 / 13)
        for layer, depth in target_depths.items()
    }
    phase = phase_a(rows(repo / "results/fedphoenix_specialization_diagnostic/persistence_events.csv"))
    summary = {
        "sources": {
            "historical_log": "result_other_method/cifar10/0.3/vgg_FedPhoenix.log",
            "current_fedphoenix_observer": "results/fedphoenix_specialization_diagnostic/round_metrics.csv",
            "l2": "results/targeted_fedphoenix_longrun1000/L2_E5_r1000_seed1/round_metrics.csv",
            "l3": "results/targeted_fedphoenix_longrun1000/L3_E2_r1000_seed1/round_metrics.csv",
            "phase_a": "results/fedphoenix_specialization_diagnostic/persistence_events.csv",
        },
        "curves": {
            "historical_first1000": summarize_curve(historical_first1000),
            "historical_full_log": summarize_curve(historical),
            "current_observer_300": summarize_curve(diagnostic_acc),
            "l2": summarize_curve(l2_acc),
            "l3": summarize_curve(l3_acc),
        },
        "first150_parity_l2_vs_current_observer": {"rounds_compared": len(prefix), **parity},
        "historical_vs_current_first300": {
            "compared_rounds": len(compared),
            "accuracy_mismatch_rounds": len(mismatches),
            "first_accuracy_mismatch_round": mismatches[0] if mismatches else None,
            "mean_absolute_accuracy_difference": mean(abs_differences),
            "median_absolute_accuracy_difference": median(abs_differences),
            "p95_absolute_accuracy_difference": abs_differences[int(0.95 * (len(abs_differences) - 1))],
            "maximum_absolute_accuracy_difference": abs_differences[-1],
            "maximum_difference_round": maximum_difference_round,
            "maximum_difference_historical_accuracy": historical[maximum_difference_round],
            "maximum_difference_current_accuracy": diagnostic_acc[maximum_difference_round],
            "rounds_within_0p01pp": sum(abs(historical[r] - diagnostic_acc[r]) <= 0.01 for r in compared),
            "metadata_available_in_log": False,
        },
        "stages": stages,
        "l2_last_targeted_round": last_targeted,
        "l2_targeted_slots_771_1000": sum(int(l2[r]["targeted_reset_slots"]) for r in range(771, 1001)),
        "e5_target_layer_last_active_rounds": stop_rounds,
        "phase_a": phase,
        "shuffled_history_exact_age_availability": donor_match_feasibility(l2),
        "limitations": [
            "No paired current-code 1000-round FedPhoenix trajectory exists in these artifacts.",
            "Historical log lacks per-round selected clients, task seeds, reset masks and model hashes.",
            "Phase-A cross overlap is stored only as a mean across unmatched client histories; exact gap-matched cross overlap cannot be reconstructed from this CSV.",
            "All accuracy comparisons are single-seed, with repeated test-set-guided configuration selection.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(output)
    print(json.dumps({
        "historical_vs_current_first300": summary["historical_vs_current_first300"],
        "first150_parity": summary["first150_parity_l2_vs_current_observer"],
        "l2_last_targeted_round": last_targeted,
        "phase_a_own_gap_mean": phase["own_gap_mean"],
        "phase_a_cross_gap_mean": phase["cross_gap_mean_per_comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
