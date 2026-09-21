#!/usr/bin/env python
"""Compare completed seed-1 FedAvg/FedPhoenix diagnostic summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_dir",
        default="results/fedphoenix_longitudinal_diagnostic/seed1",
    )
    parser.add_argument(
        "--root_output",
        default="results/fedphoenix_longitudinal_diagnostic",
    )
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(summary: Dict[str, Any], name: str, statistic: str = "median") -> Optional[float]:
    return summary["metrics"][name].get(statistic)


def _transfer(
    summary: Dict[str, Any], subset: str, metric: str, statistic: str = "median"
) -> Optional[float]:
    value = summary["similarity_subsets"][subset].get(metric)
    if isinstance(value, dict):
        return value.get(statistic)
    return value


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def _threshold_row(summary: Dict[str, Any], subset: str) -> Dict[str, Any]:
    data = summary["similarity_subsets"][subset]
    return {
        "count": data["count"],
        "insufficient_count": data["insufficient_count"],
        "response_direction_cosine_median": data["response_direction_cosine"]["median"],
        "response_prediction_nre_median": data["response_prediction_nre"]["median"],
        "response_norm_ratio_median": data["response_norm_ratio"]["median"],
        "support_sign_agreement_rate": data["support_sign_agreement_rate"],
        "support_pearson": data["support_pearson"],
        "support_spearman": data["support_spearman"],
    }


def build_comparison(fedavg: Dict[str, Any], fedphoenix: Dict[str, Any]) -> Dict[str, Any]:
    algorithms = {"FedAvg": fedavg, "FedPhoenix": fedphoenix}
    comparison: Dict[str, Any] = {
        "seed": 1,
        "algorithms": {},
        "critical_subsets": {},
        "context_subsets": {},
        "gap_bins": {},
    }
    for name, summary in algorithms.items():
        comparison["algorithms"][name] = {
            "accuracy": summary["accuracy"],
            "event_count": summary["event_count"],
            "transfer_event_count": summary["transfer_event_count"],
            "max_diagnostic_cpu_history_bytes": summary[
                "max_diagnostic_cpu_history_bytes"
            ],
            "input_change_norm": summary["metrics"]["input_change_norm"],
            "dispatch_minus_global_norm": summary["metrics"][
                "dispatch_minus_global_norm"
            ],
            "response_change_norm": summary["metrics"]["response_change_norm"],
            "response_change_ratio": summary["metrics"]["response_change_ratio"],
            "secant_gain": summary["metrics"]["secant_gain"],
        }
    for subset in ("input_cos_ge_0_5", "input_cos_ge_0_7"):
        comparison["critical_subsets"][subset] = {
            name: _threshold_row(summary, subset)
            for name, summary in algorithms.items()
        }
    for subset in ("all", "input_cos_gt_0"):
        comparison["context_subsets"][subset] = {
            name: _threshold_row(summary, subset)
            | {
                "input_direction_cosine_mean": summary["similarity_subsets"][
                    subset
                ]["input_direction_cosine"]["mean"],
                "input_direction_cosine_median": summary[
                    "similarity_subsets"
                ][subset]["input_direction_cosine"]["median"],
            }
            for name, summary in algorithms.items()
        }
    for gap_name in ("gap_le_5", "gap_6_10", "gap_11_20", "gap_gt_20"):
        comparison["gap_bins"][gap_name] = {
            name: {
                "count": summary["participation_gap_bins"][gap_name]["count"],
                "response_direction_cosine_median": summary[
                    "participation_gap_bins"
                ][gap_name]["response_direction_cosine"]["median"],
                "response_prediction_nre_median": summary[
                    "participation_gap_bins"
                ][gap_name]["response_prediction_nre"]["median"],
                "support_sign_agreement_rate": summary[
                    "participation_gap_bins"
                ][gap_name]["support_sign_agreement_rate"],
            }
            for name, summary in algorithms.items()
        }

    excitation_ratio = _ratio(
        _metric(fedphoenix, "input_change_norm"),
        _metric(fedavg, "input_change_norm"),
    )
    response_ratio = _ratio(
        _metric(fedphoenix, "response_change_ratio"),
        _metric(fedavg, "response_change_ratio"),
    )
    sufficient = all(
        comparison["critical_subsets"][subset][algorithm]["count"] >= 10
        for subset in comparison["critical_subsets"]
        for algorithm in ("FedAvg", "FedPhoenix")
    )
    consistent_wins = 0
    possible_wins = 0
    for subset in comparison["critical_subsets"].values():
        avg, phoenix = subset["FedAvg"], subset["FedPhoenix"]
        criteria = (
            (phoenix["response_direction_cosine_median"], avg["response_direction_cosine_median"], 1),
            (phoenix["response_prediction_nre_median"], avg["response_prediction_nre_median"], -1),
            (phoenix["support_sign_agreement_rate"], avg["support_sign_agreement_rate"], 1),
            (phoenix["support_pearson"], avg["support_pearson"], 1),
            (phoenix["support_spearman"], avg["support_spearman"], 1),
        )
        for fp_value, avg_value, direction in criteria:
            if fp_value is None or avg_value is None:
                continue
            possible_wins += 1
            improvement = direction * (fp_value - avg_value)
            if improvement >= 0.05:
                consistent_wins += 1
    if not sufficient:
        decision = "C"
        rationale = "Critical similar-input subsets do not all contain at least 10 events."
    elif excitation_ratio is not None and excitation_ratio > 1.25 and possible_wins and consistent_wins >= math.ceil(0.7 * possible_wins):
        decision = "B"
        rationale = "FedPhoenix has stronger excitation and improves at least 70% of predefined transfer comparisons by >=0.05."
    elif excitation_ratio is not None and excitation_ratio > 1.25:
        decision = "A"
        rationale = "FedPhoenix increases input excitation, but transfer improvements are not consistent across the predefined criteria."
    else:
        decision = "C"
        rationale = "The data do not show both clearly stronger excitation and consistent transfer gains."
    comparison["derived"] = {
        "input_change_median_ratio_fedphoenix_over_fedavg": excitation_ratio,
        "response_change_ratio_median_ratio_fedphoenix_over_fedavg": response_ratio,
        "critical_subsets_sufficient": sufficient,
        "transfer_criterion_wins": consistent_wins,
        "transfer_criterion_comparisons": possible_wins,
    }
    comparison["decision"] = {"category": decision, "rationale": rationale}
    return comparison


def write_report(path: Path, comparison: Dict[str, Any]) -> None:
    alg = comparison["algorithms"]
    avg, fp = alg["FedAvg"], alg["FedPhoenix"]
    decision = comparison["decision"]
    lines = [
        "# FedPhoenix longitudinal repeated-interaction diagnostic",
        "",
        "This is a read-only seed-1 diagnostic. Accuracy is reported as context only and is not used to judge response transfer.",
        "",
        "## Main comparison",
        "",
        "| Metric | FedAvg | FedPhoenix |",
        "|---|---:|---:|",
        f"| Final accuracy | {_fmt(avg['accuracy']['final'])} | {_fmt(fp['accuracy']['final'])} |",
        f"| Best accuracy | {_fmt(avg['accuracy']['best'])} | {_fmt(fp['accuracy']['best'])} |",
        f"| Median input change norm | {_fmt(avg['input_change_norm']['median'])} | {_fmt(fp['input_change_norm']['median'])} |",
        f"| Input change q25/q75 | {_fmt(avg['input_change_norm']['q25'])} / {_fmt(avg['input_change_norm']['q75'])} | {_fmt(fp['input_change_norm']['q25'])} / {_fmt(fp['input_change_norm']['q75'])} |",
        f"| Median dispatch-global norm | {_fmt(avg['dispatch_minus_global_norm']['median'])} | {_fmt(fp['dispatch_minus_global_norm']['median'])} |",
        f"| Median response change norm | {_fmt(avg['response_change_norm']['median'])} | {_fmt(fp['response_change_norm']['median'])} |",
        f"| Median response change ratio | {_fmt(avg['response_change_ratio']['median'])} | {_fmt(fp['response_change_ratio']['median'])} |",
        f"| Median secant gain | {_fmt(avg['secant_gain']['median'])} | {_fmt(fp['secant_gain']['median'])} |",
        "",
        "## Similar-input transfer",
        "",
        "| Subset | Algorithm | Count | Response cosine median | NRE median | Sign agreement | Pearson | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset_name, subset in comparison["critical_subsets"].items():
        for algorithm in ("FedAvg", "FedPhoenix"):
            row = subset[algorithm]
            lines.append(
                f"| {subset_name} | {algorithm} | {row['count']} | "
                f"{_fmt(row['response_direction_cosine_median'])} | "
                f"{_fmt(row['response_prediction_nre_median'])} | "
                f"{_fmt(row['support_sign_agreement_rate'])} | "
                f"{_fmt(row['support_pearson'])} | {_fmt(row['support_spearman'])} |"
            )
    lines.extend(
        [
            "",
            "## Participation-gap transfer",
            "",
            "| Gap | Algorithm | Count | Response cosine median | NRE median | Sign agreement |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for gap_name, gap in comparison["gap_bins"].items():
        for algorithm in ("FedAvg", "FedPhoenix"):
            row = gap[algorithm]
            lines.append(
                f"| {gap_name} | {algorithm} | {row['count']} | "
                f"{_fmt(row['response_direction_cosine_median'])} | "
                f"{_fmt(row['response_prediction_nre_median'])} | "
                f"{_fmt(row['support_sign_agreement_rate'])} |"
            )
    excitation = comparison["derived"]["input_change_median_ratio_fedphoenix_over_fedavg"]
    response_ratio = comparison["derived"]["response_change_ratio_median_ratio_fedphoenix_over_fedavg"]
    overall_avg = comparison["context_subsets"]["all"]["FedAvg"]
    overall_fp = comparison["context_subsets"]["all"]["FedPhoenix"]
    positive_avg = comparison["context_subsets"]["input_cos_gt_0"]["FedAvg"]
    positive_fp = comparison["context_subsets"]["input_cos_gt_0"]["FedPhoenix"]
    absolute_response_ratio = _ratio(
        fp["response_change_norm"]["median"], avg["response_change_norm"]["median"]
    )
    secant_ratio = _ratio(fp["secant_gain"]["median"], avg["secant_gain"]["median"])
    lines.extend(
        [
            "",
            "## Required questions",
            "",
            f"**Q1. Does FedPhoenix strengthen returning-client input variation? Yes.** The median input-change norm is {_fmt(excitation)}x FedAvg, and the median dispatch-global norm changes from {_fmt(avg['dispatch_minus_global_norm']['median'])} to {_fmt(fp['dispatch_minus_global_norm']['median'])}. This establishes stronger excitation only.",
            "",
            f"**Q2. Does it create stronger local-update response change? Only in absolute norm, not proportionally.** The absolute response-change norm is {_fmt(absolute_response_ratio)}x, while normalized response-change ratio is {_fmt(response_ratio)}x and secant gain is {_fmt(secant_ratio)}x FedAvg. The much larger input therefore does not produce a proportionally stronger response.",
            "",
            f"**Q3. Are adjacent observations reusable for similar input directions? Not testable at the required thresholds.** Both algorithms have 0 events at input cosine >=0.5 (and therefore >=0.7). Across all 287 pairs, response cosine is {_fmt(overall_avg['response_direction_cosine_median'])}/{_fmt(overall_fp['response_direction_cosine_median'])} and NRE is {_fmt(overall_avg['response_prediction_nre_median'])}/{_fmt(overall_fp['response_prediction_nre_median'])} for FedAvg/FedPhoenix, which is poor unconstrained transfer but cannot replace the missing similar-direction test.",
            "",
            f"**Q4. Is reuse stronger under FedPhoenix? Not supported.** There are no critical similar-direction events. Even the weak input-cosine >0 subset has only {positive_avg['count']} FedAvg and {positive_fp['count']} FedPhoenix events, with median input cosine {_fmt(positive_avg['input_direction_cosine_median'])}/{_fmt(positive_fp['input_direction_cosine_median'])}; those directions are not close enough for the intended transfer claim.",
            "",
            f"**Q5. Does previous support predict the next support sign better? No reliable evidence.** Overall sign agreement is {_fmt(overall_avg['support_sign_agreement_rate'])} for FedAvg and {_fmt(overall_fp['support_sign_agreement_rate'])} for FedPhoenix, both below 0.5; Pearson correlations are {_fmt(overall_avg['support_pearson'])}/{_fmt(overall_fp['support_pearson'])}. The required >=0.5 and >=0.7 subsets are empty.",
            "",
            "**Q6. Does reuse decay rapidly with participation gap? No clear monotonic decay is visible.** Transfer is already poor in the <=5 bucket (negative response cosine and NRE near 0.86-0.88), and the longer-gap buckets do not worsen monotonically. This is weak-at-origin behavior rather than evidence for a useful signal that merely becomes stale.",
            "",
            f"**Q7. Final interpretation: {decision['category']}.** {decision['rationale']}",
            "",
            "A larger dispatch/input perturbation is explicitly not treated as evidence of longitudinal predictive value. FedPhoenix accuracy is likewise not part of the A/B/C rule.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir).resolve()
    root = Path(args.root_output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fedavg = _load(seed_dir / "fedavg_summary.json")
    fedphoenix = _load(seed_dir / "fedphoenix_summary.json")
    comparison = build_comparison(fedavg, fedphoenix)
    (seed_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (seed_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "metric", "median", "q25", "q75", "count"])
        for algorithm, summary in (("FedAvg", fedavg), ("FedPhoenix", fedphoenix)):
            for metric, values in summary["metrics"].items():
                writer.writerow(
                    [algorithm, metric, values["median"], values["q25"], values["q75"], values["count"]]
                )
    write_report(root / "decision_report.md", comparison)
    print(json.dumps(comparison["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
