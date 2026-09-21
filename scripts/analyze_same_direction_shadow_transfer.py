#!/usr/bin/env python
"""Seed-1 report for the same-direction shadow transfer diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


RATIOS = (0.02, 0.05, 0.10, 0.20, 0.40)
GAPS = ("gap_le_5", "gap_6_10", "gap_11_20", "gap_gt_20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_dir", default="results/same_direction_shadow_transfer/seed1"
    )
    parser.add_argument(
        "--output_dir", default="results/same_direction_shadow_transfer"
    )
    return parser.parse_args()


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _ratio_value(summary: Dict[str, Any], ratio: float) -> Dict[str, Any]:
    return summary["ratio_summaries"][str(ratio)]


def _or_default(value: Optional[float], default: float) -> float:
    return default if value is None else float(value)


def _meets_strong(values: Dict[str, Any]) -> bool:
    correlations = [values.get("support_pearson"), values.get("support_spearman")]
    return (
        values["event_count"] >= 30
        and values["predicted_active_count"] >= 10
        and _or_default(values["response_cosine"]["median"], -math.inf) >= 0.30
        and _or_default(values["response_nre"]["median"], math.inf) <= 0.70
        and _or_default(values["support_sign_agreement"], -math.inf) >= 0.60
        and _or_default(values["predicted_active_helpful_fraction"], -math.inf) >= 0.60
        and max((value for value in correlations if value is not None), default=-math.inf)
        >= 0.20
    )


def _meets_limited(values: Dict[str, Any]) -> bool:
    correlations = [values.get("support_pearson"), values.get("support_spearman")]
    return (
        values["event_count"] >= 20
        and values["predicted_active_count"] >= 5
        and _or_default(values["response_cosine"]["median"], -math.inf) > 0.10
        and _or_default(values["response_nre"]["median"], math.inf) < 0.90
        and _or_default(values["support_sign_agreement"], -math.inf) > 0.50
        and _or_default(values["predicted_active_helpful_fraction"], -math.inf) > 0.50
        and max((value for value in correlations if value is not None), default=-math.inf)
        > 0.0
    )


def decide(summary: Dict[str, Any]) -> Dict[str, Any]:
    small = [_ratio_value(summary, ratio) for ratio in (0.02, 0.05)]
    if all(_meets_strong(values) for values in small):
        return {
            "category": "STRONG SUPPORT",
            "reason": (
                "Both fixed small ratios satisfy all pre-registered direction, "
                "magnitude, support-sign, active-helpfulness, count, and correlation criteria."
            ),
        }
    if any(_meets_limited(values) for values in small):
        return {
            "category": "LIMITED SUPPORT",
            "reason": (
                "At least one fixed small ratio satisfies the pre-registered weaker "
                "transfer criteria, but the strong criteria do not hold for both."
            ),
        }
    return {
        "category": "NOT SUPPORTED",
        "reason": (
            "Neither fixed small ratio satisfies even the pre-registered limited-support "
            "combination of response, support, helpfulness, and count criteria."
        ),
    }


def _mean_available(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def write_report(
    path: Path,
    summary: Dict[str, Any],
    parity: Dict[str, Any],
    decision: Dict[str, Any],
) -> None:
    rows = {ratio: _ratio_value(summary, ratio) for ratio in RATIOS}
    small_cos = _mean_available(
        [rows[ratio]["response_cosine"]["median"] for ratio in (0.02, 0.05)]
    )
    large_cos = _mean_available(
        [rows[ratio]["response_cosine"]["median"] for ratio in (0.20, 0.40)]
    )
    small_nre = _mean_available(
        [rows[ratio]["response_nre"]["median"] for ratio in (0.02, 0.05)]
    )
    large_nre = _mean_available(
        [rows[ratio]["response_nre"]["median"] for ratio in (0.20, 0.40)]
    )
    smaller_more_reliable = (
        small_cos is not None
        and large_cos is not None
        and small_nre is not None
        and large_nre is not None
        and small_cos > large_cos
        and small_nre < large_nre
    )
    max_alignment_error = max(
        (rows[ratio]["alignment_identity_abs_error"]["q75"] or 0.0)
        for ratio in RATIOS
    )
    lines = [
        "# Same-experienced-direction shadow transfer diagnostic",
        "",
        "This is a seed-1 falsification-style diagnostic. The main trajectory is ordinary FedAvg; every shadow result is excluded from aggregation.",
        "",
        "## Isolation and parity",
        "",
        f"- 10-round parity passed: **{parity['passed']}**",
        f"- Selected clients match: {parity['selected_clients_match']}",
        f"- Global-state SHA-256 matches every round: {parity['global_state_sha256_match']}",
        f"- Maximum accuracy absolute delta: {parity['max_accuracy_abs_delta']}",
        f"- Shadow events exercised during parity: {parity['shadow_event_count']}",
        "",
        "## Ratio summary",
        "",
        "| Ratio | Events | Cosine median | NRE median | Norm ratio median | Sign agreement | Pearson | Spearman | Active count | Helpful fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in RATIOS:
        values = rows[ratio]
        lines.append(
            f"| {ratio:.2f} | {values['event_count']} | "
            f"{_fmt(values['response_cosine']['median'])} | "
            f"{_fmt(values['response_nre']['median'])} | "
            f"{_fmt(values['response_norm_ratio']['median'])} | "
            f"{_fmt(values['support_sign_agreement'])} | "
            f"{_fmt(values['support_pearson'])} | {_fmt(values['support_spearman'])} | "
            f"{values['predicted_active_count']} | "
            f"{_fmt(values['predicted_active_helpful_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Participation-gap breakdown",
            "",
            "| Ratio | Gap | Count | Cosine median | NRE median | Sign agreement | Active count | Helpful fraction |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in RATIOS:
        for gap in GAPS:
            values = summary["gap_summaries"][str(ratio)][gap]
            lines.append(
                f"| {ratio:.2f} | {gap} | {values['event_count']} | "
                f"{_fmt(values['response_cosine']['median'])} | "
                f"{_fmt(values['response_nre']['median'])} | "
                f"{_fmt(values['support_sign_agreement'])} | "
                f"{values['predicted_active_count']} | "
                f"{_fmt(values['predicted_active_helpful_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment does not ask whether the next natural global displacement resembles the old direction. It actively reuses the old experienced direction at a future nearby global state and tests whether the old response observation remains predictive. That is the direct test of Assumption 3.",
            "",
            "## Required questions",
            "",
            f"**Q1. Is old h directionally consistent with future same-direction h? No meaningful full-vector consistency.** Median cosine only ranges from {_fmt(min(rows[ratio]['response_cosine']['median'] for ratio in RATIOS))} to {_fmt(max(rows[ratio]['response_cosine']['median'] for ratio in RATIOS))}, i.e. nearly orthogonal even though the sign is slightly positive.",
            "",
            f"**Q2. Does magnitude transfer? No at the small local ratios.** At ratio 0.02/0.05, median norm ratios are {_fmt(rows[0.02]['response_norm_ratio']['median'])}/{_fmt(rows[0.05]['response_norm_ratio']['median'])} and median NRE is {_fmt(rows[0.02]['response_nre']['median'])}/{_fmt(rows[0.05]['response_nre']['median'])}. Even ratio 0.40 retains NRE {_fmt(rows[0.40]['response_nre']['median'])} and norm ratio {_fmt(rows[0.40]['response_norm_ratio']['median'])}.",
            "",
            f"**Q3. Does predicted support predict realized support? Weakly as a one-dimensional projection, but not as a reusable response vector.** Sign agreement rises from {_fmt(rows[0.02]['support_sign_agreement'])} to {_fmt(rows[0.40]['support_sign_agreement'])}; Spearman rises from {_fmt(rows[0.02]['support_spearman'])} to {_fmt(rows[0.40]['support_spearman'])}. This scalar signal coexists with near-zero full-vector cosine.",
            "",
            f"**Q4. Are controller-relevant predicted-active actions actually helpful? Not reliably in the most local regime.** All ratios have {rows[0.02]['predicted_active_count']} predicted-active events. Helpful fraction is {_fmt(rows[0.02]['predicted_active_helpful_fraction'])} at 0.02 and {_fmt(rows[0.05]['predicted_active_helpful_fraction'])} at 0.05, versus {_fmt(rows[0.40]['predicted_active_helpful_fraction'])} at 0.40. The smallest step is near chance and the 0.05 result is only modest.",
            "",
            f"**Q5. Is transfer more reliable at smaller ratios?** {'Yes by the predeclared joint cosine-up/NRE-down comparison.' if smaller_more_reliable else 'No consistent joint cosine-up/NRE-down advantage is established for 0.02/0.05 over 0.20/0.40.'}",
            "",
            "**Q6. Does transfer decay with participation gap? Only a mild vector-quality trend, not a clean action-level decay.** Cosine generally falls and NRE generally rises with gap, but sign agreement and predicted-active helpfulness are non-monotonic; the >20 active subset has only 10 events per ratio.",
            "",
            "## Accounting and resources",
            "",
            f"- Valid scalar events: {summary['valid_event_count']} (attempted {summary['event_count']})",
            f"- Skipped events: {summary['event_count'] - summary['valid_event_count']}",
            f"- Maximum diagnostic CPU history: {summary['max_diagnostic_cpu_history_gib']:.3f} GiB",
            f"- Representative upper-quartile alignment identity error bound across ratios: {_fmt(max_alignment_error, 8)}",
            f"- Context-only final/best FedAvg accuracy: {_fmt(summary['accuracy']['final'])} / {_fmt(summary['accuracy']['best'])}",
            "",
            "## Pre-registered decision rule",
            "",
            "STRONG SUPPORT requires both ratios 0.02 and 0.05 to have at least 30 events and 10 predicted-active events, median cosine >=0.30, median NRE <=0.70, sign agreement >=0.60, helpful fraction >=0.60, and at least one support correlation >=0.20. LIMITED SUPPORT requires at least one of those ratios to have 20/5 events, cosine >0.10, NRE <0.90, sign/helpful fractions >0.50, and a positive support correlation. Otherwise the result is NOT SUPPORTED.",
            "",
            f"## Final judgment: {decision['category']}",
            "",
            decision["reason"],
            "",
            "Accuracy is not part of this judgment, and no InteractionDispatch or FedPhoenix behavior was modified.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
    parity = json.loads(
        (output / "parity_seed1" / "parity_report.json").read_text(encoding="utf-8")
    )
    decision = decide(summary)
    result = {
        "seed": 1,
        "decision": decision,
        "parity": parity,
        "max_diagnostic_cpu_history_gib": summary["max_diagnostic_cpu_history_gib"],
        "event_count": summary["event_count"],
        "valid_event_count": summary["valid_event_count"],
    }
    (output / "decision_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output / "decision_report.md", summary, parity, decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
