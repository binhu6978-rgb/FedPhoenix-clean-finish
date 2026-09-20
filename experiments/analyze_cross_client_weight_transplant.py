#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consolidate the completed cross-client transplantation prestudy runs."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = {
    "probe_0.2": RESULTS / "cross_client_transplant_vgg_cifar10_a03_seed1",
    "probe_0.5": RESULTS / "cross_client_transplant_vgg_cifar10_a03_probe05_seed1",
}
PURE_DIR = RESULTS / "fedavg_vgg_cifar10_a03_seed1_500"
OUTPUT = RESULTS / "cross_client_transplant_comparison"

SCOPES = ("all", "conv", "classifier")
BASE_METRICS = (
    "norm_WA_minus_Wt",
    "norm_WB_minus_Wt",
    "norm_WBA_minus_WB",
    "norm_WB_minus_WA",
    "norm_WBA_minus_WA",
    "recovery_pull_score",
    "cos_WBA_minus_WB__WA_minus_Wt",
    "cos_WBA_minus_WB__WB_minus_Wt",
    "cos_WBA_minus_Wt__WA_minus_Wt",
    "cos_WBA_minus_Wt__WB_minus_Wt",
)
SIGNATURE_BASES = (
    "recovery_pull_score",
    "cos_WBA_minus_WB__WA_minus_Wt",
    "cos_WBA_minus_WB__WB_minus_Wt",
    "cos_WBA_minus_Wt__WA_minus_Wt",
    "cos_WBA_minus_Wt__WB_minus_Wt",
)
STAGES = ("rounds_1_150", "rounds_151_300", "rounds_301_500")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_pure_fedavg() -> pd.DataFrame:
    text = (PURE_DIR / "run.log").read_text(encoding="utf-8")
    matches = re.findall(
        r"ROUND_ACCURACY method=FedAvg round=(\d+) accuracy=([0-9.]+)", text
    )
    return pd.DataFrame(
        [(int(round_number), float(accuracy)) for round_number, accuracy in matches],
        columns=["round", "test_accuracy"],
    )


def accuracy_summary(name: str, frame: pd.DataFrame, probe_events: int) -> dict:
    values = frame["test_accuracy"].to_numpy(dtype=float)
    best_index = int(np.argmax(values))
    return {
        "run": name,
        "rounds": len(frame),
        "probe_events": probe_events,
        "final_accuracy": float(values[-1]),
        "best_accuracy": float(values[best_index]),
        "best_round": int(frame.iloc[best_index]["round"]),
        "mean_accuracy_all_rounds": float(values.mean()),
        "mean_accuracy_last_20": float(values[-20:].mean()),
        "mean_accuracy_last_50": float(values[-50:].mean()),
        "std_accuracy_last_50": float(values[-50:].std(ddof=1)),
    }


def mean_ci(values: pd.Series) -> dict:
    clean = values.dropna().to_numpy(dtype=float)
    n = len(clean)
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n else float("nan")
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    test = stats.ttest_1samp(clean, 0.0) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(clean)),
        "q05": float(np.quantile(clean, 0.05)),
        "q25": float(np.quantile(clean, 0.25)),
        "q75": float(np.quantile(clean, 0.75)),
        "q95": float(np.quantile(clean, 0.95)),
        "sem": sem,
        "ci95_low": mean - critical * sem if n > 1 else float("nan"),
        "ci95_high": mean + critical * sem if n > 1 else float("nan"),
        "positive_fraction": float(np.mean(clean > 0)),
        "t_statistic_vs_zero": float(test.statistic) if test else float("nan"),
        "p_value_vs_zero": float(test.pvalue) if test else float("nan"),
    }


def categorical_additive_sse(frame: pd.DataFrame, y: np.ndarray, factors) -> float:
    prediction = np.full(y.shape, float(y.mean()), dtype=float)
    effects = {
        factor: {level: 0.0 for level in frame[factor].unique()} for factor in factors
    }
    for _ in range(300):
        max_change = 0.0
        intercept_delta = float(np.mean(y - prediction))
        prediction += intercept_delta
        max_change = max(max_change, abs(intercept_delta))
        for factor in factors:
            old = effects[factor]
            new = {}
            groups = frame.groupby(factor, sort=False).indices
            for level, indices in groups.items():
                idx = np.asarray(indices, dtype=int)
                new[level] = float(np.mean(y[idx] - prediction[idx] + old[level]))
            weighted_mean = sum(new[level] * len(indices) for level, indices in groups.items()) / len(frame)
            for level in new:
                new[level] -= weighted_mean
            for level, indices in groups.items():
                delta = new[level] - old[level]
                prediction[np.asarray(indices, dtype=int)] += delta
                max_change = max(max_change, abs(delta))
            effects[factor] = new
        if max_change < 1e-12:
            break
    residual = y - prediction
    return float(np.sum(residual * residual))


def partial_r2(frame: pd.DataFrame, metric: str, tested_factor: str, controls) -> float:
    clean = frame[[metric, tested_factor, *controls]].dropna().reset_index(drop=True)
    y = clean[metric].to_numpy(dtype=float)
    reduced = categorical_additive_sse(clean, y, tuple(controls))
    full = categorical_additive_sse(clean, y, tuple(controls) + (tested_factor,))
    return float((reduced - full) / reduced) if reduced > 0 else float("nan")


def variance_decomposition(frame: pd.DataFrame, metric: str) -> dict:
    groups = frame.groupby("target_client_id")[metric]
    means = groups.mean()
    within_ss = 0.0
    within_df = 0
    for _, values in groups:
        array = values.dropna().to_numpy(dtype=float)
        if len(array) > 1:
            within_ss += float(np.sum((array - array.mean()) ** 2))
            within_df += len(array) - 1
    within = within_ss / within_df if within_df else float("nan")
    between = float(means.var(ddof=1))
    ratio = between / within if within > 0 else float("nan")
    icc = (between - within) / (between + within) if between + within > 0 else float("nan")
    return {
        "metric": metric,
        "target_count": int(frame["target_client_id"].nunique()),
        "between_client_variance_of_target_means": between,
        "pooled_within_client_event_variance": within,
        "between_to_within_ratio": ratio,
        "descriptive_icc": icc,
        "target_partial_r2_controlling_donor_type_and_stage": partial_r2(
            frame,
            metric,
            "target_client_id",
            ("donor_dominant_class", "stage"),
        ),
        "donor_type_partial_r2_controlling_target_and_stage": partial_r2(
            frame,
            metric,
            "donor_dominant_class",
            ("target_client_id", "stage"),
        ),
    }


def stability_rows(run_name: str, frame: pd.DataFrame, metric: str):
    pivot = frame.pivot_table(
        index="target_client_id", columns="stage", values=metric, aggfunc="mean"
    )
    rows = []
    for left_index in range(len(STAGES)):
        for right_index in range(left_index + 1, len(STAGES)):
            left_stage = STAGES[left_index]
            right_stage = STAGES[right_index]
            paired = pivot[[left_stage, right_stage]].dropna()
            left = paired[left_stage].to_numpy(dtype=float)
            right = paired[right_stage].to_numpy(dtype=float)
            if len(paired) >= 3 and left.std() > 0 and right.std() > 0:
                left_centered = left - left.mean()
                right_centered = right - right.mean()
                pearson_r = float(
                    np.sum(left_centered * right_centered)
                    / math.sqrt(
                        float(np.sum(left_centered**2))
                        * float(np.sum(right_centered**2))
                    )
                )
                left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float).copy()
                right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float).copy()
                left_rank -= left_rank.mean()
                right_rank -= right_rank.mean()
                spearman_r = float(
                    np.sum(left_rank * right_rank)
                    / math.sqrt(
                        float(np.sum(left_rank**2))
                        * float(np.sum(right_rank**2))
                    )
                )
                # Correlations are descriptive here; inference is carried by
                # event-level confidence intervals and zero tests.
                pearson_p = spearman_p = float("nan")
            else:
                pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
            rows.append(
                {
                    "run": run_name,
                    "metric": metric,
                    "stage_left": left_stage,
                    "stage_right": right_stage,
                    "common_targets": len(paired),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    return rows


def draw_accuracy_plot(frame: pd.DataFrame, path: Path) -> None:
    width, height = 1600, 920
    left, right, top, bottom = 125, 55, 95, 120
    plot_width, plot_height = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 28), "CIFAR-10 FedAvg accuracy comparison", fill=(20, 20, 20), font=font)
    colors = {
        "transplant_probe_0.2": (31, 119, 180),
        "transplant_probe_0.5": (44, 160, 44),
        "pure_fedavg": (214, 39, 40),
    }
    for tick in range(0, 101, 10):
        y = top + plot_height - tick / 100.0 * plot_height
        draw.line((left, y, left + plot_width, y), fill=(226, 226, 226), width=1)
        draw.text((65, y - 6), str(tick), fill=(50, 50, 50), font=font)
    for tick in range(0, 501, 50):
        x = left + tick / 500.0 * plot_width
        draw.line((x, top, x, top + plot_height), fill=(238, 238, 238), width=1)
        draw.text((x - 9, top + plot_height + 15), str(tick), fill=(50, 50, 50), font=font)
    draw.line((left, top, left, top + plot_height), fill=(25, 25, 25), width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill=(25, 25, 25), width=2)
    for column, color in colors.items():
        points = []
        for round_number, accuracy in zip(frame["round"], frame[column]):
            x = left + float(round_number) / 500.0 * plot_width
            y = top + plot_height - float(accuracy) / 100.0 * plot_height
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    legend_x, legend_y = left + 25, top + 20
    for index, (column, color) in enumerate(colors.items()):
        y = legend_y + index * 24
        draw.line((legend_x, y + 5, legend_x + 30, y + 5), fill=color, width=4)
        draw.text((legend_x + 40, y), column, fill=(30, 30, 30), font=font)
    draw.text((left + plot_width // 2 - 55, height - 45), "Communication round", fill=(20, 20, 20), font=font)
    draw.text((12, top + plot_height // 2), "Accuracy (%)", fill=(20, 20, 20), font=font)
    image.save(path, format="PNG")


def draw_stage_plot(stage_frame: pd.DataFrame, path: Path) -> None:
    data = stage_frame[
        (stage_frame["run"] == "probe_0.5")
        & (stage_frame["metric"].isin([
            "recovery_pull_score",
            "cos_WBA_minus_WB__WA_minus_Wt",
            "cos_WBA_minus_WB__WB_minus_Wt",
        ]))
    ].copy()
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((70, 25), "Probe 0.5: key metrics by stage and parameter scope", fill=(20, 20, 20), font=font)
    panel_specs = [
        ("recovery_pull_score", 70, 110, 720, 680, -0.65, 0.2, "Recovery / pull score"),
        ("cos_WBA_minus_WB__WA_minus_Wt", 850, 110, 680, 290, 0.0, 0.6, "Shadow step cosine with A update"),
        ("cos_WBA_minus_WB__WB_minus_Wt", 850, 500, 680, 290, -0.4, 0.1, "Shadow step cosine with B update"),
    ]
    colors = {"all": (31, 119, 180), "conv": (255, 127, 14), "classifier": (44, 160, 44)}
    stage_labels = ["1-150", "151-300", "301-500"]
    for metric, x0, y0, pw, ph, ymin, ymax, title in panel_specs:
        draw.text((x0, y0 - 28), title, fill=(30, 30, 30), font=font)
        zero_y = y0 + ph - (0 - ymin) / (ymax - ymin) * ph
        draw.line((x0, zero_y, x0 + pw, zero_y), fill=(120, 120, 120), width=1)
        for idx, stage in enumerate(STAGES):
            center = x0 + (idx + 0.5) * pw / 3
            draw.text((center - 23, y0 + ph + 15), stage_labels[idx], fill=(40, 40, 40), font=font)
            for scope_index, scope in enumerate(SCOPES):
                row = data[(data["stage"] == stage) & (data["scope"] == scope) & (data["metric"] == metric)].iloc[0]
                value = float(row["mean"])
                bar_width = 42
                x = center + (scope_index - 1) * 52 - bar_width / 2
                value_y = y0 + ph - (value - ymin) / (ymax - ymin) * ph
                top_y, bottom_y = min(value_y, zero_y), max(value_y, zero_y)
                draw.rectangle((x, top_y, x + bar_width, bottom_y), fill=colors[scope])
        draw.rectangle((x0, y0, x0 + pw, y0 + ph), outline=(30, 30, 30), width=2)
    lx, ly = 90, 820
    for idx, scope in enumerate(SCOPES):
        x = lx + idx * 170
        draw.rectangle((x, ly, x + 22, ly + 14), fill=colors[scope])
        draw.text((x + 30, ly), scope, fill=(30, 30, 30), font=font)
    image.save(path, format="PNG")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    event_frames = {name: pd.read_csv(path / "probe_events.csv") for name, path in RUNS.items()}
    accuracy_frames = {
        name: pd.read_csv(path / "fedavg_accuracy_curve.csv") for name, path in RUNS.items()
    }
    pure = parse_pure_fedavg()

    qc = {
        "stderr_empty": {
            name: (path / "run.err.log").stat().st_size == 0 for name, path in RUNS.items()
        },
        "pure_fedavg_stderr_empty": (PURE_DIR / "run.err.log").stat().st_size == 0,
        "round_counts": {name: len(frame) for name, frame in accuracy_frames.items()},
        "pure_fedavg_round_count": len(pure),
        "probe_event_counts": {name: len(frame) for name, frame in event_frames.items()},
        "target_ne_donor": {
            name: bool((frame["target_client_id"] != frame["donor_client_id"]).all())
            for name, frame in event_frames.items()
        },
        "all_metric_values_finite": {
            name: bool(np.isfinite(frame[[f"{base}_{scope}" for base in BASE_METRICS for scope in SCOPES]].to_numpy()).all())
            for name, frame in event_frames.items()
        },
    }
    p02_accuracy = accuracy_frames["probe_0.2"]["test_accuracy"].to_numpy()
    p05_accuracy = accuracy_frames["probe_0.5"]["test_accuracy"].to_numpy()
    qc["transplant_accuracy_exact_match"] = bool(np.array_equal(p02_accuracy, p05_accuracy))
    qc["transplant_accuracy_max_abs_difference"] = float(np.max(np.abs(p02_accuracy - p05_accuracy)))
    expected = {
        "round_counts": {"probe_0.2": 500, "probe_0.5": 500},
        "events": {"probe_0.2": 1000, "probe_0.5": 2500},
    }
    qc["passed"] = bool(
        all(qc["stderr_empty"].values())
        and qc["pure_fedavg_stderr_empty"]
        and qc["round_counts"] == expected["round_counts"]
        and qc["pure_fedavg_round_count"] == 500
        and qc["probe_event_counts"] == expected["events"]
        and all(qc["target_ne_donor"].values())
        and all(qc["all_metric_values_finite"].values())
        and qc["transplant_accuracy_exact_match"]
    )
    if not qc["passed"]:
        raise RuntimeError(f"QC failed: {qc}")

    accuracy_comparison = pd.DataFrame(
        {
            "round": accuracy_frames["probe_0.2"]["round"].astype(int),
            "transplant_probe_0.2": p02_accuracy,
            "transplant_probe_0.5": p05_accuracy,
            "pure_fedavg": pure["test_accuracy"].to_numpy(),
        }
    )
    accuracy_comparison.to_csv(OUTPUT / "accuracy_comparison.csv", index=False)
    pure.to_csv(OUTPUT / "pure_fedavg_accuracy_curve.csv", index=False)

    run_rows = [
        accuracy_summary("transplant_probe_0.2", accuracy_frames["probe_0.2"], 1000),
        accuracy_summary("transplant_probe_0.5", accuracy_frames["probe_0.5"], 2500),
        accuracy_summary("pure_fedavg", pure, 0),
    ]
    pd.DataFrame(run_rows).to_csv(OUTPUT / "run_summary.csv", index=False)

    metric_rows = []
    stage_key_rows = []
    per_client_rows = []
    variance_rows = []
    stability = []
    for run_name, frame in event_frames.items():
        for stage in ("overall", *STAGES):
            subset = frame if stage == "overall" else frame[frame["stage"] == stage]
            for scope in SCOPES:
                for base in BASE_METRICS:
                    metric = f"{base}_{scope}"
                    metric_rows.append(
                        {
                            "run": run_name,
                            "stage": stage,
                            "scope": scope,
                            "metric": base,
                            **mean_ci(subset[metric]),
                        }
                    )
                    if stage != "overall" and base in SIGNATURE_BASES:
                        stage_key_rows.append(
                            {
                                "run": run_name,
                                "stage": stage,
                                "scope": scope,
                                "metric": base,
                                **mean_ci(subset[metric]),
                            }
                        )
        for client_id, subset in frame.groupby("target_client_id"):
            row = {
                "run": run_name,
                "target_client_id": int(client_id),
                "probe_count": len(subset),
                "unique_donor_count": int(subset["donor_client_id"].nunique()),
            }
            for scope in SCOPES:
                for base in SIGNATURE_BASES:
                    values = subset[f"{base}_{scope}"]
                    row[f"{base}_{scope}_mean"] = float(values.mean())
                    row[f"{base}_{scope}_std"] = float(values.std(ddof=1))
            per_client_rows.append(row)
        for scope in SCOPES:
            for base in SIGNATURE_BASES:
                metric = f"{base}_{scope}"
                variance_rows.append({"run": run_name, "scope": scope, **variance_decomposition(frame, metric)})
                stability.extend(stability_rows(run_name, frame, metric))

    metric_summary = pd.DataFrame(metric_rows)
    stage_key = pd.DataFrame(stage_key_rows)
    per_client = pd.DataFrame(per_client_rows)
    variance_frame = pd.DataFrame(variance_rows)
    stability_frame = pd.DataFrame(stability)
    metric_summary.to_csv(OUTPUT / "metric_summary.csv", index=False)
    stage_key.to_csv(OUTPUT / "stage_summary.csv", index=False)
    per_client.to_csv(OUTPUT / "per_client_summary.csv", index=False)
    variance_frame.to_csv(OUTPUT / "variance_decomposition.csv", index=False)
    stability_frame.to_csv(OUTPUT / "stage_stability.csv", index=False)

    draw_accuracy_plot(accuracy_comparison, OUTPUT / "accuracy_comparison.png")
    draw_stage_plot(stage_key, OUTPUT / "key_metrics_by_stage.png")

    def overall(run_name, base, scope):
        row = metric_summary[
            (metric_summary["run"] == run_name)
            & (metric_summary["stage"] == "overall")
            & (metric_summary["metric"] == base)
            & (metric_summary["scope"] == scope)
        ].iloc[0]
        return row.to_dict()

    def variance_item(run_name, metric):
        return variance_frame[(variance_frame["run"] == run_name) & (variance_frame["metric"] == metric)].iloc[0].to_dict()

    p05 = event_frames["probe_0.5"]
    coverage = p05.groupby("target_client_id").agg(
        probe_count=("round", "size"), unique_donor_count=("donor_client_id", "nunique")
    )
    recovery = {scope: overall("probe_0.5", "recovery_pull_score", scope) for scope in SCOPES}
    step_a = {scope: overall("probe_0.5", "cos_WBA_minus_WB__WA_minus_Wt", scope) for scope in SCOPES}
    step_b = {scope: overall("probe_0.5", "cos_WBA_minus_WB__WB_minus_Wt", scope) for scope in SCOPES}
    endpoint_a = {scope: overall("probe_0.5", "cos_WBA_minus_Wt__WA_minus_Wt", scope) for scope in SCOPES}
    endpoint_b = {scope: overall("probe_0.5", "cos_WBA_minus_Wt__WB_minus_Wt", scope) for scope in SCOPES}
    all_recovery_var = variance_item("probe_0.5", "recovery_pull_score_all")
    all_step_a_var = variance_item("probe_0.5", "cos_WBA_minus_WB__WA_minus_Wt_all")

    p05_stability = stability_frame[
        (stability_frame["run"] == "probe_0.5")
        & (stability_frame["metric"].isin([
            "recovery_pull_score_all",
            "cos_WBA_minus_WB__WA_minus_Wt_all",
        ]))
    ]
    stability_ranges = {
        metric: {
            "pearson_min": float(group["pearson_r"].min()),
            "pearson_max": float(group["pearson_r"].max()),
            "spearman_min": float(group["spearman_r"].min()),
            "spearman_max": float(group["spearman_r"].max()),
        }
        for metric, group in p05_stability.groupby("metric")
    }

    summary = {
        "qc": qc,
        "primary_run_for_inference": "probe_0.5 (2500 events; denser per-target coverage)",
        "replication": {
            "probe_0.2_events": 1000,
            "probe_0.5_events": 2500,
            "global_accuracy_trajectories_exactly_equal": qc["transplant_accuracy_exact_match"],
            "max_absolute_accuracy_difference": qc["transplant_accuracy_max_abs_difference"],
        },
        "accuracy": {row["run"]: row for row in run_rows},
        "core_metrics_probe_0.5": {
            "recovery": recovery,
            "shadow_step_alignment_with_A": step_a,
            "shadow_step_alignment_with_B": step_b,
            "endpoint_alignment_with_A": endpoint_a,
            "endpoint_alignment_with_B": endpoint_b,
        },
        "questions": {
            "q1_same_target_across_donors": {
                "answer": "存在弱到中等 target signal，但远非 donor-invariant pattern；同一 target 内事件方差明显更大。",
                "mean_probes_per_target": float(coverage["probe_count"].mean()),
                "mean_unique_donors_per_target": float(coverage["unique_donor_count"].mean()),
                "recovery_target_partial_r2": all_recovery_var["target_partial_r2_controlling_donor_type_and_stage"],
                "step_A_target_partial_r2": all_step_a_var["target_partial_r2_controlling_donor_type_and_stage"],
            },
            "q2_different_targets_same_donor_type": {
                "answer": "target identity 在控制 donor dominant class 和阶段后仍解释约一成响应变异，差异可检测但不占主导。",
                "recovery_target_partial_r2": all_recovery_var["target_partial_r2_controlling_donor_type_and_stage"],
                "step_A_target_partial_r2": all_step_a_var["target_partial_r2_controlling_donor_type_and_stage"],
                "definition_of_same_type": "donor client 固定本地标签分布的 dominant class 相同",
            },
            "q3_stage_stability": {
                "answer": "存在中等稳定性，probe 0.5 中 recovery 和 A-alignment 的跨阶段 target-mean Pearson 相关约 0.4-0.52，但不是强稳定。",
                "correlation_ranges": stability_ranges,
            },
            "q4_between_vs_within": {
                "answer": "否。within-client event variance 明显大于 between-client variance of means。",
                "recovery_between_to_within_ratio": all_recovery_var["between_to_within_ratio"],
                "step_A_between_to_within_ratio": all_step_a_var["between_to_within_ratio"],
            },
            "q5_recovery_nonzero": {
                "answer": "系统性非零，但方向为负：整体/卷积/分类器的均值 95% CI 均低于 0。",
                "by_scope": recovery,
            },
        },
        "overall_interpretation": (
            "A 对 donor model 的训练步具有 A-specific directional signature：shadow step 与 A update 正相关、"
            "与 B update 负相关，classifier 最强。但 5 个 local epochs 后，W_B->A 通常没有回到 W_A 附近；"
            "recovery/pull score 在整体和卷积参数上显著为负。因此支持弱/局部的 client-specific transformation，"
            "不支持强 endpoint recovery 或 donor-invariant client operator。"
        ),
        "important_comparison_limit": (
            "transplantation 脚本为保证 A baseline/shadow 配对而显式按 round/client 重置本地 RNG；"
            "原生 main_fed.py 使用连续全局 RNG。因此纯 FedAvg 曲线可作性能 sanity check，不能作逐轮 bitwise control。"
        ),
    }
    write_json(OUTPUT / "comparison_summary.json", summary)
    write_json(OUTPUT / "analysis_qc.json", qc)

    r_all, r_conv, r_cls = recovery["all"], recovery["conv"], recovery["classifier"]
    run_table = pd.DataFrame(run_rows).set_index("run")
    report = f"""# Cross-client weight transplantation prestudy 结果

## 结论

实验观察到 **A-specific 的局部方向性响应**，但没有观察到强意义上的 endpoint recovery。

- 在更高统计功效的 `probe_fraction=0.5` 运行中，shadow training step 与 A 的正常 update 呈正相关：all `{step_a['all']['mean']:.3f}`、conv `{step_a['conv']['mean']:.3f}`、classifier `{step_a['classifier']['mean']:.3f}`。
- 同一个 shadow step 与 donor B 的 update 呈负相关：all `{step_b['all']['mean']:.3f}`、conv `{step_b['conv']['mean']:.3f}`、classifier `{step_b['classifier']['mean']:.3f}`。分类器的 A-specific 方向信号最强。
- 但 recovery/pull score 的均值为负：all `{r_all['mean']:.3f}`（95% CI `{r_all['ci95_low']:.3f}`–`{r_all['ci95_high']:.3f}`）、conv `{r_conv['mean']:.3f}`、classifier `{r_cls['mean']:.3f}`。`W_B→A` 通常比 donor 起点 `W_B` 更远离 A 的正常终点 `W_A`。
- 因此，数据支持“客户端数据会系统性改变输入模型的更新方向”，不支持“5 个 local epochs 足以把任意同轮 donor model 拉回 A 自己的正常 endpoint”。

## 数据完整性

| Run | Rounds | Probe events | Final accuracy | Best accuracy | Last-50 mean |
|---|---:|---:|---:|---:|---:|
| transplant, probe 0.2 | 500 | 1,000 | {run_table.loc['transplant_probe_0.2','final_accuracy']:.2f}% | {run_table.loc['transplant_probe_0.2','best_accuracy']:.2f}% | {run_table.loc['transplant_probe_0.2','mean_accuracy_last_50']:.2f}% |
| transplant, probe 0.5 | 500 | 2,500 | {run_table.loc['transplant_probe_0.5','final_accuracy']:.2f}% | {run_table.loc['transplant_probe_0.5','best_accuracy']:.2f}% | {run_table.loc['transplant_probe_0.5','mean_accuracy_last_50']:.2f}% |
| original FedAvg | 500 | 0 | {run_table.loc['pure_fedavg','final_accuracy']:.2f}% | {run_table.loc['pure_fedavg','best_accuracy']:.2f}% | {run_table.loc['pure_fedavg','mean_accuracy_last_50']:.2f}% |

三份 stderr 均为空。两个 transplantation 运行的 500 点 global accuracy 逐点完全相同，最大绝对差为 0。这证明增加 shadow probe 数量没有改变 global trajectory。完整 500 轮未启用逐轮 full-state SHA-256；5-round smoke test 已完成该哈希隔离验证。

原生 FedAvg 与 transplantation 的总体性能接近，但不能把逐轮差异解释成 probe 效应。transplantation 脚本为了让 A baseline 与 shadow 使用相同 shuffle/dropout 轨迹，按 round/client 显式重置本地 RNG；原生 `main_fed.py` 使用连续全局 RNG。

## 五个核心问题

### 1. 同一个 target 换不同 donor 后，是否有相似 transformation pattern？

有弱到中等的 target signal，但不够强，不能称为 donor-invariant。`probe 0.5` 中每个 target 平均被 probe `{coverage['probe_count'].mean():.1f}` 次，覆盖 `{coverage['unique_donor_count'].mean():.1f}` 个不同 donor。控制 donor dominant class 和阶段后，target identity 对 recovery 的 partial R² 为 `{all_recovery_var['target_partial_r2_controlling_donor_type_and_stage']:.3f}`，对 shadow-step/A-update cosine 为 `{all_step_a_var['target_partial_r2_controlling_donor_type_and_stage']:.3f}`。大部分变异仍来自同一 target 内的 event/donor/round 差异。

### 2. 不同 target 对同类 donor input 的处理是否明显不同？

差异可检测，但不占主导。这里“同类 donor”定义为 donor 本地标签分布的 dominant class 相同。target identity 在控制 donor 类型与阶段后解释约 9%–11% 的关键响应变异。

### 3. 同一个 target 的 response 在训练前中后期是否稳定？

存在中等稳定性，但不是强稳定。`probe 0.5` 中 target-level recovery 跨阶段 Pearson 相关为 `{stability_ranges['recovery_pull_score_all']['pearson_min']:.3f}`–`{stability_ranges['recovery_pull_score_all']['pearson_max']:.3f}`；shadow-step/A-update cosine 为 `{stability_ranges['cos_WBA_minus_WB__WA_minus_Wt_all']['pearson_min']:.3f}`–`{stability_ranges['cos_WBA_minus_WB__WA_minus_Wt_all']['pearson_max']:.3f}`。

阶段均值也在变化：all-parameter recovery 从前期 `{stage_key[(stage_key.run=='probe_0.5') & (stage_key.stage=='rounds_1_150') & (stage_key.scope=='all') & (stage_key.metric=='recovery_pull_score')]['mean'].iloc[0]:.3f}`，变为中期 `{stage_key[(stage_key.run=='probe_0.5') & (stage_key.stage=='rounds_151_300') & (stage_key.scope=='all') & (stage_key.metric=='recovery_pull_score')]['mean'].iloc[0]:.3f}`、后期 `{stage_key[(stage_key.run=='probe_0.5') & (stage_key.stage=='rounds_301_500') & (stage_key.scope=='all') & (stage_key.metric=='recovery_pull_score')]['mean'].iloc[0]:.3f}`。后期 endpoint recovery 更弱。

### 4. Between-client variance 是否明显大于 within-client variance？

否，方向相反。`probe 0.5` 的 recovery between/within variance ratio 为 `{all_recovery_var['between_to_within_ratio']:.3f}`，shadow-step/A-update cosine 为 `{all_step_a_var['between_to_within_ratio']:.3f}`。within-client event variance 分别约为 between-target-means variance 的 `{1/all_recovery_var['between_to_within_ratio']:.1f}` 倍和 `{1/all_step_a_var['between_to_within_ratio']:.1f}` 倍。

### 5. Recovery/pull score 是否系统性非零？

是，但为系统性负值，不是期望的正 pull。all 参数仅 `{100*r_all['positive_fraction']:.1f}%` 事件为正，conv `{100*r_conv['positive_fraction']:.1f}%`，classifier `{100*r_cls['positive_fraction']:.1f}%`。classifier 的中位数接近 0，存在较强 A-specific 方向信号，但负尾部使均值显著为负。

## 参数范围差异

- Conv 参数主导 all-parameter 结果：recovery 明显为负，A-alignment 为正但较弱。
- Classifier 参数的转换最像 A：shadow step 与 A update cosine 为 `{step_a['classifier']['mean']:.3f}`，与 B update 为 `{step_b['classifier']['mean']:.3f}`；endpoint 与 A update cosine `{endpoint_a['classifier']['mean']:.3f}`，高于与 B update 的 `{endpoint_b['classifier']['mean']:.3f}`。
- 即便如此，classifier recovery 均值仍为 `{r_cls['mean']:.3f}`，且正 recovery 事件比例为 `{100*r_cls['positive_fraction']:.1f}%`。方向像 A 不等于端点距离一定更接近 `W_A`。

## 解释边界

- 每个事件只有一个 donor 输入和一次配对 stochastic trajectory；结论描述本实验定义下的平均现象。
- donor “同类”使用 dominant label class，是粗粒度 operational definition。
- `probe 0.2` 与 `0.5` 的总体统计高度一致；主要推断优先采用事件更多的 `probe 0.5`。
- 这些结果用于验证现象，不构成新方法或优化结论。
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")

    print(f"ANALYSIS_COMPLETE {OUTPUT}")
    print(f"QC_PASSED={qc['passed']}")


if __name__ == "__main__":
    main()
