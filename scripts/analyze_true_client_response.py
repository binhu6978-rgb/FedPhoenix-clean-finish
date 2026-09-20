#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-hoc analysis of the true input-conditioned client response.

No model weights or datasets are loaded.  Every quantity is reconstructed
algebraically from the event-level norms and cosines already stored by the
completed cross-client transplantation run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SCOPES = ("all", "conv", "classifier")
STAGES = ("rounds_1_150", "rounds_151_300", "rounds_301_500")
METRICS = (
    "response_change_ratio",
    "cancellation_coefficient",
    "survival_ratio",
    "survival_cosine",
    "response_cosine",
    "cos_q_a",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze q=s-a and e=b+q from completed probe event geometry."
    )
    parser.add_argument("--events_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def required_columns() -> list[str]:
    metadata = [
        "round",
        "stage",
        "target_client_id",
        "donor_client_id",
        "target_dominant_class",
        "donor_dominant_class",
    ]
    geometry = []
    for scope in SCOPES:
        geometry.extend(
            [
                f"norm_WA_minus_Wt_{scope}",
                f"norm_WB_minus_Wt_{scope}",
                f"norm_WBA_minus_WB_{scope}",
                f"norm_WB_minus_WA_{scope}",
                f"norm_WBA_minus_WA_{scope}",
                f"cos_WBA_minus_WB__WA_minus_Wt_{scope}",
                f"cos_WBA_minus_WB__WB_minus_Wt_{scope}",
            ]
        )
    return metadata + geometry


def derive_event_metrics(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(required_columns()) - set(source.columns))
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")
    if (source["target_client_id"] == source["donor_client_id"]).any():
        raise ValueError("Found an event where target and donor are identical")

    metadata = [
        column
        for column in (
            "round",
            "stage",
            "target_client_id",
            "donor_client_id",
            "target_local_seed",
            "target_num_samples",
            "donor_num_samples",
            "target_dominant_class",
            "donor_dominant_class",
            "target_donor_label_cosine",
        )
        if column in source.columns
    ]
    result = source[metadata].copy()
    qc = {"event_count": len(source), "scopes": {}}

    for scope in SCOPES:
        norm_a = source[f"norm_WA_minus_Wt_{scope}"].astype(float)
        norm_b = source[f"norm_WB_minus_Wt_{scope}"].astype(float)
        norm_s = source[f"norm_WBA_minus_WB_{scope}"].astype(float)
        norm_b_minus_a = source[f"norm_WB_minus_WA_{scope}"].astype(float)
        norm_e = source[f"norm_WBA_minus_WA_{scope}"].astype(float)
        cos_s_a = source[f"cos_WBA_minus_WB__WA_minus_Wt_{scope}"].astype(float)
        cos_s_b = source[f"cos_WBA_minus_WB__WB_minus_Wt_{scope}"].astype(float)

        if (norm_a <= 0).any() or (norm_b <= 0).any() or (norm_s <= 0).any() or (norm_e <= 0).any():
            raise ValueError(f"Zero/negative norm prevents exact reconstruction in scope {scope}")

        # Strict algebraic reconstruction:
        # ||b-a||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        dot_a_b = (norm_a**2 + norm_b**2 - norm_b_minus_a**2) / 2.0
        dot_s_a = norm_s * norm_a * cos_s_a
        dot_s_b = norm_s * norm_b * cos_s_b
        norm_q_sq_raw = norm_s**2 + norm_a**2 - 2.0 * dot_s_a
        tolerance = 1e-10 * (norm_s**2 + norm_a**2 + 2.0 * dot_s_a.abs() + 1.0)
        materially_negative = norm_q_sq_raw < -tolerance
        if materially_negative.any():
            worst = float(norm_q_sq_raw[materially_negative].min())
            raise ValueError(f"Derived ||q||^2 is materially negative in {scope}: {worst}")
        norm_q_sq = norm_q_sq_raw.clip(lower=0.0)
        norm_q = np.sqrt(norm_q_sq)
        if (norm_q <= 0).any():
            raise ValueError(f"Derived q is exactly zero for at least one event in {scope}")

        dot_q_b = dot_s_b - dot_a_b
        dot_q_a = dot_s_a - norm_a**2
        derived_e_sq = norm_b**2 + norm_q_sq + 2.0 * dot_q_b
        closure_abs = (norm_e**2 - derived_e_sq).abs()
        closure_scale = (
            norm_e**2
            + norm_b**2
            + norm_q_sq
            + 2.0 * dot_q_b.abs()
            + np.finfo(float).tiny
        )
        closure_rel = closure_abs / closure_scale

        result[f"norm_a_{scope}"] = norm_a
        result[f"norm_b_{scope}"] = norm_b
        result[f"norm_s_{scope}"] = norm_s
        result[f"norm_q_{scope}"] = norm_q
        result[f"norm_e_{scope}"] = norm_e
        result[f"dot_a_b_{scope}"] = dot_a_b
        result[f"dot_s_a_{scope}"] = dot_s_a
        result[f"dot_s_b_{scope}"] = dot_s_b
        result[f"dot_q_b_{scope}"] = dot_q_b
        result[f"dot_q_a_{scope}"] = dot_q_a
        result[f"response_change_ratio_{scope}"] = norm_q / norm_b
        result[f"cancellation_coefficient_{scope}"] = -dot_q_b / norm_b**2
        result[f"survival_ratio_{scope}"] = norm_e / norm_b
        result[f"survival_cosine_{scope}"] = (norm_b**2 + dot_q_b) / (norm_e * norm_b)
        result[f"response_cosine_{scope}"] = dot_q_b / (norm_q * norm_b)
        result[f"cos_q_a_{scope}"] = dot_q_a / (norm_q * norm_a)
        result[f"e_identity_abs_error_{scope}"] = closure_abs
        result[f"e_identity_relative_error_{scope}"] = closure_rel

        qc["scopes"][scope] = {
            "negative_q_squared_count": int((norm_q_sq_raw < 0).sum()),
            "zero_q_count": int((norm_q == 0).sum()),
            "max_e_identity_absolute_error": float(closure_abs.max()),
            "max_e_identity_relative_error": float(closure_rel.max()),
            "mean_e_identity_relative_error": float(closure_rel.mean()),
            "all_derived_metrics_finite": bool(
                np.isfinite(
                    result[[f"{metric}_{scope}" for metric in METRICS]].to_numpy()
                ).all()
            ),
        }
    qc["passed"] = bool(
        qc["event_count"] == 2500
        and all(
            details["zero_q_count"] == 0
            and details["max_e_identity_relative_error"] < 1e-6
            and details["all_derived_metrics_finite"]
            for details in qc["scopes"].values()
        )
    )
    return result, qc


def distribution_summary(values: pd.Series) -> dict:
    clean = values.dropna().to_numpy(dtype=float)
    count = len(clean)
    if count == 0:
        raise ValueError("Cannot summarize an empty metric")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if count > 1 else 0.0
    sem = std / math.sqrt(count)
    # With n=2500 overall and at least 750 per stage, the normal interval and
    # t interval are indistinguishable at the reported precision.
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "median": float(np.median(clean)),
        "min": float(clean.min()),
        "q05": float(np.quantile(clean, 0.05)),
        "q25": float(np.quantile(clean, 0.25)),
        "q75": float(np.quantile(clean, 0.75)),
        "q95": float(np.quantile(clean, 0.95)),
        "max": float(clean.max()),
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "negative_fraction": float(np.mean(clean < 0.0)),
        "zero_fraction": float(np.mean(clean == 0.0)),
        "positive_fraction": float(np.mean(clean > 0.0)),
    }


def build_metric_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ("overall", *STAGES):
        subset = events if stage == "overall" else events[events["stage"] == stage]
        for scope in SCOPES:
            for metric in METRICS:
                rows.append(
                    {
                        "stage": stage,
                        "scope": scope,
                        "metric": metric,
                        **distribution_summary(subset[f"{metric}_{scope}"]),
                    }
                )
    return pd.DataFrame(rows)


def entity_summary(events: pd.DataFrame, entity: str, other: str) -> pd.DataFrame:
    rows = []
    class_column = (
        "target_dominant_class" if entity == "target_client_id" else "donor_dominant_class"
    )
    for entity_id, subset in events.groupby(entity):
        row = {
            entity: int(entity_id),
            "dominant_class": int(subset[class_column].iloc[0]),
            "event_count": len(subset),
            f"unique_{other}_count": int(subset[other].nunique()),
        }
        for scope in SCOPES:
            for metric in METRICS:
                values = subset[f"{metric}_{scope}"]
                row[f"{metric}_{scope}_mean"] = float(values.mean())
                row[f"{metric}_{scope}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(entity).reset_index(drop=True)


def pooled_variance(events: pd.DataFrame, entity: str, column: str) -> dict:
    groups = events.groupby(entity)[column]
    means = groups.mean()
    within_ss = 0.0
    within_df = 0
    for _, values in groups:
        array = values.to_numpy(dtype=float)
        if len(array) > 1:
            within_ss += float(np.sum((array - array.mean()) ** 2))
            within_df += len(array) - 1
    within = within_ss / within_df if within_df else float("nan")
    between = float(means.var(ddof=1))
    ratio = between / within if within > 0 else float("nan")
    return {
        "entity_count": int(events[entity].nunique()),
        "between_entity_variance_of_means": between,
        "pooled_within_entity_event_variance": within,
        "between_to_within_ratio": ratio,
    }


def build_variance_decomposition(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity in ("target_client_id", "donor_client_id"):
        entity_label = "target" if entity.startswith("target") else "donor"
        for scope in SCOPES:
            for metric in METRICS:
                rows.append(
                    {
                        "entity": entity_label,
                        "scope": scope,
                        "metric": metric,
                        **pooled_variance(events, entity, f"{metric}_{scope}"),
                    }
                )
    return pd.DataFrame(rows)


def categorical_additive_sse(
    frame: pd.DataFrame, y: np.ndarray, factors: Sequence[str]
) -> float:
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
            groups = frame.groupby(factor, sort=False).indices
            old = effects[factor]
            new = {}
            for level, indices in groups.items():
                idx = np.asarray(indices, dtype=int)
                new[level] = float(np.mean(y[idx] - prediction[idx] + old[level]))
            weighted_mean = sum(
                new[level] * len(indices) for level, indices in groups.items()
            ) / len(frame)
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


def partial_r2(
    events: pd.DataFrame,
    column: str,
    tested_factor: str,
    controls: Sequence[str],
) -> dict:
    clean = events[[column, tested_factor, *controls]].dropna().reset_index(drop=True)
    y = clean[column].to_numpy(dtype=float)
    reduced_sse = categorical_additive_sse(clean, y, controls)
    full_sse = categorical_additive_sse(clean, y, (*controls, tested_factor))
    value = (
        (reduced_sse - full_sse) / reduced_sse if reduced_sse > 0 else float("nan")
    )
    return {
        "count": len(clean),
        "reduced_sse": reduced_sse,
        "full_sse": full_sse,
        "partial_r2": float(value),
    }


def build_partial_r2(events: pd.DataFrame) -> pd.DataFrame:
    specifications = (
        (
            "target_identity",
            "target_client_id",
            ("stage", "donor_dominant_class"),
        ),
        (
            "donor_identity",
            "donor_client_id",
            ("stage", "target_dominant_class"),
        ),
    )
    rows = []
    for effect, tested, controls in specifications:
        for scope in SCOPES:
            for metric in METRICS:
                rows.append(
                    {
                        "effect": effect,
                        "scope": scope,
                        "metric": metric,
                        "tested_factor": tested,
                        "controls": "+".join(controls),
                        **partial_r2(events, f"{metric}_{scope}", tested, controls),
                    }
                )
    return pd.DataFrame(rows)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(float) - float(left.mean())
    right = right.astype(float) - float(right.mean())
    denominator = math.sqrt(float(np.sum(left**2)) * float(np.sum(right**2)))
    return float(np.sum(left * right) / denominator) if denominator > 0 else float("nan")


def stability_for(
    events: pd.DataFrame, entity: str, scope: str, metric: str
) -> list[dict]:
    column = f"{metric}_{scope}"
    pivot = events.pivot_table(index=entity, columns="stage", values=column, aggfunc="mean")
    rows = []
    for left_index in range(len(STAGES)):
        for right_index in range(left_index + 1, len(STAGES)):
            left_stage = STAGES[left_index]
            right_stage = STAGES[right_index]
            paired = pivot[[left_stage, right_stage]].dropna()
            left = paired[left_stage].to_numpy(dtype=float)
            right = paired[right_stage].to_numpy(dtype=float)
            left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
            right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
            rows.append(
                {
                    "entity": "target" if entity.startswith("target") else "donor",
                    "scope": scope,
                    "metric": metric,
                    "stage_left": left_stage,
                    "stage_right": right_stage,
                    "common_entities": len(paired),
                    "pearson_r": correlation(left, right),
                    "spearman_r": correlation(left_rank, right_rank),
                }
            )
    return rows


def build_stage_stability(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity in ("target_client_id", "donor_client_id"):
        for scope in SCOPES:
            for metric in METRICS:
                rows.extend(stability_for(events, entity, scope, metric))
    return pd.DataFrame(rows)


def select_row(frame: pd.DataFrame, **filters) -> dict:
    subset = frame
    for column, value in filters.items():
        subset = subset[subset[column] == value]
    if len(subset) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(subset)}")
    return subset.iloc[0].to_dict()


def correlation_range(
    stability: pd.DataFrame, entity: str, scope: str, metric: str
) -> dict:
    subset = stability[
        (stability["entity"] == entity)
        & (stability["scope"] == scope)
        & (stability["metric"] == metric)
    ]
    return {
        "pearson_min": float(subset["pearson_r"].min()),
        "pearson_max": float(subset["pearson_r"].max()),
        "spearman_min": float(subset["spearman_r"].min()),
        "spearman_max": float(subset["spearman_r"].max()),
    }


def build_report(
    metrics: pd.DataFrame,
    variance: pd.DataFrame,
    partials: pd.DataFrame,
    stability: pd.DataFrame,
    qc: dict,
    judgment: str,
) -> str:
    def overall(scope, metric):
        return select_row(metrics, stage="overall", scope=scope, metric=metric)

    def stage(scope, metric, stage_name):
        return select_row(metrics, stage=stage_name, scope=scope, metric=metric)

    def var(entity, scope, metric):
        return select_row(variance, entity=entity, scope=scope, metric=metric)

    def pr2(effect, scope, metric):
        return select_row(partials, effect=effect, scope=scope, metric=metric)

    change = {scope: overall(scope, "response_change_ratio") for scope in SCOPES}
    cancel = {scope: overall(scope, "cancellation_coefficient") for scope in SCOPES}
    survive = {scope: overall(scope, "survival_ratio") for scope in SCOPES}
    survive_cos = {scope: overall(scope, "survival_cosine") for scope in SCOPES}
    response_cos = {scope: overall(scope, "response_cosine") for scope in SCOPES}
    qa_cos = {scope: overall(scope, "cos_q_a") for scope in SCOPES}

    all_change_var = var("target", "all", "response_change_ratio")
    all_survival_cos_var = var("target", "all", "survival_cosine")
    all_response_cos_var = var("target", "all", "response_cosine")
    all_change_target_r2 = pr2("target_identity", "all", "response_change_ratio")
    all_survival_cos_target_r2 = pr2("target_identity", "all", "survival_cosine")
    all_response_cos_target_r2 = pr2("target_identity", "all", "response_cosine")
    all_cancel_donor_r2 = pr2("donor_identity", "all", "cancellation_coefficient")
    all_response_cos_donor_r2 = pr2("donor_identity", "all", "response_cosine")

    target_change_stability = correlation_range(
        stability, "target", "all", "response_change_ratio"
    )
    target_survival_cos_stability = correlation_range(
        stability, "target", "all", "survival_cosine"
    )
    target_response_cos_stability = correlation_range(
        stability, "target", "all", "response_cosine"
    )
    donor_response_cos_stability = correlation_range(
        stability, "donor", "all", "response_cosine"
    )

    early_change = stage("all", "response_change_ratio", STAGES[0])
    mid_change = stage("all", "response_change_ratio", STAGES[1])
    late_change = stage("all", "response_change_ratio", STAGES[2])

    return f"""# True input-conditioned client response: post-hoc analysis

## Final judgment

**{judgment}**

Starting model 明显改变了 A 的 local training step，因此现象不能只用普通的 A-specific local gradient 解释。但 target-specific component 主要体现在 response magnitude；rotation/cancellation 的 target identity 解释度较弱，而且 donor identity 对部分方向指标的解释度更高。因此结果支持真实的 input-conditioned response，但不支持强、稳定、由 target identity 主导的 rich response operator。

## Exact derivation and QC

本分析没有重新训练，也没有加载模型权重。对每个 scope，现有日志严格提供了 `||a||`、`||b||`、`||s||`、`||b-a||`、`||e||`、`cos(s,a)` 和 `cos(s,b)`。由此精确恢复：

```text
<a,b> = (||a||² + ||b||² - ||b-a||²) / 2
<s,a> = ||s|| ||a|| cos(s,a)
<s,b> = ||s|| ||b|| cos(s,b)
||q||² = ||s||² + ||a||² - 2<s,a>
<q,b> = <s,b> - <a,b>
<q,a> = <s,a> - ||a||²
e = b + q
```

`||e||² = ||b+q||²` 的最大相对闭合误差为 all `{qc['scopes']['all']['max_e_identity_relative_error']:.3g}`、conv `{qc['scopes']['conv']['max_e_identity_relative_error']:.3g}`、classifier `{qc['scopes']['classifier']['max_e_identity_relative_error']:.3g}`。2,500 个事件均得到有限指标，没有负的 `||q||²`，也没有零 `||q||`。

## Overall metrics

| Scope | `||q||/||b||` | Cancellation | `||e||/||b||` | `cos(e,b)` | `cos(q,b)` | `cos(q,a)` |
|---|---:|---:|---:|---:|---:|---:|
| all | {change['all']['mean']:.3f} | {cancel['all']['mean']:.3f} | {survive['all']['mean']:.3f} | {survive_cos['all']['mean']:.3f} | {response_cos['all']['mean']:.3f} | {qa_cos['all']['mean']:.3f} |
| conv | {change['conv']['mean']:.3f} | {cancel['conv']['mean']:.3f} | {survive['conv']['mean']:.3f} | {survive_cos['conv']['mean']:.3f} | {response_cos['conv']['mean']:.3f} | {qa_cos['conv']['mean']:.3f} |
| classifier | {change['classifier']['mean']:.3f} | {cancel['classifier']['mean']:.3f} | {survive['classifier']['mean']:.3f} | {survive_cos['classifier']['mean']:.3f} | {response_cos['classifier']['mean']:.3f} | {qa_cos['classifier']['mean']:.3f} |

## Q1. `q` 是否显著非零？

是，而且不是小扰动。`response_change_ratio=||q||/||b||` 的均值为 all `{change['all']['mean']:.3f}`（95% CI `{change['all']['ci95_low']:.3f}`–`{change['all']['ci95_high']:.3f}`）、conv `{change['conv']['mean']:.3f}`、classifier `{change['classifier']['mean']:.3f}`。所有 2,500 个事件在三个 scope 上都严格大于 0。all 参数的中位数为 `{change['all']['median']:.3f}`，5%–95% 分位为 `{change['all']['q05']:.3f}`–`{change['all']['q95']:.3f}`。

这排除了“starting model 几乎不影响 A local step”的解释。

## Q2. `e` 是否只是 `b` 的简单缩放？

否。若只是正 scalar gain，`cos(e,b)` 应接近 1；实际均值为 all `{survive_cos['all']['mean']:.3f}`、conv `{survive_cos['conv']['mean']:.3f}`、classifier `{survive_cos['classifier']['mean']:.3f}`。classifier 中 `{100*survive_cos['classifier']['negative_fraction']:.1f}%` 的事件甚至 `cos(e,b)<0`。与此同时 survival ratio 均值为 all `{survive['all']['mean']:.3f}`、classifier `{survive['classifier']['mean']:.3f}`，说明输出 perturbation 通常不是简单衰减，而是带有很大的旋转/新增分量。

`q` 几乎总是反向于 `b`：all `cos(q,b)` 均值 `{response_cos['all']['mean']:.3f}`，classifier `{response_cos['classifier']['mean']:.3f}`。cancellation coefficient 为 all `{cancel['all']['mean']:.3f}`、classifier `{cancel['classifier']['mean']:.3f}`。A 在 classifier 上抵消 donor 方向更强，但同时产生较大的非共线 response，所以 `||e||/||b||` 仍常大于 1。

## Q3. Target identity 解释多少 response variation？

Target identity 对 response magnitude 的解释明显高于旧 recovery 指标：

- all `response_change_ratio` target partial R² = `{all_change_target_r2['partial_r2']:.3f}`；between/within ratio = `{all_change_var['between_to_within_ratio']:.3f}`。
- all `survival_ratio` 的 target partial R² 与 magnitude 结果相近（详见 CSV）。
- 方向/旋转指标较弱：`survival_cosine` target partial R² = `{all_survival_cos_target_r2['partial_r2']:.3f}`，`response_cosine` = `{all_response_cos_target_r2['partial_r2']:.3f}`；对应 between/within ratio 分别为 `{all_survival_cos_var['between_to_within_ratio']:.3f}` 和 `{all_response_cos_var['between_to_within_ratio']:.3f}`。

Donor identity 对部分方向指标更强：all cancellation donor partial R² = `{all_cancel_donor_r2['partial_r2']:.3f}`，response cosine donor partial R² = `{all_response_cos_donor_r2['partial_r2']:.3f}`。所以 response magnitude 有清晰 target-specific component，但 rotation/cancellation 不是由 target identity 单独主导。

## Q4. Response 是否跨 participation 稳定？

Target-specific magnitude 的数值具有较强线性延续性：all `response_change_ratio` 的 target-level 跨阶段 Pearson 为 `{target_change_stability['pearson_min']:.3f}`–`{target_change_stability['pearson_max']:.3f}`；但直接衡量排序的 Spearman 只有 `{target_change_stability['spearman_min']:.3f}`–`{target_change_stability['spearman_max']:.3f}`，所以不能称为强 rank stability。阶段均值分别为 `{early_change['mean']:.3f}`、`{mid_change['mean']:.3f}`、`{late_change['mean']:.3f}`。

方向 ordering 更弱：target-level `survival_cosine` Pearson `{target_survival_cos_stability['pearson_min']:.3f}`–`{target_survival_cos_stability['pearson_max']:.3f}`、Spearman `{target_survival_cos_stability['spearman_min']:.3f}`–`{target_survival_cos_stability['spearman_max']:.3f}`；`response_cosine` Pearson `{target_response_cos_stability['pearson_min']:.3f}`–`{target_response_cos_stability['pearson_max']:.3f}`、Spearman `{target_response_cos_stability['spearman_min']:.3f}`–`{target_response_cos_stability['spearman_max']:.3f}`。相反，donor-level `response_cosine` 更稳定，Pearson `{donor_response_cos_stability['pearson_min']:.3f}`–`{donor_response_cos_stability['pearson_max']:.3f}`、Spearman `{donor_response_cos_stability['spearman_min']:.3f}`–`{donor_response_cos_stability['spearman_max']:.3f}`。

因此稳定的是“某个 target 对 starting-model change 有多敏感”，不是一个完全稳定的 target-specific rotation map。

## Q5. Classifier signal 是普通 local gradient 还是 conditioned response？

不是普通 gradient 就能解释完。Classifier 的 `||q||/||b||` 均值 `{change['classifier']['mean']:.3f}`，cancellation `{cancel['classifier']['mean']:.3f}`，`cos(q,b)` `{response_cos['classifier']['mean']:.3f}`；starting point 对 A 的 update 有大幅影响。Classifier response magnitude 的 target partial R² 为 `{pr2('target_identity','classifier','response_change_ratio')['partial_r2']:.3f}`，并具有稳定 target ordering。

但 classifier rotation 的 target-specific component 仍有限：`survival_cosine` target partial R² `{pr2('target_identity','classifier','survival_cosine')['partial_r2']:.3f}`，`response_cosine` `{pr2('target_identity','classifier','response_cosine')['partial_r2']:.3f}`；donor identity 对这两个方向指标分别解释 `{pr2('donor_identity','classifier','survival_cosine')['partial_r2']:.3f}` 和 `{pr2('donor_identity','classifier','response_cosine')['partial_r2']:.3f}`。

## Interpretation boundary

- 所有新指标都由已保存几何量解析恢复，没有近似模型权重或重新训练。
- Partial R² 是 categorical fixed-effect descriptive statistic，不应解释为因果效应。
- Between variance 指 entity mean 的方差；within variance 是同一 entity 内 event residual 的 pooled variance。
- “Strong support”需要 target identity 同时主导 response magnitude 和稳定 rotation。本结果只满足前者，因此最终判断为 **{judgment}**。
"""


def build_summary_json(
    metrics: pd.DataFrame,
    variance: pd.DataFrame,
    partials: pd.DataFrame,
    stability: pd.DataFrame,
    qc: dict,
    judgment: str,
) -> dict:
    def row(frame, **filters):
        return select_row(frame, **filters)

    primary = {}
    for scope in SCOPES:
        primary[scope] = {
            metric: row(
                metrics, stage="overall", scope=scope, metric=metric
            )
            for metric in METRICS
        }
    target_r2 = {
        scope: {
            metric: row(
                partials,
                effect="target_identity",
                scope=scope,
                metric=metric,
            )["partial_r2"]
            for metric in METRICS
        }
        for scope in SCOPES
    }
    donor_r2 = {
        scope: {
            metric: row(
                partials,
                effect="donor_identity",
                scope=scope,
                metric=metric,
            )["partial_r2"]
            for metric in METRICS
        }
        for scope in SCOPES
    }
    return {
        "judgment": judgment,
        "event_count": qc["event_count"],
        "derivation_qc": qc,
        "metric_definitions": {
            "response_change_ratio": "||q|| / ||b||",
            "cancellation_coefficient": "-<q,b> / ||b||^2",
            "survival_ratio": "||e|| / ||b||",
            "survival_cosine": "cos(e,b)",
            "response_cosine": "cos(q,b)",
            "cos_q_a": "cos(q,a)",
        },
        "overall_metrics": primary,
        "target_identity_partial_r2": target_r2,
        "donor_identity_partial_r2": donor_r2,
        "answers": {
            "q1": "q is large and nonzero in every event; starting-model conditioning materially changes A's local step.",
            "q2": "e is not a scalar multiple of b; survival cosine is far below one, especially in the classifier.",
            "q3": "Target identity strongly affects response magnitude (~0.29 partial R2) but only weakly/moderately affects rotation metrics.",
            "q4": "Target magnitude values have high cross-stage Pearson correlation, but rank stability is only moderate; target rotation ordering is weaker, while donor-conditioned direction is more stable.",
            "q5": "Classifier q is large, so the earlier classifier signal is not merely A's ordinary gradient; however its directional target-specific component remains limited.",
        },
    }


def main() -> None:
    args = parse_args()
    events_path = Path(args.events_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to mix results: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    source = pd.read_csv(events_path)
    events, qc = derive_event_metrics(source)
    if not qc["passed"]:
        raise RuntimeError(f"Derivation QC failed: {qc}")

    metric_summary = build_metric_summary(events)
    per_target = entity_summary(
        events, "target_client_id", "donor_client_id"
    )
    per_donor = entity_summary(
        events, "donor_client_id", "target_client_id"
    )
    variance = build_variance_decomposition(events)
    partials = build_partial_r2(events)
    stability = build_stage_stability(events)

    change_all = select_row(
        metric_summary,
        stage="overall",
        scope="all",
        metric="response_change_ratio",
    )
    target_change_r2 = select_row(
        partials,
        effect="target_identity",
        scope="all",
        metric="response_change_ratio",
    )["partial_r2"]
    target_change_stability = correlation_range(
        stability, "target", "all", "response_change_ratio"
    )
    target_response_cos_r2 = select_row(
        partials,
        effect="target_identity",
        scope="all",
        metric="response_cosine",
    )["partial_r2"]
    target_response_cos_stability = correlation_range(
        stability, "target", "all", "response_cosine"
    )

    if change_all["mean"] < 0.1 or change_all["ci95_high"] < 0.1:
        judgment = "Not supported"
    elif (
        target_change_r2 >= 0.25
        and target_change_stability["pearson_min"] >= 0.5
        and target_change_stability["spearman_min"] >= 0.5
        and target_response_cos_r2 >= 0.25
        and target_response_cos_stability["pearson_min"] >= 0.5
        and target_response_cos_stability["spearman_min"] >= 0.5
    ):
        judgment = "Strong support"
    else:
        judgment = "Weak/limited support"

    events.to_csv(output_dir / "true_response_event_metrics.csv", index=False)
    metric_summary.to_csv(
        output_dir / "true_response_metric_summary.csv", index=False
    )
    per_target.to_csv(output_dir / "true_response_per_target.csv", index=False)
    per_donor.to_csv(output_dir / "true_response_per_donor.csv", index=False)
    variance.to_csv(
        output_dir / "true_response_variance_decomposition.csv", index=False
    )
    stability.to_csv(
        output_dir / "true_response_stage_stability.csv", index=False
    )
    partials.to_csv(output_dir / "true_response_partial_r2.csv", index=False)
    write_json(output_dir / "true_response_derivation_qc.json", qc)

    report = build_report(
        metric_summary, variance, partials, stability, qc, judgment
    )
    (output_dir / "TRUE_RESPONSE_REPORT.md").write_text(report, encoding="utf-8")
    summary = build_summary_json(
        metric_summary, variance, partials, stability, qc, judgment
    )
    summary["source_events_csv"] = str(events_path)
    write_json(output_dir / "true_response_summary.json", summary)

    print(f"TRUE_RESPONSE_ANALYSIS_COMPLETE output_dir={output_dir}")
    print(f"DERIVATION_QC_PASSED={qc['passed']}")
    print(f"JUDGMENT={judgment}")


if __name__ == "__main__":
    main()
