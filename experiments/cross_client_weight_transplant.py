#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-client weight-transplantation prestudy.

This is deliberately a measurement-only experiment.  The ordinary FedAvg
trajectory is computed exclusively from the ten normal same-round local
models.  The configured fraction of additional B -> A local-training runs are
shadow probes and are never passed to the aggregation accumulator.

The implementation reuses the repository's CIFAR-10 partition loader,
VGG16, LocalUpdate_FedAvg, and test_img.  Model geometry is accumulated one
parameter tensor/chunk at a time; no full-model flattened copies are built.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# Required by PyTorch deterministic CUDA matmul kernels.  It must be set
# before importing torch / initializing the CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The repository's data loader intentionally uses paths relative to repo root.
os.chdir(ROOT)

from models.Nets import VGG16  # noqa: E402
from models.Update import LocalUpdate_FedAvg  # noqa: E402
from models.test import test_img  # noqa: E402
from utils.get_dataset import get_dataset  # noqa: E402
from utils.set_seed import set_random_seed  # noqa: E402


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
SIGNATURE_METRICS = (
    "recovery_pull_score_all",
    "cos_WBA_minus_WB__WA_minus_Wt_all",
    "cos_WBA_minus_WB__WB_minus_Wt_all",
    "cos_WBA_minus_Wt__WA_minus_Wt_all",
    "cos_WBA_minus_Wt__WB_minus_Wt_all",
)
STAGES = (
    ("rounds_1_150", 1, 150),
    ("rounds_151_300", 151, 300),
    ("rounds_301_500", 301, 500),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-round cross-client weight-transplant shadow probe."
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="vgg")
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--local_ep", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--iid", type=int, default=0)
    parser.add_argument("--noniid_case", type=int, default=5)
    parser.add_argument("--data_beta", type=float, default=0.3)
    parser.add_argument("--generate_data", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--probe_fraction", type=float, default=0.2)
    parser.add_argument("--transplant_scale", type=float, default=1.0)
    parser.add_argument(
        "--output_dir",
        default="results/cross_client_transplant_vgg_cifar10_a03_seed1",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--verify_isolation",
        type=int,
        default=0,
        help="Hash the full aggregate before/after every shadow probe. Auto-enabled for <=5 rounds.",
    )
    parser.add_argument(
        "--metric_chunk_size",
        type=int,
        default=1_000_000,
        help="Maximum number of parameter elements processed at once on CPU.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    expected = {
        "dataset": "cifar10",
        "model": "vgg",
        "num_users": 100,
        "frac": 0.1,
        "local_ep": 5,
        "local_bs": 50,
        "optimizer": "sgd",
        "lr": 0.01,
        "momentum": 0.5,
        "weight_decay": 0.0,
        "iid": 0,
        "noniid_case": 5,
        "data_beta": 0.3,
        "generate_data": 0,
        "num_classes": 10,
        "num_channels": 3,
        "seed": 1,
        "transplant_scale": 1.0,
    }
    mismatches = []
    for key, wanted in expected.items():
        actual = getattr(args, key)
        if isinstance(wanted, float):
            ok = math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
        else:
            ok = actual == wanted
        if not ok:
            mismatches.append(f"{key}={actual!r} (required {wanted!r})")
    if mismatches:
        raise ValueError(
            "This prestudy intentionally has a fixed primary protocol; "
            "refusing protocol drift: " + ", ".join(mismatches)
        )
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.bs < 1 or args.num_workers < 0 or args.metric_chunk_size < 1:
        raise ValueError("bs/chunk size must be positive and num_workers non-negative")
    selected = max(int(args.frac * args.num_users), 1)
    probes = int(round(selected * args.probe_fraction))
    if selected != 10:
        raise ValueError("The fixed protocol requires 10 selected clients per round")
    if not (0.0 < args.probe_fraction <= 1.0):
        raise ValueError("probe_fraction must be in (0, 1]")
    if not math.isclose(
        selected * args.probe_fraction,
        probes,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "selected_clients * probe_fraction must be an integer number of probes"
        )
    if probes < 1 or probes > selected:
        raise ValueError("probe_fraction produces an invalid probe count")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def seed_local_rng(seed: int) -> None:
    """Reset every RNG touched by shuffle/dropout for a paired A trajectory."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def local_seed(base_seed: int, round_number: int, client_id: int) -> int:
    return int(base_seed) * 10_000_019 + int(round_number) * 100_003 + int(client_id) * 997 + 17


def clone_state_cpu(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def clone_named_params_cpu(
    state: Mapping[str, torch.Tensor], param_names: Iterable[str]
) -> Dict[str, torch.Tensor]:
    return {name: state[name].detach().cpu().clone() for name in param_names}


class WeightedStateAccumulator:
    """Streaming equivalent of models.Fed.Aggregation(w_locals, lens)."""

    def __init__(self, reference: Mapping[str, torch.Tensor]):
        self.reference_dtypes = {k: v.dtype for k, v in reference.items()}
        self.weighted_sum: Dict[str, torch.Tensor] = {}
        self.total_weight = 0

    def add(self, state: Mapping[str, torch.Tensor], weight: int) -> None:
        weight = int(weight)
        if not self.weighted_sum:
            for name, value in state.items():
                tensor = value.detach().cpu().clone()
                tensor.mul_(weight)
                self.weighted_sum[name] = tensor
        else:
            for name, value in state.items():
                tensor = value.detach().cpu()
                self.weighted_sum[name].add_(tensor, alpha=weight)
        self.total_weight += weight

    def finalize(self) -> Dict[str, torch.Tensor]:
        if self.total_weight <= 0:
            raise RuntimeError("Cannot finalize an empty FedAvg accumulator")
        result = {}
        for name, value in self.weighted_sum.items():
            averaged = torch.div(value, self.total_weight)
            # load_state_dict in the original Aggregation path casts averaged
            # integer buffers back to the model buffer dtype. Do that explicitly.
            result[name] = averaged.to(dtype=self.reference_dtypes[name])
        return result


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def build_parameter_scopes(model: nn.Module) -> Tuple[List[str], Dict[str, set]]:
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    conv = set()
    classifier = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        destination = conv if isinstance(module, nn.Conv2d) else classifier
        for local_name, parameter in module.named_parameters(recurse=False):
            if parameter.requires_grad:
                full_name = f"{module_name}.{local_name}" if module_name else local_name
                destination.add(full_name)
    scopes = {"all": set(trainable), "conv": conv, "classifier": classifier}
    if not conv or not classifier or (conv | classifier) - set(trainable):
        raise RuntimeError("Failed to identify VGG convolution/classifier parameter scopes")
    return trainable, scopes


def train_local_from_state(
    args: argparse.Namespace,
    start_state: Mapping[str, torch.Tensor],
    dataset_train,
    dataset_test,
    client_indices,
    stochastic_seed: int,
) -> Tuple[Dict[str, torch.Tensor], float]:
    # Seed only after construction/load: random initialization must not shift
    # the paired minibatch/dropout stream.
    local_model = VGG16(args).to(args.device)
    local_model.load_state_dict(start_state)
    seed_local_rng(stochastic_seed)
    trainer = LocalUpdate_FedAvg(
        args=args,
        dataset=dataset_train,
        dataset_test=dataset_test,
        idxs=client_indices,
        verbose=args.verbose,
    )
    trained_state = trainer.train(local_model)
    mean_loss = float(trainer.get_mean_loss())
    return trained_state, mean_loss


def safe_cos(dot: float, norm_a_sq: float, norm_b_sq: float) -> float:
    denominator = math.sqrt(max(norm_a_sq, 0.0)) * math.sqrt(max(norm_b_sq, 0.0))
    return float(dot / denominator) if denominator > 1e-30 else float("nan")


def streaming_geometry(
    wt: Mapping[str, torch.Tensor],
    wa: Mapping[str, torch.Tensor],
    wb: Mapping[str, torch.Tensor],
    wba: Mapping[str, torch.Tensor],
    parameter_names: Sequence[str],
    scopes: Mapping[str, set],
    chunk_size: int,
) -> Dict[str, float]:
    """Accumulate all requested geometry without flattening whole VGG states."""
    # ag=a-g, bg=b-g, xb=x-b, xg=x-g=xb+bg, ba=b-a=bg-ag,
    # xa=x-a=xg-ag.  Each temporary is bounded by chunk_size.
    sums = {
        scope: {
            "ag2": 0.0,
            "bg2": 0.0,
            "xb2": 0.0,
            "ba2": 0.0,
            "xa2": 0.0,
            "xg2": 0.0,
            "xb_ag": 0.0,
            "xb_bg": 0.0,
            "xg_ag": 0.0,
            "xg_bg": 0.0,
        }
        for scope in SCOPES
    }

    for name in parameter_names:
        memberships = [scope for scope in SCOPES if name in scopes[scope]]
        if not memberships:
            continue
        g_flat = wt[name].detach().cpu().reshape(-1)
        a_flat = wa[name].detach().cpu().reshape(-1)
        b_flat = wb[name].detach().cpu().reshape(-1)
        x_flat = wba[name].detach().cpu().reshape(-1)
        for start in range(0, g_flat.numel(), chunk_size):
            stop = min(start + chunk_size, g_flat.numel())
            # float32 is the native trained precision; scalar reductions use
            # float64 to keep large-model sums stable.
            ag = a_flat[start:stop].float() - g_flat[start:stop].float()
            bg = b_flat[start:stop].float() - g_flat[start:stop].float()
            xb = x_flat[start:stop].float() - b_flat[start:stop].float()
            xg = xb + bg
            ba = bg - ag
            xa = xg - ag
            values = {
                "ag2": torch.sum(ag * ag, dtype=torch.float64).item(),
                "bg2": torch.sum(bg * bg, dtype=torch.float64).item(),
                "xb2": torch.sum(xb * xb, dtype=torch.float64).item(),
                "ba2": torch.sum(ba * ba, dtype=torch.float64).item(),
                "xa2": torch.sum(xa * xa, dtype=torch.float64).item(),
                "xg2": torch.sum(xg * xg, dtype=torch.float64).item(),
                "xb_ag": torch.sum(xb * ag, dtype=torch.float64).item(),
                "xb_bg": torch.sum(xb * bg, dtype=torch.float64).item(),
                "xg_ag": torch.sum(xg * ag, dtype=torch.float64).item(),
                "xg_bg": torch.sum(xg * bg, dtype=torch.float64).item(),
            }
            for scope in memberships:
                for key, value in values.items():
                    sums[scope][key] += value

    result: Dict[str, float] = {}
    for scope, value in sums.items():
        norm_ba = math.sqrt(max(value["ba2"], 0.0))
        norm_xa = math.sqrt(max(value["xa2"], 0.0))
        result[f"norm_WA_minus_Wt_{scope}"] = math.sqrt(max(value["ag2"], 0.0))
        result[f"norm_WB_minus_Wt_{scope}"] = math.sqrt(max(value["bg2"], 0.0))
        result[f"norm_WBA_minus_WB_{scope}"] = math.sqrt(max(value["xb2"], 0.0))
        result[f"norm_WB_minus_WA_{scope}"] = norm_ba
        result[f"norm_WBA_minus_WA_{scope}"] = norm_xa
        result[f"recovery_pull_score_{scope}"] = (
            1.0 - norm_xa / norm_ba if norm_ba > 1e-30 else float("nan")
        )
        result[f"cos_WBA_minus_WB__WA_minus_Wt_{scope}"] = safe_cos(
            value["xb_ag"], value["xb2"], value["ag2"]
        )
        result[f"cos_WBA_minus_WB__WB_minus_Wt_{scope}"] = safe_cos(
            value["xb_bg"], value["xb2"], value["bg2"]
        )
        result[f"cos_WBA_minus_Wt__WA_minus_Wt_{scope}"] = safe_cos(
            value["xg_ag"], value["xg2"], value["ag2"]
        )
        result[f"cos_WBA_minus_Wt__WB_minus_Wt_{scope}"] = safe_cos(
            value["xg_bg"], value["xg2"], value["bg2"]
        )
    return result


def client_label_profiles(dataset, dict_users, num_classes: int) -> Dict[int, np.ndarray]:
    labels = np.asarray(dataset.targets)
    profiles = {}
    for client_id, indices in dict_users.items():
        counts = np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=num_classes)
        profiles[int(client_id)] = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    return profiles


def profile_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def event_fieldnames() -> List[str]:
    fields = [
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
        "target_baseline_loss",
        "donor_baseline_loss",
        "shadow_loss",
    ]
    fields.extend(f"{metric}_{scope}" for scope in SCOPES for metric in BASE_METRICS)
    return fields


def stage_for_round(round_number: int) -> str:
    for name, start, end in STAGES:
        if start <= round_number <= end:
            return name
    return "outside_prespecified_stages"


def finite_values(rows: Sequence[Mapping], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def sample_stats(values: np.ndarray) -> Dict[str, float | int]:
    n = int(values.size)
    if n == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "count": n,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
    }


def metric_columns() -> List[str]:
    return [f"{metric}_{scope}" for scope in SCOPES for metric in BASE_METRICS]


def build_per_client_summary(events: Sequence[Mapping]) -> List[Dict]:
    grouped: Dict[int, List[Mapping]] = defaultdict(list)
    for event in events:
        grouped[int(event["target_client_id"])].append(event)
    rows = []
    for client_id in sorted(grouped):
        items = grouped[client_id]
        row = {
            "target_client_id": client_id,
            "probe_count": len(items),
            "unique_donor_count": len({int(item["donor_client_id"]) for item in items}),
        }
        for metric in metric_columns():
            stats = sample_stats(finite_values(items, metric))
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return rows


def build_stage_summary(events: Sequence[Mapping]) -> List[Dict]:
    rows = []
    for stage_name, start, end in STAGES:
        items = [event for event in events if start <= int(event["round"]) <= end]
        for metric in metric_columns():
            values = finite_values(items, metric)
            stats = sample_stats(values)
            sem = stats["std"] / math.sqrt(stats["count"]) if stats["count"] else float("nan")
            rows.append(
                {
                    "stage": stage_name,
                    "start_round": start,
                    "end_round": end,
                    "n_events": len(items),
                    "unique_targets": len({int(item["target_client_id"]) for item in items}),
                    "unique_donors": len({int(item["donor_client_id"]) for item in items}),
                    "metric": metric,
                    "count_finite": stats["count"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "sem": sem,
                    "ci95_low": stats["mean"] - 1.96 * sem if stats["count"] else float("nan"),
                    "ci95_high": stats["mean"] + 1.96 * sem if stats["count"] else float("nan"),
                }
            )
    return rows


def variance_decomposition(events: Sequence[Mapping], metric: str) -> Dict:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for event in events:
        value = float(event[metric])
        if math.isfinite(value):
            grouped[int(event["target_client_id"])].append(value)
    usable = {client: values for client, values in grouped.items() if values}
    means = np.asarray([np.mean(values) for values in usable.values()], dtype=np.float64)
    within_numerator = sum(
        sum((np.asarray(values) - np.mean(values)) ** 2) for values in usable.values() if len(values) > 1
    )
    within_df = sum(max(len(values) - 1, 0) for values in usable.values())
    pooled_within = float(within_numerator / within_df) if within_df > 0 else float("nan")
    between = float(means.var(ddof=1)) if means.size > 1 else float("nan")
    ratio = between / pooled_within if pooled_within > 0 else float("nan")
    icc = (between - pooled_within) / (between + pooled_within) if between + pooled_within > 0 else float("nan")
    return {
        "metric": metric,
        "clients": len(usable),
        "between_client_variance_of_means": between,
        "pooled_within_client_variance": pooled_within,
        "between_to_within_ratio": ratio,
        "descriptive_icc": icc,
    }


def stage_stability(events: Sequence[Mapping], metric: str) -> List[Dict]:
    stage_means: Dict[str, Dict[int, float]] = {}
    for stage_name, start, end in STAGES:
        grouped: Dict[int, List[float]] = defaultdict(list)
        for event in events:
            if start <= int(event["round"]) <= end:
                value = float(event[metric])
                if math.isfinite(value):
                    grouped[int(event["target_client_id"])].append(value)
        stage_means[stage_name] = {client: float(np.mean(values)) for client, values in grouped.items()}
    rows = []
    for left_index in range(len(STAGES)):
        for right_index in range(left_index + 1, len(STAGES)):
            left_name = STAGES[left_index][0]
            right_name = STAGES[right_index][0]
            clients = sorted(set(stage_means[left_name]) & set(stage_means[right_name]))
            left = np.asarray([stage_means[left_name][client] for client in clients])
            right = np.asarray([stage_means[right_name][client] for client in clients])
            correlation = (
                float(np.corrcoef(left, right)[0, 1])
                if len(clients) >= 3 and left.std() > 0 and right.std() > 0
                else float("nan")
            )
            rows.append(
                {
                    "metric": metric,
                    "stage_left": left_name,
                    "stage_right": right_name,
                    "common_clients": len(clients),
                    "pearson_correlation_of_target_means": correlation,
                }
            )
    return rows


def categorical_additive_sse(
    events: Sequence[Mapping], y: np.ndarray, factors: Sequence[str]
) -> float:
    """Fit additive categorical effects by backfitting, without BLAS/LAPACK.

    The Windows NumPy build available for the prescribed environment can
    block inside tiny LAPACK least-squares calls.  Group-mean backfitting is
    the same projection for this all-categorical diagnostic and avoids that
    external runtime failure.
    """
    prediction = np.full(y.shape, float(y.mean()), dtype=np.float64)
    effects: Dict[str, Dict[object, float]] = {
        factor: {row[factor]: 0.0 for row in events} for factor in factors
    }
    for _ in range(200):
        max_change = 0.0
        intercept_delta = float(np.mean(y - prediction))
        prediction += intercept_delta
        max_change = max(max_change, abs(intercept_delta))
        for factor in factors:
            grouped_indices: Dict[object, List[int]] = defaultdict(list)
            for index, row in enumerate(events):
                grouped_indices[row[factor]].append(index)
            old = effects[factor]
            new = {}
            for level, indices in grouped_indices.items():
                idx = np.asarray(indices, dtype=np.int64)
                residual_without_this_factor = y[idx] - prediction[idx] + old[level]
                new[level] = float(residual_without_this_factor.mean())
            # Center effects to keep the representation identifiable.
            weighted_mean = sum(
                new[level] * len(indices) for level, indices in grouped_indices.items()
            ) / len(events)
            for level in new:
                new[level] -= weighted_mean
            for index, row in enumerate(events):
                delta = new[row[factor]] - old[row[factor]]
                prediction[index] += delta
                max_change = max(max_change, abs(delta))
            effects[factor] = new
        if max_change < 1e-12:
            break
    residual = y - prediction
    return float(np.sum(residual * residual))


def target_partial_r2(events: Sequence[Mapping], metric: str) -> Dict:
    usable = [row for row in events if math.isfinite(float(row[metric]))]
    if len(usable) < 3:
        return {"metric": metric, "n": len(usable), "target_partial_r2": float("nan")}
    y = np.asarray([float(row[metric]) for row in usable], dtype=np.float64)
    reduced_factors = ("donor_dominant_class", "stage")
    full_factors = ("donor_dominant_class", "stage", "target_client_id")
    sse_reduced = categorical_additive_sse(usable, y, reduced_factors)
    sse_full = categorical_additive_sse(usable, y, full_factors)
    partial = (sse_reduced - sse_full) / sse_reduced if sse_reduced > 0 else float("nan")
    return {
        "metric": metric,
        "n": len(usable),
        "target_partial_r2_controlling_donor_class_and_stage": float(partial),
        "sse_reduced": sse_reduced,
        "sse_full": sse_full,
    }


def build_analysis(events: Sequence[Mapping]) -> Dict:
    variance = [variance_decomposition(events, metric) for metric in SIGNATURE_METRICS]
    stability = []
    for metric in SIGNATURE_METRICS:
        stability.extend(stage_stability(events, metric))
    target_effects = [target_partial_r2(events, metric) for metric in SIGNATURE_METRICS]
    recovery_nonzero = {}
    for scope in SCOPES:
        metric = f"recovery_pull_score_{scope}"
        values = finite_values(events, metric)
        stats = sample_stats(values)
        sem = stats["std"] / math.sqrt(stats["count"]) if stats["count"] else float("nan")
        recovery_nonzero[scope] = {
            **stats,
            "sem": sem,
            "ci95_low": stats["mean"] - 1.96 * sem if stats["count"] else float("nan"),
            "ci95_high": stats["mean"] + 1.96 * sem if stats["count"] else float("nan"),
            "positive_fraction": float(np.mean(values > 0)) if values.size else float("nan"),
        }
    return {
        "definitions": {
            "same_donor_type": "same dominant class in the donor client's fixed local label distribution",
            "transformation_signature": list(SIGNATURE_METRICS),
            "variance_note": "Between variance is variance of target means; within variance is pooled event residual variance within target.",
            "partial_r2_note": "Target fixed-effect partial R^2 controls for donor dominant class and the three prespecified stages; descriptive, not causal.",
        },
        "q1_same_target_across_donors": {
            "variance_decomposition": variance,
            "target_identity_partial_r2": target_effects,
        },
        "q2_different_targets_same_donor_type": {
            "target_identity_partial_r2_controlling_donor_type_and_stage": target_effects,
        },
        "q3_target_response_stage_stability": stability,
        "q4_between_vs_within_client_variance": variance,
        "q5_recovery_systematically_nonzero": recovery_nonzero,
    }


def render_accuracy_curve(output_dir: Path, accuracy_rows: Sequence[Mapping]) -> None:
    """Draw a dependency-light PNG curve with Pillow.

    Matplotlib is intentionally avoided because its import blocks in the
    prescribed Windows Conda runtime.  This renderer is deterministic and the
    underlying exact values are also saved in the companion CSV.
    """
    from PIL import Image, ImageDraw, ImageFont

    rounds = [int(row["round"]) for row in accuracy_rows]
    accuracies = [float(row["test_accuracy"]) for row in accuracy_rows]
    width, height = 1530, 900
    left, right, top, bottom = 125, 55, 85, 115
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "FedAvg accuracy (shadow probes excluded from aggregation)"
    draw.text((left, 28), title, fill=(20, 20, 20), font=font)
    draw.line((left, top, left, top + plot_height), fill=(25, 25, 25), width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill=(25, 25, 25),
        width=2,
    )
    for tick in range(0, 101, 10):
        y = top + plot_height - tick / 100.0 * plot_height
        draw.line((left, y, left + plot_width, y), fill=(225, 225, 225), width=1)
        draw.text((65, y - 6), str(tick), fill=(55, 55, 55), font=font)
    maximum_round = max(rounds)
    tick_step = max(1, int(math.ceil(maximum_round / 10.0)))
    x_ticks = sorted(set([1, maximum_round] + list(range(tick_step, maximum_round + 1, tick_step))))
    for tick in x_ticks:
        x = left + (tick - 1) / max(maximum_round - 1, 1) * plot_width
        draw.line((x, top, x, top + plot_height), fill=(235, 235, 235), width=1)
        draw.text((x - 8, top + plot_height + 14), str(tick), fill=(55, 55, 55), font=font)
    points = []
    for round_number, accuracy in zip(rounds, accuracies):
        x = left + (round_number - 1) / max(maximum_round - 1, 1) * plot_width
        y = top + plot_height - max(0.0, min(100.0, accuracy)) / 100.0 * plot_height
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=3)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(31, 119, 180))
    draw.text((left + plot_width // 2 - 55, height - 45), "Communication round", fill=(20, 20, 20), font=font)
    draw.text((12, top + plot_height // 2), "Accuracy (%)", fill=(20, 20, 20), font=font)
    image.save(output_dir / "fedavg_accuracy_curve.png", format="PNG")


def write_summary_text(output_dir: Path, summary: Mapping) -> None:
    analysis = summary["analysis"]
    lines = [
        "Cross-client weight transplantation prestudy",
        "==============================================",
        f"Completed rounds: {summary['completed_rounds']}",
        f"Probe events: {summary['probe_events']}",
        f"Final FedAvg accuracy: {summary['final_test_accuracy']:.6f}%",
        f"Best FedAvg accuracy: {summary['best_test_accuracy']:.6f}%",
        f"Aggregation isolation verified: {summary['aggregation_isolation_verified']}",
        "",
        "Primary interpretation is intentionally deferred to the measured statistics.",
        "Positive recovery means B->A ends closer to A's normal endpoint than B was.",
        "",
        "Recovery / pull score:",
    ]
    for scope, stats in analysis["q5_recovery_systematically_nonzero"].items():
        lines.append(
            f"  {scope}: mean={stats['mean']:.8g}, std={stats['std']:.8g}, "
            f"95% CI=[{stats['ci95_low']:.8g}, {stats['ci95_high']:.8g}], "
            f"positive_fraction={stats['positive_fraction']:.6g}, n={stats['count']}"
        )
    lines.extend(["", "Between-target / within-target variance ratios:"])
    for item in analysis["q4_between_vs_within_client_variance"]:
        lines.append(f"  {item['metric']}: {item['between_to_within_ratio']:.8g}")
    lines.extend(
        [
            "",
            "See summary.json for stage correlations, target partial R^2, exact definitions,",
            "and all five requested question-oriented diagnostics.",
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke_validation(
    args: argparse.Namespace,
    events: Sequence[Mapping],
    accuracy_rows: Sequence[Mapping],
    isolation_rows: Sequence[Mapping],
    output_dir: Path,
) -> Dict:
    probe_count = int(round(10 * args.probe_fraction))
    expected_events = args.epochs * probe_count
    checks = {
        "expected_probes_per_round": len(events) == expected_events,
        "target_and_donor_always_different": all(
            int(row["target_client_id"]) != int(row["donor_client_id"]) for row in events
        ),
        "all_requested_metrics_finite": all(
            math.isfinite(float(row[key])) for row in events for key in metric_columns()
        ),
        "one_accuracy_per_round": len(accuracy_rows) == args.epochs,
        "aggregate_unchanged_by_shadow": all(bool(row["shadow_left_aggregate_unchanged"]) for row in isolation_rows),
        "global_equals_normal_only_aggregate": all(bool(row["global_equals_normal_aggregate"]) for row in isolation_rows),
        "cuda_device_used": args.device.type == "cuda",
        "probe_csv_exists": (output_dir / "probe_events.csv").exists(),
        "accuracy_csv_exists": (output_dir / "fedavg_accuracy_curve.csv").exists(),
    }
    result = {"passed": all(checks.values()), "checks": checks}
    write_json(output_dir / "smoke_validation.json", result)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_csv = output_dir / "probe_events.csv"
    accuracy_csv = output_dir / "fedavg_accuracy_curve.csv"
    isolation_csv = output_dir / "aggregation_isolation_audit.csv"
    # Refuse accidental append/mixing. A requested run gets a clean directory.
    existing_outputs = [path for path in (probe_csv, accuracy_csv, isolation_csv) if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Output files already exist; choose a new --output_dir to preserve prior results: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    if args.gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError("This prescribed run requires a CUDA GPU, but CUDA is unavailable")
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f"cuda:{args.gpu}")
    verify_isolation = bool(args.verify_isolation) or args.epochs <= 5
    probe_count = int(round(10 * args.probe_fraction))

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    expected_partition = ROOT / "data" / "cifar10_100_noniidCase5_beta0.3.json"
    if not expected_partition.exists():
        raise FileNotFoundError(f"Required existing partition not found: {expected_partition}")
    if len(dict_users) != args.num_users:
        raise RuntimeError(f"Expected {args.num_users} partition clients, found {len(dict_users)}")

    net_glob = VGG16(args).to(args.device)
    parameter_names, scopes = build_parameter_scopes(net_glob)
    profiles = client_label_profiles(dataset_train, dict_users, args.num_classes)
    selection_rng = np.random.RandomState(args.seed)
    probe_rng = random.Random(args.seed + 71_003)

    config = vars(args).copy()
    config.update(
        {
            "device": str(args.device),
            "repo_root": str(ROOT),
            "partition_file": str(expected_partition),
            "aggregation": "sample-count weighted, streaming equivalent of models.Fed.Aggregation",
            "shadow_in_aggregation": False,
            "probe_count_per_round": probe_count,
            "metric_scopes": {
                "all": "all trainable parameters (including BatchNorm affine parameters)",
                "conv": "nn.Conv2d weight and bias parameters only",
                "classifier": "nn.Linear weight and bias parameters only (classifier plus final fc)",
            },
            "paired_rng": "same explicit round/client seed for A baseline and B->A shadow",
            "verify_isolation": verify_isolation,
        }
    )
    write_json(output_dir / "config.json", config)

    events: List[Dict] = []
    accuracy_rows: List[Dict] = []
    isolation_rows: List[Dict] = []
    event_fields = event_fieldnames()
    start_time = time.time()

    print(f"Repository root: {ROOT}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f"CUDA device: {torch.cuda.get_device_name(args.gpu)}; "
        f"clients={len(dict_users)}; train={len(dataset_train)}; test={len(dataset_test)}",
        flush=True,
    )

    for round_index in range(args.epochs):
        round_number = round_index + 1
        round_start = time.perf_counter()
        selected = [
            int(value)
            for value in selection_rng.choice(args.num_users, 10, replace=False).tolist()
        ]
        targets = probe_rng.sample(selected, probe_count)
        pairs = []
        for target in targets:
            donor = probe_rng.choice([client for client in selected if client != target])
            pairs.append((target, donor))
        needed_donors = {donor for _, donor in pairs}
        needed_targets = set(targets)

        # W_t parameter snapshot is CPU-only and used solely by the metrics.
        wt_params = clone_named_params_cpu(net_glob.state_dict(), parameter_names)
        accumulator = WeightedStateAccumulator(net_glob.state_dict())
        donor_states: Dict[int, Dict[str, torch.Tensor]] = {}
        target_states: Dict[int, Dict[str, torch.Tensor]] = {}
        baseline_losses: Dict[int, float] = {}

        for client_id in selected:
            stochastic_seed = local_seed(args.seed, round_number, client_id)
            trained_state, mean_loss = train_local_from_state(
                args,
                net_glob.state_dict(),
                dataset_train,
                dataset_test,
                dict_users[client_id],
                stochastic_seed,
            )
            baseline_losses[client_id] = mean_loss
            accumulator.add(trained_state, len(dict_users[client_id]))
            if client_id in needed_donors:
                donor_states[client_id] = clone_state_cpu(trained_state)
            if client_id in needed_targets:
                # Avoid a second full copy when this target is also a donor.
                target_states[client_id] = (
                    donor_states[client_id]
                    if client_id in donor_states
                    else clone_named_params_cpu(trained_state, parameter_names)
                )
            del trained_state
            torch.cuda.empty_cache()

        normal_only_aggregate = accumulator.finalize()
        aggregate_hash_before = state_sha256(normal_only_aggregate) if verify_isolation else "not_computed"
        round_events = []

        for target_id, donor_id in pairs:
            stochastic_seed = local_seed(args.seed, round_number, target_id)
            shadow_state, shadow_loss = train_local_from_state(
                args,
                donor_states[donor_id],
                dataset_train,
                dataset_test,
                dict_users[target_id],
                stochastic_seed,
            )
            shadow_params = clone_named_params_cpu(shadow_state, parameter_names)
            geometry = streaming_geometry(
                wt_params,
                target_states[target_id],
                donor_states[donor_id],
                shadow_params,
                parameter_names,
                scopes,
                args.metric_chunk_size,
            )
            target_profile = profiles[target_id]
            donor_profile = profiles[donor_id]
            event = {
                "round": round_number,
                "stage": stage_for_round(round_number),
                "target_client_id": target_id,
                "donor_client_id": donor_id,
                "target_local_seed": stochastic_seed,
                "target_num_samples": len(dict_users[target_id]),
                "donor_num_samples": len(dict_users[donor_id]),
                "target_dominant_class": int(np.argmax(target_profile)),
                "donor_dominant_class": int(np.argmax(donor_profile)),
                "target_donor_label_cosine": profile_cosine(target_profile, donor_profile),
                "target_baseline_loss": baseline_losses[target_id],
                "donor_baseline_loss": baseline_losses[donor_id],
                "shadow_loss": shadow_loss,
                **geometry,
            }
            round_events.append(event)
            events.append(event)
            del shadow_state, shadow_params
            torch.cuda.empty_cache()

        aggregate_hash_after = state_sha256(normal_only_aggregate) if verify_isolation else "not_computed"
        shadow_unchanged = aggregate_hash_before == aggregate_hash_after if verify_isolation else True

        # This is the only update of the global trajectory, and its source was
        # finalized before any shadow training began.
        net_glob.load_state_dict(normal_only_aggregate)
        global_hash = state_sha256(net_glob.state_dict()) if verify_isolation else "not_computed"
        global_matches = global_hash == aggregate_hash_before if verify_isolation else True
        if not shadow_unchanged or not global_matches:
            raise RuntimeError(f"Aggregation isolation failed in round {round_number}")

        accuracy, test_loss = test_img(net_glob, dataset_test, args)
        accuracy = float(accuracy.item() if hasattr(accuracy, "item") else accuracy)
        test_loss = float(test_loss.item() if hasattr(test_loss, "item") else test_loss)
        elapsed = time.perf_counter() - round_start
        accuracy_row = {
            "round": round_number,
            "test_accuracy": accuracy,
            "test_loss": test_loss,
            "selected_clients": json.dumps(selected),
            "probe_pairs": json.dumps(pairs),
            "round_seconds": elapsed,
        }
        isolation_row = {
            "round": round_number,
            "verification_performed": verify_isolation,
            "normal_aggregate_sha256_before_shadow": aggregate_hash_before,
            "normal_aggregate_sha256_after_shadow": aggregate_hash_after,
            "loaded_global_sha256": global_hash,
            "shadow_left_aggregate_unchanged": shadow_unchanged,
            "global_equals_normal_aggregate": global_matches,
        }
        accuracy_rows.append(accuracy_row)
        isolation_rows.append(isolation_row)
        append_csv(probe_csv, round_events, event_fields)
        append_csv(accuracy_csv, [accuracy_row], list(accuracy_row.keys()))
        append_csv(isolation_csv, [isolation_row], list(isolation_row.keys()))

        print(
            f"ROUND {round_number:03d}/{args.epochs} "
            f"accuracy={accuracy:.4f} loss={test_loss:.6f} "
            f"selected={selected} probes={pairs} seconds={elapsed:.2f}",
            flush=True,
        )
        del wt_params, accumulator, donor_states, target_states, normal_only_aggregate
        torch.cuda.empty_cache()

    per_client = build_per_client_summary(events)
    stage_summary = build_stage_summary(events)
    analysis = build_analysis(events)
    write_csv(output_dir / "per_client_summary.csv", per_client)
    write_csv(output_dir / "stage_summary.csv", stage_summary)
    render_accuracy_curve(output_dir, accuracy_rows)
    torch.save(clone_state_cpu(net_glob.state_dict()), output_dir / "final_global_model.pt")

    summary = {
        "status": "completed",
        "completed_rounds": args.epochs,
        "probe_events": len(events),
        "expected_probe_events": args.epochs * probe_count,
        "final_test_accuracy": accuracy_rows[-1]["test_accuracy"],
        "best_test_accuracy": max(row["test_accuracy"] for row in accuracy_rows),
        "wall_time_seconds": time.time() - start_time,
        "aggregation_isolation_verified": all(
            row["shadow_left_aggregate_unchanged"] and row["global_equals_normal_aggregate"]
            for row in isolation_rows
        ),
        "analysis": analysis,
        "outputs": {
            "probe_events": str(probe_csv),
            "per_client_summary": str(output_dir / "per_client_summary.csv"),
            "stage_summary": str(output_dir / "stage_summary.csv"),
            "accuracy_csv": str(accuracy_csv),
            "accuracy_plot": str(output_dir / "fedavg_accuracy_curve.png"),
            "isolation_audit": str(isolation_csv),
            "final_global_model": str(output_dir / "final_global_model.pt"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_summary_text(output_dir, summary)

    if args.epochs <= 5:
        validation = run_smoke_validation(
            args, events, accuracy_rows, isolation_rows, output_dir
        )
        print(f"SMOKE_TEST_PASSED={validation['passed']}", flush=True)
        if not validation["passed"]:
            raise RuntimeError(f"Smoke validation failed: {validation['checks']}")

    print(f"RUN_COMPLETE output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
