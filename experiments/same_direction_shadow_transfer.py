#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Same-experienced-direction shadow transfer diagnostic.

The server trajectory is ordinary FedAvg.  For monitored returning clients,
the script replays local training from small shadow displacements along the
previously experienced input direction.  Shadow results are scalar-only
measurements and are never aggregated.  Full RNG state is restored to the
post-baseline state before the main trajectory continues.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from models.Fed import Aggregation  # noqa: E402
from models.Nets import VGG16  # noqa: E402
from models.Update import LocalUpdate_FedAvg  # noqa: E402
from models.test import evaluate_round_accuracy, print_peak_accuracy  # noqa: E402
from utils.get_dataset import get_dataset  # noqa: E402
from utils.set_seed import set_random_seed  # noqa: E402


EPS = 1.0e-12
FIXED_RATIOS = (0.02, 0.05, 0.10, 0.20, 0.40)
GAP_BINS = (
    ("gap_le_5", lambda gap: gap <= 5),
    ("gap_6_10", lambda gap: 6 <= gap <= 10),
    ("gap_11_20", lambda gap: 11 <= gap <= 20),
    ("gap_gt_20", lambda gap: gap > 20),
)
EVENT_FIELDS = (
    "seed",
    "round",
    "client_id",
    "client_visit_index",
    "participation_gap",
    "probe_ratio",
    "epsilon",
    "previous_obs_input_norm",
    "previous_obs_age_rounds",
    "status",
    "skip_reason",
    "predicted_response_norm",
    "realized_response_norm",
    "response_cosine",
    "response_nre",
    "response_norm_ratio",
    "actual_delta_update_norm",
    "predicted_delta_update_norm",
    "conflict",
    "predicted_support",
    "realized_support",
    "support_sign_agreement",
    "predicted_active",
    "base_alignment",
    "shadow_alignment",
    "alignment_change",
    "alignment_identity_abs_error",
    "shadow_displacement_norm",
    "epsilon_geometry_abs_error",
    "response_identity_abs_error",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-experienced-direction shadow transfer diagnostic"
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_monitored_clients", type=int, default=10)
    parser.add_argument(
        "--probe_ratios", default=",".join(str(value) for value in FIXED_RATIOS)
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--parity_mode", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_protocol(args: argparse.Namespace) -> Tuple[float, ...]:
    expected = {
        "dataset": "cifar10",
        "model": "vgg",
        "num_users": 100,
        "frac": 0.1,
        "local_ep": 5,
        "local_bs": 50,
        "bs": 256,
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
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual = getattr(args, name)
        matches = (
            math.isclose(float(actual), expected_value, abs_tol=1e-12)
            if isinstance(expected_value, float)
            else actual == expected_value
        )
        if not matches:
            mismatches.append(f"{name}={actual!r} (required {expected_value!r})")
    if mismatches:
        raise ValueError("Fixed diagnostic protocol violated: " + ", ".join(mismatches))
    if args.seed != 1:
        raise ValueError("This diagnostic is fixed to seed=1")
    required_epochs = 10 if args.parity_mode else 300
    if args.epochs != required_epochs:
        raise ValueError(f"epochs must be {required_epochs} in this mode")
    if args.num_monitored_clients not in (8, 10):
        raise ValueError("num_monitored_clients must be 10, or 8 for reported memory pressure")
    ratios = tuple(float(value) for value in args.probe_ratios.split(","))
    if len(ratios) != len(FIXED_RATIOS) or any(
        not math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(ratios, FIXED_RATIOS)
    ):
        raise ValueError(f"probe_ratios must be exactly {FIXED_RATIOS}")
    return ratios


@dataclass
class RNGState:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: Optional[List[torch.Tensor]]


def capture_rng_state() -> RNGState:
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_states,
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def rng_states_equal(left: RNGState, right: RNGState) -> bool:
    numpy_equal = (
        left.numpy[0] == right.numpy[0]
        and np.array_equal(left.numpy[1], right.numpy[1])
        and left.numpy[2:] == right.numpy[2:]
    )
    cuda_equal = left.torch_cuda is None and right.torch_cuda is None
    if left.torch_cuda is not None and right.torch_cuda is not None:
        cuda_equal = len(left.torch_cuda) == len(right.torch_cuda) and all(
            torch.equal(a, b) for a, b in zip(left.torch_cuda, right.torch_cuda)
        )
    return (
        left.python == right.python
        and numpy_equal
        and torch.equal(left.torch_cpu, right.torch_cpu)
        and cuda_equal
    )


def choose_monitored_clients(seed: int, num_users: int, count: int) -> List[int]:
    before = capture_rng_state()
    diagnostic_rng = np.random.default_rng(int(seed) + 910001)
    clients = sorted(
        int(item)
        for item in diagnostic_rng.choice(num_users, size=count, replace=False).tolist()
    )
    after = capture_rng_state()
    if not rng_states_equal(before, after):
        raise RuntimeError("Private monitored-client selection consumed training RNG")
    return clients


class ParameterGeometry:
    """Fixed named-parameter order; state_dict buffers are never flattened."""

    def __init__(self, model: torch.nn.Module):
        parameters = list(model.named_parameters())
        self.names = [name for name, _ in parameters]
        self.numels = [parameter.numel() for _, parameter in parameters]
        self.total_numel = int(sum(self.numels))

    def flatten(
        self, source: torch.nn.Module | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        tensors = dict(source.named_parameters()) if isinstance(source, torch.nn.Module) else source
        output = torch.empty(self.total_numel, dtype=torch.float32, device="cpu")
        offset = 0
        for name, numel in zip(self.names, self.numels):
            output[offset : offset + numel].copy_(
                tensors[name].detach().reshape(-1), non_blocking=False
            )
            offset += numel
        return output

    def add_direction_(
        self, model: torch.nn.Module, direction: torch.Tensor, epsilon: float
    ) -> None:
        named = dict(model.named_parameters())
        offset = 0
        with torch.no_grad():
            for name, numel in zip(self.names, self.numels):
                parameter = named[name]
                delta = direction[offset : offset + numel].reshape(parameter.shape)
                parameter.add_(delta.to(parameter.device, dtype=parameter.dtype), alpha=epsilon)
                offset += numel


def build_shadow_model(
    global_model: torch.nn.Module,
    geometry: ParameterGeometry,
    direction: torch.Tensor,
    epsilon: float,
    device: torch.device | str,
) -> torch.nn.Module:
    shadow = copy.deepcopy(global_model).to(device)
    geometry.add_direction_(shadow, direction, epsilon)
    return shadow


def server_update(returned: torch.Tensor, actual_dispatch: torch.Tensor) -> torch.Tensor:
    return returned - actual_dispatch


def realized_response(
    shadow_update: torch.Tensor, base_update: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if epsilon <= EPS:
        raise ValueError("epsilon must be positive")
    return (shadow_update - base_update) / epsilon


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.float()).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= EPS:
        return math.nan
    return float(torch.dot(left.float(), right.float()).item() / denominator)


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ClientState:
    num_visits: int = 0
    last_dispatch: Optional[torch.Tensor] = None
    last_update: Optional[torch.Tensor] = None
    obs_direction: Optional[torch.Tensor] = None
    obs_response: Optional[torch.Tensor] = None
    obs_input_norm: float = math.nan
    last_round: int = 0


@dataclass
class PreviousObservation:
    direction: torch.Tensor
    response: torch.Tensor
    input_norm: float
    last_update: torch.Tensor
    age_rounds: int


class ShadowObserver:
    def __init__(self, model: torch.nn.Module, monitored_clients: Iterable[int]):
        self.geometry = ParameterGeometry(model)
        self.monitored = set(int(item) for item in monitored_clients)
        self.states = {client: ClientState() for client in self.monitored}
        self.current_global: Optional[torch.Tensor] = None
        self.previous_round_global: Optional[torch.Tensor] = None
        self.global_direction: Optional[torch.Tensor] = None
        self.max_history_bytes = 0

    def begin_round(self, global_model: torch.nn.Module) -> None:
        current = self.geometry.flatten(global_model)
        if self.current_global is None:
            self.global_direction = None
        else:
            progress = current - self.current_global
            progress_norm = _norm(progress)
            self.global_direction = progress.div(progress_norm) if progress_norm > EPS else None
        self.previous_round_global = self.current_global
        self.current_global = current
        self._update_max_memory()

    def previous_observation(
        self, client_id: int, round_number: int
    ) -> Optional[PreviousObservation]:
        state = self.states[client_id]
        if (
            state.obs_direction is None
            or state.obs_response is None
            or state.last_update is None
            or not math.isfinite(state.obs_input_norm)
            or state.obs_input_norm <= EPS
        ):
            return None
        direction = state.obs_direction.float()
        direction_norm = _norm(direction)
        if direction_norm <= EPS or not _finite_tensor(direction):
            return None
        direction.div_(direction_norm)
        response = state.obs_response.float()
        if not _finite_tensor(response):
            return None
        return PreviousObservation(
            direction=direction,
            response=response,
            input_norm=float(state.obs_input_norm),
            last_update=state.last_update,
            age_rounds=int(round_number - state.last_round),
        )

    def update_after_shadows(
        self,
        client_id: int,
        round_number: int,
        dispatch: torch.Tensor,
        update: torch.Tensor,
    ) -> None:
        state = self.states[client_id]
        if state.last_dispatch is not None and state.last_update is not None:
            input_change = dispatch - state.last_dispatch
            response_change = update - state.last_update
            input_norm = _norm(input_change)
            if input_norm > EPS and _finite_tensor(input_change) and _finite_tensor(response_change):
                direction = input_change.div(input_norm)
                response = response_change.div(input_norm)
                state.obs_direction = direction.to(torch.float16)
                state.obs_response = response.to(torch.float16)
                state.obs_input_norm = input_norm
        state.num_visits += 1
        state.last_dispatch = dispatch
        state.last_update = update
        state.last_round = int(round_number)
        self._update_max_memory()

    def history_bytes(self) -> int:
        total = sum(
            _tensor_bytes(tensor)
            for tensor in (
                self.current_global,
                self.previous_round_global,
                self.global_direction,
            )
        )
        for state in self.states.values():
            total += sum(
                _tensor_bytes(tensor)
                for tensor in (
                    state.last_dispatch,
                    state.last_update,
                    state.obs_direction,
                    state.obs_response,
                )
            )
        return int(total)

    def _update_max_memory(self) -> None:
        self.max_history_bytes = max(self.max_history_bytes, self.history_bytes())


def _blank_event() -> Dict[str, Any]:
    return {field: "" for field in EVENT_FIELDS}


def compute_shadow_event(
    *,
    seed: int,
    round_number: int,
    client_id: int,
    visit_index: int,
    participation_gap: int,
    ratio: float,
    epsilon: float,
    previous: PreviousObservation,
    base_update: torch.Tensor,
    shadow_update: torch.Tensor,
    shadow_displacement_norm: float,
    global_direction: Optional[torch.Tensor],
) -> Dict[str, Any]:
    row = _blank_event()
    row.update(
        {
            "seed": int(seed),
            "round": int(round_number),
            "client_id": int(client_id),
            "client_visit_index": int(visit_index),
            "participation_gap": int(participation_gap),
            "probe_ratio": float(ratio),
            "epsilon": float(epsilon),
            "previous_obs_input_norm": float(previous.input_norm),
            "previous_obs_age_rounds": int(previous.age_rounds),
            "status": "ok",
            "skip_reason": "",
            "shadow_displacement_norm": float(shadow_displacement_norm),
            "epsilon_geometry_abs_error": abs(shadow_displacement_norm - epsilon),
        }
    )
    h_previous = previous.response
    delta_update = shadow_update - base_update
    h_real = realized_response(shadow_update, base_update, epsilon)
    predicted_norm = _norm(h_previous)
    realized_norm = _norm(h_real)
    row.update(
        {
            "predicted_response_norm": predicted_norm,
            "realized_response_norm": realized_norm,
            "response_cosine": _cosine(h_previous, h_real),
            "response_nre": _norm(h_real - h_previous)
            / (realized_norm + predicted_norm + EPS),
            "response_norm_ratio": realized_norm / (predicted_norm + EPS),
            "actual_delta_update_norm": _norm(delta_update),
            "predicted_delta_update_norm": epsilon * predicted_norm,
            "response_identity_abs_error": _norm(h_real.mul(epsilon) - delta_update),
        }
    )
    if global_direction is not None:
        conflict = -float(torch.dot(previous.last_update, global_direction).item())
        predicted_support = float(torch.dot(h_previous, global_direction).item())
        realized_support = float(torch.dot(h_real, global_direction).item())
        base_alignment = float(torch.dot(base_update, global_direction).item())
        shadow_alignment = float(torch.dot(shadow_update, global_direction).item())
        alignment_change = shadow_alignment - base_alignment
        row.update(
            {
                "conflict": conflict,
                "predicted_support": predicted_support,
                "realized_support": realized_support,
                "support_sign_agreement": int(
                    np.sign(predicted_support) == np.sign(realized_support)
                ),
                "predicted_active": int(conflict > 0 and predicted_support > 0),
                "base_alignment": base_alignment,
                "shadow_alignment": shadow_alignment,
                "alignment_change": alignment_change,
                "alignment_identity_abs_error": abs(
                    alignment_change - epsilon * realized_support
                ),
            }
        )
    return row


def skipped_event(
    *,
    seed: int,
    round_number: int,
    client_id: int,
    visit_index: int,
    participation_gap: int,
    ratio: float,
    epsilon: float,
    previous: PreviousObservation,
    reason: str,
) -> Dict[str, Any]:
    row = _blank_event()
    row.update(
        {
            "seed": seed,
            "round": round_number,
            "client_id": client_id,
            "client_visit_index": visit_index,
            "participation_gap": participation_gap,
            "probe_ratio": ratio,
            "epsilon": epsilon,
            "previous_obs_input_norm": previous.input_norm,
            "previous_obs_age_rounds": previous.age_rounds,
            "status": "skipped",
            "skip_reason": reason,
        }
    )
    return row


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value in ("", None):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def _stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> Optional[float]:
    if left.size < 2 or right.size != left.size:
        return None
    if spearman:
        left, right = _average_ranks(left), _average_ranks(right)
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
    if denominator <= EPS:
        return None
    return float(np.dot(left, right) / denominator)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    result: Dict[str, Any] = {
        "event_count": len(valid),
        "attempted_count": len(rows),
        "skipped_count": len(rows) - len(valid),
    }
    for field in (
        "response_cosine",
        "response_nre",
        "response_norm_ratio",
        "actual_delta_update_norm",
        "predicted_delta_update_norm",
        "realized_support",
        "alignment_change",
        "alignment_identity_abs_error",
        "epsilon_geometry_abs_error",
        "response_identity_abs_error",
    ):
        result[field] = _stats(_finite_values(valid, field))

    support_pairs = []
    for row in valid:
        try:
            predicted = float(row["predicted_support"])
            realized = float(row["realized_support"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(predicted) and math.isfinite(realized):
            support_pairs.append((predicted, realized))
    if support_pairs:
        pairs = np.asarray(support_pairs, dtype=np.float64)
        result["support_pair_count"] = len(support_pairs)
        result["support_pearson"] = _correlation(pairs[:, 0], pairs[:, 1])
        result["support_spearman"] = _correlation(
            pairs[:, 0], pairs[:, 1], spearman=True
        )
    else:
        result.update(
            {"support_pair_count": 0, "support_pearson": None, "support_spearman": None}
        )
    signs = _finite_values(valid, "support_sign_agreement")
    result["support_sign_agreement"] = float(np.mean(signs)) if signs.size else None
    active = [
        row
        for row in valid
        if row.get("predicted_active", "") not in ("", None)
        and int(row["predicted_active"]) == 1
    ]
    helpful = [float(row["realized_support"]) > 0 for row in active]
    result["predicted_active_count"] = len(active)
    result["predicted_active_helpful_fraction"] = (
        float(np.mean(helpful)) if helpful else None
    )
    return result


def build_summary(
    events: Sequence[Mapping[str, Any]],
    ratios: Sequence[float],
    accuracies: Sequence[float],
    monitored_clients: Sequence[int],
    max_history_bytes: int,
    parameter_numel: int,
    seed: int,
) -> Dict[str, Any]:
    ratio_summaries = {}
    gap_summaries = {}
    for ratio in ratios:
        ratio_rows = [
            row for row in events if math.isclose(float(row["probe_ratio"]), ratio, abs_tol=1e-12)
        ]
        ratio_summaries[str(ratio)] = summarize_rows(ratio_rows)
        gap_summaries[str(ratio)] = {}
        for name, predicate in GAP_BINS:
            gap_summaries[str(ratio)][name] = summarize_rows(
                [row for row in ratio_rows if predicate(int(row["participation_gap"]))]
            )
    accuracy = np.asarray(accuracies, dtype=np.float64)
    return {
        "seed": int(seed),
        "monitored_clients": [int(item) for item in monitored_clients],
        "parameter_numel": int(parameter_numel),
        "event_count": len(events),
        "valid_event_count": sum(row.get("status") == "ok" for row in events),
        "max_diagnostic_cpu_history_bytes": int(max_history_bytes),
        "max_diagnostic_cpu_history_gib": float(max_history_bytes / (1024**3)),
        "accuracy": {
            "rounds": int(accuracy.size),
            "final": float(accuracy[-1]) if accuracy.size else None,
            "best": float(np.max(accuracy)) if accuracy.size else None,
            "best_round": int(np.argmax(accuracy) + 1) if accuracy.size else None,
        },
        "ratio_summaries": ratio_summaries,
        "gap_summaries": gap_summaries,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _write_summary_csvs(output: Path, summary: Mapping[str, Any]) -> None:
    ratio_path = output / "ratio_summary.csv"
    with ratio_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "probe_ratio", "event_count", "attempted_count", "skipped_count",
            "response_cosine_mean", "response_cosine_median", "response_cosine_q25",
            "response_cosine_q75", "response_nre_mean", "response_nre_median",
            "response_norm_ratio_median", "support_pearson", "support_spearman",
            "support_sign_agreement", "predicted_active_count",
            "predicted_active_helpful_fraction", "realized_support_median",
            "alignment_change_median",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ratio, values in summary["ratio_summaries"].items():
            writer.writerow(
                {
                    "probe_ratio": ratio,
                    "event_count": values["event_count"],
                    "attempted_count": values["attempted_count"],
                    "skipped_count": values["skipped_count"],
                    "response_cosine_mean": values["response_cosine"]["mean"],
                    "response_cosine_median": values["response_cosine"]["median"],
                    "response_cosine_q25": values["response_cosine"]["q25"],
                    "response_cosine_q75": values["response_cosine"]["q75"],
                    "response_nre_mean": values["response_nre"]["mean"],
                    "response_nre_median": values["response_nre"]["median"],
                    "response_norm_ratio_median": values["response_norm_ratio"]["median"],
                    "support_pearson": values["support_pearson"],
                    "support_spearman": values["support_spearman"],
                    "support_sign_agreement": values["support_sign_agreement"],
                    "predicted_active_count": values["predicted_active_count"],
                    "predicted_active_helpful_fraction": values[
                        "predicted_active_helpful_fraction"
                    ],
                    "realized_support_median": values["realized_support"]["median"],
                    "alignment_change_median": values["alignment_change"]["median"],
                }
            )
    gap_path = output / "gap_summary.csv"
    with gap_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "probe_ratio", "gap_bin", "count", "response_cosine_median",
            "response_nre_median", "support_sign_agreement",
            "predicted_active_count", "predicted_active_helpful_fraction",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ratio, bins in summary["gap_summaries"].items():
            for name, values in bins.items():
                writer.writerow(
                    {
                        "probe_ratio": ratio,
                        "gap_bin": name,
                        "count": values["event_count"],
                        "response_cosine_median": values["response_cosine"]["median"],
                        "response_nre_median": values["response_nre"]["median"],
                        "support_sign_agreement": values["support_sign_agreement"],
                        "predicted_active_count": values["predicted_active_count"],
                        "predicted_active_helpful_fraction": values[
                            "predicted_active_helpful_fraction"
                        ],
                    }
                )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ratios = validate_protocol(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    if args.gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError("The fixed VGG protocol requires a CUDA GPU")
    torch.cuda.set_device(args.gpu)
    args.device = torch.device("cuda", args.gpu)
    args.algorithm = "FedAvg"
    monitored_clients = choose_monitored_clients(
        args.seed, args.num_users, args.num_monitored_clients
    )
    dataset_train, dataset_test, dict_users = get_dataset(args)
    global_model = VGG16(args).to(args.device)
    global_model.train()
    observer = ShadowObserver(global_model, monitored_clients)

    config = vars(args).copy()
    config["device"] = str(args.device)
    config["probe_ratios"] = list(ratios)
    config["monitored_clients"] = monitored_clients
    config["shadow_enters_aggregation"] = False
    config["geometry_scope"] = "named_parameters_only"
    _write_json(output / "config.json", config)

    events: List[Dict[str, Any]] = []
    accuracies: List[float] = []
    trajectory_fields = (
        "round", "test_accuracy", "round_train_seconds", "selected_clients",
        "global_state_sha256", "round_shadow_event_count", "observer_history_bytes",
        "max_alignment_identity_abs_error", "max_response_identity_abs_error",
        "max_epsilon_geometry_abs_error",
    )
    with (output / "events.csv").open("w", newline="", encoding="utf-8") as event_handle, (
        output / "trajectory.csv"
    ).open("w", newline="", encoding="utf-8") as trajectory_handle:
        event_writer = csv.DictWriter(event_handle, fieldnames=EVENT_FIELDS)
        trajectory_writer = csv.DictWriter(trajectory_handle, fieldnames=trajectory_fields)
        event_writer.writeheader()
        trajectory_writer.writeheader()

        for round_idx in range(args.epochs):
            round_number = round_idx + 1
            round_start = time.perf_counter()
            observer.begin_round(global_model)
            selected_count = max(int(args.frac * args.num_users), 1)
            selected = np.random.choice(
                range(args.num_users), selected_count, replace=False
            )
            local_states = []
            local_sizes = []
            round_events: List[Dict[str, Any]] = []

            for raw_client_id in selected:
                client_id = int(raw_client_id)
                monitored = client_id in observer.monitored
                previous = (
                    observer.previous_observation(client_id, round_number)
                    if monitored
                    else None
                )
                state = observer.states.get(client_id)
                visit_index = state.num_visits + 1 if state is not None else 0
                gap = round_number - state.last_round if state is not None and state.num_visits else 0
                rng_pre = capture_rng_state() if previous is not None else None

                base_model = copy.deepcopy(global_model).to(args.device)
                base_local = LocalUpdate_FedAvg(
                    args=args,
                    dataset=dataset_train,
                    idxs=dict_users[client_id],
                    dataset_test=dataset_test,
                )
                base_state = base_local.train(net=base_model)
                rng_post = capture_rng_state() if previous is not None else None
                local_states.append(copy.deepcopy(base_state))
                local_sizes.append(len(dict_users[client_id]))

                base_update = None
                base_dispatch = None
                if monitored:
                    if observer.current_global is None:
                        raise RuntimeError("Observer global state is unavailable")
                    base_dispatch = observer.current_global.clone()
                    base_returned = observer.geometry.flatten(base_state)
                    base_update = server_update(base_returned, base_dispatch)
                    del base_returned
                del base_state, base_local, base_model

                if previous is not None:
                    if rng_pre is None or rng_post is None or base_update is None:
                        raise RuntimeError("Incomplete paired-shadow state")
                    try:
                        for ratio in ratios:
                            epsilon = float(ratio * previous.input_norm)
                            if epsilon <= EPS or not math.isfinite(epsilon):
                                event = skipped_event(
                                    seed=args.seed,
                                    round_number=round_number,
                                    client_id=client_id,
                                    visit_index=visit_index,
                                    participation_gap=gap,
                                    ratio=ratio,
                                    epsilon=epsilon,
                                    previous=previous,
                                    reason="invalid_epsilon",
                                )
                                round_events.append(event)
                                continue
                            restore_rng_state(rng_pre)
                            shadow_model = build_shadow_model(
                                global_model,
                                observer.geometry,
                                previous.direction,
                                epsilon,
                                args.device,
                            )
                            shadow_dispatch = observer.geometry.flatten(shadow_model)
                            displacement_norm = _norm(
                                shadow_dispatch - observer.current_global
                            )
                            shadow_local = LocalUpdate_FedAvg(
                                args=args,
                                dataset=dataset_train,
                                idxs=dict_users[client_id],
                                dataset_test=dataset_test,
                            )
                            shadow_state = shadow_local.train(net=shadow_model)
                            shadow_returned = observer.geometry.flatten(shadow_state)
                            shadow_update = server_update(
                                shadow_returned, shadow_dispatch
                            )
                            if not _finite_tensor(shadow_update):
                                event = skipped_event(
                                    seed=args.seed,
                                    round_number=round_number,
                                    client_id=client_id,
                                    visit_index=visit_index,
                                    participation_gap=gap,
                                    ratio=ratio,
                                    epsilon=epsilon,
                                    previous=previous,
                                    reason="nonfinite_shadow_update",
                                )
                            else:
                                event = compute_shadow_event(
                                    seed=args.seed,
                                    round_number=round_number,
                                    client_id=client_id,
                                    visit_index=visit_index,
                                    participation_gap=gap,
                                    ratio=ratio,
                                    epsilon=epsilon,
                                    previous=previous,
                                    base_update=base_update,
                                    shadow_update=shadow_update,
                                    shadow_displacement_norm=displacement_norm,
                                    global_direction=observer.global_direction,
                                )
                            round_events.append(event)
                            del (
                                shadow_state,
                                shadow_local,
                                shadow_model,
                                shadow_dispatch,
                                shadow_returned,
                                shadow_update,
                            )
                    finally:
                        restore_rng_state(rng_post)

                if monitored:
                    if base_dispatch is None or base_update is None:
                        raise RuntimeError("Missing monitored base interaction")
                    observer.update_after_shadows(
                        client_id, round_number, base_dispatch, base_update
                    )
                del previous, base_update, base_dispatch

            global_state = Aggregation(local_states, local_sizes)
            global_model.load_state_dict(global_state)
            global_hash = state_sha256(global_model)
            round_seconds = time.perf_counter() - round_start
            accuracy = float(
                evaluate_round_accuracy(global_model, dataset_test, args, round_number)
            )
            accuracies.append(accuracy)
            events.extend(round_events)
            for event in round_events:
                event_writer.writerow(event)
            event_handle.flush()
            trajectory_writer.writerow(
                {
                    "round": round_number,
                    "test_accuracy": accuracy,
                    "round_train_seconds": round_seconds,
                    "selected_clients": json.dumps(
                        [int(item) for item in selected.tolist()]
                    ),
                    "global_state_sha256": global_hash,
                    "round_shadow_event_count": len(round_events),
                    "observer_history_bytes": observer.history_bytes(),
                    "max_alignment_identity_abs_error": max(
                        _finite_values(round_events, "alignment_identity_abs_error"),
                        default=math.nan,
                    ),
                    "max_response_identity_abs_error": max(
                        _finite_values(round_events, "response_identity_abs_error"),
                        default=math.nan,
                    ),
                    "max_epsilon_geometry_abs_error": max(
                        _finite_values(round_events, "epsilon_geometry_abs_error"),
                        default=math.nan,
                    ),
                }
            )
            trajectory_handle.flush()
            print(
                f"SHADOW_DIAGNOSTIC_ROUND seed={args.seed} round={round_number} "
                f"events={len(events)} history_gib={observer.history_bytes() / (1024**3):.3f}",
                flush=True,
            )
            del local_states, local_sizes, global_state, round_events
            torch.cuda.empty_cache()

    print_peak_accuracy(accuracies, "FedAvg")
    summary = build_summary(
        events,
        ratios,
        accuracies,
        monitored_clients,
        observer.max_history_bytes,
        observer.geometry.total_numel,
        args.seed,
    )
    _write_json(output / "summary.json", summary)
    _write_summary_csvs(output, summary)
    print(f"Completed seed {args.seed}: {output}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
