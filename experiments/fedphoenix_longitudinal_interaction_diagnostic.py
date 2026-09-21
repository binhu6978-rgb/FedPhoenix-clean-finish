#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only longitudinal interaction diagnostic for FedAvg/FedPhoenix.

The training path deliberately mirrors ``main_fed.py``.  The observer only
reads ``named_parameters()`` and keeps the most recent full-vector interaction
state for a fixed, privately sampled subset of clients.  No observer quantity
is used for client selection, local training, aggregation, or dispatch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from main_fed import _build_fedphoenix_tasks  # noqa: E402
from models.Fed import Aggregation  # noqa: E402
from models.Nets import VGG16  # noqa: E402
from models.Update import LocalUpdate_FedAvg  # noqa: E402
from models.test import evaluate_round_accuracy, print_peak_accuracy  # noqa: E402
from utils.get_dataset import get_dataset  # noqa: E402
from utils.set_seed import set_random_seed  # noqa: E402


EPS = 1.0e-12
EVENT_FIELDS = (
    "source_algorithm",
    "seed",
    "round",
    "client_id",
    "client_visit_index",
    "participation_gap",
    "task_id",
    "task_seed",
    "num_reset_kernels",
    "local_update_norm",
    "dispatch_minus_global_norm",
    "input_change_norm",
    "response_change_norm",
    "secant_gain",
    "response_change_ratio",
    "h_norm",
    "input_direction_cosine",
    "response_direction_cosine",
    "response_norm_ratio",
    "response_prediction_nre",
    "predicted_support",
    "realized_support",
    "support_sign_agreement",
)
SIMILARITY_SUBSETS = (
    ("all", None),
    ("input_cos_gt_0", 0.0),
    ("input_cos_ge_0_5", 0.5),
    ("input_cos_ge_0_7", 0.7),
    ("input_cos_ge_0_9", 0.9),
)
GAP_BINS = (
    ("gap_le_5", lambda gap: gap <= 5),
    ("gap_6_10", lambda gap: 6 <= gap <= 10),
    ("gap_11_20", lambda gap: 11 <= gap <= 20),
    ("gap_gt_20", lambda gap: gap > 20),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedAvg/FedPhoenix longitudinal repeated-interaction diagnostic"
    )
    parser.add_argument(
        "--source_algorithm", choices=("FedAvg", "FedPhoenix"), required=True
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
    parser.add_argument("--FP_conv", type=int, default=1000)
    parser.add_argument("--FP_fc", type=int, default=0)
    parser.add_argument("--reset", type=float, default=0.015625)
    parser.add_argument("--remethod", default="ori_normal")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_protocol(args: argparse.Namespace) -> None:
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
        "seed": 1,
        "FP_conv": 1000,
        "FP_fc": 0,
        "reset": 0.015625,
        "remethod": "ori_normal",
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual = getattr(args, name)
        if isinstance(expected_value, float):
            matches = math.isclose(float(actual), expected_value, abs_tol=1e-12)
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(f"{name}={actual!r} (required {expected_value!r})")
    if mismatches:
        raise ValueError("Fixed diagnostic protocol violated: " + ", ".join(mismatches))
    if not 1 <= args.num_monitored_clients <= args.num_users:
        raise ValueError("num_monitored_clients must be in [1, num_users]")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")


def _numpy_state_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def snapshot_global_rng(include_cuda: bool = True) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if include_cuda and torch.cuda.is_available() and torch.cuda.is_initialized():
        snapshot["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return snapshot


def assert_global_rng_unchanged(before: Mapping[str, Any], context: str) -> None:
    after = snapshot_global_rng(include_cuda="cuda" in before)
    same = before["python"] == after["python"]
    same = same and _numpy_state_equal(before["numpy"], after["numpy"])
    same = same and torch.equal(before["torch"], after["torch"])
    if "cuda" in before:
        same = same and len(before["cuda"]) == len(after.get("cuda", []))
        same = same and all(
            torch.equal(left, right)
            for left, right in zip(before["cuda"], after.get("cuda", []))
        )
    if not same:
        raise RuntimeError(f"Diagnostic observer consumed global RNG during {context}")


def choose_monitored_clients(seed: int, num_users: int, count: int) -> List[int]:
    before = snapshot_global_rng(include_cuda=False)
    diagnostic_rng = np.random.default_rng(int(seed) + 880001)
    clients = sorted(
        int(value)
        for value in diagnostic_rng.choice(num_users, size=count, replace=False).tolist()
    )
    assert_global_rng_unchanged(before, "monitored-client selection")
    return clients


def _safe_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.float()).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.float()
    right_f = right.float()
    denom = _safe_norm(left_f) * _safe_norm(right_f)
    if denom <= EPS:
        return math.nan
    return float(torch.dot(left_f, right_f).item() / denom)


def _tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


@dataclass
class ClientHistory:
    num_visits: int = 0
    last_dispatch: Optional[torch.Tensor] = None
    last_update: Optional[torch.Tensor] = None
    last_obs_direction: Optional[torch.Tensor] = None
    last_obs_response: Optional[torch.Tensor] = None
    last_obs_input_norm: float = math.nan
    last_round: int = 0


class LongitudinalObserver:
    """CPU-only full-parameter observer with one-step client histories."""

    def __init__(self, model: torch.nn.Module, monitored_clients: Iterable[int]):
        self.param_names = [name for name, _ in model.named_parameters()]
        self.param_numels = [parameter.numel() for _, parameter in model.named_parameters()]
        self.total_numel = int(sum(self.param_numels))
        self.monitored = set(int(item) for item in monitored_clients)
        self.history = {client: ClientHistory() for client in self.monitored}
        self.current_global: Optional[torch.Tensor] = None
        self.previous_global: Optional[torch.Tensor] = None
        self.global_direction: Optional[torch.Tensor] = None
        self.max_history_bytes = 0

    def flatten_named_parameters(
        self, source: torch.nn.Module | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(source, torch.nn.Module):
            tensors = dict(source.named_parameters())
        else:
            tensors = source
        result = torch.empty(self.total_numel, dtype=torch.float32, device="cpu")
        offset = 0
        for name, numel in zip(self.param_names, self.param_numels):
            tensor = tensors[name].detach().reshape(-1)
            result[offset : offset + numel].copy_(tensor, non_blocking=False)
            offset += numel
        return result

    def begin_round(self, global_model: torch.nn.Module) -> None:
        before = snapshot_global_rng()
        current = self.flatten_named_parameters(global_model)
        if self.current_global is None:
            self.previous_global = None
            self.global_direction = None
        else:
            self.previous_global = self.current_global
            progress = current - self.previous_global
            progress_norm = _safe_norm(progress)
            self.global_direction = (
                progress.div(progress_norm) if progress_norm > EPS else None
            )
        self.current_global = current
        self._update_max_memory()
        assert_global_rng_unchanged(before, "round setup")

    def observe(
        self,
        *,
        algorithm: str,
        seed: int,
        round_number: int,
        client_id: int,
        dispatch: torch.Tensor,
        returned_state: Mapping[str, torch.Tensor],
        task_id: Optional[int],
        task_seed: Optional[int],
        num_reset_kernels: Optional[int],
    ) -> Dict[str, Any]:
        if client_id not in self.monitored:
            raise ValueError("observe called for an unmonitored client")
        if self.current_global is None:
            raise RuntimeError("begin_round must be called before observe")
        before = snapshot_global_rng()
        returned = self.flatten_named_parameters(returned_state)
        update = returned.sub_(dispatch)
        state = self.history[client_id]
        visit_index = state.num_visits + 1
        gap = round_number - state.last_round if state.num_visits else None
        row: Dict[str, Any] = {field: "" for field in EVENT_FIELDS}
        row.update(
            {
                "source_algorithm": algorithm,
                "seed": int(seed),
                "round": int(round_number),
                "client_id": int(client_id),
                "client_visit_index": int(visit_index),
                "participation_gap": gap if gap is not None else "",
                "task_id": task_id if task_id is not None else "",
                "task_seed": task_seed if task_seed is not None else "",
                "num_reset_kernels": (
                    num_reset_kernels if num_reset_kernels is not None else ""
                ),
                "local_update_norm": _safe_norm(update),
                "dispatch_minus_global_norm": _safe_norm(
                    dispatch - self.current_global
                ),
            }
        )

        if state.last_dispatch is not None and state.last_update is not None:
            direction = dispatch - state.last_dispatch
            response = update - state.last_update
            direction_norm = _safe_norm(direction)
            response_norm = _safe_norm(response)
            update_denominator = (
                _safe_norm(update) + _safe_norm(state.last_update) + EPS
            )
            row.update(
                {
                    "input_change_norm": direction_norm,
                    "response_change_norm": response_norm,
                    "secant_gain": response_norm / (direction_norm + EPS),
                    "response_change_ratio": response_norm / update_denominator,
                }
            )
            if direction_norm > EPS:
                current_direction = direction.div(direction_norm)
                current_response = response.div(direction_norm)
                row["h_norm"] = _safe_norm(current_response)

                if (
                    state.last_obs_direction is not None
                    and state.last_obs_response is not None
                ):
                    previous_direction = state.last_obs_direction.float()
                    previous_direction_norm = _safe_norm(previous_direction)
                    if previous_direction_norm > EPS:
                        previous_direction.div_(previous_direction_norm)
                    previous_response = state.last_obs_response.float()
                    previous_response_norm = _safe_norm(previous_response)
                    current_response_norm = _safe_norm(current_response)
                    row.update(
                        {
                            "input_direction_cosine": _cosine(
                                previous_direction, current_direction
                            ),
                            "response_direction_cosine": _cosine(
                                previous_response, current_response
                            ),
                            "response_norm_ratio": current_response_norm
                            / (previous_response_norm + EPS),
                            "response_prediction_nre": _safe_norm(
                                current_response - previous_response
                            )
                            / (
                                current_response_norm
                                + previous_response_norm
                                + EPS
                            ),
                        }
                    )
                    if self.global_direction is not None:
                        predicted = float(
                            torch.dot(previous_response, self.global_direction).item()
                        )
                        realized = float(
                            torch.dot(current_response, self.global_direction).item()
                        )
                        row.update(
                            {
                                "predicted_support": predicted,
                                "realized_support": realized,
                                "support_sign_agreement": int(
                                    np.sign(predicted) == np.sign(realized)
                                ),
                            }
                        )
                    del previous_direction, previous_response

                # These are explicitly the only lossy persistent tensors.
                state.last_obs_direction = current_direction.to(torch.float16)
                state.last_obs_response = current_response.to(torch.float16)
                state.last_obs_input_norm = direction_norm
                del current_direction, current_response
            del direction, response

        state.num_visits = visit_index
        state.last_dispatch = dispatch
        state.last_update = update
        state.last_round = int(round_number)
        self._update_max_memory()
        assert_global_rng_unchanged(before, f"client {client_id} observation")
        return row

    def history_bytes(self) -> int:
        total = sum(
            _tensor_bytes(tensor)
            for tensor in (
                self.current_global,
                self.previous_global,
                self.global_direction,
            )
        )
        for state in self.history.values():
            total += sum(
                _tensor_bytes(tensor)
                for tensor in (
                    state.last_dispatch,
                    state.last_update,
                    state.last_obs_direction,
                    state.last_obs_response,
                )
            )
        return int(total)

    def _update_max_memory(self) -> None:
        self.max_history_bytes = max(self.max_history_bytes, self.history_bytes())


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


def _descriptive(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
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
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= EPS:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _transfer_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"count": len(rows), "insufficient_count": len(rows) < 10}
    for field in (
        "input_direction_cosine",
        "response_direction_cosine",
        "response_prediction_nre",
        "response_norm_ratio",
    ):
        result[field] = _descriptive(_finite_values(rows, field))
    support_pairs = []
    for row in rows:
        try:
            predicted = float(row.get("predicted_support", ""))
            realized = float(row.get("realized_support", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(predicted) and math.isfinite(realized):
            support_pairs.append((predicted, realized))
    if support_pairs:
        support = np.asarray(support_pairs, dtype=np.float64)
        result["support_pair_count"] = len(support_pairs)
        result["support_pearson"] = _correlation(support[:, 0], support[:, 1])
        result["support_spearman"] = _correlation(
            support[:, 0], support[:, 1], spearman=True
        )
    else:
        result.update(
            {"support_pair_count": 0, "support_pearson": None, "support_spearman": None}
        )
    signs = _finite_values(rows, "support_sign_agreement")
    result["support_sign_agreement_rate"] = (
        float(np.mean(signs)) if signs.size else None
    )
    return result


def build_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    algorithm: str,
    seed: int,
    monitored_clients: Sequence[int],
    accuracies: Sequence[float],
    max_history_bytes: int,
    total_numel: int,
) -> Dict[str, Any]:
    scalar_metrics = {}
    for field in (
        "local_update_norm",
        "dispatch_minus_global_norm",
        "input_change_norm",
        "response_change_norm",
        "secant_gain",
        "response_change_ratio",
        "h_norm",
    ):
        scalar_metrics[field] = _descriptive(_finite_values(events, field))

    transferable = [
        row
        for row in events
        if row.get("input_direction_cosine", "") not in ("", None)
    ]
    similarity = {}
    for name, threshold in SIMILARITY_SUBSETS:
        subset = transferable
        if threshold is not None:
            subset = [
                row
                for row in transferable
                if float(row["input_direction_cosine"])
                >= threshold
                + (np.finfo(float).eps if name == "input_cos_gt_0" else 0.0)
            ]
        similarity[name] = _transfer_summary(subset)

    gaps = {}
    for name, predicate in GAP_BINS:
        subset = [
            row
            for row in transferable
            if row.get("participation_gap", "") not in ("", None)
            and predicate(int(row["participation_gap"]))
        ]
        gaps[name] = _transfer_summary(subset)

    accuracy_values = np.asarray(accuracies, dtype=np.float64)
    visits = {str(client): 0 for client in monitored_clients}
    for row in events:
        visits[str(int(row["client_id"]))] += 1
    return {
        "source_algorithm": algorithm,
        "seed": int(seed),
        "monitored_clients": [int(item) for item in monitored_clients],
        "monitored_client_visits": visits,
        "event_count": len(events),
        "transfer_event_count": len(transferable),
        "parameter_numel": int(total_numel),
        "max_diagnostic_cpu_history_bytes": int(max_history_bytes),
        "max_diagnostic_cpu_history_gib": float(max_history_bytes / (1024**3)),
        "accuracy": {
            "rounds": int(len(accuracy_values)),
            "final": float(accuracy_values[-1]) if accuracy_values.size else None,
            "best": float(np.max(accuracy_values)) if accuracy_values.size else None,
            "best_round": int(np.argmax(accuracy_values) + 1) if accuracy_values.size else None,
        },
        "metrics": scalar_metrics,
        "similarity_subsets": similarity,
        "participation_gap_bins": gaps,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.source_algorithm.lower()
    events_path = output_dir / f"{prefix}_events.csv"
    trajectory_path = output_dir / f"{prefix}_trajectory.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    config_path = output_dir / f"{prefix}_config.json"

    set_random_seed(args.seed)
    if args.gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError("This fixed VGG diagnostic requires a CUDA GPU")
    torch.cuda.set_device(args.gpu)
    args.device = torch.device("cuda", args.gpu)
    args.algorithm = args.source_algorithm
    args.density_local = 0.01

    monitored_clients = choose_monitored_clients(
        args.seed, args.num_users, args.num_monitored_clients
    )
    dataset_train, dataset_test, dict_users = get_dataset(args)
    global_model = VGG16(args).to(args.device)
    global_model.train()
    observer = LongitudinalObserver(global_model, monitored_clients)

    config = vars(args).copy()
    config["device"] = str(args.device)
    config["monitored_clients"] = monitored_clients
    config["diagnostic_only_named_parameters"] = True
    config["observer_affects_training"] = False
    _write_json(config_path, config)

    events: List[Dict[str, Any]] = []
    accuracies: List[float] = []
    trajectory_fields = (
        "round",
        "source_algorithm",
        "test_accuracy",
        "round_train_seconds",
        "selected_clients",
        "assignments",
        "task_seeds",
        "observer_history_bytes",
    )
    with events_path.open("w", newline="", encoding="utf-8") as event_handle, trajectory_path.open(
        "w", newline="", encoding="utf-8"
    ) as trajectory_handle:
        event_writer = csv.DictWriter(event_handle, fieldnames=EVENT_FIELDS)
        trajectory_writer = csv.DictWriter(
            trajectory_handle, fieldnames=trajectory_fields
        )
        event_writer.writeheader()
        trajectory_writer.writeheader()

        for round_idx in range(args.epochs):
            round_number = round_idx + 1
            round_start = time.perf_counter()
            observer.begin_round(global_model)
            local_states = []
            local_sizes = []
            selected_count = max(int(args.frac * args.num_users), 1)
            selected = np.random.choice(
                range(args.num_users), selected_count, replace=False
            )
            task_models = None
            task_traces: List[Mapping[str, Any]] = []
            if args.source_algorithm == "FedPhoenix":
                task_models, task_traces = _build_fedphoenix_tasks(
                    global_model, round_idx, selected_count, args
                )

            assignments = []
            for task_id, raw_client_id in enumerate(selected):
                client_id = int(raw_client_id)
                if task_models is None:
                    dispatch_source = global_model
                    task_seed = None
                    reset_count = None
                else:
                    dispatch_source = task_models[task_id]
                    task_seed = int(task_traces[task_id]["seed"])
                    reset_count = int(task_traces[task_id]["num_reset_kernels"])

                dispatch = None
                if client_id in observer.monitored:
                    observer_before = snapshot_global_rng()
                    dispatch = observer.flatten_named_parameters(dispatch_source)
                    assert_global_rng_unchanged(
                        observer_before, f"client {client_id} dispatch capture"
                    )

                local_model = copy.deepcopy(dispatch_source).to(args.device)
                local = LocalUpdate_FedAvg(
                    args=args,
                    dataset=dataset_train,
                    idxs=dict_users[client_id],
                    dataset_test=dataset_test,
                )
                returned_state = local.train(net=local_model)
                local_states.append(copy.deepcopy(returned_state))
                local_sizes.append(len(dict_users[client_id]))
                assignments.append({"client_id": client_id, "task_id": task_id})

                if dispatch is not None:
                    event = observer.observe(
                        algorithm=args.source_algorithm,
                        seed=args.seed,
                        round_number=round_number,
                        client_id=client_id,
                        dispatch=dispatch,
                        returned_state=returned_state,
                        task_id=(task_id if task_models is not None else None),
                        task_seed=task_seed,
                        num_reset_kernels=reset_count,
                    )
                    events.append(event)
                    event_writer.writerow(event)
                    event_handle.flush()
                del local, local_model, returned_state

            global_state = Aggregation(local_states, local_sizes)
            global_model.load_state_dict(global_state)
            round_seconds = time.perf_counter() - round_start
            accuracy = evaluate_round_accuracy(
                global_model, dataset_test, args, round_number
            )
            accuracies.append(float(accuracy))
            trajectory_writer.writerow(
                {
                    "round": round_number,
                    "source_algorithm": args.source_algorithm,
                    "test_accuracy": float(accuracy),
                    "round_train_seconds": float(round_seconds),
                    "selected_clients": json.dumps(
                        [int(item) for item in selected.tolist()]
                    ),
                    "assignments": json.dumps(assignments),
                    "task_seeds": json.dumps(
                        [int(trace["seed"]) for trace in task_traces]
                    ),
                    "observer_history_bytes": observer.history_bytes(),
                }
            )
            trajectory_handle.flush()
            print(
                "DIAGNOSTIC_ROUND "
                f"algorithm={args.source_algorithm} round={round_number} "
                f"events={len(events)} history_gib={observer.history_bytes() / (1024**3):.3f}",
                flush=True,
            )
            del local_states, local_sizes, global_state, task_models
            torch.cuda.empty_cache()

    print_peak_accuracy(accuracies, args.source_algorithm)
    summary = build_summary(
        events,
        algorithm=args.source_algorithm,
        seed=args.seed,
        monitored_clients=monitored_clients,
        accuracies=accuracies,
        max_history_bytes=observer.max_history_bytes,
        total_numel=observer.total_numel,
    )
    _write_json(summary_path, summary)
    print(f"Events: {events_path}")
    print(f"Summary: {summary_path}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
