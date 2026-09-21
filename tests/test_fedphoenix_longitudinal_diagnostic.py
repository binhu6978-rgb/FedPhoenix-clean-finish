from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fedphoenix_longitudinal_interaction_diagnostic import (  # noqa: E402
    LongitudinalObserver,
    build_summary,
    choose_monitored_clients,
    snapshot_global_rng,
)


def test_private_monitor_selection_does_not_consume_global_rng():
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    before = snapshot_global_rng(include_cuda=False)
    first = choose_monitored_clients(1, 100, 10)
    second = choose_monitored_clients(1, 100, 10)
    after = snapshot_global_rng(include_cuda=False)
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])


def test_observer_geometry_uses_parameters_and_forms_transfer_event():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    observer = LongitudinalObserver(model, [7])

    observer.begin_round(model)
    dispatch1 = observer.flatten_named_parameters(model)
    returned1 = {"weight": torch.tensor([[2.0, 2.0]])}
    first = observer.observe(
        algorithm="FedAvg", seed=1, round_number=1, client_id=7,
        dispatch=dispatch1, returned_state=returned1,
        task_id=None, task_seed=None, num_reset_kernels=None,
    )
    assert first["client_visit_index"] == 1
    assert first["input_change_norm"] == ""

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 2.0]]))
    observer.begin_round(model)
    dispatch2 = observer.flatten_named_parameters(model)
    returned2 = {"weight": torch.tensor([[4.0, 2.0]])}
    second = observer.observe(
        algorithm="FedAvg", seed=1, round_number=2, client_id=7,
        dispatch=dispatch2, returned_state=returned2,
        task_id=None, task_seed=None, num_reset_kernels=None,
    )
    assert math.isclose(second["input_change_norm"], 1.0)
    assert math.isclose(second["response_change_norm"], 1.0)
    assert math.isclose(second["secant_gain"], 1.0)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 2.0]]))
    observer.begin_round(model)
    dispatch3 = observer.flatten_named_parameters(model)
    returned3 = {"weight": torch.tensor([[6.0, 2.0]])}
    third = observer.observe(
        algorithm="FedAvg", seed=1, round_number=3, client_id=7,
        dispatch=dispatch3, returned_state=returned3,
        task_id=None, task_seed=None, num_reset_kernels=None,
    )
    assert math.isclose(third["input_direction_cosine"], 1.0)
    assert math.isclose(third["response_direction_cosine"], 1.0)
    assert math.isclose(third["response_prediction_nre"], 0.0)
    assert third["support_sign_agreement"] == 1


def test_summary_marks_small_similarity_subsets_insufficient():
    rows = []
    for index in range(3):
        rows.append(
            {
                "client_id": 1,
                "input_direction_cosine": 0.8,
                "response_direction_cosine": 0.5,
                "response_prediction_nre": 0.25,
                "response_norm_ratio": 1.0,
                "predicted_support": index + 1,
                "realized_support": index + 2,
                "support_sign_agreement": 1,
                "participation_gap": 4,
                "local_update_norm": 1.0,
                "dispatch_minus_global_norm": 0.0,
                "input_change_norm": 1.0,
                "response_change_norm": 0.5,
                "secant_gain": 0.5,
                "response_change_ratio": 0.2,
                "h_norm": 0.5,
            }
        )
    summary = build_summary(
        rows,
        algorithm="FedAvg",
        seed=1,
        monitored_clients=[1],
        accuracies=[10.0],
        max_history_bytes=100,
        total_numel=2,
    )
    subset = summary["similarity_subsets"]["input_cos_ge_0_7"]
    assert subset["count"] == 3
    assert subset["insufficient_count"] is True
    assert math.isclose(subset["support_pearson"], 1.0)
