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

from experiments.same_direction_shadow_transfer import (  # noqa: E402
    ParameterGeometry,
    build_shadow_model,
    capture_rng_state,
    choose_monitored_clients,
    realized_response,
    restore_rng_state,
    rng_states_equal,
    server_update,
)


def test_rng_capture_restore_replays_python_numpy_and_torch():
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    state = capture_rng_state()
    expected = (
        [random.random() for _ in range(3)],
        np.random.rand(3),
        torch.rand(3),
    )
    restore_rng_state(state)
    actual = (
        [random.random() for _ in range(3)],
        np.random.rand(3),
        torch.rand(3),
    )
    assert expected[0] == actual[0]
    assert np.array_equal(expected[1], actual[1])
    assert torch.equal(expected[2], actual[2])


def test_private_monitor_selection_preserves_all_global_rng_states():
    before = capture_rng_state()
    first = choose_monitored_clients(1, 100, 10)
    second = choose_monitored_clients(1, 100, 10)
    after = capture_rng_state()
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert rng_states_equal(before, after)


def test_shadow_has_no_alias_and_epsilon_geometry_is_exact():
    global_model = torch.nn.Linear(3, 2, bias=True)
    geometry = ParameterGeometry(global_model)
    original_state = {
        name: tensor.detach().clone() for name, tensor in global_model.state_dict().items()
    }
    direction = torch.arange(1, geometry.total_numel + 1, dtype=torch.float32)
    direction /= torch.linalg.vector_norm(direction)
    epsilon = 0.125
    global_flat = geometry.flatten(global_model)
    shadow = build_shadow_model(
        global_model, geometry, direction, epsilon, torch.device("cpu")
    )
    shadow_flat = geometry.flatten(shadow)
    displacement = shadow_flat - global_flat
    assert math.isclose(
        float(torch.linalg.vector_norm(displacement)), epsilon, rel_tol=1e-6, abs_tol=1e-7
    )
    with torch.no_grad():
        next(shadow.parameters()).add_(100.0)
    for name, tensor in global_model.state_dict().items():
        assert torch.equal(tensor, original_state[name])


def test_server_accounting_response_and_alignment_identities():
    base_dispatch = torch.tensor([1.0, 2.0, 3.0])
    base_returned = torch.tensor([1.5, 1.0, 4.0])
    shadow_dispatch = torch.tensor([1.1, 2.0, 3.0])
    shadow_returned = torch.tensor([1.8, 0.9, 4.2])
    epsilon = 0.1
    base_update = server_update(base_returned, base_dispatch)
    shadow_update = server_update(shadow_returned, shadow_dispatch)
    assert torch.equal(shadow_update, shadow_returned - shadow_dispatch)

    h_real = realized_response(shadow_update, base_update, epsilon)
    delta_update = shadow_update - base_update
    assert torch.allclose(h_real * epsilon, delta_update, atol=1e-7, rtol=1e-7)

    global_direction = torch.tensor([1.0, 2.0, -1.0])
    global_direction /= torch.linalg.vector_norm(global_direction)
    realized_support = torch.dot(h_real, global_direction)
    alignment_change = torch.dot(delta_update, global_direction)
    assert torch.allclose(
        epsilon * realized_support, alignment_change, atol=1e-7, rtol=1e-7
    )
