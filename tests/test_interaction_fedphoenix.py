import copy
import inspect
import math
import random
import unittest

import numpy as np
import torch
from torch import nn

from Algorithm.InteractionFedPhoenix import (
    ClientHistory, InteractionController, InteractionGeometry,
    ResetLedger, corrected_state,
)
from models.Fed import Aggregation


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, 1, bias=True), nn.BatchNorm2d(2)
        )
        self.fc = nn.Linear(2, 2)


def trace(indices=()):
    return {"layers": [{"name": "features.0", "num_kernels": 2,
                        "reset_indices": list(indices)}]}


class InteractionFedPhoenixTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyNet()
        self.head = InteractionGeometry(self.model, "head")
        self.full = InteractionGeometry(self.model, "reset_aware")

    def test_geometry_excludes_conv_bias_and_buffers(self):
        self.assertEqual(
            self.head.names,
            ("features.1.weight", "features.1.bias", "fc.weight", "fc.bias"),
        )
        self.assertEqual(self.full.names[0], "features.0.weight")
        self.assertNotIn("features.0.bias", self.full.names)
        self.assertFalse(any("running" in name or "tracked" in name
                             for name in self.full.names))
        state = self.model.state_dict()
        self.assertEqual(self.full.flatten(state).numel(), self.full.numel)
        before = {name: tensor.clone() for name, tensor in state.items()}
        proposed = torch.ones(self.full.numel) * 0.125
        action = self.full.materialize(self.model, before, proposed)
        self.assertNotIn("features.0.bias", action)
        self.assertTrue(torch.equal(before["features.0.bias"], state["features.0.bias"]))
        self.assertTrue(torch.equal(before["features.1.running_mean"],
                                    state["features.1.running_mean"]))

    def test_ledger_interval_union_and_other_client_reset(self):
        ledger = ResetLedger(self.model)
        ledger.record(0, [trace([0]), trace([])])
        ledger.record(1, [trace([1])])
        self.assertEqual(ledger.interval_union(0, 1)["features.0"].tolist(),
                         [True, False])
        self.assertEqual(ledger.interval_union(0, 2)["features.0"].tolist(),
                         [True, True])
        self.assertEqual(ledger.interval_union(1, 1)["features.0"].tolist(),
                         [False, False])

    def test_history_identity_cold_start_and_no_leakage(self):
        ledger = ResetLedger(self.model)
        controller = InteractionController(self.head, ledger, rho=0.05)
        self.assertEqual(
            tuple(inspect.signature(controller.decide).parameters),
            ("client_id", "round_idx", "current_reset", "global_change"),
        )
        v = torch.zeros(self.head.numel)
        v[0] = 1
        zero = torch.zeros_like(v)
        ledger.record(0, [trace([])])
        own = ledger.from_trace(trace([]))
        self.assertEqual(controller.decide(7, 0, own, v)[1]["reason"], "cold_start")
        controller.history.record(7, 0, zero, -2 * v, own)
        ledger.record(1, [trace([])])
        self.assertEqual(controller.decide(7, 1, own, v)[1]["reason"], "cold_start")
        controller.history.record(8, 1, zero, zero, own)
        controller.history.record(7, 1, v, -v, own)
        ledger.record(2, [trace([])])
        self.assertEqual(controller.decide(8, 2, own, v)[1]["reason"], "cold_start")
        delta, info = controller.decide(7, 2, own, v)
        self.assertEqual(info["reason"], "active")
        self.assertAlmostEqual(info["lambda"], 0.05)
        self.assertAlmostEqual(float(delta[0]), 0.05, places=5)
        self.assertTrue(torch.equal(delta[1:], zero[1:]))
        self.assertEqual(controller.decide(7, 22, own, v)[1]["reason"], "stale")

    def test_observation_span_respects_max_gap_and_recovers(self):
        ledger = ResetLedger(self.model)
        controller = InteractionController(self.head, ledger, max_gap=20)
        own = ledger.from_trace(trace([]))
        vector = torch.ones(self.head.numel)
        for round_idx in range(106):
            ledger.record(round_idx, [trace([])])
            if round_idx in (0, 100, 105):
                controller.history.record(
                    7, round_idx, vector * round_idx,
                    -vector * (106 - round_idx), own,
                )
                if round_idx == 100:
                    item = controller.history.clients[7]
                    self.assertEqual(item["t_latest"], 100)
                    self.assertFalse(item["obs_valid"])
                    self.assertIsNone(item["d"])
                    self.assertIsNone(item["r"])
        item = controller.history.clients[7]
        self.assertTrue(item["obs_valid"])
        self.assertEqual(item["t_prev"], 100)
        self.assertEqual(item["t_latest"], 105)

    def test_historical_previous_current_and_ledger_masks(self):
        ledger = ResetLedger(self.model)
        controller = InteractionController(self.full, ledger, rho=0.05)
        size = self.full.numel
        x0 = torch.zeros(size)
        x1 = torch.ones(size)
        u0 = -2 * torch.ones(size)
        u1 = -torch.ones(size)
        # Another task's reset at t=0 and this client's reset at t=1.
        ledger.record(0, [trace([0]), trace([])])
        controller.history.record(5, 0, x0, u0, ledger.from_trace(trace([])))
        ledger.record(1, [trace([1])])
        controller.history.record(5, 1, x1, u1, ledger.from_trace(trace([1])))
        hist = controller.history.clients[5]
        self.assertEqual(hist["hist_valid"]["features.0"].tolist(), [False, False])
        ledger.record(2, [trace([])])
        delta, _ = controller.decide(5, 2, ledger.from_trace(trace([])),
                                     torch.ones(size))
        self.assertTrue(torch.equal(delta[:2], torch.zeros(2)))

        # Fresh clean history: previous-round union and current own reset
        # independently remove their Conv kernels from the action.
        ledger2 = ResetLedger(self.model)
        ctl2 = InteractionController(self.full, ledger2, rho=0.05)
        ledger2.record(0, [trace([])])
        ctl2.history.record(5, 0, x0, u0, ledger2.from_trace(trace([])))
        ledger2.record(1, [trace([])])
        ctl2.history.record(5, 1, x1, u1, ledger2.from_trace(trace([])))
        ledger2.record(2, [trace([0])])
        ledger2.record(3, [trace([])])
        delta, _ = ctl2.decide(5, 3, ledger2.from_trace(trace([1])),
                               torch.ones(size))
        self.assertTrue(torch.equal(delta[:2], torch.zeros(2)))

    def test_current_other_client_reset_is_not_visible_to_decision(self):
        ledger = ResetLedger(self.model)
        ctl = InteractionController(self.full, ledger, rho=0.05)
        vector = torch.ones(self.full.numel)
        own = ledger.from_trace(trace([]))
        ledger.record(0, [trace([])])
        ctl.history.record(5, 0, torch.zeros_like(vector), -2 * vector, own)
        ledger.record(1, [trace([])])
        ctl.history.record(5, 1, vector, -vector, own)
        # L[2] contains another client's reset. Only this client's R[2]
        # belongs to the current decision mask.
        ledger.record(2, [trace([0]), trace([])])
        delta, info = ctl.decide(5, 2, own, vector)
        self.assertEqual(info["reason"], "active")
        self.assertGreater(float(delta[0]), 0)

    def test_corrected_aggregation_active_and_inactive(self):
        base = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        returned = {name: tensor.clone() for name, tensor in base.items()}
        name = "fc.weight"
        action = torch.ones_like(base[name]) * 0.25
        returned[name] += action + 0.1
        returned["features.1.running_mean"] += 0.5
        corrected = corrected_state(returned, {name: action})
        self.assertTrue(torch.allclose(corrected[name], base[name] + 0.1))
        self.assertTrue(torch.equal(corrected["features.1.running_mean"],
                                    returned["features.1.running_mean"]))
        self.assertEqual(Aggregation([corrected], [1])[name].shape, base[name].shape)
        inactive = copy.deepcopy(returned)
        self.assertTrue(all(torch.equal(inactive[key], returned[key]) for key in base))

    def test_numerical_boundaries_and_rng_state(self):
        ledger = ResetLedger(self.model)
        ctl = InteractionController(self.head, ledger, rho=0.05)
        v = torch.zeros(self.head.numel)
        v[0] = 1
        zero = torch.zeros_like(v)
        py_state = random.getstate()
        np_state = np.random.get_state()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ledger.record(0, [trace([])])
        own = ledger.from_trace(trace([]))
        ctl.history.record(3, 0, zero, -2 * v, own)
        ledger.record(1, [trace([])])
        ctl.history.record(3, 1, zero, -v, own)
        ledger.record(2, [trace([])])
        self.assertEqual(ctl.decide(3, 2, own, v)[1]["reason"], "zero_d")
        ctl.history.record(3, 2, v, -v, own)
        ledger.record(3, [trace([])])
        self.assertEqual(ctl.decide(3, 3, own, zero)[1]["reason"], "no_g")
        self.assertEqual(ctl.decide(3, 3, own, torch.full_like(v, math.nan))[1]["reason"],
                         "no_obs")
        self.assertEqual(ctl.decide(3, 3, own, torch.full_like(v, math.inf))[1]["reason"],
                         "no_obs")
        self.assertEqual(py_state, random.getstate())
        self.assertEqual(np_state[1].tolist(), np.random.get_state()[1].tolist())
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        if cuda_state is not None:
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(cuda_state, torch.cuda.get_rng_state_all())))

    def test_tiny_positive_support_remains_finite_and_bounded(self):
        ledger = ResetLedger(self.model)
        ctl = InteractionController(self.head, ledger, rho=0.05)
        v = torch.zeros(self.head.numel)
        v[0] = 1
        own = ledger.from_trace(trace([]))
        ledger.record(0, [trace([])])
        ctl.history.record(3, 0, torch.zeros_like(v), -2e-20 * v, own)
        ledger.record(1, [trace([])])
        ctl.history.record(3, 1, v, -1e-20 * v, own)
        ledger.record(2, [trace([])])
        delta, info = ctl.decide(3, 2, own, v)
        self.assertEqual(info["reason"], "active")
        self.assertTrue(math.isfinite(info["lambda"]))
        self.assertGreaterEqual(info["lambda"], 0)
        self.assertLessEqual(info["lambda"], 0.05)
        self.assertTrue(bool(torch.isfinite(delta).all()))


if __name__ == "__main__":
    unittest.main()
