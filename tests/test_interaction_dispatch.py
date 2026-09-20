#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import torch
from torch import nn

from Algorithm.InteractionDispatch import (
    InteractionDispatchController,
    add_flat_delta_to_state,
    build_corrected_local_state,
    clone_state_to_cpu,
    flatten_trainable_state,
)
from models.Fed import Aggregation


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)


class VectorModel(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(size))


class InteractionDispatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_no_action_aggregation_equivalence(self):
        model = TinyModel()
        controller = InteractionDispatchController(model)
        global_state = clone_state_to_cpu(model.state_dict())
        returned_states = []
        for shift in (0.25, -0.4):
            returned = clone_state_to_cpu(global_state)
            for name in controller.param_names:
                returned[name].add_(shift)
            returned_states.append(returned)

        corrected_states = [
            build_corrected_local_state(
                global_state,
                clone_state_to_cpu(global_state),
                returned,
                controller.param_names,
            )
            for returned in returned_states
        ]
        expected = Aggregation(returned_states, [2, 3])
        actual = Aggregation(corrected_states, [2, 3])
        for name in controller.param_names:
            self.assertTrue(torch.equal(actual[name], expected[name]))

    def test_server_action_isolation(self):
        model = TinyModel()
        controller = InteractionDispatchController(model)
        global_state = clone_state_to_cpu(model.state_dict())
        delta = torch.linspace(-0.2, 0.2, controller.num_params)
        dispatch_state = add_flat_delta_to_state(
            global_state,
            delta,
            controller.param_names,
            controller.param_shapes,
            controller.param_offsets,
        )
        returned_state = clone_state_to_cpu(dispatch_state)
        corrected = build_corrected_local_state(
            global_state,
            dispatch_state,
            returned_state,
            controller.param_names,
        )

        for name in controller.param_names:
            self.assertTrue(torch.allclose(corrected[name], global_state[name]))
        self.assertTrue(
            torch.equal(
                corrected["bn.num_batches_tracked"],
                returned_state["bn.num_batches_tracked"],
            )
        )

    def test_cold_start_timing_and_history_precision(self):
        controller = InteractionDispatchController(
            VectorModel(), tau=0.0, max_step_ratio=1.0
        )
        global_direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
        x1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        u1 = torch.tensor([-2.0, 0.0], dtype=torch.float32)
        controller.record_interaction(5, x1, u1, round_idx=0)

        delta, info = controller.make_dispatch_delta(5, global_direction)
        self.assertIsNone(delta)
        self.assertEqual(info["reason"], "cold_start")
        self.assertEqual(info["num_visits_before"], 1)

        x2 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        u2 = torch.tensor([-1.0, 0.0], dtype=torch.float32)
        controller.record_interaction(5, x2, u2, round_idx=1)
        state = controller.client_states[5]
        self.assertEqual(state["num_visits"], 2)
        for key in ("last_dispatch", "last_update"):
            self.assertEqual(state[key].device.type, "cpu")
            self.assertEqual(state[key].dtype, torch.float32)
        for key in ("obs_direction", "obs_response"):
            self.assertEqual(state[key].device.type, "cpu")
            self.assertEqual(state[key].dtype, torch.float16)

        delta, info = controller.make_dispatch_delta(5, global_direction)
        self.assertEqual(info["reason"], "active")
        self.assertGreater(info["alpha"], 0.0)
        self.assertTrue(torch.allclose(delta, torch.tensor([1.0, 0.0])))
        self.assertGreater(float(torch.dot(delta, state["obs_direction"].float())), 0.0)
        self.assertLessEqual(
            float(torch.linalg.vector_norm(delta)),
            controller.max_step_ratio * state["obs_input_norm"] + 1e-7,
        )

    def test_support_gate(self):
        controller = InteractionDispatchController(
            VectorModel(), tau=0.0, max_step_ratio=1.0
        )
        controller.record_interaction(
            9,
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([-2.0, 0.0], dtype=torch.float32),
            round_idx=0,
        )
        controller.record_interaction(
            9,
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            torch.tensor([-1.0, 0.0], dtype=torch.float32),
            round_idx=1,
        )
        global_direction = torch.tensor([1.0, 0.0], dtype=torch.float32)

        delta, info = controller.make_dispatch_delta(9, global_direction)
        self.assertIsNotNone(delta)
        self.assertEqual(info["reason"], "active")
        self.assertGreater(info["support"], 0.0)

        controller.client_states[9]["obs_response"] = torch.tensor(
            [-1.0, 0.0], dtype=torch.float16
        )
        delta, info = controller.make_dispatch_delta(9, global_direction)
        self.assertIsNone(delta)
        self.assertEqual(info["reason"], "no_support")
        self.assertLess(info["support"], 0.0)
        self.assertEqual(info["alpha"], 0.0)

    def test_direction_is_renormalized_after_float16_storage(self):
        size = 257
        controller = InteractionDispatchController(
            VectorModel(size), tau=0.0, max_step_ratio=0.25
        )
        direction = torch.linspace(0.001, 1.0, size, dtype=torch.float32)
        direction = direction / torch.linalg.vector_norm(direction)
        stored_direction = direction.to(dtype=torch.float16)
        restored_norm = float(
            torch.linalg.vector_norm(stored_direction.float()).item()
        )
        self.assertGreater(abs(restored_norm - 1.0), 1e-7)

        controller.client_states[3] = {
            "num_visits": 2,
            "last_dispatch": torch.zeros(size, dtype=torch.float32),
            "last_update": -direction.clone(),
            "obs_direction": stored_direction.clone(),
            "obs_response": direction.to(dtype=torch.float16),
            "obs_input_norm": 2.0,
            "last_round": 1,
        }
        delta, info = controller.make_dispatch_delta(3, direction)
        self.assertEqual(info["reason"], "active")
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        tolerance = 1e-6
        self.assertLessEqual(abs(delta_norm - info["alpha"]), tolerance)
        self.assertLessEqual(
            delta_norm,
            controller.max_step_ratio * info["obs_input_norm"] + tolerance,
        )

    def test_flatten_uses_parameters_only_and_state_helpers_do_not_alias(self):
        model = TinyModel()
        controller = InteractionDispatchController(model)
        state = clone_state_to_cpu(model.state_dict())
        flat = flatten_trainable_state(state, controller.param_names)
        self.assertEqual(flat.dtype, torch.float32)
        self.assertEqual(flat.device.type, "cpu")
        self.assertEqual(flat.numel(), controller.num_params)
        self.assertNotIn("bn.running_mean", controller.param_names)
        self.assertNotIn("bn.num_batches_tracked", controller.param_names)

        cloned = clone_state_to_cpu(state)
        cloned[controller.param_names[0]].add_(1.0)
        self.assertFalse(
            torch.equal(cloned[controller.param_names[0]], state[controller.param_names[0]])
        )


if __name__ == "__main__":
    unittest.main()
