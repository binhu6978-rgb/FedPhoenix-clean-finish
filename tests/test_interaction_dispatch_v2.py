#!/usr/bin/env python

import copy
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from Algorithm.InteractionDispatchV2 import (
    InteractionDispatchV2Controller,
    ParameterGeometry,
    clone_state_to_cpu,
)
from models.Aggregation_InteractionDispatch import (
    aggregate_buffers_legacy_compatible,
    aggregate_client_responses,
)
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg
from models.Update_InteractionDispatch import LocalUpdate_InteractionDispatch


class VectorModel(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(size))

    def forward(self, value):
        return {"output": value @ self.vector.reshape(-1, 1)}


class TinyBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.classifier = nn.Linear(4, 2)

    def forward(self, value):
        return {"output": self.classifier(self.features(value))}


class TinyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(31)
        self.values = torch.randn(12, 4, generator=generator)
        self.targets = [int(i % 2) for i in range(12)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.values[index], self.targets[index]


class InteractionDispatchV2Tests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

    @staticmethod
    def _record_two_valid(controller):
        controller.record_interaction(
            3,
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([3.0, 0.0], dtype=torch.float32),
            0,
        )
        controller.record_interaction(
            3,
            torch.tensor([2.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            1,
        )

    def test_first_visit_has_no_observation(self):
        controller = InteractionDispatchV2Controller(VectorModel())
        controller.record_interaction(1, torch.zeros(2), torch.ones(2), 0)
        state = controller.client_states[1]
        self.assertEqual(state["num_visits"], 1)
        self.assertIsNone(state["obs_direction"])
        self.assertIsNone(state["obs_response_direction"])

    def test_second_visit_creates_first_valid_observation(self):
        controller = InteractionDispatchV2Controller(VectorModel())
        self._record_two_valid(controller)
        state = controller.client_states[3]
        self.assertEqual(state["num_visits"], 2)
        self.assertIsNotNone(state["obs_direction"])
        self.assertIsNotNone(state["obs_response_direction"])
        self.assertAlmostEqual(state["obs_input_norm"], 2.0)

    def test_third_visit_is_active_only_with_need_and_support(self):
        controller = InteractionDispatchV2Controller(VectorModel(), step_ratio=0.2)
        self._record_two_valid(controller)
        delta, info = controller.make_dispatch_delta(3, torch.tensor([-1.0, 0.0]))
        self.assertEqual(info["reason"], "active")
        self.assertGreater(info["conflict"], 0)
        self.assertGreater(info["support"], 0)
        self.assertIsNotNone(delta)

    def test_conflict_nonpositive_is_no_need(self):
        controller = InteractionDispatchV2Controller(VectorModel())
        self._record_two_valid(controller)
        delta, info = controller.make_dispatch_delta(3, torch.tensor([1.0, 0.0]))
        self.assertIsNone(delta)
        self.assertEqual(info["reason"], "no_need")

    def test_support_nonpositive_is_no_support(self):
        controller = InteractionDispatchV2Controller(VectorModel())
        self._record_two_valid(controller)
        controller.client_states[3]["obs_response_direction"] = torch.tensor(
            [1.0, 0.0], dtype=torch.float16
        )
        delta, info = controller.make_dispatch_delta(3, torch.tensor([-1.0, 0.0]))
        self.assertIsNone(delta)
        self.assertEqual(info["reason"], "no_support")

    def test_active_alpha_is_ratio_times_observed_input_norm(self):
        controller = InteractionDispatchV2Controller(VectorModel(), step_ratio=0.4)
        self._record_two_valid(controller)
        _, info = controller.make_dispatch_delta(3, torch.tensor([-1.0, 0.0]))
        self.assertAlmostEqual(info["alpha"], 0.8, places=6)
        self.assertAlmostEqual(info["effective_ratio"], 0.4, places=6)

    def test_delta_norm_matches_alpha_after_half_restore(self):
        size = 257
        controller = InteractionDispatchV2Controller(VectorModel(size), step_ratio=0.1)
        direction = torch.linspace(0.001, 1.0, size)
        direction /= torch.linalg.vector_norm(direction)
        response = -direction
        controller.client_states[4] = {
            "num_visits": 2,
            "last_dispatch": torch.zeros(size),
            "last_update": direction.clone(),
            "obs_direction": direction.half(),
            "obs_response_direction": response.half(),
            "obs_input_norm": 3.0,
            "last_round": 1,
        }
        delta, info = controller.make_dispatch_delta(4, response)
        self.assertEqual(info["reason"], "active")
        self.assertAlmostEqual(float(torch.linalg.vector_norm(delta)), 0.3, places=5)

    def test_server_action_isolation_when_returned_equals_dispatch(self):
        model = TinyBNModel()
        geometry = ParameterGeometry(model)
        global_state = clone_state_to_cpu(model.state_dict())
        delta = torch.linspace(-0.2, 0.2, geometry.num_params)
        dispatch = geometry.add_delta_to_state(global_state, delta)
        result = aggregate_client_responses(
            global_state,
            [{"dispatch_state": dispatch, "returned_state": clone_state_to_cpu(dispatch), "weight": 5}],
            geometry.param_names,
        )
        for name in geometry.param_names:
            self.assertTrue(torch.equal(result[name], global_state[name]))

    def test_no_action_full_state_matches_legacy_aggregation(self):
        model = TinyBNModel()
        geometry = ParameterGeometry(model)
        global_state = clone_state_to_cpu(model.state_dict())
        returned = []
        for shift, tracked in ((0.25, 3), (-0.4, 8)):
            state = clone_state_to_cpu(global_state)
            for name in geometry.param_names:
                state[name].add_(shift)
            state["features.1.running_mean"].add_(shift)
            state["features.1.num_batches_tracked"].fill_(tracked)
            returned.append(state)
        expected = Aggregation(returned, [2, 3])
        records = [
            {"dispatch_state": clone_state_to_cpu(global_state), "returned_state": state, "weight": weight}
            for state, weight in zip(returned, [2, 3])
        ]
        actual = aggregate_client_responses(global_state, records, geometry.param_names)
        expected_model = copy.deepcopy(model)
        actual_model = copy.deepcopy(model)
        expected_model.load_state_dict(expected)
        actual_model.load_state_dict(actual)
        for name, tensor in expected_model.state_dict().items():
            self.assertTrue(torch.equal(tensor, actual_model.state_dict()[name]), name)

    def test_buffer_helper_matches_legacy_values(self):
        model = TinyBNModel()
        geometry = ParameterGeometry(model)
        states = [clone_state_to_cpu(model.state_dict()) for _ in range(2)]
        states[0]["features.1.running_var"].fill_(2.0)
        states[1]["features.1.running_var"].fill_(5.0)
        expected = Aggregation(states, [1, 4])
        buffer_names = [name for name in states[0] if name not in set(geometry.param_names)]
        actual = aggregate_buffers_legacy_compatible(states, [1, 4], buffer_names)
        for name in buffer_names:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_local_trainer_matches_fedavg_under_restored_rng(self):
        args = SimpleNamespace(
            local_bs=4,
            num_workers=0,
            optimizer="sgd",
            lr=0.01,
            momentum=0.5,
            weight_decay=0.0,
            local_ep=2,
            device=torch.device("cpu"),
        )
        dataset = TinyDataset()
        initial = TinyBNModel()
        old_model = copy.deepcopy(initial)
        new_model = copy.deepcopy(initial)
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        old = LocalUpdate_FedAvg(args, None, dataset, range(len(dataset)))
        old_state = clone_state_to_cpu(old.train(old_model))
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        new = LocalUpdate_InteractionDispatch(args, None, dataset, range(len(dataset)))
        new_state = new.train_from_dispatch(new_model)["returned_state"]
        for name in old_state:
            self.assertTrue(torch.equal(old_state[name], new_state[name]), name)

    def test_persistent_state_tensors_are_cpu_and_not_aliased(self):
        controller = InteractionDispatchV2Controller(VectorModel())
        self._record_two_valid(controller)
        state = controller.client_states[3]
        for key in ("last_dispatch", "last_update", "obs_direction", "obs_response_direction"):
            self.assertEqual(state[key].device.type, "cpu")
        external = torch.tensor([9.0, 9.0])
        controller.record_interaction(5, external, external, 2)
        external.zero_()
        self.assertFalse(torch.equal(controller.client_states[5]["last_dispatch"], external))

    def test_geometry_ignores_batchnorm_buffers_and_splits_parameters(self):
        model = TinyBNModel()
        geometry = ParameterGeometry(model)
        self.assertNotIn("features.1.running_mean", geometry.param_names)
        self.assertNotIn("features.1.running_var", geometry.param_names)
        self.assertNotIn("features.1.num_batches_tracked", geometry.param_names)
        split = geometry.split_vector_by_parameter(geometry.flatten_state(model))
        self.assertEqual(list(split), geometry.param_names)
        self.assertEqual(set(geometry.group_names), {"features", "classifier"})


if __name__ == "__main__":
    unittest.main()
