#!/usr/bin/env python

import copy
import random
import unittest

import numpy as np
import torch
from torch import nn

from Algorithm.FedPhoenixHistoryObserver import (
    FedPhoenixHistoryObserver,
    capture_global_rng_state,
    global_rng_state_equal,
    topk_comparison,
)


class TinyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.fc = nn.Linear(2, 2)


def clone_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


class FedPhoenixHistoryObserverTests(unittest.TestCase):
    def setUp(self):
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        self.model = TinyConvModel()
        self.observer = FedPhoenixHistoryObserver(self.model, reset_ratio=0.5)

    def trace(self, reset=None, include_conv2=True, seed=1):
        layers = [
            {
                "name": "conv1",
                "num_kernels": 4,
                "reset_indices": sorted(reset or []),
            }
        ]
        if include_conv2:
            layers.append(
                {"name": "conv2", "num_kernels": 2, "reset_indices": []}
            )
        return {"seed": seed, "layers": layers}

    def interaction(self, client_id, values, weight=1, reset=None, vectors=None):
        dispatch = clone_state(self.model)
        returned = {name: tensor.clone() for name, tensor in dispatch.items()}
        update = torch.zeros_like(returned["conv1.weight"])
        if vectors is None:
            for index, value in enumerate(values):
                update[index].fill_(float(value))
        else:
            for index, vector in enumerate(vectors):
                update[index, :, 0, 0] = torch.tensor(vector, dtype=torch.float32)
        returned["conv1.weight"] += update
        return self.observer.build_interaction(
            client_id, weight, dispatch, returned, self.trace(reset), task_id=client_id
        )

    def test_update_is_returned_minus_actual_dispatch(self):
        item = self.interaction(0, [1, 2, 3, 4])
        actual = item["updates"]["conv1"][:, 0, 0, 0]
        self.assertTrue(torch.allclose(actual, torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_geometry_reads_only_conv2d_weight(self):
        names = [item["parameter_name"] for item in self.observer.geometry.layers]
        self.assertEqual(names, ["conv1.weight", "conv2.weight"])
        self.assertNotIn("conv1.bias", names)
        self.assertNotIn("bn.weight", names)
        self.assertNotIn("fc.weight", names)

    def test_task_trace_builds_exact_reset_mask(self):
        masks, eligible = self.observer.geometry.reset_masks_from_trace(
            self.trace([1, 3], include_conv2=False)
        )
        self.assertEqual(masks["conv1"].tolist(), [False, True, False, True])
        self.assertFalse(eligible["conv2"])
        self.assertFalse(bool(masks["conv2"].any()))

    def test_own_reset_kernel_score_is_invalid(self):
        items = [
            self.interaction(0, [1, 2, 3, 4], reset=[1]),
            self.interaction(1, [2, 3, 4, 5]),
        ]
        self.observer.process_round(items, 1)
        layer = self.observer.history[0]["conv1"]
        for valid in layer["validity"].values():
            self.assertFalse(bool(valid[1]))

    def test_reset_peer_is_excluded_from_consensus(self):
        items = [
            self.interaction(0, [1, 1, 1, 1]),
            self.interaction(1, [100, 2, 2, 2], weight=9, reset=[0]),
            self.interaction(2, [5, 3, 3, 3], weight=2),
        ]
        self.observer.process_round(items, 1)
        residual = self.observer.history[0]["conv1"]["scores"]["residual"]
        # Each scalar fills two input coefficients: ||[1,1]-[5,5]||.
        self.assertAlmostEqual(float(residual[0]), np.sqrt(32.0), places=5)

    def test_leave_one_out_excludes_client_itself(self):
        items = [
            self.interaction(0, [1, 1, 1, 1], weight=100),
            self.interaction(1, [3, 3, 3, 3], weight=1),
        ]
        self.observer.process_round(items, 1)
        residual = self.observer.history[0]["conv1"]["scores"]["residual"]
        self.assertAlmostEqual(float(residual[0]), np.sqrt(8.0), places=5)

    def test_data_size_weighted_consensus(self):
        items = [
            self.interaction(0, [1, 1, 1, 1], weight=1),
            self.interaction(1, [3, 3, 3, 3], weight=1),
            self.interaction(2, [7, 7, 7, 7], weight=3),
        ]
        self.observer.process_round(items, 1)
        # Peer consensus for client 0 is (1*3 + 3*7)/4 = 6.
        residual = self.observer.history[0]["conv1"]["scores"]["residual"]
        self.assertAlmostEqual(float(residual[0]), np.sqrt(50.0), places=5)

    def test_no_clean_peer_is_invalid(self):
        items = [
            self.interaction(0, [1, 1, 1, 1]),
            self.interaction(1, [2, 2, 2, 2], reset=[0]),
        ]
        self.observer.process_round(items, 1)
        self.assertFalse(
            bool(self.observer.history[0]["conv1"]["validity"]["residual"][0])
        )

    def test_normalized_residual_formula(self):
        items = [
            self.interaction(0, [1, 1, 1, 1]),
            self.interaction(1, [3, 3, 3, 3]),
        ]
        self.observer.process_round(items, 1)
        score = self.observer.history[0]["conv1"]["scores"]["norm_residual"]
        self.assertAlmostEqual(float(score[0]), 0.5, places=6)

    def test_angular_score_formula(self):
        first = self.interaction(
            0, [], vectors=[[1, 0], [1, 0], [1, 0], [1, 0]]
        )
        second = self.interaction(
            1, [], vectors=[[0, 1], [0, 1], [0, 1], [0, 1]]
        )
        self.observer.process_round([first, second], 1)
        angular = self.observer.history[0]["conv1"]["scores"]["angular"]
        self.assertAlmostEqual(float(angular[0]), 1.0, places=6)

    def test_history_comparison_precedes_update(self):
        self.observer.process_round(
            [self.interaction(0, [1, 2, 3, 4]), self.interaction(1, [4, 3, 2, 1])],
            1,
        )
        events = self.observer.process_round(
            [self.interaction(0, [2, 3, 4, 5]), self.interaction(1, [5, 4, 3, 2])],
            2,
        )
        self.assertTrue(events)
        self.assertTrue(all(event["previous_round"] == 1 for event in events))
        self.assertEqual(self.observer.history[0]["conv1"]["round"], 2)

    def test_current_round_other_client_is_not_cross_history(self):
        self.observer.process_round(
            [
                self.interaction(0, [1, 2, 3, 4]),
                self.interaction(1, [4, 3, 2, 1]),
                self.interaction(2, [2, 4, 1, 3]),
            ],
            1,
        )
        events = self.observer.process_round(
            [self.interaction(0, [2, 3, 4, 5]), self.interaction(1, [5, 4, 3, 2])],
            2,
        )
        for event in events:
            rounds = json_load(event["cross_history_rounds"])
            self.assertTrue(all(round_number < 2 for round_number in rounds))

    def test_topk_overlap(self):
        result = topk_comparison(
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            torch.tensor([4.0, 1.0, 3.0, 2.0]),
            torch.ones(4, dtype=torch.bool),
            2,
        )
        self.assertEqual(result["previous_topk"], [0, 1])
        self.assertEqual(result["current_topk"], [0, 2])
        self.assertEqual(result["intersection_count"], 1)
        self.assertEqual(result["overlap_rate"], 0.5)

    def test_random_expectation_is_k_over_n(self):
        result = topk_comparison(
            torch.arange(5, dtype=torch.float32),
            torch.arange(5, dtype=torch.float32),
            torch.ones(5, dtype=torch.bool),
            2,
        )
        self.assertAlmostEqual(result["random_overlap_rate"], 0.4)

    def test_cross_client_history_is_strictly_older(self):
        self.observer.process_round(
            [self.interaction(0, [1, 2, 3, 4]), self.interaction(1, [4, 3, 2, 1])],
            3,
        )
        events = self.observer.process_round(
            [self.interaction(0, [2, 3, 4, 5]), self.interaction(2, [5, 4, 3, 2])],
            7,
        )
        self.assertTrue(events)
        for event in events:
            self.assertTrue(all(value < 7 for value in json_load(event["cross_history_rounds"])))

    def test_observer_does_not_change_global_rng(self):
        before = capture_global_rng_state(include_cuda=False)
        interactions = [
            self.interaction(0, [1, 2, 3, 4]),
            self.interaction(1, [4, 3, 2, 1]),
        ]
        self.observer.process_round(interactions, 1)
        after = capture_global_rng_state(include_cuda=False)
        self.assertTrue(global_rng_state_equal(before, after))

    def test_persistent_history_is_cpu(self):
        self.observer.process_round(
            [self.interaction(0, [1, 2, 3, 4]), self.interaction(1, [4, 3, 2, 1])],
            1,
        )
        self.assertTrue(
            all(item["device"] == "cpu" for item in self.observer.history_tensor_inventory())
        )

    def test_history_contains_only_scalar_kernel_arrays(self):
        self.observer.process_round(
            [self.interaction(0, [1, 2, 3, 4]), self.interaction(1, [4, 3, 2, 1])],
            1,
        )
        inventory = self.observer.history_tensor_inventory()
        self.assertTrue(inventory)
        for item in inventory:
            self.assertEqual(len(item["shape"]), 1)
            self.assertEqual(item["numel"], item["output_kernels"])


def json_load(value):
    import json

    return json.loads(value)


if __name__ == "__main__":
    unittest.main()
