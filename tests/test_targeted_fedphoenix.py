import copy
import random
import unittest

import numpy as np
import torch
from torch import nn

from Algorithm.FedPhoenixHistoryObserver import (
    capture_global_rng_state,
    global_rng_state_equal,
)
from Algorithm.TargetedFedPhoenix import (
    TargetedFedPhoenixController,
    deterministic_reset_priority,
)
from Algorithm.Training_TargetedFedPhoenix import (
    Aggregation as TargetedAggregation,
    LocalUpdate_FedAvg as TargetedLocalUpdate,
)
from models.Fed import Aggregation
from models.Update import LocalUpdate_FedAvg


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 1, bias=False)
        self.conv2 = nn.Conv2d(4, 4, 1, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1))
            self.conv2.weight.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(4, 4, 1, 1))


class WideTinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(1, 11, dtype=torch.float32).reshape(10, 1, 1, 1)
            )


def baseline_fixture(active_layer="conv1"):
    global_model = TinyConvNet()
    task = copy.deepcopy(global_model)
    layer = getattr(task, active_layer)
    with torch.no_grad():
        layer.weight[0].fill_(-10.0)
        layer.weight[2].fill_(-20.0)
    trace = {
        "seed": 7,
        "layers": [
            {
                "name": active_layer,
                "num_kernels": 4,
                "reset_indices": [0, 2],
                "actual_reset_ratio": 0.5,
            }
        ],
        "num_reset_layers": 1,
        "num_reset_kernels": 2,
    }
    return global_model, task, trace


def state_equal(left, right):
    return all(torch.equal(left[key], right[key]) for key in left)


def add_history(controller, scores, validity=None, round_number=1, client_id=3, layer="conv1"):
    validity = validity if validity is not None else [True] * len(scores)
    controller.history[client_id] = {
        "last_round": round_number,
        "layers": {
            layer: {
                "score": torch.tensor(scores, dtype=torch.float32),
                "validity": torch.tensor(validity, dtype=torch.bool),
            }
        },
    }


def wide_fixture(task_seed):
    global_model = WideTinyConvNet()
    task = copy.deepcopy(global_model)
    reset_indices = list(range(1, 9))
    with torch.no_grad():
        for offset, index in enumerate(reset_indices, start=1):
            task.conv.weight[index].fill_(-100.0 - offset)
    trace = {
        "seed": int(task_seed),
        "layers": [
            {
                "name": "conv",
                "num_kernels": 10,
                "reset_indices": reset_indices,
                "actual_reset_ratio": 0.8,
            }
        ],
        "num_reset_layers": 1,
        "num_reset_kernels": 8,
    }
    return global_model, task, trace


class TargetedFedPhoenixTests(unittest.TestCase):
    def test_01_target_count_uses_floor(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 4])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertEqual(final["layers"][0]["target_requested"], 1)

    def test_02_target_comes_only_from_valid_history(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 100, 3, 4], [True, False, True, True])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertEqual(final["layers"][0]["targeted_indices"], [3])

    def test_03_highest_score_has_priority(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 9, 3, 4])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertEqual(final["layers"][0]["targeted_indices"], [1])

    def test_04_tie_breaks_by_kernel_index(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 9, 9, 4])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertEqual(final["layers"][0]["targeted_indices"], [1])

    def test_05_cold_start_is_exact_baseline(self):
        global_model, task, trace = baseline_fixture()
        before = copy.deepcopy(task.state_dict())
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        controller.retarget_task(global_model, task, trace, 3, 1)
        self.assertTrue(state_equal(before, task.state_dict()))

    def test_06_stale_history_is_exact_baseline(self):
        global_model, task, trace = baseline_fixture()
        before = copy.deepcopy(task.state_dict())
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 100], round_number=1)
        controller.retarget_task(global_model, task, trace, 3, 22)
        self.assertTrue(state_equal(before, task.state_dict()))

    def test_07_zero_ratio_is_exact_baseline(self):
        global_model, task, trace = baseline_fixture()
        before = copy.deepcopy(task.state_dict())
        controller = TargetedFedPhoenixController(global_model, 0.0, 20)
        add_history(controller, [1, 2, 3, 100])
        controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertTrue(state_equal(before, task.state_dict()))

    def test_08_zero_floor_slot_is_exact_layer(self):
        global_model, task, trace = baseline_fixture()
        before = task.conv1.weight.detach().clone()
        controller = TargetedFedPhoenixController(global_model, 0.25, 20)
        add_history(controller, [1, 2, 3, 100])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertTrue(torch.equal(before, task.conv1.weight))
        self.assertEqual(final["layers"][0]["fallback_reason"], "no_target_slots")

    def _retarget_to_three(self):
        global_model, task, trace = baseline_fixture()
        baseline = copy.deepcopy(task)
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 100])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        return global_model, baseline, task, final

    def test_09_reset_total_count_unchanged(self):
        _, _, _, final = self._retarget_to_three()
        layer = final["layers"][0]
        self.assertEqual(len(layer["final_reset_indices"]), len(layer["baseline_reset_indices"]))

    def test_10_final_reset_has_no_duplicates(self):
        _, _, _, final = self._retarget_to_three()
        indices = final["layers"][0]["final_reset_indices"]
        self.assertEqual(len(indices), len(set(indices)))

    def test_11_new_target_replaces_baseline_donor(self):
        _, _, _, final = self._retarget_to_three()
        layer = final["layers"][0]
        self.assertEqual(layer["final_reset_indices"], [0, 3])
        self.assertEqual(layer["donor_mapping"], [{"target_index": 3, "donor_index": 2}])

    def test_12_donor_restores_global_kernel_exact(self):
        global_model, _, task, _ = self._retarget_to_three()
        self.assertTrue(torch.equal(task.conv1.weight[2], global_model.conv1.weight[2]))

    def test_13_target_receives_donor_reset_tensor_exact(self):
        _, baseline, task, _ = self._retarget_to_three()
        self.assertTrue(torch.equal(task.conv1.weight[3], baseline.conv1.weight[2]))

    def test_14_baseline_reset_target_keeps_own_sample(self):
        global_model, task, trace = baseline_fixture()
        original = task.conv1.weight[2].detach().clone()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 100, 4])
        controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertTrue(torch.equal(task.conv1.weight[2], original))

    def test_15_reset_tensor_multiset_is_preserved(self):
        _, baseline, task, final = self._retarget_to_three()
        before = sorted(float(baseline.conv1.weight[index, 0, 0, 0]) for index in [0, 2])
        after = sorted(
            float(task.conv1.weight[index, 0, 0, 0])
            for index in final["layers"][0]["final_reset_indices"]
        )
        self.assertEqual(before, after)

    def test_16_retarget_does_not_change_global_rng(self):
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 100])
        before = capture_global_rng_state(include_cuda=False)
        controller.retarget_task(global_model, task, trace, 3, 2)
        after = capture_global_rng_state(include_cuda=False)
        self.assertTrue(global_rng_state_equal(before, after))

    def test_17_history_score_is_returned_minus_actual_dispatch_norm(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        _, final = controller.retarget_task(global_model, task, trace, 3, 1)
        returned = copy.deepcopy(task.state_dict())
        returned["conv1.weight"][1].add_(3.0)
        controller.observe(3, 1, task.state_dict(), returned, final)
        self.assertAlmostEqual(float(controller.history[3]["layers"]["conv1"]["score"][1]), 3.0)

    def test_18_final_reset_kernels_are_history_invalid(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        _, final = controller.retarget_task(global_model, task, trace, 3, 1)
        controller.observe(3, 1, task.state_dict(), task.state_dict(), final)
        validity = controller.history[3]["layers"]["conv1"]["validity"]
        self.assertFalse(bool(validity[0]))
        self.assertFalse(bool(validity[2]))

    def test_19_unreset_kernel_history_is_valid(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        _, final = controller.retarget_task(global_model, task, trace, 3, 1)
        controller.observe(3, 1, task.state_dict(), task.state_dict(), final)
        self.assertTrue(bool(controller.history[3]["layers"]["conv1"]["validity"][1]))

    def test_20_history_is_cpu_scalar_kernel_arrays(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        _, final = controller.retarget_task(global_model, task, trace, 3, 1)
        controller.observe(3, 1, task.state_dict(), task.state_dict(), final)
        inventory = controller.history_inventory()
        self.assertTrue(inventory["all_cpu"])
        self.assertTrue(inventory["scalar_kernel_arrays_only"])

    def test_21_current_interaction_cannot_change_current_dispatch(self):
        global_model, task, trace = baseline_fixture()
        before = copy.deepcopy(task.state_dict())
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        _, final = controller.retarget_task(global_model, task, trace, 3, 1)
        controller.observe(3, 1, task.state_dict(), task.state_dict(), final)
        self.assertTrue(state_equal(before, task.state_dict()))

    def test_22_aggregation_is_legacy_aggregation(self):
        self.assertIs(TargetedAggregation, Aggregation)

    def test_23_local_trainer_is_legacy_local_update(self):
        self.assertIs(TargetedLocalUpdate, LocalUpdate_FedAvg)

    def test_24_active_layer_schedule_comes_only_from_trace(self):
        global_model, task, trace = baseline_fixture(active_layer="conv2")
        conv1_before = task.conv1.weight.detach().clone()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 100], layer="conv2")
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertEqual([layer["name"] for layer in final["layers"]], ["conv2"])
        self.assertTrue(torch.equal(conv1_before, task.conv1.weight))

    def test_25_no_valid_history_falls_back_exactly(self):
        global_model, task, trace = baseline_fixture()
        before = copy.deepcopy(task.state_dict())
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 4], [False] * 4)
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertTrue(state_equal(before, task.state_dict()))
        self.assertEqual(final["layers"][0]["fallback_reason"], "no_valid_history")

    def test_26_nonfinite_scores_are_not_targets(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, float("nan"), 3, 4])
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        self.assertEqual(final["layers"][0]["targeted_indices"], [3])

    def test_27_gap_equal_to_limit_is_eligible(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        add_history(controller, [1, 2, 3, 100], round_number=1)
        _, final = controller.retarget_task(global_model, task, trace, 3, 21)
        self.assertEqual(final["layers"][0]["fallback_reason"], "targeted")

    def test_28_observe_does_not_change_global_rng(self):
        global_model, task, trace = baseline_fixture()
        controller = TargetedFedPhoenixController(global_model, 0.5, 20)
        _, final = controller.retarget_task(global_model, task, trace, 3, 1)
        before = capture_global_rng_state(include_cuda=False)
        controller.observe(3, 1, task.state_dict(), task.state_dict(), final)
        after = capture_global_rng_state(include_cuda=False)
        self.assertTrue(global_rng_state_equal(before, after))

    def test_29_sorted_reset_indices_do_not_force_prefix_retention(self):
        kept_sets = []
        for task_seed in range(1, 13):
            global_model, task, trace = wide_fixture(task_seed)
            controller = TargetedFedPhoenixController(global_model, 0.25, 20)
            add_history(
                controller,
                [100, 1, 2, 3, 4, 5, 6, 7, 8, 99],
                client_id=3,
                layer="conv",
            )
            _, final = controller.retarget_task(global_model, task, trace, 3, 2)
            layer = final["layers"][0]
            kept_sets.append(layer["random_kept_indices"])
            self.assertEqual(layer["retention_policy"], "sha256_private_priority")
        self.assertTrue(any(kept != [1, 2, 3, 4, 5, 6] for kept in kept_sets))
        self.assertGreater(len({tuple(values) for values in kept_sets}), 1)

    def test_30_corrected_retarget_is_deterministic(self):
        outputs = []
        for _ in range(2):
            global_model, task, trace = wide_fixture(41)
            controller = TargetedFedPhoenixController(global_model, 0.25, 20)
            add_history(
                controller,
                [100, 1, 2, 3, 4, 5, 6, 7, 8, 99],
                client_id=3,
                layer="conv",
            )
            _, final = controller.retarget_task(global_model, task, trace, 3, 2)
            outputs.append(
                (
                    final["layers"][0]["random_kept_indices"],
                    final["layers"][0]["donor_mapping"],
                    task.conv.weight.detach().clone(),
                )
            )
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1], outputs[1][1])
        self.assertTrue(torch.equal(outputs[0][2], outputs[1][2]))

    def test_31_priority_has_fixed_process_independent_value(self):
        self.assertEqual(
            deterministic_reset_priority(
                123, "features.14", 7, "random-retention"
            ),
            92751000886822488204589377115599076655345740921772688260799517893196794101231,
        )

    def test_32_corrected_retarget_preserves_all_global_rng_states(self):
        global_model, task, trace = wide_fixture(91)
        controller = TargetedFedPhoenixController(global_model, 0.25, 20)
        add_history(
            controller,
            [100, 1, 2, 3, 4, 5, 6, 7, 8, 99],
            client_id=3,
            layer="conv",
        )
        before = capture_global_rng_state()
        controller.retarget_task(global_model, task, trace, 3, 2)
        after = capture_global_rng_state()
        self.assertTrue(global_rng_state_equal(before, after))

    def test_33_corrected_trace_records_candidates_and_displaced_donors(self):
        global_model, task, trace = wide_fixture(19)
        controller = TargetedFedPhoenixController(global_model, 0.25, 20)
        add_history(
            controller,
            [100, 1, 2, 3, 4, 5, 6, 7, 8, 99],
            client_id=3,
            layer="conv",
        )
        _, final = controller.retarget_task(global_model, task, trace, 3, 2)
        layer = final["layers"][0]
        self.assertEqual(layer["random_candidates"], list(range(1, 9)))
        self.assertEqual(
            set(layer["displaced_donor_indices"]),
            set(layer["baseline_reset_indices"]) - set(layer["final_reset_indices"]),
        )
        self.assertEqual(layer["donor_pairing_policy"], "sha256_private_priority")


if __name__ == "__main__":
    unittest.main()
