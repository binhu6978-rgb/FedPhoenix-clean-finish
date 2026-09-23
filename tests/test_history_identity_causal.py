import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from Algorithm.HistoryIdentityCausal import (
    FrozenHistoryController,
    FrozenScoreBankReader,
    FrozenScoreBankWriter,
    build_matching_schedule,
)
from Algorithm.TargetedFedPhoenix import TargetedFedPhoenixController
from Algorithm.Phoenix_util import reset_kernels_for_task


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1, bias=False)


def score_entry(round_number, values, validity=None):
    return {
        "last_round": round_number,
        "layers": {"conv": {
            "score": torch.tensor(values, dtype=torch.float32),
            "validity": torch.tensor(
                validity if validity is not None else [1, 1, 1, 1],
                dtype=torch.bool,
            ),
        }},
    }


class HistoryIdentityTests(unittest.TestCase):
    def test_exact_age_derangement_excludes_self(self):
        rows = []
        for number, selected in enumerate(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1):
            import json
            rows.append({"round": number, "selected_clients": json.dumps(selected),
                         "task_seeds": json.dumps(list(range(10)))})
        schedule = build_matching_schedule(rows, start_round=2)
        second = schedule[1]
        donors = [second["matches"][str(client)]["donor_id"]
                  for client in second["selected_clients"]]
        self.assertEqual(set(donors), set(second["selected_clients"]))
        self.assertTrue(all(client != donor for client, donor
                            in zip(second["selected_clients"], donors)))
        self.assertTrue(all(second["matches"][str(client)]["age"] == 1
                            for client in second["selected_clients"]))

    def test_all_sources_share_exact_support_and_reset_budget(self):
        model = TinyModel()
        history = {
            1: score_entry(4, [10, 0, 1, 2]),
            2: score_entry(4, [0, 10, 1, 2]),
            3: score_entry(4, [0, 8, 1, 2], [1, 1, 1, 0]),
        }
        frozen = {"round": 5, "selected_clients": [1], "matches": {
            "1": {"age": 1, "donor_id": 2, "peers": [2, 3], "eligible": True}
        }}
        baseline = TinyModel()
        trace = reset_kernels_for_task(baseline, reset_ratio=0.5, seed=19,
                                       layer_scope="all", init_method="ori_normal")
        final_masks = []
        target_indexes = []
        supports = []
        source_values = []
        for mode in ("H_same", "S_shuffled", "P_population"):
            controller = FrozenHistoryController(
                model, mode=mode, reset_ratio=0.5, target_ratio=0.5,
                max_history_gap=10, score_type="update_norm",
                target_layers="conv", start_round=5,
            )
            controller.prepare_round(frozen, history)
            entry = controller.source_entries[1]["layers"]["conv"]
            supports.append(entry["validity"].tolist())
            source_values.append(entry["score"].tolist())
            _, retargeted = controller.retarget_task(
                global_model=model,
                baseline_task_model=self._baseline_copy(model, trace),
                baseline_trace=trace, client_id=1, current_round=5,
            )
            final_masks.append(retargeted["layers"][0]["final_reset_indices"])
            target_indexes.append(retargeted["layers"][0]["targeted_indices"])
            self.assertEqual(retargeted["targeted_reset_slots"], 1)
            self.assertEqual(retargeted["num_reset_kernels"], 2)
        self.assertEqual(supports, [[True, True, True, False]] * 3)
        self.assertNotEqual(source_values[0], source_values[1])
        self.assertEqual(target_indexes, [[0], [1], [1]])
        self.assertEqual(len(final_masks), 3)

    @staticmethod
    def _baseline_copy(model, trace):
        import copy
        baseline = copy.deepcopy(model)
        # The trace's reset tensor values are immaterial to support/budget tests;
        # make a fresh reproducible task for each controller.
        reset_kernels_for_task(baseline, reset_ratio=0.5, seed=trace["seed"],
                               layer_scope="all", init_method="ori_normal")
        return baseline

    def test_score_bank_roundtrip(self):
        model = TinyModel()
        controller = TargetedFedPhoenixController(model)
        controller.history = {1: score_entry(1, [2, 3, 5, 7], [1, 0, 1, 1])}
        with tempfile.TemporaryDirectory() as temporary:
            writer = FrozenScoreBankWriter(Path(temporary), controller.geometry)
            writer.write_round(1, [1], controller.history)
            reader = FrozenScoreBankReader(Path(temporary), wait_seconds=1)
            restored = reader.through(1)[1]
            self.assertEqual(restored["last_round"], 1)
            self.assertEqual(restored["layers"]["conv"]["score"].tolist(),
                             [2.0, 3.0, 5.0, 7.0])
            self.assertEqual(restored["layers"]["conv"]["validity"].tolist(),
                             [True, False, True, True])


if __name__ == "__main__":
    unittest.main()
