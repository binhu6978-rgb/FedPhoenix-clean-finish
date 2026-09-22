#!/usr/bin/env python
"""Synthetic active-retarget smoke for corrected TargetedFedPhoenix."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Algorithm.FedPhoenixHistoryObserver import (
    capture_global_rng_state,
    global_rng_state_equal,
)
from Algorithm.TargetedFedPhoenix import TargetedFedPhoenixController


class ActiveSmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(1, 11, dtype=torch.float32).reshape(10, 1, 1, 1)
            )


def fixture(task_seed):
    global_model = ActiveSmokeModel()
    baseline = copy.deepcopy(global_model)
    reset_indices = list(range(1, 9))
    with torch.no_grad():
        for offset, index in enumerate(reset_indices, start=1):
            baseline.conv.weight[index].fill_(-100.0 - offset)
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
    return global_model, baseline, trace


def run_once(task_seed):
    global_model, baseline, trace = fixture(task_seed)
    controller = TargetedFedPhoenixController(global_model, 0.25, 20)
    controller.history[7] = {
        "last_round": 1,
        "layers": {
            "conv": {
                "score": torch.tensor(
                    [100, 1, 2, 3, 4, 5, 6, 7, 8, 99],
                    dtype=torch.float32,
                ),
                "validity": torch.ones(10, dtype=torch.bool),
            }
        },
    }
    rng_before = capture_global_rng_state()
    final_model, final_trace = controller.retarget_task(
        global_model, baseline, trace, client_id=7, current_round=2
    )
    rng_after = capture_global_rng_state()
    return {
        "trace": final_trace,
        "weight": final_model.conv.weight.detach().clone(),
        "rng_unchanged": global_rng_state_equal(rng_before, rng_after),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "results/targeted_fedphoenix_v1_corrected/"
            "active_smoke_report.json"
        ),
    )
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    first = run_once(41)
    second = run_once(41)
    layer = first["trace"]["layers"][0]
    smallest_possible = sorted(layer["random_candidates"])[
        : len(layer["random_kept_indices"])
    ]
    seed_kept = []
    for task_seed in range(1, 13):
        item = run_once(task_seed)
        seed_kept.append(item["trace"]["layers"][0]["random_kept_indices"])

    checks = {
        "targeted_slots_positive": first["trace"]["targeted_reset_slots"] > 0,
        "private_priority_used": (
            layer["retention_policy"] == "sha256_private_priority"
            and layer["retention_priority_namespace"] == "random-retention"
        ),
        "single_seed_not_prefix": layer["random_kept_indices"] != smallest_possible,
        "multiple_seeds_not_always_prefix": any(
            kept != [1, 2, 3, 4, 5, 6] for kept in seed_kept
        ),
        "multiple_seed_outcomes": len({tuple(kept) for kept in seed_kept}) > 1,
        "repeat_trace_exact": first["trace"] == second["trace"],
        "repeat_tensor_exact": torch.equal(first["weight"], second["weight"]),
        "global_rng_unchanged": first["rng_unchanged"] and second["rng_unchanged"],
    }
    report = {
        "task_seed": 41,
        "target_ratio": 0.25,
        "targeted_indices": layer["targeted_indices"],
        "random_candidates": layer["random_candidates"],
        "random_kept_indices": layer["random_kept_indices"],
        "smallest_possible_indices": smallest_possible,
        "displaced_donor_indices": layer["displaced_donor_indices"],
        "donor_mapping": layer["donor_mapping"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("corrected active-retarget smoke failed")


if __name__ == "__main__":
    main()
