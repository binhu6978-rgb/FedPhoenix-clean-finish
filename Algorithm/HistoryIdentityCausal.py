"""Frozen-bank history-source controls for the FedPhoenix identity ablation.

All three targeted arms read scores and validity from the same *baseline*
trajectory.  No arm observes its own returned weights for future selection.
This makes the score-source assignment the only controlled intervention among
H, S and P, while the training trajectories themselves may naturally diverge.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from Algorithm.TargetedFedPhoenix import TargetedFedPhoenixController


MODES = {"H_same", "S_shuffled", "P_population"}


def deterministic_shift(seed, round_number, age, cohort_size):
    if cohort_size < 2:
        return None
    payload = f"history-identity|{seed}|{round_number}|{age}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (cohort_size - 1)


def build_matching_schedule(round_rows, seed=1, max_gap=10, start_round=151):
    """Freeze exact-age derangements from an existing RNG-parity trajectory."""
    last_seen = {}
    schedule = []
    for expected_round, row in enumerate(round_rows, 1):
        round_number = int(row["round"])
        if round_number != expected_round:
            raise ValueError("source selected-client trajectory is not sequential")
        selected = [int(value) for value in json.loads(row["selected_clients"])]
        task_seeds = [int(value) for value in json.loads(row["task_seeds"])]
        if len(selected) != 10 or len(set(selected)) != 10 or len(task_seeds) != 10:
            raise ValueError("expected ten unique selected clients and task seeds")
        by_age = {}
        for client, previous in last_seen.items():
            age = round_number - previous
            if 1 <= age <= max_gap:
                by_age.setdefault(age, []).append(client)
        for age in by_age:
            by_age[age].sort()
        matches = {}
        for client in selected:
            prior = last_seen.get(client)
            age = round_number - prior if prior is not None else None
            peers = []
            donor = None
            if age is not None and age <= max_gap:
                cohort = by_age[age]
                peers = [value for value in cohort if value != client]
                if peers:
                    shift = deterministic_shift(seed, round_number, age, len(cohort))
                    donor = cohort[(cohort.index(client) + shift) % len(cohort)]
                    assert donor != client and donor in peers
            matches[str(client)] = {
                "age": age,
                "donor_id": donor,
                "peers": peers,
                "eligible": bool(round_number >= start_round and donor is not None),
            }
        schedule.append({
            "round": round_number,
            "selected_clients": selected,
            "task_seeds": task_seeds,
            "matches": matches,
        })
        last_seen.update({client: round_number for client in selected})
    return schedule


def percentile_rank(scores, validity):
    """Rank only on the frozen common support; ties use channel index."""
    indexes = torch.nonzero(validity, as_tuple=False).reshape(-1).tolist()
    values = torch.zeros_like(scores, dtype=torch.float32)
    ordered = sorted(indexes, key=lambda idx: (float(scores[idx]), idx))
    denominator = max(len(ordered) - 1, 1)
    for rank, idx in enumerate(ordered):
        values[idx] = rank / denominator
    return values


class FrozenScoreBankWriter:
    """Atomic per-round CPU score records from the unmodified B0 trajectory."""

    def __init__(self, directory, geometry):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layers = [item["layer_name"] for item in geometry.layers]
        self.widths = [int(item["output_kernels"]) for item in geometry.layers]
        self.offsets = np.cumsum([0, *self.widths]).tolist()
        (self.directory / "bank_config.json").write_text(
            json.dumps({"layers": self.layers, "widths": self.widths,
                        "offsets": self.offsets, "score_type": "update_norm",
                        "source_algorithm": "HistoryIdentityB0"}, indent=2),
            encoding="utf-8",
        )

    def write_round(self, round_number, selected, history):
        scores = []
        validity = []
        for client in selected:
            entry = history[int(client)]
            if int(entry["last_round"]) != round_number:
                raise AssertionError("score bank contains non-current interaction")
            scores.append(np.concatenate([
                entry["layers"][name]["score"].cpu().numpy()
                for name in self.layers
            ]).astype(np.float32))
            validity.append(np.concatenate([
                entry["layers"][name]["validity"].cpu().numpy()
                for name in self.layers
            ]).astype(np.uint8))
        path = self.directory / f"round{round_number:04d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, round_number=np.asarray(round_number, dtype=np.int32),
                client_ids=np.asarray(selected, dtype=np.int16),
                scores=np.stack(scores), validity=np.stack(validity),
            )
        os.replace(temporary, path)


class FrozenScoreBankReader:
    def __init__(self, directory, wait_seconds=900):
        self.directory = Path(directory)
        config_path = self.directory / "bank_config.json"
        deadline = time.monotonic() + wait_seconds
        while not config_path.exists() and time.monotonic() < deadline:
            time.sleep(1)
        if not config_path.exists():
            raise TimeoutError("B0 score-bank config was not written")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["score_type"] != "update_norm":
            raise ValueError("frozen bank must contain update_norm")
        self.layers = config["layers"]
        self.offsets = config["offsets"]
        self.next_round = 1
        self.history = {}
        self.wait_seconds = wait_seconds

    def through(self, last_round):
        while self.next_round <= last_round:
            path = self.directory / f"round{self.next_round:04d}.npz"
            deadline = time.monotonic() + self.wait_seconds
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(1)
            if not path.exists():
                raise TimeoutError(f"B0 score bank stalled before round {self.next_round}")
            with np.load(path, allow_pickle=False) as record:
                if int(record["round_number"]) != self.next_round:
                    raise AssertionError("score-bank round mismatch")
                client_ids = record["client_ids"]
                scores = record["scores"]
                validity = record["validity"]
                for row, client in enumerate(client_ids):
                    layers = {}
                    for index, name in enumerate(self.layers):
                        left, right = self.offsets[index:index + 2]
                        layers[name] = {
                            "score": torch.from_numpy(scores[row, left:right].copy()),
                            "validity": torch.from_numpy(
                                validity[row, left:right].astype(np.bool_)
                            ),
                        }
                    self.history[int(client)] = {
                        "last_round": self.next_round, "layers": layers,
                    }
            self.next_round += 1
        return self.history


class FrozenHistoryController(TargetedFedPhoenixController):
    """Exact original reset retargeting with a frozen common-support score view."""

    def __init__(self, model, mode, reset_ratio, **kwargs):
        if mode not in MODES:
            raise ValueError(f"unknown history source: {mode}")
        super().__init__(model, **kwargs)
        self.mode = mode
        self.reset_ratio = float(reset_ratio)
        self.source_entries = {}
        self.source_metadata = {}

    def prepare_round(self, schedule_round, bank_history):
        self.source_entries = {}
        self.source_metadata = {}
        round_number = int(schedule_round["round"])
        for client in schedule_round["selected_clients"]:
            spec = schedule_round["matches"][str(client)]
            donor = spec["donor_id"]
            age = spec["age"]
            metadata = {
                "history_source": self.mode,
                "matched_donor_id": donor,
                "matched_peer_count": len(spec["peers"]),
                "matched_age": age,
                "common_valid_by_layer": {},
                "matching_fallback": None,
            }
            if not spec["eligible"]:
                metadata["matching_fallback"] = (
                    "before_start_round" if round_number < self.start_round else
                    "cold_start" if age is None else
                    "stale_history" if age > self.max_history_gap else
                    "no_exact_age_peer"
                )
                self.source_metadata[client] = metadata
                continue
            if client not in bank_history or donor not in bank_history:
                raise AssertionError("frozen matching refers to absent score history")
            peers = spec["peers"]
            if any(other not in bank_history for other in peers):
                raise AssertionError("population peer absent from score bank")
            if any(round_number - bank_history[other]["last_round"] != age
                   for other in [client, *peers]):
                raise AssertionError("score-bank ages do not match frozen schedule")
            layers = {}
            for geometry in self.geometry.layers:
                name = geometry["layer_name"]
                cohort = [client, *peers]
                common = torch.ones(geometry["output_kernels"], dtype=torch.bool)
                for other in cohort:
                    common &= bank_history[other]["layers"][name]["validity"]
                requested = math.floor(
                    self.target_ratio * int(
                        geometry["output_kernels"] * self.reset_ratio
                    )
                )
                valid_count = int(common.sum().item())
                metadata["common_valid_by_layer"][name] = valid_count
                if requested > valid_count:
                    common.zero_()
                if self.mode == "H_same":
                    source = percentile_rank(
                        bank_history[client]["layers"][name]["score"], common
                    )
                elif self.mode == "S_shuffled":
                    source = percentile_rank(
                        bank_history[donor]["layers"][name]["score"], common
                    )
                else:
                    source = torch.stack([
                        percentile_rank(
                            bank_history[other]["layers"][name]["score"], common
                        ) for other in peers
                    ]).mean(dim=0)
                layers[name] = {"score": source, "validity": common}
            self.source_entries[client] = {
                "last_round": round_number - age, "layers": layers,
            }
            self.source_metadata[client] = metadata

    def _history_status(self, client_id, current_round):
        client_id = int(client_id)
        if client_id in self.source_entries:
            entry = self.source_entries[client_id]
            age = int(current_round) - int(entry["last_round"])
            return entry, age, None
        reason = self.source_metadata.get(client_id, {}).get(
            "matching_fallback", "no_frozen_source"
        )
        return None, None, reason

    def retarget_task(self, *args, **kwargs):
        client_id = int(kwargs["client_id"])
        model, trace = super().retarget_task(*args, **kwargs)
        trace.update(self.source_metadata[client_id])
        return model, trace
