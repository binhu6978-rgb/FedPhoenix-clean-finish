#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Short-horizon client-operator falsification experiment.

Purpose
-------
This script fixes the main formulation problems in the previous
`client_operator_probe.py`:

1. It first validates the measurement itself:
   - exact repeatability of the same probe under the same seed;
   - epsilon consistency with the SAME stochastic trajectory for all epsilons.

2. It tests longitudinal operator stability on the timescale that returning
   clients actually revisit (default gaps: 5/10/20/40 rounds), not 100-round
   gaps.

3. It uses the SAME shared intervention directions for every client and every
   checkpoint, so "same client vs other client" isolates client identity
   instead of changing the input direction at the same time.

4. It estimates a small input-conditioned local response operator:
   shared basis direction u_k -> client-specific secant response J_i u_k.
   Unseen directions are fixed linear combinations of the basis directions,
   and their responses are predicted by the same linear combination of past
   basis responses. This is a valid test of local operator prediction.

5. Symmetric responses are true secants:
       [T(w + eps*d) - T(w - eps*d)] / (2*eps)

6. The script intentionally does NOT run the old Q3 utility experiment.
   Utility/control should only be tested after the local operator measurement
   and short-horizon prediction premise pass.

Phase 1 is still ordinary FedAvg using the repository's original:
- LocalUpdate_FedAvg
- Aggregation
- VGG16
- existing CIFAR-10 partition

Recommended full run:
    python experiments/client_operator_probe_v2.py \
      --dataset cifar10 --model vgg --algorithm FedAvg \
      --num_users 100 --frac 0.1 \
      --local_ep 5 --local_bs 50 --bs 256 \
      --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 0 \
      --iid 0 --noniid_case 5 --data_beta 0.3 --generate_data 0 \
      --num_classes 10 --seed 1 --gpu 0 \
      --epochs 240 \
      --anchor_round 200 \
      --target_rounds 205,210,220,240 \
      --num_probe_clients 10 \
      --num_basis 3 \
      --num_unseen 3 \
      --epsilon_fractions 0.01,0.025,0.05,0.1 \
      --primary_epsilon_fraction 0.05
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.Update import LocalUpdate_FedAvg  # noqa: E402
from models.Fed import Aggregation  # noqa: E402
from models.Nets import VGG16  # noqa: E402
from models.test import test_img  # noqa: E402
from utils.get_dataset import get_dataset  # noqa: E402
from utils.set_seed import set_random_seed  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Short-horizon client-operator falsification experiment."
    )

    # Keep the original FedAvg protocol explicit.
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="vgg")
    parser.add_argument("--algorithm", default="FedAvg")
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--local_ep", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--iid", type=int, default=0)
    parser.add_argument("--noniid_case", type=int, default=5)
    parser.add_argument("--data_beta", type=float, default=0.3)
    parser.add_argument("--generate_data", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    # Short-horizon design.
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--anchor_round", type=int, default=200)
    parser.add_argument(
        "--target_rounds",
        default="205,210,220,240",
        help="Dense post-anchor checkpoints; default lags = 5,10,20,40.",
    )
    parser.add_argument("--num_probe_clients", type=int, default=10)
    parser.add_argument("--min_pre_anchor_participations", type=int, default=5)
    parser.add_argument("--min_post_anchor_participations", type=int, default=2)

    # Shared low-dimensional intervention basis.
    parser.add_argument("--num_basis", type=int, default=3)
    parser.add_argument("--num_unseen", type=int, default=3)

    # Measurement validity.
    parser.add_argument(
        "--epsilon_fractions",
        default="0.01,0.025,0.05,0.1",
        help="Fractions of the median probe-client natural update norm at anchor.",
    )
    parser.add_argument("--primary_epsilon_fraction", type=float, default=0.05)
    parser.add_argument("--num_sanity_clients", type=int, default=3)
    parser.add_argument("--num_sanity_directions", type=int, default=2)

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: results/client_operator_probe_v2_<model>_seed<seed>",
    )
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()
    args.target_rounds = [
        int(x) for x in str(args.target_rounds).split(",") if str(x).strip()
    ]
    args.epsilon_fractions = [
        float(x) for x in str(args.epsilon_fractions).split(",") if str(x).strip()
    ]

    if args.output_dir is None:
        args.output_dir = os.path.join(
            ROOT, "results", f"client_operator_probe_v2_{args.model}_seed{args.seed}"
        )

    if args.smoke_test:
        # Keep the same code path, only compress the schedule.
        args.epochs = 12
        args.anchor_round = 5
        args.target_rounds = [6, 8, 10, 12]
        args.num_probe_clients = min(args.num_probe_clients, 3)
        args.min_pre_anchor_participations = 1
        args.min_post_anchor_participations = 0
        args.num_basis = min(args.num_basis, 2)
        args.num_unseen = min(args.num_unseen, 1)
        args.num_sanity_clients = 1
        args.num_sanity_directions = 1

    checkpoints = [args.anchor_round] + args.target_rounds
    if any(r <= 0 for r in checkpoints):
        raise ValueError("All checkpoint rounds must be positive.")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("anchor_round and target_rounds must be unique.")
    if args.target_rounds != sorted(args.target_rounds):
        raise ValueError("target_rounds must be increasing.")
    if args.target_rounds and args.target_rounds[0] <= args.anchor_round:
        raise ValueError("All target_rounds must be after anchor_round.")
    if max(checkpoints) > args.epochs:
        raise ValueError("epochs must be >= max(anchor_round, target_rounds).")
    if args.primary_epsilon_fraction not in args.epsilon_fractions:
        raise ValueError("primary_epsilon_fraction must be included in epsilon_fractions.")
    if args.num_basis < 1:
        raise ValueError("num_basis must be >= 1.")
    if args.num_unseen < 1:
        raise ValueError("num_unseen must be >= 1.")
    return args


# =============================================================================
# Parameter vector utilities
# =============================================================================

def build_vgg(args):
    if args.model != "vgg":
        raise ValueError("This v2 script is intentionally scoped to --model vgg.")
    return VGG16(args)


def param_names_shapes_offsets(model: nn.Module):
    names = []
    shapes = {}
    offsets = {}
    cursor = 0
    for name, param in model.named_parameters():
        names.append(name)
        shapes[name] = tuple(param.shape)
        numel = param.numel()
        offsets[name] = (cursor, cursor + numel)
        cursor += numel
    return names, shapes, offsets, cursor


def flatten_params(state_dict, param_names):
    return torch.cat(
        [
            state_dict[name].detach().reshape(-1).float().cpu()
            for name in param_names
        ]
    )


def apply_perturbation(state_dict, param_names, shapes, delta_flat):
    out = {key: value.clone() for key, value in state_dict.items()}
    offset = 0
    for name in param_names:
        numel = int(np.prod(shapes[name]))
        chunk = delta_flat[offset : offset + numel].reshape(shapes[name])
        out[name] = (out[name].float().cpu() + chunk).to(out[name].dtype)
        offset += numel
    return out


def normalize(vector, eps=1e-12):
    norm = vector.float().norm()
    if norm < eps:
        return None
    return vector.float() / norm


def cosine(a, b, eps=1e-12):
    a = a.float()
    b = b.float()
    denom = a.norm() * b.norm()
    if denom < eps:
        return float("nan")
    return float((a @ b) / denom)


def normalized_error(pred, target, eps=1e-12):
    pred = pred.float()
    target = target.float()
    denom = pred.norm() + target.norm()
    if denom < eps:
        return float("nan")
    return float((pred - target).norm() / denom)


def build_scope_slices(model, param_names, offsets):
    """
    Pre-registered diagnostic scopes.

    full       : all trainable parameters
    features   : VGG feature extractor (conv+BN affine)
    conv       : Conv2d trainable parameters only
    classifier : VGG classifier + final fc
    """
    total = max(end for _, end in offsets.values())
    scopes = {"full": [(0, total)], "features": [], "conv": [], "classifier": []}

    conv_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prefix = module_name + "." if module_name else ""
            for local_name, _ in module.named_parameters(recurse=False):
                conv_param_names.add(prefix + local_name)

    for name in param_names:
        start, end = offsets[name]
        if name.startswith("features."):
            scopes["features"].append((start, end))
        if name in conv_param_names:
            scopes["conv"].append((start, end))
        if name.startswith("classifier.") or name.startswith("fc."):
            scopes["classifier"].append((start, end))

    return scopes


def scoped_cosine(a, b, slices, eps=1e-12):
    a = a.float()
    b = b.float()
    dot = 0.0
    na = 0.0
    nb = 0.0
    for start, end in slices:
        aa = a[start:end]
        bb = b[start:end]
        dot += float((aa * bb).sum())
        na += float((aa * aa).sum())
        nb += float((bb * bb).sum())
    denom = math.sqrt(max(na, 0.0)) * math.sqrt(max(nb, 0.0))
    if denom < eps:
        return float("nan")
    return dot / denom


def scoped_normalized_error(a, b, slices, eps=1e-12):
    a = a.float()
    b = b.float()
    diff_sq = 0.0
    na = 0.0
    nb = 0.0
    for start, end in slices:
        aa = a[start:end]
        bb = b[start:end]
        dd = aa - bb
        diff_sq += float((dd * dd).sum())
        na += float((aa * aa).sum())
        nb += float((bb * bb).sum())
    denom = math.sqrt(max(na, 0.0)) + math.sqrt(max(nb, 0.0))
    if denom < eps:
        return float("nan")
    return math.sqrt(max(diff_sq, 0.0)) / denom


# =============================================================================
# Compact response storage
# =============================================================================

def pack_vector(vector):
    """
    Store a huge response vector as (unit_direction_fp16, fp32_norm).
    This keeps response magnitude separately while avoiding global int8
    quantization. It is materially safer than the previous max-abs int8 scheme.
    """
    vector = vector.float().cpu()
    norm = float(vector.norm())
    if norm < 1e-20:
        return {"unit": torch.zeros_like(vector, dtype=torch.float16), "norm": 0.0}
    unit = (vector / norm).to(torch.float16)
    return {"unit": unit, "norm": norm}


def unpack_vector(packed):
    return packed["unit"].float() * float(packed["norm"])


# =============================================================================
# Participation schedule and client selection
# =============================================================================

def compute_schedule(seed, epochs, num_users, frac):
    set_random_seed(seed)
    m = max(int(frac * num_users), 1)
    schedule = []
    for _ in range(epochs):
        idxs = np.random.choice(range(num_users), m, replace=False)
        schedule.append([int(i) for i in idxs])
    return schedule


def select_probe_clients(
    schedule,
    anchor_round,
    final_round,
    num_probe,
    min_pre,
    min_post,
):
    rounds_by_client = defaultdict(list)
    for round_idx, clients in enumerate(schedule, start=1):
        for client_id in clients:
            rounds_by_client[client_id].append(round_idx)

    candidates = []
    for client_id, rounds in rounds_by_client.items():
        pre = sum(r <= anchor_round for r in rounds)
        post = sum(anchor_round < r <= final_round for r in rounds)
        if pre >= min_pre and post >= min_post:
            candidates.append((client_id, pre, post, len(rounds)))

    if len(candidates) < num_probe:
        # Relax post requirement only; still require real history before anchor.
        candidates = []
        for client_id, rounds in rounds_by_client.items():
            pre = sum(r <= anchor_round for r in rounds)
            post = sum(anchor_round < r <= final_round for r in rounds)
            if pre >= min_pre:
                candidates.append((client_id, pre, post, len(rounds)))

    candidates.sort(key=lambda x: (-x[3], x[0]))
    return sorted([client_id for client_id, *_ in candidates[:num_probe]])


def participation_gap_summary(schedule, probe_clients):
    rows = []
    pooled = []
    by_client = {}
    for client_id in probe_clients:
        rounds = [
            round_idx
            for round_idx, clients in enumerate(schedule, start=1)
            if client_id in clients
        ]
        gaps = [b - a for a, b in zip(rounds[:-1], rounds[1:])]
        pooled.extend(gaps)
        by_client[str(client_id)] = {
            "participation_rounds": rounds,
            "num_participations": len(rounds),
            "mean_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "median_gap": float(np.median(gaps)) if gaps else float("nan"),
        }
        for gap in gaps:
            rows.append({"client_id": client_id, "gap": gap})

    pooled_arr = np.asarray(pooled, dtype=np.float64)
    summary = {
        "pooled_count": int(len(pooled)),
        "pooled_mean_gap": float(pooled_arr.mean()) if len(pooled_arr) else float("nan"),
        "pooled_median_gap": float(np.median(pooled_arr)) if len(pooled_arr) else float("nan"),
        "fraction_gap_le_5": float(np.mean(pooled_arr <= 5)) if len(pooled_arr) else float("nan"),
        "fraction_gap_le_10": float(np.mean(pooled_arr <= 10)) if len(pooled_arr) else float("nan"),
        "fraction_gap_le_20": float(np.mean(pooled_arr <= 20)) if len(pooled_arr) else float("nan"),
        "by_client": by_client,
    }
    return rows, summary


# =============================================================================
# Phase 1: unmodified FedAvg trajectory + history sufficient statistics
# =============================================================================

def run_phase1(
    args,
    net_glob,
    dataset_train,
    dataset_test,
    dict_users,
    param_names,
    probe_clients,
    schedule,
):
    """
    Run ordinary FedAvg.

    For selected probe clients only, before anchor_round:
    - accumulate the SUM of normalized natural updates;
    - track participation count;
    - track latest natural-update norm.

    We do not save all history vectors and do not quantize them.
    At anchor_round we snapshot one normalized history direction per client.
    """
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    probe_set = set(probe_clients)
    history_sum = {}
    history_count = {client_id: 0 for client_id in probe_clients}
    latest_update_norm = {client_id: None for client_id in probe_clients}

    anchor_history_dirs = None
    anchor_update_norms = None
    metrics_rows = []
    accuracies = []

    checkpoint_set = {args.anchor_round, *args.target_rounds}

    net_glob.train()

    for iter_idx in range(args.epochs):
        round_number = iter_idx + 1
        start = time.perf_counter()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        actual_clients = [int(i) for i in idxs_users.tolist()]
        expected_clients = schedule[iter_idx]
        if actual_clients != expected_clients:
            raise RuntimeError(
                f"Participation schedule mismatch at round {round_number}: "
                f"expected {expected_clients}, got {actual_clients}"
            )

        capture_history = round_number <= args.anchor_round
        global_before_flat = (
            flatten_params(net_glob.state_dict(), param_names)
            if capture_history
            else None
        )

        w_locals = []
        lens = []

        for idx in idxs_users:
            client_id = int(idx)

            net_local = copy.deepcopy(net_glob).to(args.device)
            local = LocalUpdate_FedAvg(
                args=args,
                dataset=dataset_train,
                idxs=dict_users[client_id],
                dataset_test=dataset_test,
            )
            w = local.train(net=net_local)
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[client_id]))

            if capture_history and client_id in probe_set:
                delta = flatten_params(w, param_names) - global_before_flat
                delta_norm = float(delta.norm())
                if delta_norm > 1e-20:
                    unit = delta / delta_norm
                    if client_id not in history_sum:
                        history_sum[client_id] = unit.clone()
                    else:
                        history_sum[client_id].add_(unit)
                    history_count[client_id] += 1
                    latest_update_norm[client_id] = delta_norm

            del net_local

        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)

        accuracy, loss = test_img(net_glob, dataset_test, args)
        accuracy = float(accuracy.item() if hasattr(accuracy, "item") else accuracy)
        loss = float(loss.item() if hasattr(loss, "item") else loss)
        accuracies.append(accuracy)

        metrics_rows.append(
            {
                "round": round_number,
                "test_accuracy": accuracy,
                "test_loss": loss,
                "selected_clients": json.dumps(actual_clients),
                "round_seconds": time.perf_counter() - start,
            }
        )

        print(
            f"PHASE1 round={round_number} accuracy={accuracy:.4f} "
            f"loss={loss:.4f}",
            flush=True,
        )

        if round_number in checkpoint_set:
            ckpt_path = os.path.join(ckpt_dir, f"round_{round_number}.pt")
            torch.save(
                {
                    key: value.detach().cpu().clone()
                    for key, value in net_glob.state_dict().items()
                },
                ckpt_path,
            )
            print(f"Saved checkpoint {ckpt_path}", flush=True)

        if round_number == args.anchor_round:
            anchor_history_dirs = {}
            anchor_update_norms = {}
            for client_id in probe_clients:
                if history_count.get(client_id, 0) <= 0 or client_id not in history_sum:
                    continue
                direction = normalize(history_sum[client_id])
                if direction is not None:
                    anchor_history_dirs[client_id] = direction
                if latest_update_norm[client_id] is not None:
                    anchor_update_norms[client_id] = float(latest_update_norm[client_id])

            # We no longer need these huge running sums after the anchor.
            history_sum.clear()
            global_before_flat = None

        del w_locals

    if anchor_history_dirs is None:
        raise RuntimeError("Failed to snapshot anchor history directions.")

    metrics_path = os.path.join(args.output_dir, "phase1_metrics.csv")
    write_csv(metrics_path, metrics_rows)

    print(
        f"PHASE1 peak accuracy={max(accuracies):.4f} "
        f"at round={int(np.argmax(accuracies)) + 1}",
        flush=True,
    )

    return anchor_history_dirs, anchor_update_norms, metrics_rows


# =============================================================================
# Shared intervention basis
# =============================================================================

def orthogonal_residual(vector, basis):
    residual = vector.float().clone()
    for b in basis:
        residual -= float(residual @ b) * b
    return residual


def private_random_direction(dim, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vector = torch.randn(dim, generator=generator, dtype=torch.float32)
    return normalize(vector)


def build_shared_basis(
    args,
    probe_clients,
    anchor_history_dirs,
    dim,
):
    """
    Build a fixed, shared, orthonormal basis.

    We preferentially use structured history directions from probe clients,
    but the resulting basis is shared across ALL clients and ALL checkpoints.
    Therefore client identity is not confounded with input direction.
    """
    basis = []
    metadata = []

    # Fixed ordering chosen before probing: sorted probe client ids.
    for client_id in sorted(probe_clients):
        if len(basis) >= args.num_basis:
            break
        candidate = anchor_history_dirs.get(client_id)
        if candidate is None:
            continue
        residual = orthogonal_residual(candidate, basis)
        residual_norm = float(residual.norm())
        if residual_norm < 1e-3:
            continue
        direction = residual / residual_norm
        basis.append(direction)
        metadata.append(
            {
                "basis_index": len(basis) - 1,
                "source": "client_history",
                "source_client_id": int(client_id),
                "pre_orthogonalization_residual_norm": residual_norm,
            }
        )

    # Supplement deterministically if history directions are too collinear.
    supplement_idx = 0
    while len(basis) < args.num_basis:
        candidate = private_random_direction(
            dim,
            seed=args.seed * 1_000_003 + 97 * supplement_idx + 17,
        )
        residual = orthogonal_residual(candidate, basis)
        residual_norm = float(residual.norm())
        supplement_idx += 1
        if residual_norm < 1e-3:
            continue
        direction = residual / residual_norm
        basis.append(direction)
        metadata.append(
            {
                "basis_index": len(basis) - 1,
                "source": "deterministic_random_supplement",
                "source_client_id": None,
                "pre_orthogonalization_residual_norm": residual_norm,
            }
        )

    # Sanity: basis orthonormality.
    gram = []
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            gram.append(
                {
                    "basis_i": i,
                    "basis_j": j,
                    "dot": float(bi @ bj),
                }
            )
    return basis, metadata, gram


def build_unseen_combinations(args, basis):
    """
    Generate fixed unseen in-span intervention directions.

    Each unseen direction is a normalized linear combination of the shared
    orthonormal basis. The coefficient vector is fixed across clients and time.
    """
    rng = np.random.default_rng(args.seed * 77_777 + 123)
    combos = []
    for unseen_id in range(args.num_unseen):
        coeffs = rng.standard_normal(len(basis)).astype(np.float32)
        coeff_norm = float(np.linalg.norm(coeffs))
        if coeff_norm < 1e-12:
            coeffs[0] = 1.0
            coeff_norm = 1.0
        coeffs = coeffs / coeff_norm

        direction = torch.zeros_like(basis[0])
        for coefficient, b in zip(coeffs, basis):
            direction.add_(b, alpha=float(coefficient))
        direction = normalize(direction)

        combos.append(
            {
                "unseen_id": unseen_id,
                "coeffs": coeffs.tolist(),
                "direction": direction,
            }
        )
    return combos


# =============================================================================
# Controlled local response probe
# =============================================================================

def seed_probe_rng(seed):
    """
    Reset every stochastic source used by the local training path.

    DataLoader shuffle and VGG dropout both ultimately depend on torch RNG in
    this configuration. We also reset Python/NumPy/CUDA defensively.
    """
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def probe_seed(args, client_id):
    # Intentionally depends ONLY on client identity.
    #
    # Every basis direction, unseen direction, checkpoint and epsilon for the
    # same client uses the same minibatch order / dropout RNG trajectory.
    # Otherwise cross-direction linear-combination tests would mix operator
    # changes with different SGD/dropout realizations.
    return int(args.seed) * 10_000_019 + int(client_id) * 9_973 + 17


def run_secant_probe(
    args,
    checkpoint_state,
    dataset_train,
    dataset_test,
    dict_users,
    client_id,
    direction,
    epsilon_abs,
    param_names,
    shapes,
    stochastic_seed,
):
    """
    True symmetric secant:
      [T(w + eps*d) - T(w - eps*d)] / (2*eps)

    +/- legs use the exact same stochastic seed.
    """
    endpoints = {}

    for sign, tag in ((+1.0, "plus"), (-1.0, "minus")):
        perturbed = apply_perturbation(
            checkpoint_state,
            param_names,
            shapes,
            sign * float(epsilon_abs) * direction,
        )

        net_local = build_vgg(args)
        net_local.load_state_dict(perturbed)
        net_local.to(args.device)

        # Reset RNG AFTER model construction/load so initialization does not
        # perturb the local SGD/dropout/shuffle sequence.
        seed_probe_rng(stochastic_seed)

        local = LocalUpdate_FedAvg(
            args=args,
            dataset=dataset_train,
            idxs=dict_users[client_id],
            dataset_test=dataset_test,
        )
        w = local.train(net=net_local)
        endpoints[tag] = flatten_params(w, param_names)
        del net_local

    if epsilon_abs <= 0:
        raise ValueError("epsilon_abs must be positive.")

    secant = (endpoints["plus"] - endpoints["minus"]) / (2.0 * float(epsilon_abs))
    return secant


# =============================================================================
# Measurement validity
# =============================================================================

def run_measurement_validity(
    args,
    anchor_state,
    dataset_train,
    dataset_test,
    dict_users,
    probe_clients,
    basis,
    reference_update_norm,
    param_names,
    shapes,
):
    """
    Two checks:

    A. Exact repeat:
       same client, direction, epsilon, seed -> run twice.

    B. Epsilon consistency:
       same client, direction, SAME seed, eps in pre-registered sweep.
       Compare true secants, not raw endpoint differences.
    """
    repeat_rows = []
    epsilon_rows = []

    sanity_clients = probe_clients[: min(args.num_sanity_clients, len(probe_clients))]
    sanity_basis_ids = list(range(min(args.num_sanity_directions, len(basis))))

    primary_eps = float(args.primary_epsilon_fraction) * reference_update_norm

    for client_id in sanity_clients:
        for basis_id in sanity_basis_ids:
            direction = basis[basis_id]
            seed = probe_seed(args, client_id)

            # Exact repeat under identical conditions.
            response_a = run_secant_probe(
                args, anchor_state, dataset_train, dataset_test, dict_users,
                client_id, direction, primary_eps, param_names, shapes, seed,
            )
            response_b = run_secant_probe(
                args, anchor_state, dataset_train, dataset_test, dict_users,
                client_id, direction, primary_eps, param_names, shapes, seed,
            )
            repeat_rows.append(
                {
                    "client_id": client_id,
                    "basis_id": basis_id,
                    "epsilon_fraction": args.primary_epsilon_fraction,
                    "cosine_repeat": cosine(response_a, response_b),
                    "normalized_error_repeat": normalized_error(response_a, response_b),
                    "norm_a": float(response_a.norm()),
                    "norm_b": float(response_b.norm()),
                }
            )

            primary_response = response_a
            primary_norm = float(primary_response.norm())

            for fraction in args.epsilon_fractions:
                eps_abs = float(fraction) * reference_update_norm
                # Critically, use the SAME seed for every epsilon.
                response = run_secant_probe(
                    args, anchor_state, dataset_train, dataset_test, dict_users,
                    client_id, direction, eps_abs, param_names, shapes, seed,
                )
                response_norm = float(response.norm())
                epsilon_rows.append(
                    {
                        "client_id": client_id,
                        "basis_id": basis_id,
                        "epsilon_fraction": float(fraction),
                        "cosine_to_primary": cosine(response, primary_response),
                        "normalized_error_to_primary": normalized_error(
                            response, primary_response
                        ),
                        "secant_norm": response_norm,
                        "secant_norm_ratio_to_primary": (
                            response_norm / primary_norm
                            if primary_norm > 1e-20
                            else float("nan")
                        ),
                    }
                )

            print(
                f"MEASUREMENT client={client_id} basis={basis_id} done",
                flush=True,
            )

    return repeat_rows, epsilon_rows


# =============================================================================
# Basis response measurement
# =============================================================================

def measure_basis_responses_at_checkpoint(
    args,
    checkpoint_state,
    checkpoint_round,
    dataset_train,
    dataset_test,
    dict_users,
    probe_clients,
    basis,
    epsilon_abs,
    param_names,
    shapes,
):
    """
    Measure the SAME shared basis directions for every client at one checkpoint.
    """
    responses = defaultdict(dict)

    for client_id in probe_clients:
        for basis_id, direction in enumerate(basis):
            seed = probe_seed(args, client_id)
            secant = run_secant_probe(
                args,
                checkpoint_state,
                dataset_train,
                dataset_test,
                dict_users,
                client_id,
                direction,
                epsilon_abs,
                param_names,
                shapes,
                seed,
            )
            responses[client_id][basis_id] = pack_vector(secant)
        print(
            f"BASIS checkpoint={checkpoint_round} client={client_id} done",
            flush=True,
        )
    return responses


# =============================================================================
# Q1: identity-specific short-horizon operator stability
# =============================================================================

def analyze_q1_checkpoint(
    anchor_round,
    target_round,
    anchor_responses,
    target_responses,
    probe_clients,
    scopes,
):
    """
    For the SAME input basis direction:
      same-client  = cos(R_i^anchor(u_k), R_i^target(u_k))
      other-client = mean_{j != i} cos(R_i^anchor(u_k), R_j^target(u_k))

    This directly isolates client identity.
    """
    rows = []
    lag = target_round - anchor_round

    for client_id in probe_clients:
        if client_id not in anchor_responses or client_id not in target_responses:
            continue

        for basis_id, anchor_packed in anchor_responses[client_id].items():
            if basis_id not in target_responses[client_id]:
                continue

            anchor_vec = unpack_vector(anchor_packed)
            same_vec = unpack_vector(target_responses[client_id][basis_id])

            other_vectors = []
            for other_id in probe_clients:
                if other_id == client_id:
                    continue
                packed = target_responses.get(other_id, {}).get(basis_id)
                if packed is not None:
                    other_vectors.append((other_id, unpack_vector(packed)))

            for scope_name, slices in scopes.items():
                same_cos = scoped_cosine(anchor_vec, same_vec, slices)
                same_err = scoped_normalized_error(anchor_vec, same_vec, slices)

                other_cosines = [
                    scoped_cosine(anchor_vec, vec, slices)
                    for _oid, vec in other_vectors
                ]
                valid_other = [x for x in other_cosines if not math.isnan(x)]
                other_mean = (
                    float(np.mean(valid_other)) if valid_other else float("nan")
                )

                rows.append(
                    {
                        "anchor_round": anchor_round,
                        "target_round": target_round,
                        "lag": lag,
                        "client_id": client_id,
                        "basis_id": basis_id,
                        "scope": scope_name,
                        "same_client_cosine": same_cos,
                        "same_client_normalized_error": same_err,
                        "mean_other_client_cosine": other_mean,
                        "identity_gap_same_minus_other": (
                            same_cos - other_mean
                            if not (math.isnan(same_cos) or math.isnan(other_mean))
                            else float("nan")
                        ),
                    }
                )

    return rows


# =============================================================================
# Q2: valid input-conditioned unseen-response prediction
# =============================================================================

def combine_response_basis(response_dict_for_client, coeffs):
    """
    Given measured responses J_i u_k and coefficients a_k,
    predict J_i (sum_k a_k u_k) = sum_k a_k J_i u_k.
    """
    first = next(iter(response_dict_for_client.values()))
    dim = first["unit"].numel()
    out = torch.zeros(dim, dtype=torch.float32)
    for basis_id, coefficient in enumerate(coeffs):
        packed = response_dict_for_client.get(basis_id)
        if packed is None:
            return None
        out.add_(unpack_vector(packed), alpha=float(coefficient))
    return out


def cohort_mean_basis_response(anchor_responses, probe_clients, excluded_client):
    result = {}
    others = [c for c in probe_clients if c != excluded_client]
    if not others:
        return result

    basis_ids = sorted(
        set.intersection(
            *[
                set(anchor_responses[c].keys())
                for c in others
                if c in anchor_responses
            ]
        )
    )
    for basis_id in basis_ids:
        vectors = [
            unpack_vector(anchor_responses[c][basis_id])
            for c in others
            if basis_id in anchor_responses[c]
        ]
        if not vectors:
            continue
        total = torch.zeros_like(vectors[0])
        for vector in vectors:
            total.add_(vector)
        total.div_(len(vectors))
        result[basis_id] = pack_vector(total)
    return result


def evaluate_prediction_rows(
    stage,
    anchor_round,
    target_round,
    client_id,
    unseen_id,
    actual,
    predictions,
    scopes,
):
    rows = []
    lag = target_round - anchor_round

    for predictor_name, prediction in predictions.items():
        if prediction is None:
            continue

        for scope_name, slices in scopes.items():
            if predictor_name == "zero_baseline":
                pred_vec = torch.zeros_like(actual)
            else:
                pred_vec = prediction

            rows.append(
                {
                    "stage": stage,
                    "anchor_round": anchor_round,
                    "target_round": target_round,
                    "lag": lag,
                    "client_id": client_id,
                    "unseen_id": unseen_id,
                    "scope": scope_name,
                    "predictor": predictor_name,
                    "cosine": scoped_cosine(pred_vec, actual, slices),
                    "normalized_error": scoped_normalized_error(
                        pred_vec, actual, slices
                    ),
                }
            )
    return rows


def run_q2_same_checkpoint_interpolation(
    args,
    anchor_state,
    anchor_round,
    dataset_train,
    dataset_test,
    dict_users,
    probe_clients,
    basis,
    unseen_combos,
    anchor_responses,
    epsilon_abs,
    param_names,
    shapes,
    scopes,
):
    """
    Before testing temporal prediction, test whether the local linear basis model
    can predict an unseen in-span intervention at the SAME checkpoint.
    """
    partner_of = {
        client_id: probe_clients[(idx + 1) % len(probe_clients)]
        for idx, client_id in enumerate(probe_clients)
    }

    rows = []

    for client_id in probe_clients:
        cohort_basis = cohort_mean_basis_response(
            anchor_responses, probe_clients, client_id
        )
        partner = partner_of[client_id]

        for combo in unseen_combos:
            unseen_id = int(combo["unseen_id"])
            coeffs = combo["coeffs"]
            direction = combo["direction"]

            seed = probe_seed(args, client_id)
            actual = run_secant_probe(
                args,
                anchor_state,
                dataset_train,
                dataset_test,
                dict_users,
                client_id,
                direction,
                epsilon_abs,
                param_names,
                shapes,
                seed,
            )

            predictions = {
                "own_anchor_operator": combine_response_basis(
                    anchor_responses[client_id], coeffs
                ),
                "matched_other_operator": combine_response_basis(
                    anchor_responses[partner], coeffs
                ),
                "cohort_mean_operator": combine_response_basis(
                    cohort_basis, coeffs
                ),
                "zero_baseline": torch.zeros_like(actual),
            }

            rows.extend(
                evaluate_prediction_rows(
                    stage="same_checkpoint_interpolation",
                    anchor_round=anchor_round,
                    target_round=anchor_round,
                    client_id=client_id,
                    unseen_id=unseen_id,
                    actual=actual,
                    predictions=predictions,
                    scopes=scopes,
                )
            )

        print(f"Q2-INTERP client={client_id} done", flush=True)

    return rows


def run_q2_temporal_prediction(
    args,
    target_state,
    target_round,
    anchor_round,
    dataset_train,
    dataset_test,
    dict_users,
    probe_clients,
    unseen_combos,
    anchor_responses,
    epsilon_abs,
    param_names,
    shapes,
    scopes,
):
    """
    Use ONLY anchor-round basis responses to predict an unseen response at a
    future checkpoint.

    own_anchor_operator:
        sum_k a_k * R_i^anchor(u_k)

    matched_other_operator:
        sum_k a_k * R_j^anchor(u_k)

    cohort_mean_operator:
        mean_{j != i} anchor operator, then apply same coefficients.
    """
    partner_of = {
        client_id: probe_clients[(idx + 1) % len(probe_clients)]
        for idx, client_id in enumerate(probe_clients)
    }

    rows = []

    for client_id in probe_clients:
        cohort_basis = cohort_mean_basis_response(
            anchor_responses, probe_clients, client_id
        )
        partner = partner_of[client_id]

        for combo in unseen_combos:
            unseen_id = int(combo["unseen_id"])
            coeffs = combo["coeffs"]
            direction = combo["direction"]

            seed = probe_seed(args, client_id)
            actual = run_secant_probe(
                args,
                target_state,
                dataset_train,
                dataset_test,
                dict_users,
                client_id,
                direction,
                epsilon_abs,
                param_names,
                shapes,
                seed,
            )

            predictions = {
                "own_anchor_operator": combine_response_basis(
                    anchor_responses[client_id], coeffs
                ),
                "matched_other_operator": combine_response_basis(
                    anchor_responses[partner], coeffs
                ),
                "cohort_mean_operator": combine_response_basis(
                    cohort_basis, coeffs
                ),
                "zero_baseline": torch.zeros_like(actual),
            }

            rows.extend(
                evaluate_prediction_rows(
                    stage="temporal_prediction",
                    anchor_round=anchor_round,
                    target_round=target_round,
                    client_id=client_id,
                    unseen_id=unseen_id,
                    actual=actual,
                    predictions=predictions,
                    scopes=scopes,
                )
            )

        print(
            f"Q2-TEMP target={target_round} client={client_id} done",
            flush=True,
        )

    return rows


# =============================================================================
# Summaries / verdicts
# =============================================================================

def finite(values):
    return [
        float(v)
        for v in values
        if v is not None
        and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    ]


def mean_or_nan(values):
    values = finite(values)
    return float(np.mean(values)) if values else float("nan")


def median_or_nan(values):
    values = finite(values)
    return float(np.median(values)) if values else float("nan")


def summarize_measurement(repeat_rows, epsilon_rows, primary_fraction):
    repeat_cos = finite([r["cosine_repeat"] for r in repeat_rows])
    repeat_err = finite([r["normalized_error_repeat"] for r in repeat_rows])

    non_primary = [
        r
        for r in epsilon_rows
        if abs(float(r["epsilon_fraction"]) - float(primary_fraction)) > 1e-12
    ]
    eps_cos = finite([r["cosine_to_primary"] for r in non_primary])
    eps_err = finite([r["normalized_error_to_primary"] for r in non_primary])
    eps_ratio = finite([r["secant_norm_ratio_to_primary"] for r in non_primary])

    repeat_mean_cos = mean_or_nan(repeat_cos)
    repeat_max_err = max(repeat_err) if repeat_err else float("nan")
    eps_mean_cos = mean_or_nan(eps_cos)
    eps_median_err = median_or_nan(eps_err)

    if (
        not math.isnan(repeat_mean_cos)
        and repeat_mean_cos >= 0.999
        and not math.isnan(repeat_max_err)
        and repeat_max_err <= 1e-3
    ):
        repeat_verdict = "PASS"
    else:
        repeat_verdict = "FAIL"

    if not math.isnan(eps_mean_cos) and eps_mean_cos >= 0.80:
        epsilon_verdict = "PASS"
    elif not math.isnan(eps_mean_cos) and eps_mean_cos >= 0.50:
        epsilon_verdict = "WEAK"
    else:
        epsilon_verdict = "FAIL"

    overall = (
        "PASS"
        if repeat_verdict == "PASS" and epsilon_verdict == "PASS"
        else "WEAK"
        if repeat_verdict == "PASS" and epsilon_verdict == "WEAK"
        else "FAIL"
    )

    return {
        "repeat_mean_cosine": repeat_mean_cos,
        "repeat_max_normalized_error": repeat_max_err,
        "epsilon_mean_cosine_to_primary": eps_mean_cos,
        "epsilon_median_normalized_error_to_primary": eps_median_err,
        "epsilon_median_secant_norm_ratio_to_primary": median_or_nan(eps_ratio),
        "repeat_verdict": repeat_verdict,
        "epsilon_verdict": epsilon_verdict,
        "verdict": overall,
    }


def summarize_q1(q1_rows, primary_scope="full"):
    by_scope = {}
    scopes = sorted(set(r["scope"] for r in q1_rows))

    for scope in scopes:
        rows = [r for r in q1_rows if r["scope"] == scope]
        short = [r for r in rows if r["lag"] <= 20]

        same = finite([r["same_client_cosine"] for r in short])
        other = finite([r["mean_other_client_cosine"] for r in short])
        gap = finite([r["identity_gap_same_minus_other"] for r in short])

        by_lag = {}
        for lag in sorted(set(r["lag"] for r in rows)):
            lag_rows = [r for r in rows if r["lag"] == lag]
            by_lag[str(lag)] = {
                "same_mean_cosine": mean_or_nan(
                    [r["same_client_cosine"] for r in lag_rows]
                ),
                "other_mean_cosine": mean_or_nan(
                    [r["mean_other_client_cosine"] for r in lag_rows]
                ),
                "identity_gap": mean_or_nan(
                    [r["identity_gap_same_minus_other"] for r in lag_rows]
                ),
            }

        by_scope[scope] = {
            "short_horizon_same_mean_cosine": mean_or_nan(same),
            "short_horizon_other_mean_cosine": mean_or_nan(other),
            "short_horizon_identity_gap": mean_or_nan(gap),
            "by_lag": by_lag,
        }

    primary = by_scope.get(primary_scope, {})
    same = primary.get("short_horizon_same_mean_cosine", float("nan"))
    gap = primary.get("short_horizon_identity_gap", float("nan"))

    if not math.isnan(same) and not math.isnan(gap) and same > 0.20 and gap > 0.10:
        verdict = "PASS"
    elif not math.isnan(gap) and gap > 0.03:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    return {
        "primary_scope": primary_scope,
        "scopes": by_scope,
        "verdict": verdict,
    }


def paired_win_rate(rows, predictor_a, predictor_b, metric="normalized_error"):
    """
    Compare two predictors on identical (stage,target,client,unseen,scope) keys.
    For normalized_error lower is better; for cosine higher is better.
    """
    table = defaultdict(dict)
    for row in rows:
        key = (
            row["stage"],
            row["target_round"],
            row["client_id"],
            row["unseen_id"],
            row["scope"],
        )
        table[key][row["predictor"]] = row

    wins = 0
    total = 0
    for value in table.values():
        if predictor_a not in value or predictor_b not in value:
            continue
        a = value[predictor_a][metric]
        b = value[predictor_b][metric]
        if any(math.isnan(float(x)) for x in (a, b)):
            continue
        total += 1
        if metric == "normalized_error":
            wins += int(a < b)
        else:
            wins += int(a > b)
    return float(wins / total) if total else float("nan")


def summarize_q2(q2_rows, primary_scope="full"):
    by_stage_scope = {}
    stages = sorted(set(r["stage"] for r in q2_rows))
    scopes = sorted(set(r["scope"] for r in q2_rows))

    for stage in stages:
        by_stage_scope[stage] = {}
        for scope in scopes:
            rows = [
                r
                for r in q2_rows
                if r["stage"] == stage and r["scope"] == scope
            ]
            predictor_summary = {}
            for predictor in sorted(set(r["predictor"] for r in rows)):
                pred_rows = [r for r in rows if r["predictor"] == predictor]
                predictor_summary[predictor] = {
                    "mean_cosine": mean_or_nan([r["cosine"] for r in pred_rows]),
                    "mean_normalized_error": mean_or_nan(
                        [r["normalized_error"] for r in pred_rows]
                    ),
                }
            by_stage_scope[stage][scope] = predictor_summary

    # Same-checkpoint interpolation is the local-linearity sanity test.
    interp = by_stage_scope.get("same_checkpoint_interpolation", {}).get(
        primary_scope, {}
    )
    interp_own = interp.get("own_anchor_operator", {})
    interp_cos = interp_own.get("mean_cosine", float("nan"))
    interp_err = interp_own.get("mean_normalized_error", float("nan"))

    # Temporal identity advantage: short horizons only (<=20).
    temporal_rows = [
        r
        for r in q2_rows
        if r["stage"] == "temporal_prediction"
        and r["scope"] == primary_scope
        and r["lag"] <= 20
    ]

    own_rows = [
        r for r in temporal_rows if r["predictor"] == "own_anchor_operator"
    ]
    matched_rows = [
        r for r in temporal_rows if r["predictor"] == "matched_other_operator"
    ]
    cohort_rows = [
        r for r in temporal_rows if r["predictor"] == "cohort_mean_operator"
    ]

    own_cos = mean_or_nan([r["cosine"] for r in own_rows])
    matched_cos = mean_or_nan([r["cosine"] for r in matched_rows])
    cohort_cos = mean_or_nan([r["cosine"] for r in cohort_rows])

    own_err = mean_or_nan([r["normalized_error"] for r in own_rows])
    matched_err = mean_or_nan([r["normalized_error"] for r in matched_rows])
    cohort_err = mean_or_nan([r["normalized_error"] for r in cohort_rows])

    temporal_primary_rows = [
        r
        for r in q2_rows
        if r["stage"] == "temporal_prediction"
        and r["scope"] == primary_scope
        and r["lag"] <= 20
    ]

    win_vs_matched = paired_win_rate(
        temporal_primary_rows,
        "own_anchor_operator",
        "matched_other_operator",
        metric="normalized_error",
    )
    win_vs_cohort = paired_win_rate(
        temporal_primary_rows,
        "own_anchor_operator",
        "cohort_mean_operator",
        metric="normalized_error",
    )

    if not math.isnan(interp_cos) and interp_cos >= 0.50:
        interpolation_verdict = "PASS"
    elif not math.isnan(interp_cos) and interp_cos >= 0.20:
        interpolation_verdict = "WEAK"
    else:
        interpolation_verdict = "FAIL"

    own_advantage_cos = (
        own_cos - max(matched_cos, cohort_cos)
        if not any(math.isnan(x) for x in (own_cos, matched_cos, cohort_cos))
        else float("nan")
    )

    if (
        interpolation_verdict in ("PASS", "WEAK")
        and not math.isnan(own_cos)
        and not math.isnan(own_advantage_cos)
        and own_cos > 0.20
        and own_advantage_cos > 0.08
        and not math.isnan(win_vs_matched)
        and win_vs_matched >= 0.65
    ):
        temporal_verdict = "PASS"
    elif (
        interpolation_verdict != "FAIL"
        and (
            (not math.isnan(own_advantage_cos) and own_advantage_cos > 0.02)
            or (not math.isnan(win_vs_matched) and win_vs_matched > 0.55)
        )
    ):
        temporal_verdict = "WEAK"
    else:
        temporal_verdict = "FAIL"

    return {
        "primary_scope": primary_scope,
        "same_checkpoint_interpolation": {
            "own_mean_cosine": interp_cos,
            "own_mean_normalized_error": interp_err,
            "verdict": interpolation_verdict,
        },
        "short_horizon_temporal_prediction": {
            "own_mean_cosine": own_cos,
            "matched_mean_cosine": matched_cos,
            "cohort_mean_cosine": cohort_cos,
            "own_mean_normalized_error": own_err,
            "matched_mean_normalized_error": matched_err,
            "cohort_mean_normalized_error": cohort_err,
            "own_cosine_advantage_vs_best_control": own_advantage_cos,
            "own_error_win_rate_vs_matched": win_vs_matched,
            "own_error_win_rate_vs_cohort": win_vs_cohort,
            "verdict": temporal_verdict,
        },
        "full_breakdown": by_stage_scope,
        "verdict": temporal_verdict,
    }


# =============================================================================
# Output
# =============================================================================

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_report(
    args,
    phase1_rows,
    probe_clients,
    gap_summary,
    basis_metadata,
    measurement_summary,
    q1_summary,
    q2_summary,
    output_dir,
):
    accuracies = [r["test_accuracy"] for r in phase1_rows]
    peak_round = int(np.argmax(accuracies)) + 1

    if measurement_summary["verdict"] == "FAIL":
        overall = "MEASUREMENT-FAIL"
        recommendation = (
            "Do not interpret Q1/Q2 scientifically yet. Fix or further reduce "
            "the probe until exact repeatability and local epsilon consistency pass."
        )
    elif q1_summary["verdict"] == "FAIL" and q2_summary["verdict"] == "FAIL":
        overall = "NO-GO-FOR-CLIENT-SPECIFIC-OPERATOR"
        recommendation = (
            "At the correct short horizon, client identity still does not produce "
            "a useful longitudinal operator advantage. Do not proceed to dual control."
        )
    elif q1_summary["verdict"] in ("PASS", "WEAK") and q2_summary["verdict"] in (
        "PASS",
        "WEAK",
    ):
        overall = "PROCEED-TO-UTILITY-TEST"
        recommendation = (
            "The identification premise survives. The next experiment should test "
            "predictability-utility compatibility with a properly defined aggregate "
            "response/utility predictor and matched dispatch budgets."
        )
    else:
        overall = "INCONCLUSIVE"
        recommendation = (
            "The measurement is valid but identity stability and unseen-response "
            "prediction disagree. Inspect scope-wise and lag-wise results before "
            "designing a full method."
        )

    lines = [
        "# Client-operator probe v2",
        "",
        "This run tests short-horizon, input-conditioned client local-training "
        "response rather than 100-round response similarity.",
        "",
        "## Phase 1",
        "",
        f"- FedAvg rounds: {args.epochs}",
        f"- Anchor round: {args.anchor_round}",
        f"- Target rounds: {args.target_rounds}",
        f"- Probe clients: {probe_clients}",
        f"- Peak accuracy: {max(accuracies):.4f}% at round {peak_round}",
        f"- Final accuracy: {accuracies[-1]:.4f}%",
        f"- Pooled return gap mean / median: "
        f"{gap_summary['pooled_mean_gap']:.3f} / {gap_summary['pooled_median_gap']:.3f}",
        f"- Fraction of return gaps <=10 / <=20: "
        f"{gap_summary['fraction_gap_le_10']:.3f} / {gap_summary['fraction_gap_le_20']:.3f}",
        "",
        "## Shared intervention basis",
        "",
        f"- Number of basis directions: {args.num_basis}",
        "- Basis is fixed across every client and every checkpoint.",
        "- Basis sources: " + str(basis_metadata),
        "",
        "## Measurement validity",
        "",
        f"- Exact-repeat mean cosine: {measurement_summary['repeat_mean_cosine']:.6f}",
        f"- Exact-repeat max normalized error: "
        f"{measurement_summary['repeat_max_normalized_error']:.6e}",
        f"- Epsilon-sweep mean cosine to primary secant: "
        f"{measurement_summary['epsilon_mean_cosine_to_primary']:.6f}",
        f"- Epsilon-sweep median normalized error: "
        f"{measurement_summary['epsilon_median_normalized_error_to_primary']:.6f}",
        f"- Epsilon-sweep median secant-norm ratio: "
        f"{measurement_summary['epsilon_median_secant_norm_ratio_to_primary']:.6f}",
        f"**Measurement verdict: {measurement_summary['verdict']}**",
        "",
        "## Q1: short-horizon identity-specific operator stability",
        "",
        f"- Primary scope: {q1_summary['primary_scope']}",
        "- Full scope summary: "
        + str(q1_summary["scopes"].get(q1_summary["primary_scope"], {})),
        f"**Q1 verdict: {q1_summary['verdict']}**",
        "",
        "## Q2: input-conditioned unseen-response prediction",
        "",
        "- Same-checkpoint interpolation: "
        + str(q2_summary["same_checkpoint_interpolation"]),
        "- Short-horizon temporal prediction: "
        + str(q2_summary["short_horizon_temporal_prediction"]),
        f"**Q2 verdict: {q2_summary['verdict']}**",
        "",
        "## Overall",
        "",
        f"**{overall}**",
        "",
        recommendation,
        "",
        "## Interpretation guardrails",
        "",
        "- Q1 always compares the same intervention direction across time and clients.",
        "- Q2 predictions explicitly condition on the unseen intervention coefficients.",
        "- All epsilon comparisons use the same stochastic seed.",
        "- Responses are symmetric secants divided by 2*epsilon.",
        "- No old Q3 utility claim is made in this script.",
        "- full/features/conv/classifier scopes are all reported; full is the primary scope.",
        "",
    ]

    with open(os.path.join(output_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return overall


# =============================================================================
# Main
# =============================================================================

def main():
    os.chdir(ROOT)
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_random_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by this experiment configuration.")
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f"cuda:{args.gpu}")

    write_json(os.path.join(args.output_dir, "config.json"), vars(args))

    # Dataset / fixed partition.
    dataset_train, dataset_test, dict_users = get_dataset(args)
    print(f"Loaded partition: {len(dict_users)} clients", flush=True)

    # Pre-compute schedule only for client selection / validation.
    schedule = compute_schedule(args.seed, args.epochs, args.num_users, args.frac)
    final_round = max([args.anchor_round] + args.target_rounds)
    probe_clients = select_probe_clients(
        schedule,
        anchor_round=args.anchor_round,
        final_round=final_round,
        num_probe=args.num_probe_clients,
        min_pre=args.min_pre_anchor_participations,
        min_post=args.min_post_anchor_participations,
    )
    if len(probe_clients) < args.num_probe_clients:
        raise RuntimeError(
            f"Could only select {len(probe_clients)} probe clients, "
            f"requested {args.num_probe_clients}."
        )

    write_json(os.path.join(args.output_dir, "probe_clients.json"), probe_clients)
    gap_rows, gap_summary = participation_gap_summary(schedule, probe_clients)
    write_csv(os.path.join(args.output_dir, "participation_gaps.csv"), gap_rows)
    write_json(
        os.path.join(args.output_dir, "participation_gap_summary.json"),
        gap_summary,
    )
    print(
        f"Probe clients: {probe_clients}; pooled mean gap="
        f"{gap_summary['pooled_mean_gap']:.3f}, median="
        f"{gap_summary['pooled_median_gap']:.3f}",
        flush=True,
    )

    # Re-seed so Phase 1 exactly replays the schedule.
    set_random_seed(args.seed)
    net_glob = build_vgg(args).to(args.device)
    param_names, shapes, offsets, dim = param_names_shapes_offsets(net_glob)
    scopes = build_scope_slices(net_glob, param_names, offsets)

    anchor_history_dirs, anchor_update_norms, phase1_rows = run_phase1(
        args,
        net_glob,
        dataset_train,
        dataset_test,
        dict_users,
        param_names,
        probe_clients,
        schedule,
    )

    if len(anchor_update_norms) < len(probe_clients):
        missing = [c for c in probe_clients if c not in anchor_update_norms]
        raise RuntimeError(f"Missing anchor update norms for clients: {missing}")

    reference_update_norm = float(
        np.median([anchor_update_norms[c] for c in probe_clients])
    )
    primary_epsilon_abs = (
        float(args.primary_epsilon_fraction) * reference_update_norm
    )

    print(
        f"Reference update norm={reference_update_norm:.6e}; "
        f"primary epsilon abs={primary_epsilon_abs:.6e}",
        flush=True,
    )

    # Fixed shared basis.
    basis, basis_metadata, basis_gram = build_shared_basis(
        args,
        probe_clients,
        anchor_history_dirs,
        dim,
    )
    # History directions are only needed to construct the fixed basis.
    del anchor_history_dirs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(
        os.path.join(args.output_dir, "basis_metadata.json"),
        {
            "metadata": basis_metadata,
            "gram": basis_gram,
            "reference_update_norm": reference_update_norm,
            "primary_epsilon_abs": primary_epsilon_abs,
        },
    )

    unseen_combos = build_unseen_combinations(args, basis)
    write_json(
        os.path.join(args.output_dir, "unseen_combinations.json"),
        [
            {
                "unseen_id": c["unseen_id"],
                "coeffs": c["coeffs"],
            }
            for c in unseen_combos
        ],
    )

    anchor_path = os.path.join(
        args.output_dir, "checkpoints", f"round_{args.anchor_round}.pt"
    )
    anchor_state = torch.load(anchor_path, map_location="cpu")

    # Phase 1 is finished; all probes instantiate fresh local models.
    del net_glob
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Measurement validity
    # ----------------------------------------------------------------------
    repeat_rows, epsilon_rows = run_measurement_validity(
        args,
        anchor_state,
        dataset_train,
        dataset_test,
        dict_users,
        probe_clients,
        basis,
        reference_update_norm,
        param_names,
        shapes,
    )
    write_csv(
        os.path.join(args.output_dir, "measurement_repeatability.csv"),
        repeat_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "measurement_epsilon_consistency.csv"),
        epsilon_rows,
    )
    measurement_summary = summarize_measurement(
        repeat_rows,
        epsilon_rows,
        args.primary_epsilon_fraction,
    )
    write_json(
        os.path.join(args.output_dir, "measurement_summary.json"),
        measurement_summary,
    )

    # ----------------------------------------------------------------------
    # Anchor operator dictionary
    # ----------------------------------------------------------------------
    anchor_responses = measure_basis_responses_at_checkpoint(
        args,
        anchor_state,
        args.anchor_round,
        dataset_train,
        dataset_test,
        dict_users,
        probe_clients,
        basis,
        primary_epsilon_abs,
        param_names,
        shapes,
    )

    # Same-checkpoint unseen interpolation sanity.
    q2_rows = run_q2_same_checkpoint_interpolation(
        args,
        anchor_state,
        args.anchor_round,
        dataset_train,
        dataset_test,
        dict_users,
        probe_clients,
        basis,
        unseen_combos,
        anchor_responses,
        primary_epsilon_abs,
        param_names,
        shapes,
        scopes,
    )

    # ----------------------------------------------------------------------
    # Short-horizon targets
    # ----------------------------------------------------------------------
    q1_rows = []

    for target_round in args.target_rounds:
        target_path = os.path.join(
            args.output_dir, "checkpoints", f"round_{target_round}.pt"
        )
        target_state = torch.load(target_path, map_location="cpu")

        target_basis_responses = measure_basis_responses_at_checkpoint(
            args,
            target_state,
            target_round,
            dataset_train,
            dataset_test,
            dict_users,
            probe_clients,
            basis,
            primary_epsilon_abs,
            param_names,
            shapes,
        )

        q1_rows.extend(
            analyze_q1_checkpoint(
                anchor_round=args.anchor_round,
                target_round=target_round,
                anchor_responses=anchor_responses,
                target_responses=target_basis_responses,
                probe_clients=probe_clients,
                scopes=scopes,
            )
        )

        q2_rows.extend(
            run_q2_temporal_prediction(
                args,
                target_state,
                target_round,
                args.anchor_round,
                dataset_train,
                dataset_test,
                dict_users,
                probe_clients,
                unseen_combos,
                anchor_responses,
                primary_epsilon_abs,
                param_names,
                shapes,
                scopes,
            )
        )

        # Free the target dictionary before loading the next checkpoint.
        del target_basis_responses
        del target_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(os.path.join(args.output_dir, "q1_identity_stability.csv"), q1_rows)
    write_csv(os.path.join(args.output_dir, "q2_unseen_prediction.csv"), q2_rows)

    q1_summary = summarize_q1(q1_rows, primary_scope="full")
    q2_summary = summarize_q2(q2_rows, primary_scope="full")
    write_json(os.path.join(args.output_dir, "q1_summary.json"), q1_summary)
    write_json(os.path.join(args.output_dir, "q2_summary.json"), q2_summary)

    overall = write_report(
        args,
        phase1_rows,
        probe_clients,
        gap_summary,
        basis_metadata,
        measurement_summary,
        q1_summary,
        q2_summary,
        args.output_dir,
    )

    summary = {
        "finished_at": datetime.now().isoformat(),
        "output_dir": args.output_dir,
        "probe_clients": probe_clients,
        "reference_update_norm": reference_update_norm,
        "primary_epsilon_abs": primary_epsilon_abs,
        "participation_gap_summary": gap_summary,
        "measurement": measurement_summary,
        "q1": q1_summary,
        "q2": q2_summary,
        "overall_judgment": overall,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nOVERALL JUDGMENT: {overall}", flush=True)


if __name__ == "__main__":
    main()
