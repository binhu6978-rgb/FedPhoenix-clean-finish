"""Horizontal Conv weight symmetry for the CIFAR10 VGG model view."""

import copy
import hashlib
import random

import torch
from torch import nn


def _conv_weight_names(model):
    return [f"{name}.weight" if name else "weight"
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)]


def apply_horizontal_view_(model):
    """Flip only the width dimension of each Conv2d weight, in place."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.copy_(torch.flip(module.weight, dims=[3]))
    return model


def map_back_state(returned_state, model):
    """Map an F-view client state back to canonical parameter coordinates."""
    result = copy.deepcopy(returned_state)
    for name in _conv_weight_names(model):
        result[name] = torch.flip(result[name], dims=[3])
    return result


def verify_flip_equivariance(model, input_shape=(2, 3, 32, 32), tolerance=1e-4):
    """Check f_F(theta)(x) = f_theta(flip(x)) without changing global RNG."""
    canonical = copy.deepcopy(model).eval()
    flipped = copy.deepcopy(model).eval()
    apply_horizontal_view_(flipped)
    generator = torch.Generator(device="cpu").manual_seed(20260924)
    x = torch.randn(input_shape, generator=generator).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        expected = canonical(torch.flip(x, dims=[3]))["logit"]
        actual = flipped(x)["logit"]
    error = float((actual - expected).abs().max())
    if not torch.isfinite(torch.tensor(error)) or error > tolerance:
        raise RuntimeError(
            f"VGG horizontal-flip equivariance failed: max error={error} "
            f"> {tolerance}"
        )
    return error


class ViewScheduler:
    """Private view decisions; alt stores only each client's last view/round."""

    def __init__(self, mode, seed):
        if mode not in {"none", "rand", "alt"}:
            raise ValueError("view mode must be none, rand, or alt")
        self.mode = mode
        self.seed = int(seed)
        self.private_rng = random.Random(self.seed + 710023) if mode == "rand" else None
        self.last = {}

    def choose(self, client_id, round_idx):
        client_id = int(client_id)
        round_idx = int(round_idx)
        if self.mode == "none":
            return "I", False
        if self.mode == "rand":
            return ("F" if self.private_rng.getrandbits(1) else "I"), False
        previous = self.last.get(client_id)
        if previous is None:
            digest = hashlib.sha256(f"{self.seed}:{client_id}".encode()).digest()
            view = "F" if digest[0] & 1 else "I"
        else:
            if round_idx <= previous[1]:
                raise ValueError("client participations must be chronological")
            view = "I" if previous[0] == "F" else "F"
        violation = previous is not None and view == previous[0]
        self.last[client_id] = (view, round_idx)
        return view, violation
