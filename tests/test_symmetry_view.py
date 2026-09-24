"""Small correctness checks for the horizontal Conv model view."""

import copy
import random
import types
import unittest

import numpy as np
import torch
from torch import nn

from Algorithm.SymmetryView import (
    ViewScheduler, apply_horizontal_view_, map_back_state,
    verify_flip_equivariance,
)
from models.Nets import VGG16


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 2, 3, bias=True), nn.BatchNorm2d(2))
        self.fc = nn.Linear(2, 2)


class SymmetryViewTests(unittest.TestCase):
    def test_involution_and_conv_only(self):
        model = TinyModel()
        original = copy.deepcopy(model.state_dict())
        apply_horizontal_view_(model)
        flipped = model.state_dict()
        self.assertTrue(torch.equal(
            flipped["features.0.weight"],
            torch.flip(original["features.0.weight"], dims=[3]),
        ))
        self.assertTrue(all(
            torch.equal(flipped[name], value)
            for name, value in original.items()
            if name != "features.0.weight"
        ))
        mapped = map_back_state(flipped, model)
        self.assertTrue(all(torch.equal(mapped[name], value)
                            for name, value in original.items()))
        apply_horizontal_view_(model)
        self.assertTrue(all(torch.equal(model.state_dict()[name], value)
                            for name, value in original.items()))

    def test_real_vgg_equivariance_and_rng(self):
        args = types.SimpleNamespace(dataset="cifar10", num_channels=3, num_classes=10)
        model = VGG16(args).eval()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        error = verify_flip_equivariance(model)
        self.assertLessEqual(error, 1e-4)
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        if cuda_before is not None:
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(cuda_before, torch.cuda.get_rng_state_all())))

    def test_alt_and_private_rand(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        scheduler = ViewScheduler("alt", seed=1)
        first, _ = scheduler.choose(7, 0)
        scheduler.choose(8, 1)
        second, violation = scheduler.choose(7, 5)
        self.assertNotEqual(first, second)
        self.assertFalse(violation)
        self.assertEqual(ViewScheduler("none", 1).choose(7, 0)[0], "I")
        random_views = [ViewScheduler("rand", 1).choose(7, 0)[0] for _ in range(2)]
        self.assertEqual(random_views[0], random_views[1])
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
