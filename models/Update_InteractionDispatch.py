#!/usr/bin/env python
"""Independent local optimization path for InteractionDispatchV2."""

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from optimizer.Adabelief import AdaBelief


class InteractionDatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.labels = [self.dataset.targets[i] for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate_InteractionDispatch:
    """Same local optimizer semantics as LocalUpdate_FedAvg, with CPU output."""

    def __init__(self, args, dataset_test=None, dataset=None, idxs=None, verbose=False):
        del dataset_test
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.verbose = verbose
        loader_args = {
            "batch_size": args.local_bs,
            "num_workers": args.num_workers,
            "shuffle": True,
        }
        if args.num_workers > 0:
            loader_args.update({"pin_memory": True, "persistent_workers": True})
        self.ldr_train = DataLoader(InteractionDatasetSplit(dataset, idxs), **loader_args)

    def train_from_dispatch(self, net):
        net.train()
        parameters = filter(lambda parameter: parameter.requires_grad, net.parameters())
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.args.lr)
        elif self.args.optimizer == "adaBelief":
            optimizer = AdaBelief(parameters, lr=self.args.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

        accumulated_loss = 0.0
        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                net.zero_grad()
                loss = self.loss_func(net(images)["output"], labels)
                loss.backward()
                optimizer.step()
                accumulated_loss += loss.item()
        mean_loss = accumulated_loss / (self.args.local_ep * len(self.ldr_train))
        if self.verbose:
            print(f"\nUser predict Loss={mean_loss:.4f}")
        returned_state = OrderedDict(
            (name, tensor.detach().to("cpu").clone())
            for name, tensor in net.state_dict().items()
        )
        return {"returned_state": returned_state, "mean_loss": float(mean_loss)}
