#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib


from Algorithm.Phoenix_util import *

matplotlib.use('Agg')
import copy
import csv
import json
import math
import os
import random
import time
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset

from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.Update import *
from models.Nets import *
from models.Fed import Aggregation,AggregationMut
from models.test import *
from models.resnetcifar import *

from utils.get_dataset import * 
from utils.utils import save_result,save_model
from Algorithm.Training_FedGen import FedGen

from Algorithm.Training_FedMut import FedMut
from Algorithm.RecoveryAware import (
    build_recovery_score_matrix,
    confidence_gate_assignment,
    joint_match,
    mark_selected_features,
)

from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

import sys
import io   


def _run_suffix(args):
    if not args.run_name:
        return ""
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(args.run_name)
    )
    return f"_{safe}" if safe else ""


def _metrics_stem(args):
    return (
        f"{args.dataset}_{args.model}_{args.algorithm}_seed{args.seed}"
        f"{_run_suffix(args)}"
    )


def _build_fedphoenix_tasks(net_glob, round_idx, task_count, args):
    """Build a deterministic task bank shared by baseline and proposed method."""
    task_models = []
    task_traces = []
    for task_id in range(task_count):
        task_model = copy.deepcopy(net_glob).to("cpu")
        task_seed = (
            int(args.seed)
            + 200000
            + int(round_idx) * 100003
            + int(task_id) * 997
        )
        trace = reset_kernels_for_task(
            task_model,
            reset_ratio=args.reset,
            seed=task_seed,
            layer_scope="all",
            init_method=args.remethod,
            at_least_one=False,
            current_iter=round_idx,
            conv_transition_period=args.FP_conv,
        )
        task_models.append(task_model)
        task_traces.append(trace)
    return task_models, task_traces


def _write_training_metrics(args, rows):
    os.makedirs(args.metrics_log_dir, exist_ok=True)
    stem = _metrics_stem(args)
    csv_path = os.path.abspath(os.path.join(args.metrics_log_dir, f"{stem}.csv"))
    config_path = os.path.abspath(
        os.path.join(args.metrics_log_dir, f"{stem}_config.json")
    )
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    print(f"Unified training metrics saved to {csv_path}")
    return csv_path


def _split_evaluation_dataset(dataset_test, args):
    """Create a deterministic validation/test split for checkpoint selection."""
    validation_samples = int(args.validation_samples)
    if validation_samples <= 0 or validation_samples >= len(dataset_test):
        raise ValueError(
            "validation_samples must be between 1 and len(dataset_test) - 1"
        )
    rng = np.random.default_rng(int(args.seed) + 700001)
    indices = rng.permutation(len(dataset_test))
    validation_indices = indices[:validation_samples].tolist()
    test_indices = indices[validation_samples:].tolist()
    return (
        Subset(dataset_test, validation_indices),
        Subset(dataset_test, test_indices),
    )


class _EvaluationTracker:
    """Track validation-selected and diagnostic peak test checkpoints."""

    def __init__(self, args):
        self.args = args
        self.rows = []
        self.best_validation_accuracy = float("-inf")
        self.best_validation_loss = float("inf")
        self.best_validation_round = None
        self.selected_test_accuracy = None
        self.selected_test_loss = None
        self.peak_test_accuracy = float("-inf")
        self.peak_test_round = None
        self.final_test_accuracy = None
        self.final_test_loss = None
        os.makedirs(args.metrics_log_dir, exist_ok=True)
        stem = _metrics_stem(args)
        self.checkpoint_path = os.path.abspath(
            os.path.join(
                args.metrics_log_dir,
                f"{stem}_best_validation.pt",
            )
        )
        self.summary_path = os.path.abspath(
            os.path.join(args.metrics_log_dir, f"{stem}_summary.json")
        )

    def evaluate(
        self,
        model,
        validation_dataset,
        test_dataset,
        round_number,
    ):
        validation_accuracy, validation_loss = test_with_loss(
            model, validation_dataset, self.args
        )
        test_accuracy, test_loss = test_with_loss(
            model, test_dataset, self.args
        )
        validation_accuracy = float(validation_accuracy)
        validation_loss = float(validation_loss)
        test_accuracy = float(test_accuracy)
        test_loss = float(test_loss)

        is_best = (
            validation_accuracy > self.best_validation_accuracy
            or (
                validation_accuracy == self.best_validation_accuracy
                and validation_loss < self.best_validation_loss
            )
        )
        if is_best:
            self.best_validation_accuracy = validation_accuracy
            self.best_validation_loss = validation_loss
            self.best_validation_round = int(round_number)
            self.selected_test_accuracy = test_accuracy
            self.selected_test_loss = test_loss
            cpu_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(cpu_state, self.checkpoint_path)

        if test_accuracy > self.peak_test_accuracy:
            self.peak_test_accuracy = test_accuracy
            self.peak_test_round = int(round_number)
        self.final_test_accuracy = test_accuracy
        self.final_test_loss = test_loss
        return {
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "is_best_validation": bool(is_best),
        }

    def write_summary(self, evaluated_rows):
        summary = {
            "selection_rule": (
                "maximum validation accuracy, breaking ties by minimum "
                "validation loss"
            ),
            "validation_samples": int(self.args.validation_samples),
            "test_samples": int(10000 - self.args.validation_samples)
            if self.args.dataset == "cifar10"
            else None,
            "best_validation_round": self.best_validation_round,
            "best_validation_accuracy": self.best_validation_accuracy,
            "best_validation_loss": self.best_validation_loss,
            "selected_test_accuracy": self.selected_test_accuracy,
            "selected_test_loss": self.selected_test_loss,
            "diagnostic_peak_test_round": self.peak_test_round,
            "diagnostic_peak_test_accuracy": self.peak_test_accuracy,
            "final_test_accuracy": self.final_test_accuracy,
            "final_test_loss": self.final_test_loss,
            "evaluations": int(evaluated_rows),
            "best_checkpoint": self.checkpoint_path,
        }
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"Run summary saved to {self.summary_path}")
        print(
            "Validation-selected test accuracy: "
            f"{self.selected_test_accuracy:.2f}% "
            f"(round {self.best_validation_round}); "
            "diagnostic peak test accuracy: "
            f"{self.peak_test_accuracy:.2f}% "
            f"(round {self.peak_test_round})"
        )
        return summary

def FedPhoenix(
    net_glob,
    dataset_train,
    dataset_validation,
    dataset_test,
    dict_users,
):
    
    net_glob.train()
    # training
    acc = []
    train_loss=[]
    train_acc=[]
    test_loss=[]
    metrics_rows=[]

    args.density_local=0.01


    for iter in range(args.epochs):
        round_start = time.perf_counter()
        if args.density_local>1 or args.density_local<0 :
            args.density_local=0
  
        print('*'*80)
        print('Round {:3d}'.format(iter))
       
        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        task_models, task_traces = _build_fedphoenix_tasks(
            net_glob, iter, m, args
        )
        clients_mloss = 0  
        assignments = []
        for task_id, idx in enumerate(idxs_users):
            net_local = copy.deepcopy(task_models[task_id]).to(args.device)
            local = LocalUpdate_FedAvg(
                args=args,
                dataset=dataset_train,
                idxs=dict_users[idx],
                dataset_test=dataset_test,
            )
            w = local.train(net=net_local)
            w_locals.append(copy.deepcopy(w))   
            lens.append(len(dict_users[idx]))
            assignments.append(
                {"client_id": int(idx), "task_id": int(task_id)}
            )
        # update global weights 
        w_glob = Aggregation(w_locals, lens )
     
        

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        round_train_seconds = time.perf_counter() - round_start

        item_acc = evaluate_round_accuracy(
            net_glob, dataset_test, args, iter + 1
        )
        acc.append(item_acc)
        metrics_rows.append(
            {
                "round": int(iter + 1),
                "algorithm": args.algorithm,
                "seed": int(args.seed),
                "test_accuracy": item_acc,
                "round_train_seconds": float(round_train_seconds),
                "matching_seconds": 0.0,
                "selected_clients": json.dumps(
                    [int(item) for item in idxs_users.tolist()]
                ),
                "assignments": json.dumps(assignments),
                "task_seeds": json.dumps(
                    [int(trace["seed"]) for trace in task_traces]
                ),
            }
        )
        del task_models
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    print_peak_accuracy(acc, args.algorithm)
    _write_training_metrics(args, metrics_rows)


def FedPhoenixRecovery(
    net_glob,
    dataset_train,
    dataset_validation,
    dataset_test,
    dict_users,
):
    """FedPhoenix with recovery-aware joint client/copy matching.

    This is the deployable non-Agent baseline: the matching signal uses only a
    small support/query probe from each candidate client's private training
    data.  No global test metric is used for selection.
    """
    net_glob.train()
    acc = []
    test_loss = []
    metrics_rows = []
    log_dir = os.path.abspath(args.recovery_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        (
            f"{args.dataset}_{args.model}_seed{args.seed}"
            f"{_run_suffix(args)}_matching.csv"
        ),
    )
    log_handle = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = None

    try:
        for iter in range(args.epochs):
            round_start = time.perf_counter()
            print('*' * 80)
            print('Round {:3d} (FedPhoenixRecovery)'.format(iter))
            m = max(int(args.frac * args.num_users), 1)
            if m > args.num_users:
                raise ValueError("Participation budget exceeds num_users")

            task_models, task_traces = _build_fedphoenix_tasks(
                net_glob, iter, m, args
            )
            matching_seconds = 0.0
            assignment_score_gain = None
            matching_applied = False

            reset_active = any(
                trace["num_reset_kernels"] > 0 for trace in task_traces
            )
            matching_window_active = (
                reset_active
                and iter >= args.recovery_start_round
                and (
                    args.recovery_end_round < 0
                    or iter < args.recovery_end_round
                )
            )
            if matching_window_active:
                if args.recovery_mode == "assignment_only":
                    pool_size = m
                else:
                    pool_size = min(
                        args.num_users,
                        max(
                            m,
                            int(math.ceil(m * args.recovery_pool_multiplier)),
                        ),
                    )
                candidate_clients = np.random.choice(
                    range(args.num_users), pool_size, replace=False
                ).tolist()
                matching_start = time.perf_counter()
                score_matrix, feature_rows = build_recovery_score_matrix(
                    args,
                    net_glob,
                    task_models,
                    task_traces,
                    dataset_train,
                    dict_users,
                    candidate_clients,
                    iter,
                )
                matching_score_matrix = score_matrix
                if args.recovery_mode == "interaction_joint":
                    matching_score_matrix = (
                        score_matrix
                        - score_matrix.mean(axis=1, keepdims=True)
                        - score_matrix.mean(axis=0, keepdims=True)
                        + score_matrix.mean()
                    )
                assignment_row_weights = np.ones(
                    len(candidate_clients), dtype=np.float64
                )
                if (
                    args.recovery_mode == "assignment_only"
                    and args.recovery_assignment_weighting == "data_size"
                ):
                    assignment_row_weights = np.asarray(
                        [
                            len(dict_users[int(client_id)])
                            for client_id in candidate_clients
                        ],
                        dtype=np.float64,
                    )
                    assignment_row_weights /= max(
                        float(assignment_row_weights.mean()), 1e-12
                    )
                    matching_score_matrix = (
                        matching_score_matrix
                        * assignment_row_weights[:, np.newaxis]
                    )
                assignment_objective_matrix = matching_score_matrix
                if (
                    args.recovery_mode == "assignment_only"
                    and args.recovery_assignment_inertia > 0
                ):
                    assignment_objective_matrix = (
                        matching_score_matrix
                        + float(args.recovery_assignment_inertia)
                        * np.eye(matching_score_matrix.shape[0])
                    )
                if args.recovery_mode in {
                    "joint",
                    "interaction_joint",
                    "assignment_only",
                }:
                    assignments = joint_match(
                        assignment_objective_matrix, candidate_clients
                    )
                    matching_applied = True
                    if args.recovery_mode == "assignment_only":
                        (
                            assignments,
                            assignment_score_gain,
                            matching_applied,
                        ) = confidence_gate_assignment(
                            matching_score_matrix,
                            candidate_clients,
                            assignments,
                            args.recovery_min_assignment_gain,
                        )
                else:
                    client_scores = score_matrix.mean(axis=1)
                    selected_rows = np.argsort(-client_scores)[:m]
                    task_rng = np.random.default_rng(
                        int(args.seed) + 500000 + int(iter) * 1009
                    )
                    random_tasks = task_rng.permutation(m)
                    assignments = [
                        {
                            "client_row": int(client_row),
                            "client_id": int(candidate_clients[int(client_row)]),
                            "task_id": int(task_id),
                            "score": float(
                                score_matrix[int(client_row), int(task_id)]
                            ),
                        }
                        for client_row, task_id in zip(
                            selected_rows, random_tasks
                        )
                    ]
                    assignments.sort(key=lambda item: item["task_id"])
                    matching_applied = True
                feature_rows = mark_selected_features(feature_rows, assignments)
                matching_seconds = time.perf_counter() - matching_start
                for row, matching_score in zip(
                    feature_rows, matching_score_matrix.reshape(-1)
                ):
                    row["recovery_mode"] = args.recovery_mode
                    row["matching_score"] = float(matching_score)
                    row["candidate_pool_size"] = pool_size
                    row["matching_seconds"] = matching_seconds
                    row["assignment_score_gain_per_task"] = (
                        assignment_score_gain
                    )
                    row["matching_applied"] = matching_applied
                    row["assignment_weighting"] = (
                        args.recovery_assignment_weighting
                    )
                    row["assignment_inertia"] = float(
                        args.recovery_assignment_inertia
                    )
                    row["assignment_row_weight"] = float(
                        assignment_row_weights[
                            candidate_clients.index(int(row["client_id"]))
                        ]
                    )
                if log_writer is None and feature_rows:
                    log_writer = csv.DictWriter(
                        log_handle, fieldnames=list(feature_rows[0].keys())
                    )
                    log_writer.writeheader()
                if log_writer is not None:
                    log_writer.writerows(feature_rows)
                    log_handle.flush()
                print(
                    f"Assignments ({args.recovery_mode}):",
                    [(item["client_id"], item["task_id"]) for item in assignments],
                )
                if assignment_score_gain is not None:
                    print(
                        "Assignment score gain per task: "
                        f"{assignment_score_gain:.4f}; "
                        f"matching applied: {matching_applied}"
                    )
                print(f"Recovery matching overhead: {matching_seconds:.3f}s")
            else:
                selected_clients = np.random.choice(
                    range(args.num_users), m, replace=False
                ).tolist()
                assignments = [
                    {
                        "client_id": int(client_id),
                        "task_id": task_id,
                        "score": 0.0,
                    }
                    for task_id, client_id in enumerate(selected_clients)
                ]

            w_locals = []
            lens = []
            for assignment in assignments:
                client_id = assignment["client_id"]
                task_id = assignment["task_id"]
                net_local = copy.deepcopy(task_models[task_id]).to(args.device)
                local = LocalUpdate_FedAvg(
                    args=args,
                    dataset=dataset_train,
                    idxs=dict_users[client_id],
                    dataset_test=dataset_test,
                )
                local_weights = local.train(net=net_local)
                w_locals.append(copy.deepcopy(local_weights))
                lens.append(len(dict_users[client_id]))
                del net_local

            w_glob = Aggregation(w_locals, lens)
            net_glob.load_state_dict(w_glob)
            round_train_seconds = time.perf_counter() - round_start
            item_acc = evaluate_round_accuracy(
                net_glob, dataset_test, args, iter + 1
            )
            acc.append(item_acc)

            metrics_rows.append(
                {
                    "round": int(iter + 1),
                    "algorithm": args.algorithm,
                    "recovery_mode": args.recovery_mode,
                    "seed": int(args.seed),
                    "test_accuracy": item_acc,
                    "round_train_seconds": float(round_train_seconds),
                    "matching_seconds": float(matching_seconds),
                    "matching_window_active": bool(matching_window_active),
                    "matching_applied": bool(matching_applied),
                    "assignment_score_gain_per_task": assignment_score_gain,
                    "assignment_weighting": args.recovery_assignment_weighting,
                    "assignment_inertia": float(
                        args.recovery_assignment_inertia
                    ),
                    "selected_clients": json.dumps(
                        [int(item["client_id"]) for item in assignments]
                    ),
                    "assignments": json.dumps(
                        [
                            {
                                "client_id": int(item["client_id"]),
                                "task_id": int(item["task_id"]),
                            }
                            for item in assignments
                        ]
                    ),
                    "task_seeds": json.dumps(
                        [int(trace["seed"]) for trace in task_traces]
                    ),
                }
            )

            del task_models
            if args.device.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        log_handle.close()

    print_peak_accuracy(acc, args.algorithm)
    _write_training_metrics(args, metrics_rows)
    print(f"Recovery matching diagnostics saved to {log_path}")

def FedAvg(net_glob, dataset_train, dataset_test, dict_users):
    
    net_glob.train()
    # training
    acc = []
    for iter in range(args.epochs):
        print('*'*80)
        print('Round {:3d}'.format(iter))

        w_locals = []
      
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            net_local = None
            net_local = copy.deepcopy(net_glob).to(args.device)
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx],dataset_test=dataset_test)
            w = local.train(net=net_local)
 
            w_locals.append(copy.deepcopy(w))   
            lens.append(len(dict_users[idx]))
        # update global weights   
        w_glob = Aggregation(w_locals, lens )
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        item_acc = evaluate_round_accuracy(
            net_glob, dataset_test, args, iter + 1
        )
        acc.append(item_acc)

    print_peak_accuracy(acc, args.algorithm)

def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []
    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        item_acc = evaluate_round_accuracy(
            net_glob, dataset_test, args, iter + 1
        )
        acc.append(item_acc)
    print_peak_accuracy(acc, args.algorithm)

from utils.clustering import *
from scipy.cluster.hierarchy import linkage


def ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.to('cpu')

    n_samples = np.array([len(dict_users[idx]) for idx in dict_users.keys()])
    weights = n_samples / np.sum(n_samples)
    n_sampled = max(int(args.frac * args.num_users), 1)

    gradients = get_gradients('', net_glob, [net_glob] * len(dict_users))

    net_glob.train()

    # training
    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        previous_global_model = copy.deepcopy(net_glob)
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' SIMILARITY MATRIX
        if iter == 0:
            sim_matrix = get_matrix_similarity_from_grads(
                gradients, distance_type=args.sim_type
            )

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = get_clusters_with_alg2(
            linkage_matrix, n_sampled, weights
        )

        w_locals = []
        lens = []
        idxs_users = sample_clients(distri_clusters)
        for idx in idxs_users:
            local = LocalUpdate_ClientSampling(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_model = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.to('cpu')

            w_locals.append(copy.deepcopy(local_model.state_dict()))
            lens.append(len(dict_users[idx]))

            clients_models.append(copy.deepcopy(local_model))
            sampled_clients_for_grad.append(idx)

            del local_model
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        gradients_i = get_gradients(
            '', previous_global_model, clients_models
        )
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            gradients[idx] = gradient

        sim_matrix = get_matrix_similarity_from_grads_new(
            gradients, distance_type=args.sim_type, idx=idxs_users, metric_matrix=sim_matrix
        )

        net_glob.to(args.device)
        item_acc = evaluate_round_accuracy(
            net_glob, dataset_test, args, iter + 1
        )
        acc.append(item_acc)
        net_glob.to('cpu')

        del clients_models

    print_peak_accuracy(acc, args.algorithm)




def test(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()

def test_with_loss(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing Loss: {:.2f}".format(loss_test))

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(), loss_test

def tSNE(net_glob,dataset,args):
                                                    
    net_glob.eval()
    # features, labels = get_features(net_glob, dataset, args)
    # visualize_decision_boundary(features, labels, net_glob, args, epoch=0)
    zero_out_model_params(net_glob, 0.1)  
    features, labels = get_features(net_glob, dataset, args)
    visualize_decision_boundary(features, labels, net_glob, args, epoch=0)
def get_model(dataset, model_name, args=None):
    """
    Get the corresponding model instance based on the dataset and model name.
    
    Args:
        dataset (str): Dataset name
        model_name (str): Model name
        args: Model initialization parameters (optional)
    
    Returns:
        Model instance
    """
    for key in MODEL_FACTORY:
        if key in dataset:
            model_dict = MODEL_FACTORY[key]
            if model_name in model_dict:
                return model_dict[model_name](args) if args else model_dict[model_name]()
    raise ValueError(f"Unsupported dataset {dataset} or model {model_name}")
if __name__ == '__main__':
    # parse args  
    args = args_parser()
    if args.eval_every < 1:
        raise ValueError("eval_every must be at least one")
    if args.recovery_start_round < 0:
        raise ValueError("recovery_start_round must be non-negative")
    if (
        args.recovery_end_round >= 0
        and args.recovery_end_round <= args.recovery_start_round
    ):
        raise ValueError(
            "recovery_end_round must be -1 or greater than recovery_start_round"
        )
    if args.recovery_min_assignment_gain < 0:
        raise ValueError("recovery_min_assignment_gain must be non-negative")
    if args.recovery_assignment_inertia < 0:
        raise ValueError("recovery_assignment_inertia must be non-negative")
    if args.recovery_probe_steps < 1:
        raise ValueError("recovery_probe_steps must be at least one")
    set_random_seed(args.seed)
    device_index = args.gpu
    torch.cuda.set_device(device_index)
    args.device = torch.device('cuda')

    if 'timage'in args.dataset:
        dataset_train, dataset_test, dict_users = get_tiny_imagenet_data(args)
    else:
        dataset_train, dataset_test, dict_users = get_dataset(args)

    client_sizes = [len(dict_users[client_id]) for client_id in sorted(dict_users)]
    print(
        "Client partition: "
        f"{len(client_sizes)} clients, "
        f"min={min(client_sizes)}, max={max(client_sizes)}, "
        f"mean={np.mean(client_sizes):.2f}, std={np.std(client_sizes):.2f}"
    )
    if args.prepare_data_only:
        print("Partition preparation completed; exiting before model training.")
        sys.exit(0)

    dataset_validation = None
    dataset_final_test = dataset_test
    print(
        "Evaluation protocol: full server test set evaluated every round "
        f"({len(dataset_final_test)} samples)."
    )

    MODEL_FACTORY = {
    'timage': {
        'resnet18': lambda args=None: ResNetTinyImageNet(BasicBlock, [2, 2, 2, 2]),
        'resnet_drop': lambda args=None: ResNetTinyImageNet_Drop(BasicBlock, [2, 2, 2, 2]),
        'resnet_drop25': lambda args=None: ResNetTinyImageNet_Drop25(BasicBlock, [2, 2, 2, 2]),
        'vgg': lambda args: VGG16_timage(args),
        'vggdrop': lambda args: VGG16_timage_Drop(args),
        'mobnet': lambda args: MobileNet(args),
        'mobnet_drop': lambda args: MobileNet_Drop(args),
    },
    'cifar': {
        'cnn': lambda args: CNNCifar(args),
        'cnndrop': lambda args: CNNCifarDrop(args),
        'resnet18': lambda args: ResNet18_cifar10(num_classes=args.num_classes),
        'resnet_drop': lambda args: ResNet18_cifar10_drop(num_classes=args.num_classes),
        'mobnet': lambda args: MobileNet(args),
        'mobnet_drop': lambda args: MobileNet_Drop(args),
        'vgg': lambda args: VGG16(args),
        'vggdrop': lambda args: VGG16_Drop(args),
     
    }
}
    try:
        net_glob = get_model(args.dataset, args.model, args)
        print(net_glob)
    except ValueError as e:
        print(e) 
    net_glob.to(args.device)
    


    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_final_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_final_test, dict_users)
    elif args.algorithm == 'ClusteredSampling':
        ClusteredSampling(net_glob, dataset_train, dataset_final_test, dict_users)
    elif args.algorithm == 'FedGen':

        FedGen(args, net_glob, dataset_train, dataset_final_test, dict_users)
    elif args.algorithm == 'FedMut':
        FedMut(args, net_glob, dataset_train, dataset_final_test, dict_users)
    elif args.algorithm == 'FedPhoenix':
        FedPhoenix(
            net_glob,
            dataset_train,
            dataset_validation,
            dataset_final_test,
            dict_users,
        )
    elif args.algorithm == 'FedPhoenixRecovery':
        FedPhoenixRecovery(
            net_glob,
            dataset_train,
            dataset_validation,
            dataset_final_test,
            dict_users,
        )

    elif args.algorithm == 'test':
        test(net_glob, dataset_test , args)
    elif args.algorithm == 'tSNE_train':
        test(net_glob, dataset_test, args)
        tSNE(net_glob,dataset_train,args)
    elif args.algorithm == 'tSNE_test':
        test(net_glob, dataset_test, args)
        tSNE(net_glob,dataset_test,args)
