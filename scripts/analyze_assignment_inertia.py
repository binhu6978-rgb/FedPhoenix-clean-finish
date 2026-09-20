#!/usr/bin/env python
"""Estimate how identity inertia changes recovery assignments offline."""

import argparse
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment


def analyze_round(rows, inertias, min_gain):
    clients = []
    by_client = {}
    for row in rows:
        client = int(row["client_id"])
        task = int(row["task_id"])
        if client not in by_client:
            clients.append(client)
            by_client[client] = {}
        by_client[client][task] = float(row["score"])
    matrix = np.asarray(
        [[by_client[client][task] for task in range(len(clients))]
         for client in clients],
        dtype=np.float64,
    )
    baseline = float(np.trace(matrix))
    result = {}
    identity = np.eye(len(clients), dtype=np.float64)
    for inertia in inertias:
        rows_idx, tasks = linear_sum_assignment(
            -(matrix + inertia * identity)
        )
        raw_gain = (
            float(matrix[rows_idx, tasks].sum()) - baseline
        ) / len(clients)
        applied = raw_gain >= min_gain
        changed = int(np.sum(rows_idx != tasks)) if applied else 0
        result[inertia] = {
            "raw_gain": raw_gain if applied else 0.0,
            "changed": changed,
            "applied": applied,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_csv")
    parser.add_argument(
        "--inertias", default="0,0.1,0.2,0.3,0.5,0.75,1.0"
    )
    parser.add_argument("--min_gain", type=float, default=0.0)
    args = parser.parse_args()
    inertias = [float(value) for value in args.inertias.split(",")]
    accum = {
        inertia: {"raw_gain": [], "changed": [], "applied": []}
        for inertia in inertias
    }
    current_round = None
    round_rows = []
    with open(args.diagnostics_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_round = int(row["round"])
            if current_round is not None and row_round != current_round:
                result = analyze_round(round_rows, inertias, args.min_gain)
                for inertia, values in result.items():
                    for key, value in values.items():
                        accum[inertia][key].append(value)
                round_rows = []
            current_round = row_round
            round_rows.append(row)
    if round_rows:
        result = analyze_round(round_rows, inertias, args.min_gain)
        for inertia, values in result.items():
            for key, value in values.items():
                accum[inertia][key].append(value)

    print("inertia,apply_rate,mean_changed,mean_raw_gain")
    for inertia in inertias:
        values = accum[inertia]
        print(
            f"{inertia:.3f},"
            f"{np.mean(values['applied']):.4f},"
            f"{np.mean(values['changed']):.4f},"
            f"{np.mean(values['raw_gain']):.6f}"
        )


if __name__ == "__main__":
    main()
