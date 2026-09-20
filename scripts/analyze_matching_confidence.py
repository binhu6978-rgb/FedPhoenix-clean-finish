#!/usr/bin/env python
"""Summarize assignment-score gains from recovery matching diagnostics."""

import argparse
import csv
import json
import os

import numpy as np


def summarize_round(rows):
    client_order = []
    scores = {}
    selected = []
    for row in rows:
        client_id = int(row["client_id"])
        task_id = int(row["task_id"])
        if client_id not in scores:
            client_order.append(client_id)
            scores[client_id] = {}
        scores[client_id][task_id] = float(row["matching_score"])
        if row["selected"].lower() == "true":
            selected.append((client_id, task_id))

    task_count = len(client_order)
    baseline = sum(
        scores[client_order[task_id]][task_id]
        for task_id in range(task_count)
    )
    matched = sum(scores[client_id][task_id] for client_id, task_id in selected)
    selected_map = {task_id: client_id for client_id, task_id in selected}
    changed = sum(
        selected_map.get(task_id) != client_order[task_id]
        for task_id in range(task_count)
    )
    return {
        "round": int(rows[0]["round"]),
        "score_gain": matched - baseline,
        "score_gain_per_task": (matched - baseline) / task_count,
        "changed_tasks": changed,
        "task_count": task_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment_size", type=int, default=200)
    args = parser.parse_args()

    round_rows = []
    summaries = []
    current_round = None
    with open(args.diagnostics_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_round = int(row["round"])
            if current_round is not None and row_round != current_round:
                summaries.append(summarize_round(round_rows))
                round_rows = []
            current_round = row_round
            round_rows.append(row)
    if round_rows:
        summaries.append(summarize_round(round_rows))

    segments = []
    for start in range(0, len(summaries), args.segment_size):
        segment = summaries[start : start + args.segment_size]
        gains = np.asarray(
            [item["score_gain_per_task"] for item in segment],
            dtype=np.float64,
        )
        changes = np.asarray(
            [item["changed_tasks"] for item in segment], dtype=np.float64
        )
        segments.append(
            {
                "start_round": int(segment[0]["round"]),
                "end_round": int(segment[-1]["round"]),
                "rounds": len(segment),
                "gain_per_task_mean": float(gains.mean()),
                "gain_per_task_quantiles": {
                    str(q): float(np.quantile(gains, q))
                    for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                },
                "changed_tasks_mean": float(changes.mean()),
            }
        )

    output = {
        "source": os.path.abspath(args.diagnostics_csv),
        "rounds": len(summaries),
        "segments": segments,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(os.path.abspath(args.output))


if __name__ == "__main__":
    main()
