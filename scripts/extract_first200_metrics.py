#!/usr/bin/env python
"""Extract paired validation/test metrics from FedPhoenix console logs."""

import argparse
import csv
import json
import os
import re


LOSS_RE = re.compile(r"^Testing Loss:\s+([0-9.]+)")
ACCURACY_RE = re.compile(r"^Testing accuracy:\s+([0-9.]+)")


def parse_log(path, eval_every, max_round):
    measurements = []
    pending_loss = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            loss_match = LOSS_RE.match(line)
            if loss_match:
                pending_loss = float(loss_match.group(1))
                continue
            accuracy_match = ACCURACY_RE.match(line)
            if accuracy_match and pending_loss is not None:
                measurements.append(
                    {
                        "loss": pending_loss,
                        "accuracy": float(accuracy_match.group(1)),
                    }
                )
                pending_loss = None

    required_evaluations = max_round // eval_every
    paired_evaluations = min(len(measurements) // 2, required_evaluations)
    rows = []
    for evaluation_index in range(paired_evaluations):
        validation = measurements[2 * evaluation_index]
        test = measurements[2 * evaluation_index + 1]
        rows.append(
            {
                "round": (evaluation_index + 1) * eval_every,
                "validation_accuracy": validation["accuracy"],
                "validation_loss": validation["loss"],
                "test_accuracy": test["accuracy"],
                "test_loss": test["loss"],
            }
        )
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("No paired validation/test measurements found")
    selected = max(
        rows,
        key=lambda row: (
            row["validation_accuracy"],
            -row["validation_loss"],
        ),
    )
    peak = max(rows, key=lambda row: row["test_accuracy"])
    tail = rows[-5:]
    return {
        "evaluations": len(rows),
        "last_round": rows[-1]["round"],
        "final_test_accuracy": rows[-1]["test_accuracy"],
        "peak_test_accuracy": peak["test_accuracy"],
        "peak_test_round": peak["round"],
        "best_validation_accuracy": selected["validation_accuracy"],
        "validation_selected_round": selected["round"],
        "validation_selected_test_accuracy": selected["test_accuracy"],
        "mean_test_accuracy": sum(row["test_accuracy"] for row in rows)
        / len(rows),
        "mean_last5_test_accuracy": sum(
            row["test_accuracy"] for row in tail
        )
        / len(tail),
    }


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must use NAME=PATH")
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--max_round", type=int, default=200)
    parser.add_argument(
        "--output_dir",
        default="results/formal_beta03_first200",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    summaries = {}
    for name, path in args.run:
        rows = parse_log(path, args.eval_every, args.max_round)
        if not rows or rows[-1]["round"] < args.max_round:
            raise ValueError(
                f"{name} has not reached round {args.max_round}: "
                f"last paired round={rows[-1]['round'] if rows else None}"
            )
        for row in rows:
            all_rows.append({"run": name, **row})
        summaries[name] = summarize(rows)

    if args.reference not in summaries:
        raise ValueError("reference must match one of the --run names")
    reference = summaries[args.reference]
    for name, summary in summaries.items():
        summary["delta_peak_vs_reference"] = (
            summary["peak_test_accuracy"]
            - reference["peak_test_accuracy"]
        )
        summary["delta_selected_vs_reference"] = (
            summary["validation_selected_test_accuracy"]
            - reference["validation_selected_test_accuracy"]
        )
        summary["delta_final_vs_reference"] = (
            summary["final_test_accuracy"]
            - reference["final_test_accuracy"]
        )

    csv_path = os.path.abspath(
        os.path.join(args.output_dir, f"round{args.max_round}_curves.csv")
    )
    json_path = os.path.abspath(
        os.path.join(args.output_dir, f"round{args.max_round}_summary.json")
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "reference": args.reference,
                "max_round": args.max_round,
                "eval_every": args.eval_every,
                "runs": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
