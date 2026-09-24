"""Summarize the fixed 300-round SymmetryFedPhoenix mechanism probe."""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "B0": ROOT / "results/history_identity_causal_ablation/B0/round_metrics.csv",
    "Input-HFlip": HERE / "input_hflip_300/cifar10_vgg_FedPhoenix_seed1_input_hflip_300.csv",
    "Model-alt": HERE / "model_alt_300/cifar10_vgg_SymmetryFedPhoenix_seed1_model_alt_300.csv",
}


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:300]
    if len(rows) != 300 or [int(row["round"]) for row in rows] != list(range(1, 301)):
        raise ValueError(f"Expected complete rounds 1–300: {path}")
    if not all(math.isfinite(float(row["test_accuracy"])) for row in rows):
        raise ValueError(f"Non-finite accuracy: {path}")
    return rows


def metrics(rows):
    accuracy = [float(row["test_accuracy"]) for row in rows]
    peak = max(accuracy)
    return {
        "peak_accuracy": peak,
        "peak_round": accuracy.index(peak) + 1,
        "top10_mean": statistics.mean(sorted(accuracy, reverse=True)[:10]),
        "rounds_251_300_mean": statistics.mean(accuracy[250:300]),
        "round_300_accuracy": accuracy[-1],
    }


def main():
    series = {name: load_rows(path) for name, path in SOURCES.items()}
    baseline = series["B0"]
    protocol = {}
    for name in ("Input-HFlip", "Model-alt"):
        rows = series[name]
        protocol[name] = {
            "selected_clients_equal_B0": all(
                a["selected_clients"] == b["selected_clients"]
                for a, b in zip(rows, baseline)
            ),
            "task_seeds_equal_B0": all(
                a["task_seeds"] == b["task_seeds"]
                for a, b in zip(rows, baseline)
            ),
        }
        if not all(protocol[name].values()):
            raise ValueError(f"Sampling/task-seed protocol mismatch: {name}")

    last_view = {}
    f_count = 0
    alt_violations = 0
    for row in series["Model-alt"]:
        clients = json.loads(row["selected_clients"])
        views = json.loads(row["views"])
        if len(clients) != len(views):
            raise ValueError("Client/view count mismatch")
        f_count += views.count("F")
        if not math.isclose(float(row["flip_fraction"]), views.count("F") / len(views)):
            raise ValueError("Flip-fraction diagnostic mismatch")
        if int(row["alt_violations"]) != 0:
            raise ValueError("Reported alt violation")
        for client, view in zip(clients, views):
            if view not in {"I", "F"}:
                raise ValueError("Invalid view")
            if client in last_view and last_view[client] == view:
                alt_violations += 1
            last_view[client] = view
    if f_count == 0 or alt_violations:
        raise ValueError("F view absent or independently detected alt violation")

    summary = {
        "rounds": 300,
        "source_csv": {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
        "metrics": {name: metrics(rows) for name, rows in series.items()},
        "protocol_checks": protocol,
        "model_alt": {
            "f_view_count": f_count,
            "participations": 3000,
            "flip_fraction": f_count / 3000,
            "reported_alt_violations": 0,
            "independent_alt_violations": alt_violations,
        },
    }
    (HERE / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (HERE / "round_accuracy_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "B0", "Input-HFlip", "Model-alt"])
        for index in range(300):
            writer.writerow([index + 1] + [
                series[name][index]["test_accuracy"] for name in SOURCES
            ])

    colors = {"B0": "#334155", "Input-HFlip": "#2563eb", "Model-alt": "#dc2626"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for name, rows in series.items():
        accuracy = [float(row["test_accuracy"]) for row in rows]
        ax.plot(range(1, 301), accuracy, label=name, color=colors[name], linewidth=1.25)
    ax.set(xlim=(1, 300), ylim=(0, 100), xlabel="Communication round",
           ylabel="CIFAR-10 test accuracy (%)",
           title="FedPhoenix horizontal-flip probe (seed 1, first 300 rounds)")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "accuracy_curve_300.png", dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
