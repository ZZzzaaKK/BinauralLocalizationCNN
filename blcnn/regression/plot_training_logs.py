"""Plot classification vs regression training logs in a comparable way."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if v != "":
                    cols.setdefault(k, []).append(float(v))
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-log", type=str, required=True)
    parser.add_argument("--regression-log", type=str, required=True)
    parser.add_argument("--output", type=str, default="training_comparison.png")
    args = parser.parse_args()

    clf = read_log(Path(args.classification_log))
    reg = read_log(Path(args.regression_log))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: classification loss (own scale) ---
    ax = axes[0]
    ax.plot(clf["epoch"], clf["loss"], label="train")
    ax.plot(clf["epoch"], clf["val_loss"], label="val")
    ax.set_title("Classification (cross-entropy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel 2: regression loss (own scale) ---
    ax = axes[1]
    ax.plot(reg["epoch"], reg["loss"], label="train")
    ax.plot(reg["epoch"], reg["val_loss"], label="val")
    ax.set_title("Regression (MSE, normalized coords)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel 3: relative val loss (comparable convergence view) ---
    ax = axes[2]
    for name, log in [("classification", clf), ("regression", reg)]:
        v = log["val_loss"]
        ax.plot(log["epoch"], [x / v[0] for x in v], label=name)
    ax.set_title("Val loss relative to epoch 0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss / val_loss[0]")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
