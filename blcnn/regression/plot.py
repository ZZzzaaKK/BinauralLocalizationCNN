"""
Visualize regression model predictions.

Produces three figures:
  1. Hofman-style localization accuracy plot (reuses existing plot_localization_accuracy)
  2. Scatter: true vs predicted for azimuth and elevation
  3. Error distribution histograms

Usage:
    python blcnn/plot_regression.py \
        --csv  data/output/regression_predictions.csv \
        --output-dir data/output/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plotting import plot_localization_accuracy


def load_csv(path: Path) -> np.ndarray:
    """Load CSV and return (N, 4) array: [true_azim, true_elev, pred_azim, pred_elev]."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # Round true labels to nearest integer to avoid float precision issues
    # (they were integers before normalization)
    data[:, :2] = np.round(data[:, :2])
    return data


def plot_scatter(data: np.ndarray, output_path: Path) -> None:
    """Scatter plot of true vs predicted for azimuth and elevation."""
    true_azim, true_elev = data[:, 0], data[:, 1]
    pred_azim, pred_elev = data[:, 2], data[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, true, pred, label, lim in [
        (axes[0], true_azim, pred_azim, "Azimuth (deg)", (-90, 90)),
        (axes[1], true_elev, pred_elev, "Elevation (deg)", (0, 60)),
    ]:
        ax.scatter(true, pred, alpha=0.15, s=4, color="steelblue")
        ax.plot(lim, lim, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        mae = np.mean(np.abs(pred - true))
        ax.set_title(f"{label.split()[0]}   MAE = {mae:.1f}°")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    fig.suptitle("True vs Predicted Location", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved scatter plot: {output_path}")
    plt.close(fig)


def plot_error_histograms(data: np.ndarray, output_path: Path) -> None:
    """Error distribution histograms for azimuth and elevation."""
    azim_err = data[:, 2] - data[:, 0]
    elev_err = data[:, 3] - data[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, err, label, color in [
        (axes[0], azim_err, "Azimuth error (deg)", "steelblue"),
        (axes[1], elev_err, "Elevation error (deg)", "darkorange"),
    ]:
        ax.hist(err, bins=40, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(
            np.mean(err),
            color="red",
            linewidth=1,
            linestyle="-",
            label=f"Mean = {np.mean(err):.1f}°",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        mae = np.mean(np.abs(err))
        std = np.std(err)
        ax.set_title(f"{label.split()[0]}   MAE={mae:.1f}°  SD={std:.1f}°")
        ax.legend(fontsize=8)

    fig.suptitle("Prediction Error Distribution", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved error histogram: {output_path}")
    plt.close(fig)


def plot_hofman(data: np.ndarray, output_path: Path) -> None:
    """Hofman-style localization accuracy plot."""
    plt_obj = plot_localization_accuracy(
        data,
        show_single_responses=False,
        binned=False,
        style="debug",
    )
    ax = plt_obj.gca()
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Mean Predicted Location per Speaker Position")
    fig = plt_obj.gcf()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved Hofman plot: {output_path}")
    plt_obj.clf()
    plt_obj.close()


def main():
    parser = argparse.ArgumentParser(description="Plot regression model predictions")
    parser.add_argument(
        "--csv", required=True, help="Path to regression_predictions.csv"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output plots (default: same folder as CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem
    data = load_csv(csv_path)
    print(f"Loaded {len(data)} predictions")

    azim_mae = np.mean(np.abs(data[:, 2] - data[:, 0]))
    elev_mae = np.mean(np.abs(data[:, 3] - data[:, 1]))
    print(f"Azimuth MAE:   {azim_mae:.2f}°")
    print(f"Elevation MAE: {elev_mae:.2f}°")
    print(f"Combined MAE:  {(azim_mae + elev_mae) / 2:.2f}°")

    plot_scatter(data, out_dir / f"{stem}_scatter.png")
    plot_error_histograms(data, out_dir / f"{stem}_errors.png")
    plot_hofman(data, out_dir / f"{stem}_hofman.png")


if __name__ == "__main__":
    main()
