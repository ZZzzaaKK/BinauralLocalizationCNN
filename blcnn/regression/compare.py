"""
Compare classification vs regression model predictions.

Usage:
    python blcnn/regression/compare_models.py \
        --classification data/output/net3_predictions_harmonic.csv \
        --regression     data/output/regression_predictions.csv \
        --output-dir     data/output/comparison
"""

import argparse
import logging
from pathlib import Path

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def cnnpos_to_loc(target: int):
    """Invert loc_to_CNNpos: convert bin label back to (azim, elev) in degrees."""
    elev = (target // 72) * 10
    azim = (target % 72) * 5
    return azim, elev


def fold_azimuth(azim: np.ndarray) -> np.ndarray:
    """Fold azimuth from 0-360 to -90 to +90 (front-back collapse)."""
    azim = np.where(azim >= 180, azim - 360, azim)
    azim = np.where(azim > 90, 180 - azim, azim)
    azim = np.where(azim < -90, -180 - azim, azim)
    return azim


def load_classification(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    true_azim, true_elev = zip(*[cnnpos_to_loc(t) for t in df["true_class"]])
    pred_azim, pred_elev = zip(*[cnnpos_to_loc(p) for p in df["pred_class"]])
    return pd.DataFrame(
        {
            "true_azim": fold_azimuth(np.array(true_azim, dtype=float)),
            "true_elev": np.array(true_elev, dtype=float),
            "pred_azim": fold_azimuth(np.array(pred_azim, dtype=float)),
            "pred_elev": np.array(pred_elev, dtype=float),
        }
    )


def load_regression(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_metrics(df: pd.DataFrame) -> dict:
    azim_err = np.abs(df["pred_azim"] - df["true_azim"])
    elev_err = np.abs(df["pred_elev"] - df["true_elev"])
    return {
        "azim_mae": azim_err.mean(),
        "azim_std": azim_err.std(),
        "elev_mae": elev_err.mean(),
        "elev_std": elev_err.std(),
        "combined_mae": (azim_err.mean() + elev_err.mean()) / 2,
        "n": len(df),
    }


def print_metrics(name: str, m: dict):
    logger.info(
        f"\n{'=' * 40}\n"
        f"  {name}\n"
        f"{'=' * 40}\n"
        f"  N samples:     {m['n']}\n"
        f"  Azimuth MAE:   {m['azim_mae']:.2f} ± {m['azim_std']:.2f} deg\n"
        f"  Elevation MAE: {m['elev_mae']:.2f} ± {m['elev_std']:.2f} deg\n"
        f"  Combined MAE:  {m['combined_mae']:.2f} deg\n"
    )


def plot_scatter(ax, df: pd.DataFrame, title: str, coord: str):
    true_col = f"true_{coord}"
    pred_col = f"pred_{coord}"
    ax.scatter(df[true_col], df[pred_col], alpha=0.3, s=10)
    lims = [df[[true_col, pred_col]].min().min(), df[[true_col, pred_col]].max().max()]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel(f"True {coord} (deg)")
    ax.set_ylabel(f"Predicted {coord} (deg)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")


def plot_error_hist(ax, df: pd.DataFrame, title: str, coord: str, color: str):
    err = df[f"pred_{coord}"] - df[f"true_{coord}"]
    ax.hist(err, bins=40, color=color, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction error (deg)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.text(
        0.97,
        0.95,
        f"MAE={np.abs(err).mean():.1f}°\nbias={err.mean():.1f}°",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


def make_plots(clf: pd.DataFrame, reg: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Classification vs Regression Model Comparison", fontsize=14)

    # Scatter plots: true vs predicted
    plot_scatter(axes[0, 0], clf, "Classification — Azimuth", "azim")
    plot_scatter(axes[0, 1], clf, "Classification — Elevation", "elev")
    plot_scatter(axes[1, 0], reg, "Regression — Azimuth", "azim")
    plot_scatter(axes[1, 1], reg, "Regression — Elevation", "elev")

    # Error histograms
    plot_error_hist(
        axes[0, 2], clf, "Classification — Azimuth error", "azim", "steelblue"
    )
    plot_error_hist(
        axes[0, 3], clf, "Classification — Elevation error", "elev", "steelblue"
    )
    plot_error_hist(axes[1, 2], reg, "Regression — Azimuth error", "azim", "darkorange")
    plot_error_hist(
        axes[1, 3], reg, "Regression — Elevation error", "elev", "darkorange"
    )

    plt.tight_layout()
    out = output_dir / "comparison.png"
    plt.savefig(out, dpi=150)
    logger.info(f"Saved plot to {out}")
    plt.close()


def make_bar_chart(clf_m: dict, reg_m: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    width = 0.35
    clf_vals = [clf_m["azim_mae"], clf_m["elev_mae"], clf_m["combined_mae"]]
    reg_vals = [reg_m["azim_mae"], reg_m["elev_mae"], reg_m["combined_mae"]]
    clf_errs = [clf_m["azim_std"], clf_m["elev_std"], 0]
    reg_errs = [reg_m["azim_std"], reg_m["elev_std"], 0]

    ax.bar(
        x - width / 2,
        clf_vals,
        width,
        yerr=clf_errs,
        label="Classification",
        color="steelblue",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        reg_vals,
        width,
        yerr=reg_errs,
        label="Regression",
        color="darkorange",
        capsize=4,
    )

    ax.set_ylabel("MAE (degrees)")
    ax.set_title("Mean Absolute Error by model and coordinate")
    ax.set_xticks(x)
    ax.set_xticklabels(["Azimuth", "Elevation", "Combined"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "mae_comparison.png"
    plt.savefig(out, dpi=150)
    logger.info(f"Saved bar chart to {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare classification vs regression model predictions"
    )
    parser.add_argument(
        "--classification",
        required=True,
        help="Path to classification predictions CSV (true_class, pred_class)",
    )
    parser.add_argument(
        "--regression",
        required=True,
        help="Path to regression predictions CSV (true_azim, true_elev, pred_azim, pred_elev)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/comparison",
        help="Directory to save plots and summary (default: data/output/comparison)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf = load_classification(Path(args.classification))
    reg = load_regression(Path(args.regression))

    clf_m = compute_metrics(clf)
    reg_m = compute_metrics(reg)

    print_metrics("Classification model", clf_m)
    print_metrics("Regression model", reg_m)

    make_plots(clf, reg, output_dir)
    make_bar_chart(clf_m, reg_m, output_dir)

    # Save summary CSV
    summary = pd.DataFrame(
        {
            "metric": [
                "azim_mae",
                "azim_std",
                "elev_mae",
                "elev_std",
                "combined_mae",
                "n",
            ],
            "classification": [
                clf_m[k]
                for k in [
                    "azim_mae",
                    "azim_std",
                    "elev_mae",
                    "elev_std",
                    "combined_mae",
                    "n",
                ]
            ],
            "regression": [
                reg_m[k]
                for k in [
                    "azim_mae",
                    "azim_std",
                    "elev_mae",
                    "elev_std",
                    "combined_mae",
                    "n",
                ]
            ],
        }
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
