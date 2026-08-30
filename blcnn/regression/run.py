"""
Inference script for the regression-based binaural sound localization model.

Runs a trained regression model on a preprocessed TFRecord and saves
predicted vs. true (azimuth, elevation) to a CSV file.

Usage:
    python blcnn/run_regression.py \
        --model   models/regression/2026-03-01_14-51-17/final_model.keras \
        --data    data/cochleagrams/slab_kemar/cochleagrams_8k.tfrecord \
        --output  data/output/regression_predictions.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Tuple

import coloredlogs
import keras
import numpy as np
import tensorflow as tf
from data_loader import (
    OutputMode,
    load_regression_dataset,
)

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def denormalize(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_mode: OutputMode,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undo the normalization applied during training so outputs are in degrees.

    For spherical_folded: azim in [-1,1] -> [-90,90] deg, elev in [0,1] -> [0,60] deg
    For spherical:        azim in [0,1]  -> [0,360] deg,  elev in [0,1] -> [0,60] deg
    For cartesian:        already unit-sphere x/y/z, returned as-is
    """
    if output_mode == "cartesian":
        return predictions, targets

    if output_mode == "spherical_folded":
        scale = np.array([90.0, 60.0])
    else:  # spherical
        scale = np.array([360.0, 60.0])

    return predictions * scale, targets * scale


def run_inference(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    output_mode: OutputMode = "spherical_folded",
    batch_size: int = 64,
) -> None:
    logger.info(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    logger.info(f"Loading data: {data_path}")
    dataset = load_regression_dataset(
        Path(data_path),
        batch_size=batch_size,
        shuffle=False,
        drop_name=False,
    )

    all_preds = []
    all_targets = []
    all_names = []

    for images, targets, names in dataset:
        preds = model(images, training=False)
        all_preds.append(preds.numpy())
        all_targets.append(targets.numpy())
        all_names.append(names.numpy().astype(str))

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    names = np.concatenate(all_names, axis=0)

    predictions, targets = denormalize(predictions, targets, output_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["filename", "true_azim", "true_elev", "pred_azim", "pred_elev"]
    rows = np.column_stack(
        [
            names,
            targets[:, 0],
            targets[:, 1],
            predictions[:, 0],
            predictions[:, 1],
        ]
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} predictions to {output_path}")

    azim_mae = np.mean(np.abs(predictions[:, 0] - targets[:, 0]))
    elev_mae = np.mean(np.abs(predictions[:, 1] - targets[:, 1]))
    logger.info(f"Azimuth MAE:   {azim_mae:.2f} deg")
    logger.info(f"Elevation MAE: {elev_mae:.2f} deg")
    logger.info(f"Combined MAE:  {(azim_mae + elev_mae) / 2:.2f} deg")


def main():
    parser = argparse.ArgumentParser(
        description="Run regression model inference and save predictions to CSV"
    )
    parser.add_argument(
        "--model", required=True, help="Path to trained regression model (.keras)"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to preprocessed TFRecord (cochleagrams_8k.tfrecord)",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--output-mode",
        default="spherical_folded",
        choices=["spherical", "spherical_folded", "cartesian"],
        help="Output mode used during training (default: spherical_folded)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size (default: 64)"
    )
    args = parser.parse_args()

    run_inference(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        output_mode=args.output_mode,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
