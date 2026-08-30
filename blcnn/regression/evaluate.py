"""
Evaluate and compare classification vs regression models on a common
held-out set, reporting errors in degrees for both.
"""

import argparse
from pathlib import Path

import keras
import numpy as np
from data_loader import load_regression_dataset


def fold_azimuth_np(azim: np.ndarray) -> np.ndarray:
    """Fold azimuth from 0-360 to -90..+90 (numpy mirror of data_loader.fold_azimuth)."""
    azim = np.where(azim >= 180, azim - 360, azim)
    azim = np.where(azim > 90, 180 - azim, azim)
    azim = np.where(azim < -90, -180 - azim, azim)
    return azim


def class_to_degrees(class_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode 504-class label -> (azimuth 0-355, elevation 0-60) in degrees."""
    elev = (class_ids // 72) * 10.0
    azim = (class_ids % 72) * 5.0
    return azim, elev


def angular_distance_deg(az1, el1, az2, el2) -> np.ndarray:
    """Great-circle distance in degrees between two spherical directions."""
    az1, el1, az2, el2 = (np.radians(x) for x in (az1, el1, az2, el2))
    cos_d = np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def report(name: str, azim_true, elev_true, azim_pred, elev_pred) -> None:
    azim_err = np.abs(azim_true - azim_pred)
    elev_err = np.abs(elev_true - elev_pred)
    ang_err = angular_distance_deg(azim_pred, elev_pred, azim_true, elev_true)

    print(f"\n===== {name} =====")
    print(f"  Samples:                {len(azim_err)}")
    print(f"  Azimuth MAE:            {azim_err.mean():7.2f}°   (median {np.median(azim_err):.2f}°)")
    print(f"  Elevation MAE:          {elev_err.mean():7.2f}°   (median {np.median(elev_err):.2f}°)")
    print(f"  Angular distance mean:  {ang_err.mean():7.2f}°   (median {np.median(ang_err):.2f}°)")
    print(f"  Within 5°:  {(ang_err <= 5).mean() * 100:5.1f}%")
    print(f"  Within 10°: {(ang_err <= 10).mean() * 100:5.1f}%")
    print(f"  Within 20°: {(ang_err <= 20).mean() * 100:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare classification vs regression models")
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--classification-model", type=str, default=None)
    parser.add_argument("--regression-model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    if not args.classification_model and not args.regression_model:
        parser.error("Provide at least one of --classification-model / --regression-model")

    # One dataset with raw class labels as ground truth for both models
    dataset = load_regression_dataset(
        Path(args.test_data),
        output_mode="classification",
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.max_batches:
        dataset = dataset.take(args.max_batches)

    # Collect ground truth once
    labels = np.concatenate([y.numpy() for _, y in dataset])
    azim_true_raw, elev_true = class_to_degrees(labels)
    azim_true = fold_azimuth_np(azim_true_raw)

    images_only = dataset.map(lambda x, y: x)

    if args.classification_model:
        # compile=False avoids needing the original loss/metrics to deserialize
        model = keras.models.load_model(args.classification_model, compile=False)
        logits = model.predict(images_only, verbose=1)
        pred_class = np.argmax(logits, axis=-1)
        azim_pred_raw, elev_pred = class_to_degrees(pred_class)
        azim_pred = fold_azimuth_np(azim_pred_raw)
        report("CLASSIFICATION", azim_true, elev_true, azim_pred, elev_pred)
        print(f"  Exact-class accuracy:   {(pred_class == labels).mean() * 100:.1f}%")

    if args.regression_model:
        model = keras.models.load_model(args.regression_model, compile=False)
        preds = model.predict(images_only, verbose=1)
        # TODO: Remove these prints after fixing error
        print(preds[:10])
        print("raw preds min/max:", preds.min(0), preds.max(0))
        print("raw preds mean:", preds.mean(0))
        print("raw preds std:", preds.std(0))

        # Denormalize spherical_folded targets (azim/90, elev/60 in data_loader)
        azim_pred = preds[:, 0] * 90.0
        elev_pred = preds[:, 1] * 60.0
        report("REGRESSION (spherical_folded)", azim_true, elev_true, azim_pred, elev_pred)


if __name__ == "__main__":
    main()
