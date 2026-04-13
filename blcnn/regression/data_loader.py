"""
Data loading utilities for regression training.

This module provides functions to load cochleagram data from TFRecords
and convert the labels to regression targets (continuous coordinates).
"""

import logging
from pathlib import Path
from typing import Literal, Tuple

import coloredlogs
import numpy as np
import scipy.signal
import tensorflow as tf


def make_downsample_filt_tensor(
    current_rate: int, new_rate: int, window_size: int, beta: float
) -> tf.Tensor:
    """Build a low-pass FIR filter tensor for downsampling via strided convolution."""
    ds_ratio = current_rate // new_rate
    times = np.arange(-window_size / 2, int(window_size / 2))
    sinc_response = np.sinc(times / ds_ratio) / ds_ratio
    window = scipy.signal.windows.kaiser(window_size, beta)
    filt = (window * sinc_response).astype(np.float32)
    filt_tensor = tf.constant(filt)
    filt_tensor = tf.reshape(filt_tensor, [1, window_size, 1, 1])
    return filt_tensor


logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

OutputMode = Literal["spherical", "spherical_folded", "cartesian"]


def fold_azimuth(azimuth: tf.Tensor) -> tf.Tensor:
    """
    Fold azimuth from 0-360 range to -90 to +90 range.

    This handles the front-back ambiguity by folding sounds from behind
    to the equivalent frontal position.

    Args:
        azimuth: Azimuth in degrees (0-360)

    Returns:
        Folded azimuth in degrees (-90 to +90)
    """
    # First convert to -180 to +180
    azim = tf.where(azimuth >= 180, azimuth - 360, azimuth)

    # Fold back hemisphere to front
    azim = tf.where(azim > 90, 180 - azim, azim)
    azim = tf.where(azim < -90, -180 - azim, azim)

    return azim


def spherical_to_cartesian(
    azimuth: tf.Tensor, elevation: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Convert spherical coordinates to Cartesian coordinates on unit sphere.

    Args:
        azimuth: Azimuth in degrees
        elevation: Elevation in degrees

    Returns:
        Tuple of (x, y, z) tensors
    """
    # Convert to radians
    azim_rad = azimuth * np.pi / 180.0
    elev_rad = elevation * np.pi / 180.0

    # Cartesian coordinates
    # x = right/left, y = front/back, z = up/down
    x = tf.cos(elev_rad) * tf.sin(azim_rad)
    y = tf.cos(elev_rad) * tf.cos(azim_rad)
    z = tf.sin(elev_rad)

    return x, y, z


def create_regression_example_parser(
    output_mode: OutputMode = "spherical_folded",
    normalize_targets: bool = True,
    preprocessed: bool = False,
):
    """
    Create a parser function for TFRecord examples that outputs regression targets.

    Args:
        output_mode: Type of output coordinates:
            - "spherical": (azimuth, elevation) in degrees
            - "spherical_folded": (azimuth, elevation) with azimuth folded to -90 to +90
            - "cartesian": (x, y, z) on unit sphere
        normalize_targets: If True, normalize targets to roughly [-1, 1] range
        preprocessed: If True, expect data already downsampled to 8kHz with power
            compression applied (as produced by preprocess_tfrecord.py). Skips the
            expensive on-the-fly FIR filtering.

    Returns:
        Parser function for use with tf.data.Dataset.map()
    """

    def parser(serialized_example):
        feature_description = {
            "train/image": tf.io.FixedLenFeature([], tf.string),
            "train/target": tf.io.FixedLenFeature([], tf.int64),
            "train/name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        image_processed = tf.reshape(
            tf.io.decode_raw(example["train/image"], tf.float32), (39, 8000, 2)
        )
        target = example["train/target"]
        elev = tf.cast((target // 72) * 10, tf.float32)
        azim = tf.cast((target % 72) * 5, tf.float32)
        name = example["train/name"]
        image, coords = _make_target(
            image_processed, azim, elev, output_mode, normalize_targets
        )
        return image, coords, name

    return parser


def _make_target(
    image_processed: tf.Tensor,
    azim: tf.Tensor,
    elev: tf.Tensor,
    output_mode: OutputMode,
    normalize_targets: bool,
):
    """Convert azim/elev to the requested output format."""
    if output_mode == "spherical":
        if normalize_targets:
            target = tf.stack([azim / 360.0, elev / 60.0])
        else:
            target = tf.stack([azim, elev])

    elif output_mode == "spherical_folded":
        azim_folded = fold_azimuth(azim)
        if normalize_targets:
            target = tf.stack([azim_folded / 90.0, elev / 60.0])
        else:
            target = tf.stack([azim_folded, elev])

    elif output_mode == "cartesian":
        x, y, z = spherical_to_cartesian(azim, elev)
        target = tf.stack([x, y, z])

    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    return image_processed, target


def load_regression_dataset(
    tfrecord_path: Path,
    output_mode: OutputMode = "spherical_folded",
    batch_size: int = 16,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    normalize_targets: bool = True,
    preprocessed: bool = False,
) -> tf.data.Dataset:
    """
    Load a TFRecord file and create a dataset for regression training.

    Args:
        tfrecord_path: Path to the TFRecord file
        output_mode: Type of output coordinates
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        shuffle_buffer_size: Size of the shuffle buffer
        normalize_targets: Whether to normalize target values
        preprocessed: If True, data was preprocessed with preprocess_tfrecord.py
            (already at 8kHz with power compression). Skips on-the-fly FIR filtering.

    Returns:
        tf.data.Dataset yielding (cochleagram, target) tuples
    """
    logger.info(f"Loading dataset from: {tfrecord_path}")
    logger.info(
        f"Output mode: {output_mode}, batch_size: {batch_size}, preprocessed: {preprocessed}"
    )

    compression = "GZIP"
    parser = create_regression_example_parser(
        output_mode, normalize_targets, preprocessed
    )

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type=compression)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_multiple_tfrecords(
    tfrecord_paths: list,
    output_mode: OutputMode = "spherical_folded",
    batch_size: int = 16,
    shuffle: bool = True,
    normalize_targets: bool = True,
    preprocessed: bool = False,
) -> tf.data.Dataset:
    """
    Load multiple TFRecord files and combine into a single dataset.

    Args:
        tfrecord_paths: List of paths to TFRecord files
        output_mode: Type of output coordinates
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        normalize_targets: Whether to normalize target values
        preprocessed: If True, data was preprocessed with preprocess_tfrecord.py

    Returns:
        Combined tf.data.Dataset
    """
    compression = "GZIP"
    parser = create_regression_example_parser(
        output_mode, normalize_targets, preprocessed
    )

    # Interleave multiple files for better shuffling
    files = tf.data.Dataset.from_tensor_slices([str(p) for p in tfrecord_paths])

    if shuffle:
        files = files.shuffle(len(tfrecord_paths))

    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression),
        cycle_length=min(4, len(tfrecord_paths)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
