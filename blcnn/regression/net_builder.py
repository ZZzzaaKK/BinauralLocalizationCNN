"""
Regression model builder for binaural sound localization.

This module provides functions to create regression models that predict
continuous azimuth and elevation values instead of discrete class labels.
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import coloredlogs
import keras
import tensorflow as tf
from keras import layers
from tensorflow.python.eager.polymorphic_function.eager_function_run import (
    run_functions_eagerly,
)

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Output mode types
OutputMode = Literal["spherical", "spherical_folded", "cartesian"]


def load_pretrained_classification_model(model_path: Path) -> keras.Model:
    """
    Load a pretrained classification model from disk.

    Args:
        model_path: Path to the .keras model file

    Returns:
        Loaded Keras model
    """
    logger.info(f"Loading pretrained model from: {model_path}")
    return keras.models.load_model(model_path)


def create_regression_model_from_pretrained(
    pretrained_model: keras.Model,
    output_mode: OutputMode = "spherical_folded",
    freeze_conv_layers: bool = True,
    freeze_fc_layers: bool = False,
    num_unfrozen_layers: Optional[int] = None,
) -> keras.Model:
    """
    Convert a pretrained classification model to a regression model.

    This function:
    1. Removes the final classification layer (Dense 504 + softmax)
    2. Adds a new regression head with appropriate outputs
    3. Optionally freezes early layers for fine-tuning

    Args:
        pretrained_model: The pretrained classification model
        output_mode: Type of output coordinates:
            - "spherical": (azimuth, elevation) in degrees, azimuth 0-355, elevation 0-60
            - "spherical_folded": (azimuth, elevation) with azimuth folded to -90 to +90
            - "cartesian": (x, y, z) on unit sphere
        freeze_conv_layers: If True, freeze all convolutional layers
        freeze_fc_layers: If True, freeze fully connected layers (except new head)
        num_unfrozen_layers: If set, only unfreeze this many layers from the end
            (overrides freeze_conv_layers and freeze_fc_layers)

    Returns:
        New Keras model configured for regression
    """
    logger.info(f"Creating regression model with output_mode={output_mode}")

    # Determine number of outputs based on mode
    if output_mode == "cartesian":
        num_outputs = 3  # x, y, z
    else:
        num_outputs = 2  # azimuth, elevation

    # Get all layers except the last one (the 504-class output)
    base_layers = pretrained_model.layers[:-1]

    logger.info(f"Pretrained model has {len(pretrained_model.layers)} layers")
    logger.info(f"Removing final layer: {pretrained_model.layers[-1].name}")
    logger.info(f"Keeping {len(base_layers)} base layers")

    # Build new model using Functional API for more flexibility
    input_shape = pretrained_model.input_shape[1:]  # Remove batch dimension
    inputs = keras.Input(shape=input_shape, name="cochleagram_input")

    # Pass through all base layers
    x = inputs
    for layer in base_layers:
        x = layer(x)

    # Add new regression head
    x = layers.Dense(64, activation="relu", name="regression_hidden")(x)
    x = layers.Dropout(0.3, name="regression_dropout")(x)
    outputs = layers.Dense(num_outputs, activation="linear", name="regression_output")(
        x
    )

    # Create the new model
    regression_model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="binaural_regression",
    )

    # Handle layer freezing
    if num_unfrozen_layers is not None:
        # Freeze all layers except the last num_unfrozen_layers
        for layer in regression_model.layers[:-num_unfrozen_layers]:
            layer.trainable = False
        for layer in regression_model.layers[-num_unfrozen_layers:]:
            layer.trainable = True
        logger.info(f"Unfreezing last {num_unfrozen_layers} layers")
    else:
        # Freeze based on layer type
        for layer in regression_model.layers:
            if freeze_conv_layers and isinstance(
                layer, (layers.Conv2D, layers.ZeroPadding2D, layers.BatchNormalization)
            ):
                layer.trainable = False
            elif (
                freeze_fc_layers
                and isinstance(layer, layers.Dense)
                and "regression" not in layer.name
            ):
                layer.trainable = False

    # Count trainable vs frozen
    trainable_count = sum(1 for layer in regression_model.layers if layer.trainable)
    frozen_count = len(regression_model.layers) - trainable_count
    logger.info(
        f"Model has {trainable_count} trainable layers and {frozen_count} frozen layers"
    )

    return regression_model


def angular_distance_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function that computes angular distance between predictions and targets.
    Handles the wrap-around issue for azimuth.

    Args:
        y_true: Ground truth (azimuth, elevation) in degrees
        y_pred: Predicted (azimuth, elevation) in degrees

    Returns:
        Mean angular distance loss
    """
    azim_true, elev_true = y_true[:, 0], y_true[:, 1]
    azim_pred, elev_pred = y_pred[:, 0], y_pred[:, 1]

    # For azimuth, compute circular distance
    azim_diff = tf.abs(azim_true - azim_pred)
    azim_diff = tf.minimum(azim_diff, 360.0 - azim_diff)

    # For elevation, simple absolute difference
    elev_diff = tf.abs(elev_true - elev_pred)

    # Combine
    total_loss = tf.reduce_mean(azim_diff + elev_diff)

    return total_loss


def get_loss_function(output_mode: OutputMode, use_angular_loss: bool = False):
    """
    Get the appropriate loss function for the output mode.

    Args:
        output_mode: The coordinate system being used
        use_angular_loss: If True, use custom angular distance loss for spherical modes

    Returns:
        Loss function
    """
    if output_mode == "cartesian":
        return keras.losses.MeanSquaredError()
    elif use_angular_loss:
        return angular_distance_loss
    else:
        # For folded spherical, standard MSE is okay since no wrap-around
        return keras.losses.MeanSquaredError()


def compile_regression_model(
    model: keras.Model,
    output_mode: OutputMode = "spherical_folded",
    learning_rate: float = 0.001,
    use_angular_loss: bool = False,
) -> keras.Model:
    """
    Compile the regression model with appropriate loss and metrics.

    Args:
        model: The regression model to compile
        output_mode: The coordinate system being used
        learning_rate: Learning rate for the optimizer
        use_angular_loss: If True, use custom angular distance loss

    Returns:
        Compiled model
    """
    loss_fn = get_loss_function(output_mode, use_angular_loss)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="mse"),
        ],
    )

    logger.info(f"Model compiled with learning_rate={learning_rate}")

    return model


def print_model_summary(model: keras.Model) -> None:
    """Print a summary of the model architecture and trainable status."""
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    print("\nLayer trainable status:")
    print("-" * 80)
    for layer in model.layers:
        status = "TRAINABLE" if layer.trainable else "FROZEN"
        print(f"  {layer.name:40s} {status}")
    print("=" * 80 + "\n")
