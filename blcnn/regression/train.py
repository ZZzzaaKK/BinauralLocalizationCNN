"""
Training script for regression-based binaural sound localization.

This script fine-tunes a pretrained classification model for regression,
predicting continuous azimuth and elevation values.
"""

import argparse
import datetime
import logging
import time
from pathlib import Path

import coloredlogs
import keras
import tensorflow as tf
from data_loader import load_multiple_tfrecords, load_regression_dataset
from net_builder import (
    OutputMode,
    compile_regression_model,
    create_regression_model_from_pretrained,
    load_pretrained_classification_model,
    print_model_summary,
)

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def setup_gpu():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.warning("No GPU found, training will run on CPU")


def create_callbacks(output_dir: Path, patience: int = 10) -> list:
    """
    Create training callbacks.

    Args:
        output_dir: Directory to save checkpoints and logs
        patience: Patience for early stopping

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model (using .weights.h5 format for compatibility)
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # Save latest model (for resuming)
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "latest_model.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=1,
        ),
        # CSV logging
        keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_log.csv"),
        ),
    ]

    return callbacks


def train_regression_model(
    pretrained_model_path: Path,
    train_data_path: Path,
    val_data_path: Path = None,
    output_dir: Path = Path("models/regression"),
    output_mode: OutputMode = "spherical_folded",
    batch_size: int = 16,
    epochs: int = 100,
    learning_rate: float = 0.001,
    freeze_conv_layers: bool = True,
    num_unfrozen_layers: int = None,
    validation_split: float = 0.1,
    max_train_batches: int = None,
    max_val_batches: int = None,
    shuffle_buffer_size: int = 1000,
    preprocessed: bool = False,
) -> keras.Model:
    """
    Train a regression model by fine-tuning a pretrained classification model.

    Args:
        pretrained_model_path: Path to pretrained .keras model
        train_data_path: Path to training TFRecord file or directory
        val_data_path: Path to validation TFRecord (optional)
        output_dir: Directory to save outputs
        output_mode: Coordinate system for outputs
        batch_size: Training batch size
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        freeze_conv_layers: Whether to freeze convolutional layers
        validation_split: Fraction of training data for validation (if no val_data_path)
        max_train_batches: Limit training batches per epoch (None = use all data)
        max_val_batches: Limit validation batches per epoch (None = use all data)
        shuffle_buffer_size: Size of shuffle buffer (smaller = faster startup, less random)

    Returns:
        Trained model
    """
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create output directory
    output_dir = Path(output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load pretrained model
    pretrained_model = load_pretrained_classification_model(pretrained_model_path)

    # Create regression model
    regression_model = create_regression_model_from_pretrained(
        pretrained_model,
        output_mode=output_mode,
        freeze_conv_layers=freeze_conv_layers,
        num_unfrozen_layers=num_unfrozen_layers,
    )

    # Compile model
    regression_model = compile_regression_model(
        regression_model,
        output_mode=output_mode,
        learning_rate=learning_rate,
    )

    # Print model info
    print_model_summary(regression_model)

    # Load data
    train_data_path = Path(train_data_path)

    if train_data_path.is_dir():
        # Load all TFRecord files in directory
        tfrecord_files = list(train_data_path.glob("*.tfrecord"))
        logger.info(f"Found {len(tfrecord_files)} TFRecord files")
        train_dataset = load_multiple_tfrecords(
            tfrecord_files,
            output_mode=output_mode,
            batch_size=batch_size,
            shuffle=True,
            preprocessed=preprocessed,
        )
    else:
        # Single TFRecord file
        full_dataset = load_regression_dataset(
            train_data_path,
            output_mode=output_mode,
            batch_size=batch_size,
            shuffle=True,
            shuffle_buffer_size=shuffle_buffer_size,
            preprocessed=preprocessed,
        )

        # If no separate validation data, split the training data using sharding
        # (more efficient than skip - doesn't need to iterate through skipped data)
        if val_data_path is None and validation_split > 0:
            logger.info(
                f"Splitting dataset: {(1 - validation_split) * 100:.0f}% train, {validation_split * 100:.0f}% validation"
            )

            # Use shard-based split: every Nth sample goes to validation
            # This is much faster than take/skip because it doesn't need to
            # iterate through the skipped data first
            shard_size = int(1.0 / validation_split)  # e.g., 0.1 -> every 10th sample

            # Recreate datasets with sharding (need to reload to avoid consuming iterator)
            val_dataset = load_regression_dataset(
                train_data_path,
                output_mode=output_mode,
                batch_size=batch_size,
                shuffle=False,  # Don't shuffle validation
                shuffle_buffer_size=shuffle_buffer_size,
                preprocessed=preprocessed,
            ).shard(num_shards=shard_size, index=0)  # Take every Nth batch for val

            # Training uses all other shards
            train_dataset = full_dataset.shard(num_shards=shard_size, index=1)

            logger.info(f"Using shard-based split: 1/{shard_size} for validation")
        elif val_data_path is None:
            # No validation split requested
            logger.warning("No validation data - training without validation!")
            train_dataset = full_dataset
            val_dataset = None
        else:
            train_dataset = full_dataset

    # Validation data from separate file
    if val_data_path:
        val_dataset = load_regression_dataset(
            val_data_path,
            output_mode=output_mode,
            batch_size=batch_size,
            shuffle=False,
            preprocessed=preprocessed,
        )

    # Limit batches if specified (useful for quick testing)
    if max_train_batches:
        train_dataset = train_dataset.take(max_train_batches)
        logger.info(f"Limiting training to {max_train_batches} batches per epoch")

    if max_val_batches:
        val_dataset = val_dataset.take(max_val_batches)
        logger.info(f"Limiting validation to {max_val_batches} batches per epoch")

    # Create callbacks
    callbacks = create_callbacks(output_dir)

    # Train
    logger.info("Starting training...")
    history = regression_model.fit(
        train_dataset,
        validation_data=val_dataset if val_dataset is not None else None,
        epochs=epochs,
        callbacks=callbacks
        if val_dataset is not None
        else [
            c
            for c in callbacks
            if not isinstance(
                c, (keras.callbacks.EarlyStopping, keras.callbacks.ReduceLROnPlateau)
            )
        ],
        verbose=1,
    )

    # Save final model
    final_model_path = output_dir / "final_model.keras"
    regression_model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training summary
    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
    summary = f"""
##### REGRESSION TRAINING SUMMARY #####
Timestamp: {timestamp}
Elapsed time: {elapsed_time}

Configuration:
- Pretrained model: {pretrained_model_path}
- Output mode: {output_mode}
- Batch size: {batch_size}
- Initial learning rate: {learning_rate}
- Epochs trained: {len(history.history["loss"])}
- Freeze conv layers: {freeze_conv_layers}

Final metrics:
- Train loss: {history.history["loss"][-1]:.4f}
- Train MAE: {history.history["mae"][-1]:.4f}
- Val loss: {history.history.get("val_loss", [None])[-1]}
- Val MAE: {history.history.get("val_mae", [None])[-1]}

Best val_loss: {min(history.history["val_loss"]) if "val_loss" in history.history else "N/A"}
"""

    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary)

    logger.info(summary)

    return regression_model


def main():
    parser = argparse.ArgumentParser(
        description="Train regression model for sound localization"
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        required=True,
        help="Path to pretrained classification model (.keras)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training TFRecord file or directory",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation TFRecord file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/regression",
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="spherical_folded",
        choices=["spherical", "spherical_folded", "cartesian"],
        help="Output coordinate system",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--unfreeze-conv",
        action="store_true",
        help="Unfreeze convolutional layers (train full model)",
    )
    parser.add_argument(
        "--num-unfrozen-layers",
        type=int,
        default=None,
        help="Only train this many layers from the end (e.g. 3 = just the regression head)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch (for quick testing)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation batches per epoch (for quick testing)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation if no val-data provided (default: 0.1)",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1000,
        help="Shuffle buffer size (smaller = faster startup, less random; default: 1000)",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Data was preprocessed with preprocess_tfrecord.py (already 8kHz + power compressed)",
    )

    args = parser.parse_args()

    setup_gpu()

    train_regression_model(
        pretrained_model_path=Path(args.pretrained_model),
        train_data_path=Path(args.train_data),
        val_data_path=Path(args.val_data) if args.val_data else None,
        output_dir=Path(args.output_dir),
        output_mode=args.output_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_conv_layers=not args.unfreeze_conv,
        num_unfrozen_layers=args.num_unfrozen_layers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        validation_split=args.validation_split,
        shuffle_buffer_size=args.shuffle_buffer_size,
        preprocessed=args.preprocessed,
    )


if __name__ == "__main__":
    main()
