import csv
import datetime
import glob
import logging
import os
import pprint
import time
from pathlib import Path
from typing import List

import coloredlogs
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import RunModelsConfig, get_unique_folder_name, load_config

from blcnn.util import get_model_memory_usage, single_example_parser

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """
    Loads config, disables GPU if needed, and runs the testing for each HRTF label.
    """
    # Disable GPU if needed
    # Note: TF Metal (Apple Silicon) does not guarantee deterministic inference on GPU
    only_cpu = False
    if only_cpu:
        tf.config.set_visible_devices([], "GPU")

    logger.info(f"Physical devices: {tf.config.list_physical_devices()}")
    if only_cpu:
        logger.info("Removing GPU devices to force CPU usage")
    logger.info(f"Visible devices: {tf.config.get_visible_devices()}")

    # Load config
    inference_config = load_config("blcnn/config.yml").run_models
    logger.info(f"Loaded config: {inference_config}")

    # infer_models_folder(inference_config)
    for label in inference_config.labels:
        infer_multiple_models(label, inference_config)


def infer_models_folder(run_models_config: RunModelsConfig) -> None:
    """
    Run the testing for all models in the given folder.
    For now, only uses the first model in the config, but can be extended to use all models.
    """
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    path_to_models = Path(f"models/{run_models_config.folder}")
    path_to_cochleagrams = Path(f"data/cochleagrams/{run_models_config.labels[0]}")

    dest = get_unique_folder_name(f"data/output/{run_models_config.folder}/")
    Path(dest).mkdir(parents=True, exist_ok=False)

    for model_path in path_to_models.glob("*.keras"):
        logger.info(f"Testing model: {model_path}")
        infer_single_model(model_path, path_to_cochleagrams, dest)

    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
    summary = summarize_testing(
        run_models_config, path_to_cochleagrams, timestamp, elapsed_time, dest
    )
    logger.info(summary)
    with open(dest / f"_summary_{timestamp}.txt", "w") as f:
        f.write(summary)


def infer_multiple_models(label: str, run_models_config: RunModelsConfig) -> None:
    """
    Run the testing for one HRTF label using the models specified in the config.
    """
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    path_to_models = Path("models/keras/")
    path_to_cochleagrams = Path(f"data/cochleagrams/{label}")

    dest = get_unique_folder_name(f"data/output/{path_to_cochleagrams.name}/")
    Path(dest).mkdir(parents=True, exist_ok=False)

    for model_id in run_models_config.models_to_use:
        model_path = path_to_models / f"net{model_id}.keras"
        infer_single_model(model_path, path_to_cochleagrams, dest)

    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
    summary = summarize_testing(
        run_models_config, path_to_cochleagrams, timestamp, elapsed_time, dest
    )
    logger.info(summary)
    with open(dest / f"_summary_{timestamp}.txt", "w") as f:
        f.write(summary)


def infer_single_model(
    model_path: Path = None, path_to_cochleagrams: Path = None, dest: Path = None
):
    """
    Test a single model with the given cochleagrams and save the results to a CSV file.
    """
    logger.info(
        f"Testing model: {model_path}, with cochleagrams from: {path_to_cochleagrams}, saving to: {dest}"
    )

    model = keras.models.load_model(model_path)

    total_samples = None
    file_name = glob.glob((path_to_cochleagrams / "_summary_*.txt").as_posix())[0]
    for line in open(file_name, "r"):
        if "Test dataset size (nr of cochleagrams):" in line:
            total_samples = int(line.split(": ")[1].strip())
            break
    if total_samples:
        nr_examples = total_samples
    else:  # Legacy for older data that doesn't have the summary file
        nr_examples = sum(
            1
            for _ in tqdm(
                tf.data.TFRecordDataset(
                    path_to_cochleagrams / "train_cochleagrams.tfrecord",
                    compression_type="GZIP",
                ),
                unit="examples",
                desc="Counting examples",
            )
        )

    # nr_examples = tf.data.TFRecordDataset(path_to_cochleagrams / 'train_cochleagrams.tfrecord', compression_type="GZIP").reduce(np.int64(0), lambda x, _: x + 1)
    logger.info(f"Number of examples in dataset: {nr_examples}")

    # TODO: Better performance by batching Example protos and using parse_example; see if useful
    dataset = (
        tf.data.TFRecordDataset(
            path_to_cochleagrams / "test_cochleagrams.tfrecord", compression_type="GZIP"
        )
        .map(lambda serialized_example: single_example_parser(serialized_example))
        # .shuffle(64)
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    # Predict
    true_classes = []
    pred_classes = []

    for predictions, labels in tqdm(
        predict_with_ground_truth(model, dataset),
        total=nr_examples / 16,
        unit="batches",
    ):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())
        # stim_names.append(names.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    with open(dest / f"{model_path.name.split('.')[0]}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "pred_class"])
        writer.writerows(zip(true_classes, pred_classes))


def summarize_testing(
    run_models_config: RunModelsConfig,
    path_to_cochleagrams: Path,
    timestamp: str,
    elapsed_time: str,
    dest: Path,
) -> str:
    # Load cochleagram summary
    with open(
        glob.glob((path_to_cochleagrams / "_summary_*.txt").as_posix())[0], "r"
    ) as f:
        cochleagram_summary = f.read()

    # TODO: Change summary once freeze training is implemented
    summary = (
        f"##### MODEL TESTING INFO #####\n"
        f"Timestamp: {timestamp}\n"
        f"Total elapsed time: {elapsed_time}\n"
        f"Config:\n{pprint.pformat(run_models_config)}\n\n"
        f"Based on the following cochleagram generation:\n"
        f"{cochleagram_summary}\n\n"
        f"################################\n"
        f"Inference results saved to: {dest}\n"
        f"################################\n"
    )
    return summary


def predict_with_ground_truth(model, dataset):
    """
    Custom prediction loop to yield model predictions and ground truth labels.

    Args:
        model: The trained model.
        dataset: A tf.data.Dataset yielding (inputs, labels) tuples.

    Yields:
        predictions: The model's predictions for the batch.
        labels: The ground truth labels for the batch.
    """
    for batch in dataset:
        inputs, labels = batch  # Extract inputs and labels from the dataset
        predictions = model(inputs, training=False)  # Perform inference
        yield predictions, labels


if __name__ == "__main__":
    main()
