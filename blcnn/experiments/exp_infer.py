import os
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    # Specify which experiment to run inference on
    folder = Path('models/experiment_fixed_lr_trained_models')
    # folder = Path('models/experiment_opt_lr_trained_models')

    for config_folder in tqdm(folder.iterdir(), desc='Evaluating fixed LR models', total=45):
        if config_folder.is_dir():
            if 'logs' in config_folder.name:
                continue
            load_model_and_predict(config_folder / 'model.keras')


def load_model_and_predict(config_folder):
    import csv
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    from tqdm import tqdm

    only_cpu = True
    if only_cpu:
        # Force CPU inference for reproducibility.
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

    def single_example_parser(example):
        feature_description = {
            'train/image': tf.io.FixedLenFeature([], tf.string),
            'train/target': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))
        return example['train/image'], example['train/target']

    def predict_with_ground_truth(model, dataset):
        for batch in dataset:
            inputs, labels = batch
            predictions = model(inputs, training=False)
            yield predictions, labels

    print(f'\nEvaluating experiment for configuration: {config_folder.parent.name}')
    model: keras.Model = keras.models.load_model(config_folder.as_posix(), compile=False)

    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    path_to_cochleagrams = Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20')

    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'test_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    # Predict
    true_classes = []
    pred_classes = []
    for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), unit='batches', total=206):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    with open(f'{config_folder.parent}/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


if __name__ == "__main__":
    main()
