import csv
from pathlib import Path
from typing import List

from blcnn.util import get_unique_folder_name


def main() -> None:
    experiment_alpha()


def experiment_alpha():
    """
    Run inference on models before and after retraining on two different test sets.
    """
    import tensorflow as tf
    import keras

    # Force CPU inference for reproducibility on macOS/Metal.
    # This must happen before the GPU is initialized.
    tf.config.set_visible_devices([], 'GPU')

    dest = get_unique_folder_name('data/output_experiment_alpha')
    Path(dest).mkdir(parents=True, exist_ok=True)

    # Load model before retraining and after retraining
    model_before: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1.keras', compile=False)
    model_after: keras.Model = keras.models.load_model('models/experiment_opt_lr_trained_models/34/model.keras', compile=False)

    # Set all layers of after model to true
    for layer in model_after.layers:
        layer.trainable = True

    model_before.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model_after.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])




    infer_model_online(model_before, Path('data/cochleagrams/naturalsounds165_slab_kemar'), [34], dest, tf, comment='_slab_before')
    infer_model_online(model_after, Path('data/cochleagrams/naturalsounds165_slab_kemar'), [34], dest, tf, comment='_slab_after')
    infer_model_online(model_before, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [34], dest, tf, comment='_nh2_before')
    infer_model_online(model_after, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [34], dest, tf, comment='_nh2_after')


def infer_model_online(model, path_to_cochleagrams: Path, layers_to_train: List[int], dest: Path, tf, comment='') -> None:
    import numpy as np
    from tqdm import tqdm

    def single_example_parser(example):
        feature_description = {
            'train/image': tf.io.FixedLenFeature([], tf.string),
            'train/target': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))
        return example['train/image'], example['train/target']\

    def predict_with_ground_truth(model, dataset):
        for batch in dataset:
            inputs, labels = batch
            predictions = model(inputs, training=False)
            yield predictions, labels

    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'test_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )
    true_classes = []
    pred_classes = []
    for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), unit='batches'):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    print(f'Writing predictions to CSV in {dest / path_to_cochleagrams.name.split(".")[0]}_directsave{comment}.csv')
    with open(f'{dest / path_to_cochleagrams.name.split(".")[0]}_directsave{comment}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


if __name__ == '__main__':
    main()