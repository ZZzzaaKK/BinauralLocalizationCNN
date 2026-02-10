from pathlib import Path

from tqdm import tqdm

import logging

logging.getLogger("keras_tuner").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def main():
    layer_blocks = [[0], [3], [6], [11], [16], [20], [25], [29], [34]]

    for layer_block in tqdm(layer_blocks):
        tune_params(layer_block)


def tune_params(layer_block):
    import tensorflow as tf
    from tensorflow import keras
    import keras_tuner as kt

    print(f'\n\n\n\n\n\n\n\n\nTuning LR for layer block: {layer_block}')

    def single_example_parser(example):
        feature_description = {
            'train/image': tf.io.FixedLenFeature([], tf.string),
            'train/target': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))
        return example['train/image'], example['train/target']

    def model_builder(hp):

        model: keras.Model = keras.models.load_model(Path('models/keras_momentum_9e-1/net1.keras'), compile=False)

        for layer in model.layers:
            layer.trainable = False

        for layer in layer_block:
            model.layers[layer].trainable = True

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])

        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

        return model

    tuner = kt.GridSearch(model_builder,
                          objective='val_sparse_categorical_accuracy',
                          max_trials=8,
                          directory='data/tuning_results',
                          project_name=f'trainable_layers_{"_".join(map(str, layer_block))}',
                          overwrite=False)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=(Path(f'data/tuning_results/trainable_layers_{"_".join(map(str, layer_block))}/logs')).as_posix(),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        write_steps_per_second=True,
        update_freq='epoch')

    path_to_cochleagrams = Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20')

    train_dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'train_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .shuffle(64)
        .batch(16, drop_remainder=True)
        .repeat()
        .prefetch(1)
    )

    test_dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'val_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    try:
        tuner.search(train_dataset, epochs=10, validation_data=test_dataset,
                     callbacks=[tensorboard_callback], steps_per_epoch=830)
    except Exception as e:
        print(f'Exception during tuning: {e}')

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Append LR to common file
    with open(Path('data/tuning_results/tuning_summary.txt'), 'a') as f:
        f.write(f'Layers tuned: {layer_block}, Best learning rate: {best_hps.get("learning_rate")}\n')


if __name__ == '__main__':
    main()
