from tqdm import tqdm
from pathlib import Path


def main() -> None:
    configurations = [[34], [29], [25], [20], [16], [11], [6], [3], [0],
                      [25, 29], [20, 25], [16, 20], [11, 16], [6, 11], [3, 6], [0, 3],
                      [20, 25, 29], [16, 20, 25], [11, 16, 20], [6, 11, 16], [3, 6, 11], [0, 3, 6],
                      # And with 34 added everywhere
                      [34, 29], [34, 25], [34, 20], [34, 16], [34, 11], [34, 6], [34, 3], [34, 0],
                      [34, 25, 29], [34, 20, 25], [34, 16, 20], [34, 11, 16], [34, 6, 11], [34, 3, 6],
                      [34, 0, 3],
                      [34, 20, 25, 29], [34, 16, 20, 25], [34, 11, 16, 20], [34, 6, 11, 16],
                      [34, 3, 6, 11], [34, 0, 3, 6]]

    print(f'Total configurations: {len(configurations)}')
    print(f'Configurations: {configurations}')
    Path(f'models/experiment_fixed_lr_trained_models').mkdir(parents=True, exist_ok=True)

    for config in tqdm(configurations):
        print(f'\n\n\n\n\n\n\n\n\nRunning fixed LR experiment for configuration: {config}')
        run_experiment_fixed_lr(config)


def run_experiment_fixed_lr(config):
    import tensorflow as tf
    from tensorflow import keras
    import keras_tuner as kt

    def single_example_parser(example):
        feature_description = {
            'train/image': tf.io.FixedLenFeature([], tf.string),
            'train/target': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))
        return example['train/image'], example['train/target']

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=(Path(f'models/experiment_fixed_lr_trained_models/{"_".join(map(str, config))}/logs')).as_posix(),
        histogram_freq=0,
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

    validation_dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'test_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    model: keras.Model = keras.models.load_model(Path('models/keras_momentum_9e-1/net1.keras'), compile=False)

    for layer in model.layers:  # Freeze all layers
        layer.trainable = False

    for layer in config:
        model.layers[layer].trainable = True

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    try:
        model.fit(train_dataset,
                  epochs=50,
                  steps_per_epoch=830,
                  validation_data=validation_dataset,
                  callbacks=[stop_early, tensorboard_callback])
        model_save_path = Path(f'models/experiment_fixed_lr_trained_models/{"_".join(map(str, config))}/model.keras')
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_save_path)
    except Exception as e:
        print(f'Error during training: {e}')

if __name__ == "__main__":
    main()