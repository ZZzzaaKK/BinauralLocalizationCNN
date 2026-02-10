import logging
from pathlib import Path

import coloredlogs
import keras
import tensorflow as tf
from keras import layers, regularizers
import numpy as np

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    convert_tf1_checkpoints_to_tf2_keras_models(Path('models/Binaural Localization Net Weights'))


def convert_tf1_checkpoints_to_tf2_keras_models(path: Path) -> None:
    """Convert TF1 checkpoints downloaded from Dropbox to Keras models in TF2.
    Args:
        path: Path to the directory containing the checkpoints for the 10 models.
    """
    layer_indices = {}

    for i in range(1, 11):
        checkpoint_path = path / f'net{i}/model.ckpt-100000'
        logger.info(f'Converting checkpoint {checkpoint_path} to Keras model...')
        current_layer_indices = _convert_ckpt_to_model(checkpoint_path, f'models/keras/net{i}.keras')
        logger.info(f'Converted checkpoint for net{i} to Keras model at models/keras/net{i}.keras\n')
        layer_indices[f'net{i}'] = current_layer_indices

    with open(Path('models/keras/layer_indices.txt'), 'a') as f:
        f.write(str(layer_indices) + '\n')


def _convert_ckpt_to_model(checkpoint_path: Path, output_path: str) -> None:
    """Converts a TF1 checkpoint to a Keras model in TF2.
    Based on: https://www.tensorflow.org/guide/migrate/migrating_checkpoints#convert_tf1_checkpoint_to_tf2

    Args:
      checkpoint_path: Path to the TF1 checkpoint.
      output_path: Path to save the converted Keras model.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    tf1_variables = tf.train.list_variables(checkpoint_path)

    tf2_model = create_model(Path(checkpoint_path).parent)

    logger.debug(f'TF1 checkpoint variables: {tf1_variables}')
    logger.debug(f'TF2 model trainable variables: {[v.name for v in tf2_model.trainable_variables]}')
    logger.debug(f'TF2 model non-trainable variables: {[v.name for v in tf2_model.non_trainable_variables]}')

    for tf1_name, shape in tf1_variables:
        # Skip optimizer variables bc their format apparently isn't compatible with TF2
        if "/Adam" in tf1_name or tf1_name in {"beta1_power", "beta2_power"}:
            continue

        # Get the TF2 variable name
        tf2_name = _replace_name(tf1_name)

        # Find the corresponding variable in the TF2 model
        # TF2.14 -> e.g. name = 'conv2d_1/kernel:0'
        # But TF2.16 -> e.g. name = 'kernel' ...
        # We can use path in TF2.16 -> path = 'sequential/conv2d_1/kernel'

        tf2_variable = None
        for var in tf2_model.trainable_variables + tf2_model.non_trainable_variables:
            if var.name.split(":")[0] == tf2_name:  # TF2.14
            # if var.path.replace('sequential/', '') == tf2_name:  # TF2.16
                tf2_variable = var
                break

        if tf2_variable is None:
            logger.warning(f"Warning: No matching TF2 variable found for {tf1_name} -> {tf2_name}")
            continue

        # Load the value from the TF1 checkpoint and assign it
        value = reader.get_tensor(tf1_name)
        tf2_variable.assign(value)
        logger.debug(f"Mapped {tf1_name} -> {tf2_name}")

    # Save as Keras model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tf2_model.save(output_path)

    # Extract and save the indices of all Conv2D layers and the Dense layer
    conv2d_indices = []
    for i, layer in enumerate(tf2_model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            conv2d_indices.append(i)
        if layer.name == 'dense':
            dense_index = i
    logger.debug('Conv2D indices:', conv2d_indices)
    logger.debug('Dense index:', dense_index)
    indices = {'conv2d': conv2d_indices,
               'dense': dense_index}

    # Clear the session to reset numbering of layers in the TF2 model
    tf.keras.backend.clear_session()

    return indices


def create_model(ckpt_path: Path) -> keras.Sequential:
    """
    Model construction function ported from TF1 to TF2.
    Iterates over the config array in the given directory and creates a model from it.
    Populates the model with the weights from the checkpoint file and returns it.

    Model reverse engineered in TF2 from TF1 checkpoint files.
    Convolutional layers have custom padding to match the original network.
    Args:
        ckpt_path: Path to the checkpoint directory with the config_array.npy and model.ckpt-100000 (data and index,
                   meta not required) files for one network (e.g. ./resources/net_weights/net1)
    Returns:
        keras.Sequential: Model created from the config array and weights
    """

    logger.info(f'Building network from path: {ckpt_path}')

    config_array = np.load(ckpt_path / 'config_array.npy', allow_pickle=True)

    model = tf.keras.Sequential()
    # The input layer isn't really a layer, but a placeholder tensor w/ same shape as the input data
    # Network receives downsampled data (48kHz to 8kHz)
    model.add(keras.Input(shape=(39, 8000, 2), batch_size=16, name='train/image'))  # use with TF2.14
    # model.add(keras.Input(shape=(39, 8000, 2), batch_size=16, name='image'))  # use with TF2.16

    reg = regularizers.L1(l1=0.001)
    # reg = None

    for layer_config in config_array[0][1:]:
        logger.debug(f'Adding layer_config: {layer_config}')
        if layer_config[0] == 'conv':
            # Input shape for Conv2D must be: (batch_size, imageside1, imageside2, channels)
            # layer_config contains e.g. ['conv', [2, 32, 32], [1, 1]]
            # -> structure: [Conv2D, [kernel_height, kernel_width, filters], [stride_height, stride_width]]

            # Get height of the previous layer_config (39 for first layer_config)
            pl_height = model.layers[-1].get_output_shape_at(0)[1] if model.layers else 39  # use with TF2.14
            # pl_height = model.layers[-1].output.shape[1] if model.layers else 39  # use with TF2.16

            # filters is the dimensionality of the output space (i.e. the number of output filters in the convolution)

            kernel_height, kernel_width, filters = layer_config[1]
            stride_height, stride_width = layer_config[2]

            # Custom padding
            if (pl_height % stride_height == 0):
                pad_along_height = max(kernel_height - stride_height, 0)
            else:
                pad_along_height = max(kernel_height - (pl_height % stride_height), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top

            if pad_along_height == 0:  # or padding == 'SAME': # -> SAME padding is not used for now
                # logger.debug('pad_along_height == 0')
                # Note: data_format='channels_last' is correctly inferred when adding the Conv2D layer_config
                model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width),
                                        strides=(stride_height, stride_width), padding='valid',
                                        kernel_regularizer=reg))
            else:
                # logger.debug(f'pad_along_height != 0. pad_top, pad_bottom = {pad_top, pad_bottom}')
                model.add(layers.ZeroPadding2D(padding=((pad_top, pad_bottom), (0, 0))))
                model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width),
                                        strides=(stride_height, stride_width), padding='valid',
                                        kernel_regularizer=reg))
        elif layer_config[0] == 'relu' or layer_config[0] == 'fc_relu':
            # layer_config contains e.g. ['relu'] or ['fc_relu']
            # -> structure: [ReLU]
            model.add(layers.ReLU())
        elif layer_config[0] == 'bn':
            # layer_config contains e.g. ['bn']
            # -> structure: [BatchNormalization]
            model.add(layers.BatchNormalization())
        elif layer_config[0] == 'pool':
            # layer_config contains e.g. ['pool', [1, 8]]
            # -> structure: [layer_name, [kernel_height, kernel_width]]
            # Non-overlapping kernel strides -> strides = kernel_size
            kernel_size = layer_config[1]
            model.add(layers.MaxPool2D(pool_size=kernel_size, strides=kernel_size, padding='valid'))
        elif layer_config[0] == 'fc':
            # layer_config contains e.g. ['fc', 512]
            # -> structure: [FullyConnected, units]
            # Input shape must be: (batch_size, input_size) -> flatten the output of the previous layer_config first
            # (This is a huge layer_config, comprises about ~90% of the model's parameters by the looks of it)
            model.add(layers.Flatten())
            units = layer_config[1]
            model.add(layers.Dense(units=units, kernel_regularizer=reg))
        elif layer_config[0] == 'fc_bn':
            # layer_config contains e.g. ['fc_bn']
            # -> structure: [BatchNormalization]
            # Original code casts it to filter_dtype=tf.float32 after (in comp. to 'bn')
            # Look at .build() call to see what's passed as filter_dtype
            model.add(layers.BatchNormalization(momentum=0.9))
        elif layer_config[0] == 'dropout':
            # layer_config contains e.g. ['dropout']
            # -> structure: [Dropout]
            model.add(layers.Dropout(rate=0.5))
        elif layer_config[0] == 'out':
            # layer_config contains e.g. ['out']
            # -> structure: [FullyConnected, 504 units]
            model.add(layers.Dense(units=504,
                                   activation='softmax',
                                   kernel_regularizer=reg))
    return model


def _replace_name(name: str) -> str:
    """Replaces the names of the variables in the TF1 checkpoint with the names of the variables in the TF2 model.

    Args:
        name: Name of the variable in the TF1 checkpoint.

    Returns:
        Name of the variable in the TF2 model.
    """
    if 'wc_fc_0' in name:
        return name.replace('wc_fc_0', 'dense/kernel')
    elif 'wb_fc_0' in name:
        return name.replace('wb_fc_0', 'dense/bias')
    elif 'wc_out_0' in name:
        return name.replace('wc_out_0', 'dense_1/kernel')
    elif 'wb_out_0' in name:
        return name.replace('wb_out_0', 'dense_1/bias')
    elif 'wc_' in name:
        layer = name.split('_')[-1]
        if layer == '0':
            return name.replace(f'wc_0', 'conv2d/kernel')
        else:
            return name.replace(f'wc_{layer}', f'conv2d_{layer}/kernel')
    elif 'wb_' in name:
        layer = name.split('_')[-1]
        if layer == '0':
            return name.replace(f'wb_0', 'conv2d/bias')
        else:
            return name.replace(f'wb_{layer}', f'conv2d_{layer}/bias')
    else:
        return name


if __name__ == "__main__":
    main()
