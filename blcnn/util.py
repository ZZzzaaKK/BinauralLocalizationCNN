import functools
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

import tensorflow as tf
import yaml


@dataclass
class SourcePositionsConfig:
    azimuths: List[int]
    elevations: List[int]


@dataclass
class RoomConfig:
    id: int
    width: float
    length: float
    height: float

    # sort by room_id, may be unnecessary
    def __lt__(self, other):
        return self.id < other.id


@dataclass
class BRIRConfig:
    hrtfs: List[str]
    source_positions: SourcePositionsConfig
    room_configs: List[RoomConfig]
    persist_brirs_individually: bool


@dataclass
class CochleagramConfig:
    hrtf_labels: List[str]
    stim_paths: List[str]
    source_positions: SourcePositionsConfig
    bkgd_path: str
    use_bkgd: bool
    anechoic: bool
    train_test_split: float
    generation_base_probability: float


@dataclass
class FreezeTrainingConfig:
    labels: List[str]
    models_to_use: List[int]
    layer_block_lengths: List[int]


@dataclass
class RunModelsConfig:
    folder: str
    labels: List[str]
    models_to_use: List[int]


@dataclass
class PlottingConfig:
    predictions: List[str]
    data_selection: str
    folded: bool
    binned: bool
    nr_elevation_bins: int
    nr_azimuth_bins: int
    show_single_responses: bool
    style: str


@dataclass
class Config:
    generate_brirs: BRIRConfig
    generate_cochleagrams: CochleagramConfig
    freeze_training: FreezeTrainingConfig
    run_models: RunModelsConfig
    plotting: PlottingConfig


def load_config(file_path: str) -> Config:
    """
    Load the configuration from the given YAML file.
    Args:
        file_path: The path to the YAML file.

    Returns:
        The configuration data class.
    """
    with open(file_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Map the nested YAML dictionary to the data classes
    return Config(
        generate_brirs=BRIRConfig(
            hrtfs=raw_config['generate_brirs']['hrtfs'],
            source_positions=SourcePositionsConfig(
                azimuths=raw_config['generate_brirs']['source_positions']['azimuths'],
                elevations=raw_config['generate_brirs']['source_positions']['elevations']
            ),
            room_configs=[
                RoomConfig(
                    id=room['id'],
                    width=room['width'],
                    length=room['length'],
                    height=room['height']
                ) for room in raw_config['generate_brirs']['room_configs']
            ],
            persist_brirs_individually=raw_config['generate_brirs']['persist_brirs_individually']
        ),
        generate_cochleagrams=CochleagramConfig(
            hrtf_labels=raw_config['generate_cochleagrams']['hrtf_labels'],
            stim_paths=raw_config['generate_cochleagrams']['stim_paths'],
            source_positions=SourcePositionsConfig(
                azimuths=raw_config['generate_cochleagrams']['source_positions']['azimuths'],
                elevations=raw_config['generate_cochleagrams']['source_positions']['elevations']
            ),
            bkgd_path=raw_config['generate_cochleagrams']['bkgd_path'],
            use_bkgd=raw_config['generate_cochleagrams']['use_bkgd'],
            anechoic=raw_config['generate_cochleagrams']['anechoic'],
            train_test_split=raw_config['generate_cochleagrams']['train_test_split'],
            generation_base_probability= raw_config['generate_cochleagrams']['generation_base_probability']
        ),
        freeze_training=FreezeTrainingConfig(
            labels=raw_config['freeze_training']['labels'],
            models_to_use=raw_config['freeze_training']['models_to_use'],
            layer_block_lengths=raw_config['freeze_training']['layer_block_lengths']
        ),
        run_models=RunModelsConfig(
            folder=raw_config['run_models']['folder'],
            labels=raw_config['run_models']['labels'],
            models_to_use=raw_config['run_models']['models_to_use']
        ),
        plotting=PlottingConfig(
            predictions=raw_config['plotting']['predictions'],
            data_selection=raw_config['plotting']['data_selection'],
            folded=raw_config['plotting']['folded'],
            binned=raw_config['plotting']['binned'],
            nr_elevation_bins=raw_config['plotting']['nr_elevation_bins'],
            nr_azimuth_bins=raw_config['plotting']['nr_azimuth_bins'],
            show_single_responses=raw_config['plotting']['show_single_responses'],
            style=raw_config['plotting']['style']
        )
    )


def get_unique_folder_name(base_name):
    """
    Generate a unique folder name. Use the base name if it doesn't exist,
    otherwise append a numeric suffix to ensure uniqueness.
    """
    base_path = Path(base_name)
    if not base_path.exists():
        return base_path  # Return the base name directly if it doesn't exist

    counter = 1
    while (folder_path := base_path.with_name(f"{base_path.stem}_{counter}")).exists():
        counter += 1
    return folder_path


def CNNpos_to_loc(CNN_pos):
    """
    convert bin label in the CNN from Francl 2022 into [azim, elev] positions
    :param CNN_pos: int, [0, 503]
    :return: tuple, (azi, ele)

    # Old conversion to interaural polar coordinates, but might be
    # wrong if original vertical polar coords use ccw wrapping
    if azim >= 180:
        azim -= 360
    """
    div, mod = divmod(CNN_pos, 72)
    azim = mod * 5
    elev = div * 10
    return azim, elev

def loc_to_CNNpos(azim, elev):
    """
    convert [azim, elev] positions into bin label in the CNN from Francl 2022
    :param azim: int, [0, 355]
    :param elev: int, [0, 60]
    :return: int, [0, 503]
    """
    azim = azim % 360  # wrap around
    div = elev // 10
    mod = azim // 5
    return div * 72 + mod


def single_example_parser(example):
    """
    Takes a serialized Example proto and parses it into a tuple of (image, target) tensors.
    Used when loading TFRecord datasets for training and inference.

    Args:
        example: A serialized Example proto containing the features 'train/image' and 'train/target' (for TF2.14) or 'image' and 'target' (for TF2.16).

    Returns:
        A tuple (image, target) where:
        - image: A tensor of shape (39, 8000, 2) containing the cochleagram data.
        - target: A scalar tensor containing the class label (class index 0-503) for the example.

    """

    feature_description = {
        'train/image': tf.io.FixedLenFeature([], tf.string),  # use with TF2.14
        'train/target': tf.io.FixedLenFeature([], tf.int64)  # use with TF2.14

        # 'image': tf.io.FixedLenFeature([], tf.string),  # use with TF2.16
        # 'target': tf.io.FixedLenFeature([], tf.int64)  # use with TF2.16
        }
    example = tf.io.parse_single_example(example, feature_description)

    example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))  # use with TF2.14
    # example['image'] = tf.reshape(tf.io.decode_raw(example['image'], tf.float32), (39, 8000, 2))  # use with TF2.16

    return example['train/image'], example['train/target'] # use with TF2.14
    # return example['image'], example['target']  # use with TF2.16


def persistent_cache(func):
    """
    Simple persistent cache decorator.
    Creates a "cache/" directory if it does not exist and writes the
    caches of the given func to the file "cache/<func-name>.cache"
    """
    file_path = Path(f'cache/{func.__name__}.cache')
    file_path.parent.mkdir(exist_ok=True)
    try:
        with open(file_path, 'r') as f:
            cache = json.load(f)
    except (IOError, ValueError):
        cache = {}

    @functools.wraps(func)
    def wrapper(*args, persistent_cache_key=None, **kwargs):
        """
            :param persistent_cache_key: The key to use for the cache. If None, the arguments of the function are used.
        """
        if persistent_cache_key:
            persistent_cache_key = str(persistent_cache_key)
        else:
            assert args or kwargs, f'Cannot create key without arguments or explicit key. Use persistent_cache_key=<key> or provide other arguments to {func.__name__}()'
            persistent_cache_key = str(args) + str(kwargs)

        if persistent_cache_key not in cache:
            cache[persistent_cache_key] = func(*args, **kwargs)
            with open(file_path, 'w') as f:
                json.dump(cache, f)
        return cache[persistent_cache_key]

    return wrapper


def get_model_memory_usage(batch_size, model):
    """
    Usage: print(get_model_memory_usage(16, create_model(Path('../models/net_weights/net1'))))
    """
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def compute_layer_block_indices(path_to_indices: Path, net_id: int, block_lengths: List[int]) -> list:
    """
    Take a model and return a list containing the indices of consecutive layer blocks for each conv2d layer.
    Additionally return those indices with the Dense layer added.
    """
    with open(path_to_indices, 'r') as f:
        layer_indices = eval(f.read())
    print(f'Loaded layer indices from models/keras/layer_indices.txt: {layer_indices}')

    conv2d_indices = layer_indices[f'net{net_id}']['conv2d']
    dense_index = layer_indices[f'net{net_id}']['dense']

    layer_block_indices = []
    # Get the layer block indices
    for i in range(len(conv2d_indices)):
        for j in range(len(conv2d_indices)):
            if conv2d_indices[i:j + 1]:
                if (j - i + 1) in block_lengths:
                    layer_block_indices.append(conv2d_indices[i:j + 1])

    # Sort the layer blocks by length (shortest first) and then by last index (largest last index first)
    # This way we train from the back and start with small layer blocks
    layer_block_indices.sort(key=lambda x: (len(x), -x[-1]))

    # Add a copy of each layer block with the dense layer added
    for i in range(len(layer_block_indices)):
        layer_block_indices.append(layer_block_indices[i] + [dense_index])

    # Add dense layer only as well
    layer_block_indices.insert(0, [dense_index])

    return layer_block_indices
