from pathlib import Path

import tensorflow as tf


def read_names(data_path: Path) -> list[str]:
    """Do a lightweight pass to collect filenames in record order."""
    name_feature = {
        "train/name": tf.io.FixedLenFeature([], tf.string, default_value="")
    }
    names = []
    for raw in tf.data.TFRecordDataset(str(data_path), compression_type="GZIP"):
        parsed = tf.io.parse_single_example(raw, name_feature)
        names.append(parsed["train/name"].numpy().decode("utf-8"))
    return names
