"""
Preprocess cochleagram TFRecords: apply downsampling (48kHz -> 8kHz) and
power compression offline, so training loads data without any heavy CPU work.

Run once per dataset:
    python blcnn/preprocess_tfrecord.py \
        --input  data/cochleagrams/slab_kemar/cochleagrams.tfrecord \
        --output data/cochleagrams/slab_kemar/cochleagrams_8k.tfrecord
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.signal
import tensorflow as tf


def make_fir_filter(
    current_rate: int = 48000,
    new_rate: int = 8000,
    window_size: int = 4097,
    beta: float = 10.06,
) -> np.ndarray:
    ds_ratio = current_rate // new_rate
    times = np.arange(-window_size / 2, int(window_size / 2))
    sinc_response = np.sinc(times / ds_ratio) / ds_ratio
    window = scipy.signal.windows.kaiser(window_size, beta)
    return (window * sinc_response).astype(np.float32)


def process_image(image: np.ndarray, fir: np.ndarray) -> np.ndarray:
    """
    Apply the same downsampling + ReLU + power compression as the training
    data pipeline. Input: (39, 48000, 2) float32. Output: (39, 8000, 2) float32.
    """
    result = np.empty((39, 8000, 2), dtype=np.float32)
    for freq_bin in range(39):
        for ear in range(2):
            channel = image[freq_bin, :, ear]
            downsampled = scipy.signal.upfirdn(fir, channel, up=1, down=6)
            result[freq_bin, :, ear] = np.maximum(downsampled[:8000], 0.0)
    return np.power(result, 0.3)


FEATURES = {
    "train/azim": tf.io.FixedLenFeature([], tf.int64),
    "train/elev": tf.io.FixedLenFeature([], tf.int64),
    "train/image": tf.io.FixedLenFeature([], tf.string),
    "train/image_height": tf.io.FixedLenFeature([], tf.int64),
    "train/image_width": tf.io.FixedLenFeature([], tf.int64),
}


def make_tf_example(azim: int, elev: int, image: np.ndarray) -> bytes:
    feature = {
        "train/azim": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[azim])
        ),
        "train/elev": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[elev])
        ),
        "train/image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.tobytes()])
        ),
        "train/image_height": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[image.shape[0]])
        ),
        "train/image_width": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[image.shape[1]])
        ),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess cochleagram TFRecords (downsample + power compress)"
    )
    parser.add_argument("--input", required=True, help="Input TFRecord path (GZIP)")
    parser.add_argument("--output", required=True, help="Output TFRecord path")
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write uncompressed output (faster to read during training)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    compression = "" if args.no_compress else "GZIP"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path} (compression={compression or 'none'})")

    fir = make_fir_filter()
    dataset = tf.data.TFRecordDataset(str(input_path), compression_type="GZIP")
    writer_options = tf.io.TFRecordOptions(compression_type=compression)

    n = 0
    with tf.io.TFRecordWriter(str(output_path), options=writer_options) as writer:
        for raw in dataset:
            example = tf.io.parse_single_example(raw, FEATURES)
            image = tf.reshape(
                tf.io.decode_raw(example["train/image"], tf.float32), (39, 48000, 2)
            ).numpy()

            processed = process_image(image, fir)
            serialized = make_tf_example(
                int(example["train/azim"]),
                int(example["train/elev"]),
                processed,
            )
            writer.write(serialized)
            n += 1
            if n % 100 == 0:
                print(f"  {n} examples processed...", flush=True)

    print(f"Done. {n} examples written to {output_path}")


if __name__ == "__main__":
    main()
