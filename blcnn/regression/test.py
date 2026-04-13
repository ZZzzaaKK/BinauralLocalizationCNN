import tensorflow as tf

path = "data/cochleagrams/natural_selection_slab_kemar/test_cochleagrams.tfrecord"
read_features = {""}

for compression in ["GZIP", "ZLIB", ""]:
    ds = tf.data.TFRecordDataset(str(path), compression_type=compression)
    count = sum(1 for _ in ds)
    print(f"compression={repr(compression)}: {count} records")
