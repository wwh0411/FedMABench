import tensorflow as tf
from android_env.proto.a11y import android_accessibility_forest_pb2

filenames = tf.io.gfile.glob('/GPFS/data/wenhaowang-1/ms-swift/AndroidControl/android_control-android_control-00000-of-00020')
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(raw_dataset)

example = tf.train.Example.FromString(dataset_iterator.get_next().numpy())

forest = android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(example.features.feature['accessibility_trees'].bytes_list.value[0])
print(forest)