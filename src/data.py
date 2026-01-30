import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from src.setup import Environment

class Dataset:
    def __init__(self):
        self.ds_info = None
        self.train_ds = None
        self.test_ds = None
        self.batch_size = None

    def load_dataset(self, _name, _split = ["train", "validation"], _shuffle_files = True, _as_supervised = True, _with_info = True, _show_progress = False, _only_on_colab = True):
        # lokal: ~/tensorflow_datasets/food101/2.0.0
        # colab: /content/tensorflow_datasets/food101/2.0.0

        if _only_on_colab and not Environment.is_colab():
            return None

        if not _show_progress:
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            logging.getLogger("tensorflow_datasets").setLevel(logging.ERROR)

        print(f"Load dataset https://www.tensorflow.org/datasets/catalog/{_name}")

        (train_ds, test_ds), ds_info = tfds.load(
            name = _name,
            split = _split,
            shuffle_files = _shuffle_files,
            as_supervised = _as_supervised,
            with_info = _with_info,
        )
        self.ds_info = ds_info
        self.train_ds = train_ds
        self.test_ds = test_ds

        return train_ds, test_ds, ds_info

    def get_ds_info(self):
        return self.ds_info

    def get_train_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.test_ds

    def get_features_dict(self):
        if self.ds_info is not None:
            return dict(self.ds_info.features)
        return None

    def get_class_names(self, _feature_name='label'):
        if self.ds_info is not None:
            return self.ds_info.features[_feature_name].names
        return None

    def print_example_classes(self, _feature_name = 'label', _count=10):
        if self.ds_info is not None:
            class_names = self.get_class_names()
            print(f"Example Classes: {class_names[:_count]}")
        else:
            print("ds_info ist nicht geladen.")

    # def _preprocess(self, image, label):
    #     image = tf.cast(image, tf.float32) / 255.0
    #     return image, label

        # train_ds = train_ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        # train_ds = train_ds.shuffle(10_000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        # test_ds = test_ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        # test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
