import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from src.setup import Environment
from configs.config import Config

class Dataset:
    def __init__(self):
        self.ds_info = None
        self.train_ds = None
        self.train_ds_processed = None
        self.test_ds = None
        self.test_ds_processed = None
        self.batch_size = None

    def load_dataset(self, _name, _split = ["train", "validation"], _shuffle_files = True, _as_supervised = True, _with_info = True, _show_progress = False, _only_on_colab = True, _data_dir = None):
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
            data_dir = _data_dir,
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

    @staticmethod
    def preprocess_image_size(image, label, target_size=(224, 224)):
        return tf.image.resize(image, target_size), label

    @staticmethod
    def preprocess_image_cast(image, label):
        return tf.cast(image, tf.float32), label

    def preprocess_image(self, image, label):
        image, label = self.preprocess_image_size(image, label)
        image, label = self.preprocess_image_cast(image, label)
        return image, label

    def preprocess_data(self, split="train"):
        if split == "train" and self.train_ds is not None:
            self.train_ds_processed = self.train_ds.map(map_func=self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            self.train_ds_processed = self.train_ds_processed.shuffle(buffer_size=Config.buffer_size)
            self.train_ds_processed = self.train_ds_processed.batch(Config.batch_size)
            self.train_ds_processed = self.train_ds_processed.prefetch(buffer_size=tf.data.AUTOTUNE)
        elif split == "test" and self.test_ds is not None:
            self.test_ds_processed = self.test_ds.map(map_func=self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            self.test_ds_processed = self.test_ds_processed.batch(Config.batch_size)
            self.test_ds_processed = self.test_ds_processed.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            raise ValueError(f"Split '{split}' ist nicht geladen oder existiert nicht.")
