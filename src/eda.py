import matplotlib.pyplot as plt
import os
import pandas as pd
from src.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # soll oberhalb TF Import stehen
import tensorflow as tf

class EDA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.ds_info = dataset.get_ds_info()
        self.class_names = dataset.get_class_names()

    def show_features(self):
        print("\nFeatures:")
        for k, v in self.dataset.get_ds_info().features.items():
            print(f"  {k}: {v}")

    def show_splits(self):
        print("\nSplits:")
        for split_name, split_info in self.dataset.get_ds_info().splits.items():
            print(f"  {split_name}: {split_info}")

    def show_sample_classes(self, _feature_name='label', _count=10):
        if self.ds_info is None:
            print("ds_info ist nicht geladen.")
            return
        # class_names = self.ds_info.features[_feature_name].names
        print("\nSample Classes:")
        print(f"  {self.ds_info.features[_feature_name].names[:_count]}")

    def show_data_dir(self):
        if self.ds_info is None:
            print("Kein ds_info geladen.")
            return
        print("\nData Dir:")
        print(f"  {self.ds_info.data_dir}")

    def show_features_dict(self):
        ds_info = self.dataset.get_ds_info()
        if ds_info is None:
            print("Kein ds_info geladen.")
            return None
        info_dict = {
            'name': ds_info.name,
            'full_name': getattr(ds_info, 'full_name', None),
            'size': getattr(ds_info, 'download_size', None),
            'features': ds_info.features,
            'supervised_keys': getattr(ds_info, 'supervised_keys', None),
            'disable_shuffling': getattr(ds_info, 'disable_shuffling', None),
        }

        return pd.DataFrame(list(info_dict.items()), columns=["Attribut", "Wert"])

    def show_class_distribution(self, split="train", figsize=(16, 6)):
        ds = self.dataset.get_train_ds() if split == "train" else self.dataset.get_test_ds()
        labels = [label for _, label in ds]
        labels = tf.stack(labels)
        bincount = tf.math.bincount(labels, minlength=len(self.class_names)).numpy()
        plt.figure(figsize=figsize)
        plt.bar(range(len(self.class_names)), bincount, color="#1f77b4", edgecolor="black", width=0.8)
        plt.title(f"Klassenverteilung im {split}-Datensatz")
        plt.xlabel("Klasse")
        plt.ylabel("Anzahl Bilder")
        plt.tight_layout()
        plt.show()
    #
    # def show_image_shapes(self, n=100, split="train"):
    #     import tensorflow_datasets as tfds
    #     import numpy as np
    #     ds = tfds.load("food101", split=split, as_supervised=True)
    #     shapes = []
    #     for i, (image, _) in enumerate(ds):
    #         if i >= n:
    #             break
    #         shapes.append(image.shape)
    #     shapes = np.array(shapes)
    #     print(f"Beispiel-Bildgrößen (erste 10): {shapes[:10]}")
    #     print(f"Minimale Bildgröße: {shapes.min(axis=0)}")
    #     print(f"Maximale Bildgröße: {shapes.max(axis=0)}")

    def show_random_samples(self, _count=9, split="train", buffer_size=10000, target_size=(224, 224)):
        ds = self.dataset.get_train_ds() if split == "train" else self.dataset.get_test_ds()
        ds = ds.map(lambda img, lbl: Dataset.preprocess(img, lbl, target_size), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True).batch(_count).take(1)
        data = []
        for batch_images, batch_labels in ds:
            for image, label in zip(batch_images, batch_labels):
                data.append({
                    "Shape": tuple(image.shape),
                    "dtype": image.dtype.name,
                    "Class name tensor": int(label.numpy()),
                    "Class name string": self.class_names[label.numpy()],
                    "Min": int(tf.reduce_min(image).numpy()),
                    "Max": int(tf.reduce_max(image).numpy())
                })

        return pd.DataFrame(data)
