import pandas as pd

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

    def show_data_dir(self):
        ds_info = self.dataset.get_ds_info()
        if ds_info is None:
            print("Kein ds_info geladen.")
            return
        print("\nData Dir:")
        print(f"  {ds_info.data_dir}")

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

    def show_example_classes(self, n=10):
        if self.class_names is not None:
            print(f"Example Classes: {self.class_names[:n]}")
        else:
            print("Keine Klassennamen geladen.")

    def show_class_distribution(self, split="train"):
        import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow_datasets as tfds
        ds = tfds.load("food101", split=split, as_supervised=True)
        labels = [label.numpy() for _, label in ds]
        labels = np.array(labels)
        plt.figure(figsize=(12, 4))
        plt.hist(labels, bins=len(self.class_names))
        plt.title(f"Klassenverteilung im {split}-Datensatz")
        plt.xlabel("Klasse")
        plt.ylabel("Anzahl Bilder")
        plt.show()

    def show_image_shapes(self, n=100, split="train"):
        import tensorflow_datasets as tfds
        import numpy as np
        ds = tfds.load("food101", split=split, as_supervised=True)
        shapes = []
        for i, (image, _) in enumerate(ds):
            if i >= n:
                break
            shapes.append(image.shape)
        shapes = np.array(shapes)
        print(f"Beispiel-Bildgrößen (erste 10): {shapes[:10]}")
        print(f"Minimale Bildgröße: {shapes.min(axis=0)}")
        print(f"Maximale Bildgröße: {shapes.max(axis=0)}")
