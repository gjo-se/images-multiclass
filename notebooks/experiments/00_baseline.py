# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Einleitung
# - <a href="https://colab.research.google.com/github/gjo-se/images-multiclass/blob/master/notebooks/experiments/00_baseline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# - Ziel des Notebooks und des Experiments Kontext und Motivation
# - ganz zum Schluss per ChatGPT erstellen
#

# %% [markdown]
# # Imports & Setup
#

# %%
import os
import subprocess
from src.data import Dataset
from src.eda import EDA
from src.log import SuppressTFLogs
from src.setup import Environment


# %% [markdown]
# ## Clone git on Colab

# %%
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:

    repo_url = "https://github.com/gjo-se/images-multiclass.git"
    target_dir = "/content"
    notebook_dir = "notebooks/experiments/"

    if not os.path.exists(os.path.join(target_dir, "src")):
        print(f"Cloning repository {repo_url} to {target_dir}/tmp_clone ...")
        subprocess.check_call(["git", "clone", repo_url, f"{target_dir}/tmp_clone"])
        for item in os.listdir(f"{target_dir}/tmp_clone"):
            subprocess.check_call(["mv", f"{target_dir}/tmp_clone/{item}", target_dir])
        subprocess.check_call(["rm", "-rf", f"{target_dir}/tmp_clone"])
    else:
        print(f"Projekt bereits in {target_dir} vorhanden.")

    os.chdir(os.path.join(target_dir, notebook_dir))
    print(f"Changed working directory to {os.getcwd()}")
else:
    print("clone_and_cd_repo() wird nur auf Google Colab ausgeführt.")

# %% [markdown]
# ## Setup

# %%
Environment().setup();


# %% [markdown]
# # Dataset
#

# %% [markdown]
# ## Load Dataset

# %%
DATASET_NAME = "food101"
ds = Dataset()
ds.load_dataset(DATASET_NAME, _only_on_colab=False);


# %% [markdown]
# ## Explore Data

# %%
eda = EDA(ds)

# %% [markdown]
# ### Dataset

# %%
features_dict = eda.show_features_dict()
features_dict

# %%
eda.show_features()
eda.show_splits()
eda.show_data_dir()
eda.show_sample_classes()


# %%
with SuppressTFLogs():
    eda.show_class_distribution("train");
    eda.show_class_distribution("validation");


# %% [markdown]
# ### Sample Group

# %%
import tensorflow as tf
with SuppressTFLogs():
    train_one_sample = ds.get_train_ds().take(1)

for image, label in train_one_sample:
  print(f"""
  Shape: {image.shape}
  dtype: {image.dtype}
  Class name tensor: {label}
  Class name string: {ds.get_class_names()[label.numpy()]}
  Min: {tf.reduce_min(image)}
  Max: {tf.reduce_max(image)}
        """)

import matplotlib.pyplot as plt
plt.imshow(image)
plt.title(ds.get_class_names()[label.numpy()]) # add title to image by indexing on class_names list
plt.axis(False);

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

# Mehrere Bilder als Batch visualisieren - nach preprocess
# batch_sample = ds.get_train_ds().batch(9).take(1)
# for images, labels in batch_sample:
#     plt.figure(figsize=(10, 10))
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.title(ds.get_class_names()[labels[i].numpy()])
#         plt.axis("off")
#     plt.show()




# %% [markdown]
# # Modellierung
# Definition und Visualisierung des Modells
# Kompilieren des Modells (Loss, Optimizer, Metriken)
#

# %% [markdown]
# # Training
# Training des Modells mit Trainingsdaten
# Visualisierung des Trainingsverlaufs (Loss, Accuracy)
#

# %% [markdown]
# # Evaluation
# Bewertung des Modells auf Validierungs- und Testdaten
# Darstellung von Metriken und ggf. Confusion Matrix
#

# %% [markdown]
# # Ergebnisse und Interpretation
# Zusammenfassung der wichtigsten Erkenntnisse
# Diskussion von Stärken, Schwächen und möglichen Verbesserungen
#

# %% [markdown]
# # Speicherung und Laden von Modellen
# Speichern des trainierten Modells
# Laden und Testen des gespeicherten Modells
#

# %% [markdown]
# # Fazit und Ausblick
# Kurzes Fazit und mögliche nächste Schritte
#

# %% [markdown]
# # Anhang
# Zusätzliche Visualisierungen, Code-Snippets oder Referenzen

# %%
