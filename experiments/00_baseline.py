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
# - <a href="https://colab.research.google.com/github/gjo-se/images-multiclass/blob/master/experiments/00_baseline.ipynb?flush_cache=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# - Ziel des Notebooks und des Experiments Kontext und Motivation
# - ganz zum Schluss per ChatGPT erstellen
#

# %% [markdown]
# ## Colab Setup

# %%
import os
import subprocess

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
  from google.colab import drive

  REPO_URL = "https://github.com/gjo-se/images-multiclass.git"
  REPO_NAME = "images-multiclass"

  TARGET_DIR = f"/content/{REPO_NAME}"
  MOUNT_DIR = '/content/drive'
  DATA_DIR = f"{MOUNT_DIR}/MyDrive/projects/{REPO_NAME}/data"
  REPO_DATA = f"{TARGET_DIR}/data"

  if not os.path.exists(TARGET_DIR):
      subprocess.check_call(["git", "clone", REPO_URL, TARGET_DIR])
  else:
    print(f"Repository {REPO_NAME} already exists.")

  os.chdir(TARGET_DIR)
  print("Working directory 01:", os.getcwd())

  if os.path.ismount(MOUNT_DIR):
      drive.flush_and_unmount()

  drive.mount(MOUNT_DIR)
  os.makedirs(DATA_DIR, exist_ok=True)

  if os.path.exists(REPO_DATA):
      subprocess.check_call(["rm", "-rf", REPO_DATA])
  os.symlink(DATA_DIR, REPO_DATA)
  print("Symlink:", REPO_DATA, "->", DATA_DIR)

    # if os.path.exists(mount_dir) and os.listdir(mount_dir):
    #     # Falls noch Reste vorhanden sind, alles löschen
    #     import shutil
    #     shutil.rmtree(mount_dir)
    #     os.makedirs(mount_dir)


else:
    print("Google Colab Setup nur Remote ausgeführt.")


# %% [markdown]
# # Imports & Setup
#

# %%
from src.data import Dataset
from src.eda import EDA
from src.log import SuppressTFLogs
from src.setup import Environment


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
ds.load_tfds(DATASET_NAME, _only_on_colab=False, _percent=1)

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
eda.show_random_samples(_count=9)


# %% [markdown]
# ## Preprocess Data

# %%
ds.preprocess_data("train")
ds.preprocess_data("test")


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

# %%

# %%

# %%
