# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
#

# %% [markdown]
# # Imports & Setup
#

# %%
import os
import subprocess

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
from src.setup import SetupEnvironment

SetupEnvironment();


# %% [markdown]
# # Datenvorbereitung
# Laden und Vorverarbeiten der Daten
# Explorative Datenanalyse (EDA) mit Visualisierungen
# Aufteilen in Trainings-, Validierungs- und Testdaten
#

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
