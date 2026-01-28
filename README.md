# Fragen
- Wo liegen die Daten? (lokal, colab?)

# Lokal in PyCharm:
python src/train.py

# GIT einrichten
- lokal
- remote
- push

# Colab
- create new Notebook
- Runtime → Change runtime → GPU
- !git clone https://github.com/USER/project.git
- %cd project
- !pip install -r requirements.txt
- Start Training:
  - !python src/train.py

# Notebooks
- Auswertung
- Visualisierung
- keine Training

# Best Practices klären
✅ DO
- Logik in .py
- Notebook = Viewer
- if __name__ == "__main__"

❌ DON’T
- Training in Zellen
- Magics im Code (%pip, %cd)
- GPU-Checks im Notebook

# Typische Upgrades klären
- argparse → Hyperparameter
- wandb oder tensorboard
- torch.save() → Drive / GCS
- Multi-GPU → später trivial

# Typische TF-Fallen klären
✅ DO
- tf.data
- model.save()
- klare Entry Points

❌ DON’T
- tf.compat.v1.Session()
- GPU-Checks im Notebook
- %tensorflow_version
- Monolithisches Notebook

# Nächste sinnvolle Upgrades klären
- argparse → Hyperparameter von Colab setzen
- TensorBoard (in Colab genial)
- tf.keras.callbacks.ModelCheckpoint
- Mixed Precision (tf.keras.mixed_precision)