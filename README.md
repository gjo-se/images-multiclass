# GIT einrichten
- lokal
- remote

# Sync ipynb <=> py

Um Jupyter-Notebooks (.ipynb) und Python-Skripte (.py) synchron zu halten, wird [Jupytext](https://github.com/mwouts/jupytext) verwendet:

1. **Jupytext installieren**
   ```zsh
   pip install jupytext
   ```

2. **Pairing einrichten** (z.B. percent-Format):
   ```zsh
   jupytext --set-formats ipynb,py:percent notebooks/experiments/00_baseline.ipynb
   jupytext --set-formats ipynb,py:percent notebooks/experiments/*.ipynb
   ```
   Dadurch wird automatisch eine .py-Datei erstellt und das Pairing in den Metadaten des Notebooks hinterlegt.

3. **Synchronisieren** (beide Richtungen):
   ```zsh
   jupytext --sync notebooks/experiments/00_baseline.ipynb
   jupytext --sync notebooks/experiments/*.ipynb
   ```
   Änderungen in .ipynb und .py werden so abgeglichen.

# Colab
- create new Notebook
- Runtime → Change runtime → GPU
- Start Training:
  - !python src/train.py

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