import tensorflow as tf
from configs.config import TrainConfig
from data import get_datasets
from model import build_model

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    else:
        print("Running on CPU")

def train():
    setup_gpu()

    cfg = TrainConfig()
    train_ds, test_ds = get_datasets(cfg.batch_size)

    model = build_model(cfg.num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=cfg.epochs,
    )

    model.save("artifacts/model")

if __name__ == "__main__":
    train()