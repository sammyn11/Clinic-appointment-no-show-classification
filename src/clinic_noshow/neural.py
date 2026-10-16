"""Keras models fed by tf.data pipelines."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from clinic_noshow.config import SEED


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def class_weight_map(labels: np.ndarray) -> dict[int, float]:
    n = len(labels)
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(n - int(labels.sum()), 1)
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def make_flat_dataset(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    data = tf.data.Dataset.from_tensor_slices(
        (features.astype(np.float32), labels.astype(np.float32))
    )
    if shuffle:
        data = data.shuffle(buffer_size=len(labels), seed=SEED, reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_embedding_dataset(
    numeric: np.ndarray,
    neighbourhood: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    inputs = {
        "numeric": numeric.astype(np.float32),
        "neighbourhood": neighbourhood.astype(np.int32),
    }
    data = tf.data.Dataset.from_tensor_slices((inputs, labels.astype(np.float32)))
    if shuffle:
        data = data.shuffle(buffer_size=len(labels), seed=SEED, reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def scale_features(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train).astype(np.float32),
        scaler.transform(val).astype(np.float32),
        scaler.transform(test).astype(np.float32),
        scaler,
    )


def build_shallow(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="shallow_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


def build_regularized(n_features: int) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="regularized_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
    )
    return model


def build_embedding(n_numeric: int, n_neighbourhoods: int) -> tf.keras.Model:
    """Neighbourhood is an index learned as an embedding, not a hand-built rate."""
    numeric = tf.keras.layers.Input(shape=(n_numeric,), name="numeric")
    neighbourhood = tf.keras.layers.Input(shape=(), dtype="int32", name="neighbourhood")
    embedded = tf.keras.layers.Embedding(n_neighbourhoods, 8, name="neighbourhood_embedding")(neighbourhood)
    embedded = tf.keras.layers.Flatten()(embedded)
    hidden = tf.keras.layers.Dense(64, activation="relu")(numeric)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.20)(hidden)
    joined = tf.keras.layers.Concatenate()([hidden, embedded])
    joined = tf.keras.layers.Dense(32, activation="relu")(joined)
    joined = tf.keras.layers.Dropout(0.20)(joined)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(joined)
    model = tf.keras.Model(inputs={"numeric": numeric, "neighbourhood": neighbourhood}, outputs=output, name="embedding_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


class ValidationAUC(tf.keras.callbacks.Callback):
    """Epoch-wise ROC-AUC on a fixed validation set, plus early stopping on that AUC."""

    def __init__(self, predict_fn, patience: int = 4):
        super().__init__()
        self.predict_fn = predict_fn
        self.patience = patience
        self.best = -np.inf
        self.wait = 0
        self.best_weights = None
        self.history = {"loss": [], "val_loss": [], "val_auc": []}

    def on_epoch_end(self, epoch, logs=None):
        from clinic_noshow.evaluate import safe_auc

        logs = logs or {}
        scores, labels = self.predict_fn()
        auc = safe_auc(labels, scores)
        self.history["loss"].append(float(logs.get("loss", np.nan)))
        self.history["val_loss"].append(float(logs.get("val_loss", np.nan)))
        self.history["val_auc"].append(float(auc))
        if auc > self.best + 1e-4:
            self.best = auc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def predict_flat(model: tf.keras.Model, features: np.ndarray) -> np.ndarray:
    dataset = tf.data.Dataset.from_tensor_slices(features.astype(np.float32)).batch(1024)
    return model.predict(dataset, verbose=0).ravel()


def predict_embedding(model: tf.keras.Model, numeric: np.ndarray, neighbourhood: np.ndarray) -> np.ndarray:
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "numeric": numeric.astype(np.float32),
            "neighbourhood": neighbourhood.astype(np.int32),
        }
    ).batch(1024)
    return model.predict(dataset, verbose=0).ravel()
