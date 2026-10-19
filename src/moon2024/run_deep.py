"""Tables 13 and 14: ten deep networks with self-attention, run on a Kaggle GPU.

  python -m moon2024.run_deep --dataset aep [--models lstm tcn ...] [--epochs 100] [--out /kaggle/working]

Article: Adam (learning rate 0.001), LeakyReLU, self-attention, random state 42,
batch size 24, 100 epochs, many-to-many output of the next 24 hours with only the
24th hour scored. The rest follows the authors' notebook: 14 units per layer,
the previous 24 hours of the 15 inputs as the input sequence, min-max scaling
fitted on the training part, keras-tcn and keras-self-attention. Where the
notebook differs from the article (early stopping, scoring every horizon), the
article is followed and the notebook's scoring is also reported.
"""

import os

# keras-tcn and keras-self-attention are written for Keras 2
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, metrics
from .features import TARGET, TOTAL

SEED = 42
N_PAST = 24
N_FUTURE = 24
UNITS = 14
MODELS = ["lstm", "bilstm", "gru", "bigru", "1d-cnn", "tcn", "lstm-tcn", "bilstm-tcn", "gru-tcn", "bigru-tcn"]


def install_keras2_packages() -> None:
    """keras-tcn, keras-self-attention and the tf-keras release matching the installed TensorFlow.

    Installed without dependencies so Kaggle's TensorFlow is left as it is.
    """
    import subprocess
    import sys
    from importlib.metadata import version

    major, minor = version("tensorflow").split(".")[:2]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"tf-keras=={major}.{minor}.*", "keras-tcn", "keras-self-attention"], check=True)


def scale(train: np.ndarray, *others: np.ndarray):
    """Min-max scaling fitted on the training part."""
    low, high = train.min(axis=0), train.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    return [(a - low) / span for a in (train, *others)], (low, span)


def windows(X: np.ndarray, y: np.ndarray, starts: np.ndarray):
    """For each start i: inputs X[i-24:i] and targets y[i:i+24]."""
    return np.stack([X[i - N_PAST:i] for i in starts]), np.stack([y[i:i + N_FUTURE] for i in starts])


def build(name: str, n_features: int):
    from tensorflow.keras import layers, models
    from keras_self_attention import SeqSelfAttention
    from tcn import TCN

    inputs = layers.Input(shape=(N_PAST, n_features))
    if name == "1d-cnn":
        x = layers.LeakyReLU()(layers.Conv1D(UNITS, kernel_size=2, activation="linear")(inputs))
        x = layers.RepeatVector(N_FUTURE)(layers.Flatten()(x))
        x = layers.LeakyReLU()(layers.LSTM(UNITS, return_sequences=True, activation="linear")(x))
    elif name == "tcn":
        x = layers.LeakyReLU()(TCN(UNITS, return_sequences=False, activation="linear")(inputs))
        x = layers.RepeatVector(N_FUTURE)(x)
        x = layers.LeakyReLU()(TCN(UNITS, return_sequences=True, activation="linear")(x))
    else:
        base, _, decoder_tcn = name.partition("-")
        cell = layers.LSTM if base.endswith("lstm") else layers.GRU
        wrap = layers.Bidirectional if base.startswith("bi") else (lambda layer: layer)
        encoded = wrap(cell(UNITS, return_state=True, activation="linear"))(inputs)
        states = list(encoded[1:])  # every state, so a bidirectional decoder gets both directions
        x = layers.RepeatVector(N_FUTURE)(layers.LeakyReLU()(encoded[0]))
        if decoder_tcn:
            x = layers.LeakyReLU()(TCN(UNITS, return_sequences=True, activation="linear")(x))
        x = wrap(cell(UNITS, return_sequences=True, activation="linear"))(x, initial_state=states)
        x = layers.LeakyReLU()(x)
    x = layers.LeakyReLU()(SeqSelfAttention(attention_activation="linear")(x))
    outputs = layers.TimeDistributed(layers.Dense(1, activation="linear"))(x)
    return models.Model(inputs, outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=data.DATASETS, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    install_keras2_packages()
    import tensorflow as tf
    from tensorflow import keras

    tf.config.experimental.enable_op_determinism()
    train, test = data.load_dataset(args.dataset)
    full = pd.concat([train, test])
    n_train = len(train)
    (X_train, X_full), _ = scale(train[TOTAL].to_numpy(float), full[TOTAL].to_numpy(float))
    (y_train, y_full), (y_low, y_span) = scale(train[[TARGET]].to_numpy(float), full[[TARGET]].to_numpy(float))
    y_full = y_full[:, 0]
    unscale = lambda a: a * y_span[0] + y_low[0]

    # Article: windows over the whole series; train on those whose 24 targets are all in the
    # training part, score the 24th hour of every window whose 24th target is a test hour.
    last = np.arange(N_PAST, len(full) - N_FUTURE + 1)
    train_starts = last[last + N_FUTURE <= n_train]
    test_starts = last[(last + N_FUTURE - 1 >= n_train)]
    Xw_train, yw_train = windows(X_full, y_full, train_starts)
    Xw_test, _ = windows(X_full, y_full, test_starts)
    test_hours = full.iloc[test_starts + N_FUTURE - 1]

    # Authors' notebook: windows within the test part only, every horizon scored
    test_only = np.arange(N_PAST, len(test) - N_FUTURE + 1)
    Xw_code, yw_code = windows(X_full[n_train:], y_full[n_train:], test_only)

    log = {"tensorflow": tf.__version__, "epochs": args.epochs, "train_windows": len(train_starts),
           "test_hours": len(test_starts), "models": {}}
    for name in args.models:
        keras.utils.set_random_seed(SEED)
        model = build(name, len(TOTAL))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
        started = time.time()
        history = model.fit(Xw_train, yw_train[..., None], epochs=args.epochs, batch_size=24, verbose=2)
        seconds = time.time() - started

        y_hat = unscale(model.predict(Xw_test, verbose=0)[:, -1, 0])
        pd.DataFrame({"Date": test_hours.index, "Holi": test_hours["Holi"].to_numpy(),
                      "y": test_hours[TARGET].to_numpy(), "y_hat": y_hat}).to_csv(
            args.out / f"pred_deep_{args.dataset}_{name}.csv", index=False)

        code_hat = unscale(model.predict(Xw_code, verbose=0)[..., 0]).ravel()
        code_y = unscale(yw_code).ravel()
        log["models"][name] = {
            "seconds": round(seconds, 1),
            "final_loss": history.history["loss"][-1],
            "hour24": metrics.scores(test_hours[TARGET], y_hat),
            "notebook_scoring": metrics.scores(code_y, code_hat),
        }
        print(name, json.dumps(log["models"][name]), flush=True)
        (args.out / f"deep_{args.dataset}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
