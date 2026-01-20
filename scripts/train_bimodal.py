# main/scripts/train_bimodal.py
from __future__ import annotations

import math
import tensorflow as tf

from src.common.paths import ARTIFACTS, CHECKPOINTS
from src.dataio.manifests import load_manifest_csv
from src.features.normalize import NormStats
from src.dataio.dataset_tf import make_dataset, DatasetConfig
from src.models.bimodal_cnn_bilstm import build_bimodal_cnn_bilstm
from src.common.run_config import RunConfig, save_run_config



def main():
    # ---- Paths ----
    train_csv = ARTIFACTS / "manifests" / "train_3000_balanced.csv"
    dev_csv = ARTIFACTS / "manifests" / "dev_800_balanced.csv"
    stats_path = ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"

    out_dir = CHECKPOINTS / "bimodal_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"

    # ---- Load data ----
    train_samples = load_manifest_csv(train_csv, strict=True)
    dev_samples = load_manifest_csv(dev_csv, strict=True)
    stats = NormStats.load(stats_path)

    # ---- Dataset config ----
    batch_size = 16
    train_cfg = DatasetConfig(batch_size=batch_size, shuffle=True)
    dev_cfg = DatasetConfig(batch_size=batch_size, shuffle=False)

    run_cfg_path = out_dir / "run_config.json"
    save_run_config(
        RunConfig(
            batch_size=batch_size,
            learning_rate=1e-3,
            stats_file="stats_train3000_v1.npz",
        ),
        run_cfg_path,
    )
    print("Saved run config:", run_cfg_path)

    # repeat=True so datasets don't exhaust across epochs
    train_ds = make_dataset(train_samples, train_cfg, stats, repeat=True)
    dev_ds = make_dataset(dev_samples, dev_cfg, stats, repeat=True)

    steps_per_epoch = math.ceil(len(train_samples) / batch_size)
    validation_steps = math.ceil(len(dev_samples) / batch_size)

    # Peek one batch to get shapes
    x0, _ = next(iter(train_ds.take(1)))
    mel_shape = tuple(x0["mel_in"].shape[1:])  # (80, T, 1)
    seq_shape = tuple(x0["seq_in"].shape[1:])  # (T, 60)
    print("mel_shape:", mel_shape, "seq_shape:", seq_shape)
    print(f"steps_per_epoch: {steps_per_epoch} | validation_steps: {validation_steps}")

    # ---- Build model ----
    model = build_bimodal_cnn_bilstm(
        mel_shape=mel_shape,
        seq_shape=seq_shape,
        cnn_emb=128,
        lstm_units=128,
        head_units=128,
        dropout=0.3,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # ---- Callbacks ----
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            min_delta=1e-4,
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ---- Train ----
    model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=1,  # this will show x/y like 188/188
    )

    print("Training finished. Best model saved to:", model_path)


if __name__ == "__main__":
    main()
