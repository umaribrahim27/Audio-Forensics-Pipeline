# main/src/models/bimodal_cnn_bilstm.py
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers as L

def build_bimodal_cnn_bilstm(
    mel_shape: tuple[int, int, int],   # (80, T, 1)
    seq_shape: tuple[int, int],        # (T, 60)
    cnn_emb: int = 128,
    lstm_units: int = 128,
    head_units: int = 128,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """
    Two-branch model:
      - Mel spectrogram -> CNN -> embedding
      - MFCC/LFCC/CQCC fused sequence -> BiLSTM -> embedding
      - concat -> dense -> sigmoid
    """

    # ---- CNN branch (mel) ----
    mel_in = L.Input(shape=mel_shape, name="mel_in")
    x = mel_in

    x = L.Conv2D(16, (3, 3), padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.MaxPool2D((2, 2))(x)

    x = L.Conv2D(32, (3, 3), padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.MaxPool2D((2, 2))(x)

    x = x = L.Conv2D(64, (3, 3), padding="same", name="mel_conv_last")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    cnn_out = L.Dense(cnn_emb, activation="relu", name="cnn_emb")(x)

    # ---- BiLSTM branch (fused cepstral sequence) ----
    seq_in = L.Input(shape=seq_shape, name="seq_in")
    s = seq_in

    s = L.Bidirectional(L.LSTM(lstm_units, return_sequences=True))(s)
    s = L.Dropout(dropout)(s)

    # Temporal pooling: average + max (strong baseline)
    s_avg = L.GlobalAveragePooling1D()(s)
    s_max = L.GlobalMaxPooling1D()(s)
    s = L.Concatenate()([s_avg, s_max])

    s = L.Dense(head_units, activation="relu")(s)
    s = L.Dropout(dropout)(s)
    lstm_out = L.Dense(cnn_emb, activation="relu", name="lstm_emb")(s)

    # ---- Fusion head ----
    z = L.Concatenate(name="fusion")([cnn_out, lstm_out])
    z = L.Dense(head_units, activation="relu")(z)
    z = L.Dropout(dropout)(z)
    out = L.Dense(1, activation="sigmoid", name="p_fake")(z)

    model = tf.keras.Model(inputs={"mel_in": mel_in, "seq_in": seq_in}, outputs=out, name="bimodal_cnn_bilstm")
    return model
