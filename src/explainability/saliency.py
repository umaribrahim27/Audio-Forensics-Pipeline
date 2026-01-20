from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

@dataclass(frozen=True)
class SaliencyResult:
    p_fake: float
    mel_saliency: np.ndarray     # [80, T]
    seq_saliency: np.ndarray     # [T, 60]
    time_importance: np.ndarray  # [T]

def _norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.abs(x).astype(np.float32)
    return x / (x.max() + eps)

def compute_saliency(model: tf.keras.Model, x: dict[str, np.ndarray]) -> SaliencyResult:
    mel = tf.convert_to_tensor(x["mel_in"])
    seq = tf.convert_to_tensor(x["seq_in"])

    with tf.GradientTape() as tape:
        tape.watch([mel, seq])
        y = model({"mel_in": mel, "seq_in": seq}, training=False)  # [1,1]
        p_fake = y[0, 0]

    g_mel, g_seq = tape.gradient(p_fake, [mel, seq])

    mel_sal = _norm(g_mel.numpy()[0, :, :, 0])  # [80, T]
    seq_sal = _norm(g_seq.numpy()[0, :, :])     # [T, 60]

    time_imp = 0.5 * (mel_sal.mean(axis=0) + seq_sal.mean(axis=1))  # [T]
    time_imp = _norm(time_imp)

    return SaliencyResult(
        p_fake=float(p_fake.numpy()),
        mel_saliency=mel_sal,
        seq_saliency=seq_sal,
        time_importance=time_imp,
    )
