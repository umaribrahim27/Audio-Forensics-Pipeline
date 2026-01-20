# main/src/features/mel.py
from __future__ import annotations
import numpy as np
import librosa

def mel_spectrogram_db(
    y: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)  # [n_mels, T]
