# main/src/features/mfcc.py
from __future__ import annotations
import numpy as np
import librosa

def mfcc_feat(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> np.ndarray:
    X = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return X.astype(np.float32)  # [n_mfcc, T]
