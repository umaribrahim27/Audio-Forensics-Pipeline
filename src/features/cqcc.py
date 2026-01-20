# main/src/features/cqcc.py
from __future__ import annotations
import numpy as np
import librosa
from scipy.fftpack import dct

def cqcc_feat(
    y: np.ndarray,
    sr: int,
    n_cqcc: int = 20,
    hop_length: int = 160,
    fmin: float = 32.7,          # ~C1
    bins_per_octave: int = 12,
    n_bins: int = 84,            # 7 octaves * 12
) -> np.ndarray:
    C = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )  # complex [n_bins, T]
    P = (np.abs(C) ** 2).astype(np.float32)
    L = np.log(P + 1e-10)

    # DCT along frequency bins -> cepstra-like
    cep = dct(L, type=2, axis=0, norm="ortho")  # [n_bins, T]
    cep = cep[:n_cqcc, :]
    return cep.astype(np.float32)  # [n_cqcc, T]
