# main/src/features/lfcc.py
from __future__ import annotations
import numpy as np
import librosa
from scipy.fftpack import dct

def _linear_filterbank(sr: int, n_fft: int, n_filters: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    # linearly spaced filter edges
    edges = np.linspace(fmin, fmax, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)), dtype=np.float32)

    for i in range(n_filters):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        # rising slope
        l_idx = np.where((freqs >= left) & (freqs <= center))[0]
        if len(l_idx) > 0:
            fb[i, l_idx] = (freqs[l_idx] - left) / max(center - left, 1e-8)
        # falling slope
        r_idx = np.where((freqs >= center) & (freqs <= right))[0]
        if len(r_idx) > 0:
            fb[i, r_idx] = (right - freqs[r_idx]) / max(right - center, 1e-8)

    # normalize filters (optional but helps)
    fb /= (fb.sum(axis=1, keepdims=True) + 1e-8)
    return fb  # [n_filters, n_fft//2+1]

def lfcc_feat(
    y: np.ndarray,
    sr: int,
    n_lfcc: int = 20,
    n_filters: int = 40,
    n_fft: int = 1024,
    hop_length: int = 160,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2

    # magnitude spectrogram (power)
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2  # [F, T]
    S = S[: (n_fft // 2 + 1), :].astype(np.float32)

    fb = _linear_filterbank(sr, n_fft, n_filters, fmin, fmax)  # [n_filters, F]
    E = np.dot(fb, S)  # [n_filters, T]
    E = np.log(E + 1e-10)

    # DCT over filter axis -> cepstra
    C = dct(E, type=2, axis=0, norm="ortho")  # [n_filters, T]
    C = C[:n_lfcc, :]
    return C.astype(np.float32)  # [n_lfcc, T]
