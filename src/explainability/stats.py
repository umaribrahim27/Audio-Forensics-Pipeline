from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ExplainStats:
    top_time_frames: list[int]
    top_time_seconds: list[float]
    time_entropy: float
    time_concentration: float
    top_mel_bins: list[int]
    freq_entropy: float

def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    p = p / (p.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def compute_explain_stats(
    mel_importance: np.ndarray,   # [80,T]
    time_importance: np.ndarray,  # [T]
    hop_length: int = 160,
    sr: int = 16000,
    top_k: int = 10,
) -> ExplainStats:
    T = time_importance.shape[0]

    idx = np.argsort(time_importance)[::-1][:top_k].tolist()
    secs = [float(i * hop_length / sr) for i in idx]

    time_entropy = _entropy(time_importance.clip(min=0))

    k = max(1, int(0.1 * T))
    top_mass = float(np.sort(time_importance)[::-1][:k].sum())
    total_mass = float(time_importance.sum()) + 1e-12
    time_conc = top_mass / total_mass

    freq_imp = mel_importance.mean(axis=1)  # [80]
    top_bins = np.argsort(freq_imp)[::-1][:top_k].tolist()
    freq_entropy = _entropy(freq_imp.clip(min=0))

    return ExplainStats(
        top_time_frames=idx,
        top_time_seconds=secs,
        time_entropy=time_entropy,
        time_concentration=time_conc,
        top_mel_bins=top_bins,
        freq_entropy=freq_entropy,
    )
