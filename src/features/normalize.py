# main/src/features/normalize.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

EPS = 1e-8

@dataclass(frozen=True)
class NormStats:
    # mel: per-mel-bin stats over time
    mel_mean: np.ndarray  # [n_mels]
    mel_std: np.ndarray   # [n_mels]

    # seq: per-feature stats over time (MFCC+LFCC+CQCC fused)
    seq_mean: np.ndarray  # [D]
    seq_std: np.ndarray   # [D]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            seq_mean=self.seq_mean,
            seq_std=self.seq_std,
        )

    @staticmethod
    def load(path: Path) -> "NormStats":
        z = np.load(path)
        return NormStats(
            mel_mean=z["mel_mean"].astype(np.float32),
            mel_std=z["mel_std"].astype(np.float32),
            seq_mean=z["seq_mean"].astype(np.float32),
            seq_std=z["seq_std"].astype(np.float32),
        )

def apply_norm(mel: np.ndarray, seq: np.ndarray, stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    """
    mel: [n_mels, T]
    seq: [T, D]
    """
    mel_n = (mel - stats.mel_mean[:, None]) / (stats.mel_std[:, None] + EPS)
    seq_n = (seq - stats.seq_mean[None, :]) / (stats.seq_std[None, :] + EPS)
    return mel_n.astype(np.float32), seq_n.astype(np.float32)

def fit_norm_stats(mels: list[np.ndarray], seqs: list[np.ndarray]) -> NormStats:
    """
    Fit per-dimension mean/std across many utterances.
    mels: list of [n_mels, T]
    seqs: list of [T, D]
    """
    # ---- mel stats over all frames across all utts ----
    mel_cat = np.concatenate([m.T for m in mels], axis=0)  # [sumT, n_mels]
    mel_mean = mel_cat.mean(axis=0).astype(np.float32)
    mel_std = mel_cat.std(axis=0).astype(np.float32)

    # ---- seq stats over all frames across all utts ----
    seq_cat = np.concatenate(seqs, axis=0)  # [sumT, D]
    seq_mean = seq_cat.mean(axis=0).astype(np.float32)
    seq_std = seq_cat.std(axis=0).astype(np.float32)

    # avoid zeros
    mel_std = np.maximum(mel_std, EPS)
    seq_std = np.maximum(seq_std, EPS)

    return NormStats(mel_mean=mel_mean, mel_std=mel_std, seq_mean=seq_mean, seq_std=seq_std)
