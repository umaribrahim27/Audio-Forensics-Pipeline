# main/src/explainability/render.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from src.dataio.audio_reader import load_audio, AudioConfig
from src.features.mel import mel_spectrogram_db


@dataclass(frozen=True)
class RenderConfig:
    sr: int = 16000
    duration_s: float = 4.0
    hop_length: int = 160
    n_fft: int = 1024
    n_mels: int = 80

    # Visualization
    dpi: int = 160
    overlay_alpha: float = 0.45
    top_quantile: float = 0.90   # used to highlight "suspicious" regions on waveform


def _safe_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    return x / (x.max() + eps)


def _frames_to_seconds(frames: np.ndarray, hop_length: int, sr: int) -> np.ndarray:
    return frames.astype(np.float32) * (hop_length / float(sr))


def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive for contiguous True regions."""
    regions = []
    if mask.size == 0:
        return regions

    in_region = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_region:
            in_region = True
            start = i
        if not v and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, mask.size - 1))
    return regions


def render_mel_base(
    wav_path: Path,
    out_png: Path,
    cfg: RenderConfig = RenderConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Saves a clean mel spectrogram PNG.
    Returns (mel_db [80,T], times_sec [T]).
    """
    y, sr = load_audio(Path(wav_path), AudioConfig(sr=cfg.sr, duration_s=cfg.duration_s))

    mel_db = mel_spectrogram_db(
        y, sr, n_mels=cfg.n_mels, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )  # [80, T]
    T = mel_db.shape[1]
    times = _frames_to_seconds(np.arange(T), cfg.hop_length, sr)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)
    ax.imshow(mel_db, origin="lower", aspect="auto")
    ax.set_title("Mel Spectrogram (dB)")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel bin")
    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)

    return mel_db, times


def render_mel_overlay(
    mel_db: np.ndarray,
    overlay: np.ndarray,
    out_png: Path,
    title: str,
    cfg: RenderConfig = RenderConfig(),
) -> None:
    """
    Saves mel spectrogram with overlay heatmap (same shape [80,T]).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    overlay_n = _safe_norm(overlay)

    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)
    ax.imshow(mel_db, origin="lower", aspect="auto")
    ax.imshow(overlay_n, origin="lower", aspect="auto", alpha=cfg.overlay_alpha)
    ax.set_title(title)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel bin")
    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)


def render_time_importance(
    time_importance: np.ndarray,
    out_png: Path,
    cfg: RenderConfig = RenderConfig(),
    title: str = "Time Importance (Suspiciousness over Time)",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Saves a time importance curve plot.
    Returns (times_sec [T], imp_norm [T]).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    imp = _safe_norm(time_importance)
    T = imp.shape[0]
    times = _frames_to_seconds(np.arange(T), cfg.hop_length, cfg.sr)

    fig = plt.figure(figsize=(10, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(times, imp)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Importance (normalized)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)

    return times, imp


def render_waveform_with_highlights(
    wav_path: Path,
    time_importance: np.ndarray,
    out_png: Path,
    cfg: RenderConfig = RenderConfig(),
    title: str = "Waveform with Highlighted Suspicious Regions",
) -> None:
    """
    Plots waveform and highlights time regions where time_importance is in the top quantile.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(Path(wav_path), AudioConfig(sr=cfg.sr, duration_s=cfg.duration_s))
    y = y.astype(np.float32)

    # time axis for waveform
    t_wave = np.arange(y.shape[0], dtype=np.float32) / float(sr)

    imp = _safe_norm(time_importance)
    thr = float(np.quantile(imp, cfg.top_quantile))
    mask = imp >= thr
    regions = _contiguous_regions(mask)

    # Convert regions from frame indices to seconds
    # frames -> start_sec, end_sec
    frame_times = _frames_to_seconds(np.arange(imp.shape[0]), cfg.hop_length, sr)

    fig = plt.figure(figsize=(10, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(t_wave, y, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    for (a, b) in regions:
        start = float(frame_times[a])
        end = float(frame_times[b] + (cfg.hop_length / float(sr)))
        ax.axvspan(start, end, alpha=0.20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)
