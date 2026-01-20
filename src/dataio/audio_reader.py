# main/src/dataio/audio_reader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import librosa

@dataclass(frozen=True)
class AudioConfig:
    sr: int = 16000
    duration_s: float = 4.0
    mono: bool = True

def load_audio(path: Path, cfg: AudioConfig) -> tuple[np.ndarray, int]:
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=cfg.mono)
    target_len = int(cfg.sr * cfg.duration_s)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    y = y.astype(np.float32)
    return y, cfg.sr
