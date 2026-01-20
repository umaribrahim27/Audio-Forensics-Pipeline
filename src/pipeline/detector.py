# main/src/pipeline/detector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf

from src.dataio.audio_reader import load_audio, AudioConfig
from src.features.mel import mel_spectrogram_db
from src.features.mfcc import mfcc_feat
from src.features.lfcc import lfcc_feat
from src.features.cqcc import cqcc_feat
from src.features.fuse import early_fuse_time
from src.features.normalize import NormStats, apply_norm


@dataclass(frozen=True)
class DetectorConfig:
    # must match training
    sr: int = 16000
    duration_s: float = 4.0
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160
    n_mfcc: int = 20
    n_lfcc: int = 20
    n_cqcc: int = 20


class DeepfakeDetector:
    """
    Reusable inference engine:
      - loads model.keras
      - loads normalization stats
      - extracts features exactly as training did
      - outputs p_fake
    """

    def __init__(self, model_path: Path, stats_path: Path, cfg: DetectorConfig = DetectorConfig()):
        self.model_path = Path(model_path)
        self.stats_path = Path(stats_path)
        self.cfg = cfg

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats not found: {self.stats_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        self.stats = NormStats.load(self.stats_path)

    def _extract_features(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mel [80,T], seq [T,60]) BEFORE channel/batch dims."""
        mel = mel_spectrogram_db(
            y, sr, n_mels=self.cfg.n_mels, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        mfcc = mfcc_feat(
            y, sr, n_mfcc=self.cfg.n_mfcc, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        lfcc = lfcc_feat(
            y, sr, n_lfcc=self.cfg.n_lfcc, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        cqcc = cqcc_feat(
            y, sr, n_cqcc=self.cfg.n_cqcc, hop_length=self.cfg.hop_length
        )
        seq = early_fuse_time(mfcc, lfcc, cqcc)  # [T, 60]

        mel, seq = apply_norm(mel, seq, self.stats)
        return mel, seq

    def preprocess_file(self, audio_path: Path) -> Dict[str, np.ndarray]:
        """Returns model-ready inputs: mel_in [1,80,T,1], seq_in [1,T,60]."""
        y, sr = load_audio(Path(audio_path), AudioConfig(sr=self.cfg.sr, duration_s=self.cfg.duration_s))
        mel, seq = self._extract_features(y, sr)
        x = {
            "mel_in": mel[:, :, None][None, ...].astype(np.float32),
            "seq_in": seq[None, ...].astype(np.float32),
        }
        return x

    def predict_file(self, audio_path: Path) -> float:
        x = self.preprocess_file(audio_path)
        p_fake = float(self.model.predict(x, verbose=0)[0][0])
        return p_fake

    def predict_file_with_debug(self, audio_path: Path) -> Dict[str, Any]:
        """For debugging/demo: returns p_fake + shapes."""
        x = self.preprocess_file(audio_path)
        p_fake = float(self.model.predict(x, verbose=0)[0][0])
        return {
            "p_fake": p_fake,
            "mel_shape": tuple(x["mel_in"].shape),
            "seq_shape": tuple(x["seq_in"].shape),
        }
