# main/src/common/run_config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass(frozen=True)
class RunConfig:
    # audio
    sr: int = 16000
    duration_s: float = 4.0

    # features
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160

    n_mfcc: int = 20
    n_lfcc: int = 20
    n_cqcc: int = 20

    # training
    batch_size: int = 16
    learning_rate: float = 1e-3

    # bookkeeping
    stats_file: str = "stats_train3000_v1.npz"

def save_run_config(cfg: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

def load_run_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return RunConfig(**d)
