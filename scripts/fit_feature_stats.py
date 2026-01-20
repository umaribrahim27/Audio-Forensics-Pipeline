# main/scripts/fit_feature_stats.py
from __future__ import annotations

import numpy as np

from src.common.paths import ARTIFACTS
from src.dataio.audio_reader import load_audio, AudioConfig
from src.dataio.manifests import load_manifest_csv
from src.features.mel import mel_spectrogram_db
from src.features.mfcc import mfcc_feat
from src.features.lfcc import lfcc_feat
from src.features.cqcc import cqcc_feat
from src.features.fuse import early_fuse_time
from src.features.normalize import fit_norm_stats

def main():
    # Use your balanced training subset
    manifest_path = ARTIFACTS / "manifests" / "train_3000_balanced.csv"
    samples = load_manifest_csv(manifest_path)
    print("Loaded manifest:", manifest_path, "count:", len(samples))

    # Save stats with a name that reflects what they came from
    stats_path = ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = AudioConfig(sr=16000, duration_s=4.0)

    mels: list[np.ndarray] = []
    seqs: list[np.ndarray] = []

    for i, s in enumerate(samples, 1):
        y, sr = load_audio(s.wav_path, cfg)

        mel = mel_spectrogram_db(y, sr)          # [80, T]
        mfcc = mfcc_feat(y, sr)                  # [20, T]
        lfcc = lfcc_feat(y, sr)                  # [20, T]
        cqcc = cqcc_feat(y, sr)                  # [20, T]
        seq = early_fuse_time(mfcc, lfcc, cqcc)  # [T, 60]

        mels.append(mel)
        seqs.append(seq)

        if i % 500 == 0:
            print(f"Processed {i}/{len(samples)}")

    stats = fit_norm_stats(mels, seqs)
    stats.save(stats_path)
    print("Saved stats:", stats_path)

if __name__ == "__main__":
    main()
