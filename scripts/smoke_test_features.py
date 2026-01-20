# main/scripts/smoke_test_features.py
from pathlib import Path

from src.common.paths import DATA_LA
from src.dataio.protocols import build_manifest_la
from src.dataio.audio_reader import load_audio, AudioConfig
from src.features.mel import mel_spectrogram_db
from src.features.mfcc import mfcc_feat
from src.features.lfcc import lfcc_feat
from src.features.cqcc import cqcc_feat
from src.features.fuse import early_fuse_time

def main():
    samples = build_manifest_la(DATA_LA, split="train")
    print("Found samples:", len(samples))
    s = samples[0]
    print("Example:", s.utt_id, s.label, s.wav_path)

    y, sr = load_audio(s.wav_path, AudioConfig(sr=16000, duration_s=4.0))
    mel = mel_spectrogram_db(y, sr)
    mfcc = mfcc_feat(y, sr)
    lfcc = lfcc_feat(y, sr)
    cqcc = cqcc_feat(y, sr)
    fused = early_fuse_time(mfcc, lfcc, cqcc)

    print("mel:", mel.shape)       # [n_mels, T]
    print("mfcc:", mfcc.shape)     # [n_mfcc, T]
    print("lfcc:", lfcc.shape)     # [n_lfcc, T]
    print("cqcc:", cqcc.shape)     # [n_cqcc, T]
    print("fused:", fused.shape)   # [T, n_mfcc+n_lfcc+n_cqcc]

if __name__ == "__main__":
    main()
