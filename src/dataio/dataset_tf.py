# main/src/dataio/dataset_tf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import tensorflow as tf

from src.dataio.audio_reader import load_audio, AudioConfig
from src.features.mel import mel_spectrogram_db
from src.features.mfcc import mfcc_feat
from src.features.lfcc import lfcc_feat
from src.features.cqcc import cqcc_feat
from src.features.fuse import early_fuse_time
from src.features.normalize import NormStats, apply_norm
from src.dataio.protocols import Sample

@dataclass(frozen=True)
class FeatureConfig:
    # mel
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160

    # cepstral dims
    n_mfcc: int = 20
    n_lfcc: int = 20
    n_cqcc: int = 20

@dataclass(frozen=True)
class DatasetConfig:
    audio: AudioConfig = AudioConfig(sr=16000, duration_s=4.0)
    feat: FeatureConfig = FeatureConfig()
    batch_size: int = 16
    shuffle: bool = True
    seed: int = 7

def _extract_one(sample: Sample, ds_cfg: DatasetConfig, stats: NormStats):
    y, sr = load_audio(sample.wav_path, ds_cfg.audio)

    mel = mel_spectrogram_db(y, sr, n_mels=ds_cfg.feat.n_mels, n_fft=ds_cfg.feat.n_fft, hop_length=ds_cfg.feat.hop_length)
    mfcc = mfcc_feat(y, sr, n_mfcc=ds_cfg.feat.n_mfcc, n_fft=ds_cfg.feat.n_fft, hop_length=ds_cfg.feat.hop_length)
    lfcc = lfcc_feat(y, sr, n_lfcc=ds_cfg.feat.n_lfcc, n_fft=ds_cfg.feat.n_fft, hop_length=ds_cfg.feat.hop_length)
    cqcc = cqcc_feat(y, sr, n_cqcc=ds_cfg.feat.n_cqcc, hop_length=ds_cfg.feat.hop_length)

    seq = early_fuse_time(mfcc, lfcc, cqcc)  # [T, D]
    mel, seq = apply_norm(mel, seq, stats)

    # Keras Conv2D expects [H, W, C] per example
    mel_img = mel[:, :, None]  # [n_mels, T, 1]

    x = {
        "mel_in": mel_img.astype(np.float32),
        "seq_in": seq.astype(np.float32),
    }
    y = np.int32(sample.label)
    return x, y

def make_dataset(
    samples: list[Sample],
    ds_cfg: DatasetConfig,
    stats: NormStats,
    repeat: bool = False,
) -> tf.data.Dataset:
    """
    Returns tf.data.Dataset yielding ({mel_in, seq_in}, y)
    mel_in: [B, n_mels, T, 1]
    seq_in: [B, T, D]
    y: [B]

    If repeat=True, dataset repeats indefinitely (use with steps_per_epoch).
    """

    def gen():
        for s in samples:
            yield _extract_one(s, ds_cfg, stats)

    # Infer shapes from first sample
    x0, _ = _extract_one(samples[0], ds_cfg, stats)
    mel_shape = x0["mel_in"].shape
    seq_shape = x0["seq_in"].shape

    output_signature = (
        {
            "mel_in": tf.TensorSpec(shape=(mel_shape[0], mel_shape[1], mel_shape[2]), dtype=tf.float32),
            "seq_in": tf.TensorSpec(shape=(seq_shape[0], seq_shape[1]), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if ds_cfg.shuffle:
        ds = ds.shuffle(buffer_size=min(4000, len(samples)), seed=ds_cfg.seed, reshuffle_each_iteration=True)

    ds = ds.batch(ds_cfg.batch_size, drop_remainder=False)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

