# main/src/features/fuse.py
from __future__ import annotations
import numpy as np

def early_fuse_time(mfcc: np.ndarray, lfcc: np.ndarray, cqcc: np.ndarray) -> np.ndarray:
    """
    Inputs: [C, T] each
    Output: [T, C_total] for sequence models (BiLSTM)
    """
    T = min(mfcc.shape[1], lfcc.shape[1], cqcc.shape[1])
    mfcc = mfcc[:, :T]
    lfcc = lfcc[:, :T]
    cqcc = cqcc[:, :T]
    fused = np.concatenate([mfcc, lfcc, cqcc], axis=0)  # [C_total, T]
    return fused.T.astype(np.float32)  # [T, C_total]
