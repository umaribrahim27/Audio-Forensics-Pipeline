# main/scripts/eval_test.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.common.paths import ARTIFACTS, CHECKPOINTS
from src.dataio.manifests import load_manifest_csv
from src.pipeline.detector import DeepfakeDetector


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # y_true, y_pred are 0/1
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int32)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else (2 * tp) / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ARTIFACTS / "manifests" / "test_800_balanced.csv"))
    ap.add_argument("--model", default=str(CHECKPOINTS / "bimodal_v1" / "model.keras"))
    ap.add_argument("--stats", default=str(ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"))
    ap.add_argument("--find_best_threshold", action="store_true", help="Search threshold to maximize F1 on this set")
    args = ap.parse_args()

    samples = load_manifest_csv(Path(args.manifest), strict=False)
    det = DeepfakeDetector(model_path=Path(args.model), stats_path=Path(args.stats))

    y_true = np.array([s.label for s in samples], dtype=np.int32)
    p_fake = np.zeros(len(samples), dtype=np.float32)

    for i, s in enumerate(samples, 1):
        p_fake[i - 1] = det.predict_file(s.wav_path)
        if i % 100 == 0:
            print(f"Predicted {i}/{len(samples)}")

    # Metrics
    auc = float(tf.keras.metrics.AUC()(y_true, p_fake).numpy())
    y_pred = (p_fake >= 0.5).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Test Metrics (threshold=0.5) ===")
    print(f"AUC: {auc:.6f}")
    print(f"Accuracy: {acc:.6f}")
    print("Confusion Matrix [[TN,FP],[FN,TP]]:")
    print(cm)

    if args.find_best_threshold:
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            yp = (p_fake >= t).astype(np.int32)
            f1 = f1_score(y_true, yp)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        yp = (p_fake >= best_t).astype(np.int32)
        cm2 = confusion_matrix(y_true, yp)
        acc2 = float((yp == y_true).mean())

        print("\n=== Best Threshold Search (maximize F1) ===")
        print(f"Best threshold: {best_t:.2f}")
        print(f"F1: {best_f1:.6f}")
        print(f"Accuracy@best_t: {acc2:.6f}")
        print("Confusion Matrix @ best_t:")
        print(cm2)


if __name__ == "__main__":
    main()
