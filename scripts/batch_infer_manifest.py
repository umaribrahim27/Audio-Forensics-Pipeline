# main/scripts/batch_infer_manifest.py
from __future__ import annotations

import argparse
from pathlib import Path
import csv

from src.common.paths import ARTIFACTS, CHECKPOINTS
from src.dataio.manifests import load_manifest_csv
from src.pipeline.detector import DeepfakeDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=str(ARTIFACTS / "manifests" / "test_800_balanced.csv"),
        help="Path to manifest CSV",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: artifacts/preds/<manifest_stem>_preds.csv",
    )
    ap.add_argument(
        "--model",
        default=str(CHECKPOINTS / "bimodal_v1" / "model.keras"),
        help="Path to model.keras",
    )
    ap.add_argument(
        "--stats",
        default=str(ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"),
        help="Path to normalization stats .npz",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    model_path = Path(args.model)
    stats_path = Path(args.stats)

    if args.out is None:
        out_dir = ARTIFACTS / "preds"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{manifest_path.stem}_preds.csv"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_manifest_csv(manifest_path, strict=False)
    if not samples:
        raise RuntimeError(f"No samples loaded from {manifest_path}")

    det = DeepfakeDetector(model_path=model_path, stats_path=stats_path)

    print("Running batch inference on:", manifest_path)
    print("Samples:", len(samples))
    print("Writing to:", out_path)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "wav_path", "label", "p_fake", "pred"])
        w.writeheader()

        for i, s in enumerate(samples, 1):
            p = det.predict_file(s.wav_path)
            pred = 1 if p >= 0.5 else 0
            w.writerow(
                {
                    "utt_id": s.utt_id,
                    "wav_path": str(s.wav_path),
                    "label": int(s.label),
                    "p_fake": f"{p:.8f}",
                    "pred": int(pred),
                }
            )
            if i % 100 == 0:
                print(f"Done {i}/{len(samples)}")

    print("Done. Saved:", out_path)


if __name__ == "__main__":
    main()
