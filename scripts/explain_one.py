from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.common.paths import ARTIFACTS, CHECKPOINTS
from src.dataio.manifests import load_manifest_csv
from src.pipeline.detector import DeepfakeDetector
from src.explainability.saliency import compute_saliency
from src.explainability.gradcam import gradcam_mel
from src.explainability.stats import compute_explain_stats

def pick_sample(manifest_path: Path, index: int | None, pick_random: bool):
    samples = load_manifest_csv(manifest_path, strict=False)
    if pick_random:
        import random
        return random.choice(samples)
    if index is None:
        index = 0
    return samples[index]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_manifest", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--out_dir", default=str(ARTIFACTS / "explain"))

    # IMPORTANT: point this to the new retrained model folder you used
    ap.add_argument("--model", default=str(CHECKPOINTS / "bimodal_v2_explain" / "model.keras"))
    ap.add_argument("--stats", default=str(ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"))
    args = ap.parse_args()

    if args.from_manifest == "train":
        manifest = ARTIFACTS / "manifests" / "train_3000_balanced.csv"
    elif args.from_manifest == "dev":
        manifest = ARTIFACTS / "manifests" / "dev_800_balanced.csv"
    else:
        manifest = ARTIFACTS / "manifests" / "test_800_balanced.csv"

    det = DeepfakeDetector(model_path=Path(args.model), stats_path=Path(args.stats))
    sample = pick_sample(manifest, args.index, args.random)

    x = det.preprocess_file(sample.wav_path)

    sal = compute_saliency(det.model, x)
    cam = gradcam_mel(det.model, x, conv_layer_name="mel_conv_last")

    st = compute_explain_stats(
        mel_importance=sal.mel_saliency,
        time_importance=sal.time_importance,
        hop_length=det.cfg.hop_length,
        sr=det.cfg.sr,
        top_k=10,
    )

    out_root = Path(args.out_dir) / sample.utt_id
    out_root.mkdir(parents=True, exist_ok=True)

    np.save(out_root / "mel_saliency.npy", sal.mel_saliency)
    np.save(out_root / "seq_saliency.npy", sal.seq_saliency)
    np.save(out_root / "time_importance.npy", sal.time_importance)
    np.save(out_root / "mel_gradcam.npy", cam)

    report = {
        "utt_id": sample.utt_id,
        "wav_path": str(sample.wav_path),
        "label": int(sample.label),
        "p_fake": float(sal.p_fake),
        "top_time_frames": st.top_time_frames,
        "top_time_seconds": st.top_time_seconds,
        "time_entropy": st.time_entropy,
        "time_concentration": st.time_concentration,
        "top_mel_bins": st.top_mel_bins,
        "freq_entropy": st.freq_entropy,
    }
    with (out_root / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved explanation to:", out_root)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
