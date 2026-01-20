# main/scripts/explain_manifest_stats.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
import csv
import numpy as np

from src.common.paths import ARTIFACTS, CHECKPOINTS
from src.dataio.manifests import load_manifest_csv
from src.pipeline.detector import DeepfakeDetector
from src.explainability.saliency import compute_saliency
from src.explainability.gradcam import gradcam_mel
from src.explainability.stats import compute_explain_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_manifest", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=300, help="How many samples to analyze")
    ap.add_argument("--seed", type=int, default=7)  # reserved if you later shuffle/sample
    ap.add_argument("--out_dir", default=str(ARTIFACTS / "explain_runs" / "run_v1"))

    ap.add_argument("--model", default=str(CHECKPOINTS / "bimodal_v2_explain" / "model.keras"))
    ap.add_argument("--stats", default=str(ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"))
    args = ap.parse_args()

    # Pick manifest
    if args.from_manifest == "train":
        manifest = ARTIFACTS / "manifests" / "train_3000_balanced.csv"
    elif args.from_manifest == "dev":
        manifest = ARTIFACTS / "manifests" / "dev_800_balanced.csv"
    else:
        manifest = ARTIFACTS / "manifests" / "test_800_balanced.csv"

    samples = load_manifest_csv(manifest, strict=False)
    if not samples:
        raise RuntimeError(f"No samples loaded from {manifest}")

    # limit
    n = min(args.limit, len(samples))
    samples = samples[:n]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_sample_dir = out_root / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    # NEW: per-sample CSV export
    per_sample_csv = out_root / "per_sample_stats.csv"

    det = DeepfakeDetector(model_path=Path(args.model), stats_path=Path(args.stats))

    # Aggregators (overall + per class)
    def init_aggs():
        return {
            "count": 0,
            "p_fake": [],
            "time_entropy": [],
            "time_concentration": [],
            "freq_entropy": [],
            "freq_hist": Counter(),    # mel bins
            "time_hist": Counter(),    # frames
            "mel_profile_sum": None,   # [80]
            "time_profile_sum": None,  # [T]
        }

    agg_all = init_aggs()
    agg_bona = init_aggs()
    agg_spoof = init_aggs()

    def update_aggs(agg, label, p_fake, mel_sal, time_imp, stats):
        agg["count"] += 1
        agg["p_fake"].append(p_fake)
        agg["time_entropy"].append(stats.time_entropy)
        agg["time_concentration"].append(stats.time_concentration)
        agg["freq_entropy"].append(stats.freq_entropy)

        # hist: top bins + top frames
        agg["freq_hist"].update(stats.top_mel_bins)
        agg["time_hist"].update(stats.top_time_frames)

        # mean profiles
        mel_profile = mel_sal.mean(axis=1)  # [80]
        if agg["mel_profile_sum"] is None:
            agg["mel_profile_sum"] = mel_profile.astype(np.float64)
        else:
            agg["mel_profile_sum"] += mel_profile.astype(np.float64)

        if agg["time_profile_sum"] is None:
            agg["time_profile_sum"] = time_imp.astype(np.float64)
        else:
            agg["time_profile_sum"] += time_imp.astype(np.float64)

    # CSV writer wrapping main loop
    with per_sample_csv.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=[
                "utt_id", "label", "split", "attack",
                "p_fake",
                "time_entropy", "time_concentration",
                "freq_entropy",
                "top_mel_bins", "top_time_frames", "top_time_seconds",
            ],
        )
        w.writeheader()

        # Main loop
        for i, s in enumerate(samples, 1):
            x = det.preprocess_file(s.wav_path)
            sal = compute_saliency(det.model, x)

            # Grad-CAM is computed; large arrays are saved only for first few samples (gallery)
            cam = gradcam_mel(det.model, x, conv_layer_name="mel_conv_last")

            st = compute_explain_stats(
                mel_importance=sal.mel_saliency,
                time_importance=sal.time_importance,
                hop_length=det.cfg.hop_length,
                sr=det.cfg.sr,
                top_k=10,
            )

            # Save per-sample report JSON (lightweight)
            report = {
                "utt_id": s.utt_id,
                "wav_path": str(s.wav_path),
                "label": int(s.label),
                "split": getattr(s, "split", ""),
                "attack": getattr(s, "attack", None),
                "p_fake": float(sal.p_fake),
                "top_time_frames": st.top_time_frames,
                "top_time_seconds": st.top_time_seconds,
                "time_entropy": st.time_entropy,
                "time_concentration": st.time_concentration,
                "top_mel_bins": st.top_mel_bins,
                "freq_entropy": st.freq_entropy,
                "mel_shape": list(sal.mel_saliency.shape),
            }
            with (per_sample_dir / f"{s.utt_id}.json").open("w", encoding="utf-8") as fj:
                json.dump(report, fj, indent=2)

            # NEW: Write per-sample CSV row
            w.writerow(
                {
                    "utt_id": s.utt_id,
                    "label": int(s.label),
                    "split": getattr(s, "split", ""),
                    "attack": getattr(s, "attack", "") or "",
                    "p_fake": f"{float(sal.p_fake):.8f}",
                    "time_entropy": f"{float(st.time_entropy):.8f}",
                    "time_concentration": f"{float(st.time_concentration):.8f}",
                    "freq_entropy": f"{float(st.freq_entropy):.8f}",
                    "top_mel_bins": "|".join(map(str, st.top_mel_bins)),
                    "top_time_frames": "|".join(map(str, st.top_time_frames)),
                    "top_time_seconds": "|".join([f"{t:.4f}" for t in st.top_time_seconds]),
                }
            )

            # Update aggregates
            update_aggs(agg_all, s.label, float(sal.p_fake), sal.mel_saliency, sal.time_importance, st)
            if s.label == 0:
                update_aggs(agg_bona, s.label, float(sal.p_fake), sal.mel_saliency, sal.time_importance, st)
            else:
                update_aggs(agg_spoof, s.label, float(sal.p_fake), sal.mel_saliency, sal.time_importance, st)

            # Save a small gallery of raw arrays for first 10 samples
            if i <= 10:
                sample_dir = out_root / "samples" / s.utt_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / "mel_saliency.npy", sal.mel_saliency.astype(np.float32))
                np.save(sample_dir / "time_importance.npy", sal.time_importance.astype(np.float32))
                np.save(sample_dir / "mel_gradcam.npy", cam.astype(np.float32))

            if i % 25 == 0 or i == n:
                print(f"Explained {i}/{n}")

    def finalize(agg, name: str):
        count = max(1, agg["count"])
        mel_mean = (agg["mel_profile_sum"] / count).astype(np.float32) if agg["mel_profile_sum"] is not None else None
        time_mean = (agg["time_profile_sum"] / count).astype(np.float32) if agg["time_profile_sum"] is not None else None

        top_freq = agg["freq_hist"].most_common(10)
        top_time = agg["time_hist"].most_common(10)

        return {
            "name": name,
            "count": agg["count"],
            "p_fake_mean": float(np.mean(agg["p_fake"])) if agg["p_fake"] else None,
            "p_fake_std": float(np.std(agg["p_fake"])) if agg["p_fake"] else None,

            "time_entropy_mean": float(np.mean(agg["time_entropy"])) if agg["time_entropy"] else None,
            "time_entropy_std": float(np.std(agg["time_entropy"])) if agg["time_entropy"] else None,

            "time_concentration_mean": float(np.mean(agg["time_concentration"])) if agg["time_concentration"] else None,
            "time_concentration_std": float(np.std(agg["time_concentration"])) if agg["time_concentration"] else None,

            "freq_entropy_mean": float(np.mean(agg["freq_entropy"])) if agg["freq_entropy"] else None,
            "freq_entropy_std": float(np.std(agg["freq_entropy"])) if agg["freq_entropy"] else None,

            "top_mel_bins": [{"bin": b, "count": c} for b, c in top_freq],
            "top_time_frames": [{"frame": t, "count": c} for t, c in top_time],

            "mel_importance_mean": mel_mean.tolist() if mel_mean is not None else None,     # [80]
            "time_importance_mean": time_mean.tolist() if time_mean is not None else None, # [T]
            "meta": {
                "hop_length": det.cfg.hop_length,
                "sr": det.cfg.sr,
                "duration_s": det.cfg.duration_s,
            },
        }

    summary = {
        "manifest": str(manifest),
        "limit": n,
        "model": str(Path(args.model)),
        "stats": str(Path(args.stats)),
        "all": finalize(agg_all, "all"),
        "bonafide": finalize(agg_bona, "bonafide"),
        "spoof": finalize(agg_spoof, "spoof"),
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved run summary to:", out_root / "summary.json")
    print("Per-sample reports:", per_sample_dir)
    print("Saved per-sample CSV to:", per_sample_csv)


if __name__ == "__main__":
    main()
