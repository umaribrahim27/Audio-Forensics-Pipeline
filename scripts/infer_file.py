# main/scripts/infer_file.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.common.paths import ROOT, ARTIFACTS, CHECKPOINTS
from src.dataio.manifests import load_manifest_csv
from src.pipeline.detector import DeepfakeDetector


def resolve_audio_path(
    file_arg: str | None,
    utt_id: str | None,
    from_manifest: str | None,
    index: int | None,
    pick_random: bool,
) -> Path:
    """
    Priority:
      1) --file
      2) --utt_id (requires --manifest_path or --from_manifest)
      3) --from_manifest (+ index or random)
    """
    if file_arg:
        p = Path(file_arg)
        if p.exists():
            return p
        alt = ROOT / file_arg
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Audio file not found: {p} (also tried {alt})")

    # choose which manifest to load
    manifest_path: Path | None = None
    if from_manifest:
        name = from_manifest.lower()
        if name not in {"train", "dev", "test"}:
            raise ValueError("--from_manifest must be one of: train/dev/test")
        # expected filenames
        if name == "train":
            manifest_path = ARTIFACTS / "manifests" / "train_3000_balanced.csv"
        elif name == "dev":
            manifest_path = ARTIFACTS / "manifests" / "dev_800_balanced.csv"
        else:
            manifest_path = ARTIFACTS / "manifests" / "test_800_balanced.csv"

    if utt_id:
        if manifest_path is None:
            raise ValueError("For --utt_id you must also pass --from_manifest (train/dev/test).")
        samples = load_manifest_csv(manifest_path, strict=False)
        for s in samples:
            if s.utt_id == utt_id:
                return s.wav_path
        raise FileNotFoundError(f"utt_id not found in {manifest_path}: {utt_id}")

    if manifest_path:
        samples = load_manifest_csv(manifest_path, strict=False)
        if not samples:
            raise RuntimeError(f"No samples in manifest: {manifest_path}")

        if pick_random:
            s = random.choice(samples)
            return s.wav_path

        if index is None:
            index = 0
        if index < 0 or index >= len(samples):
            raise IndexError(f"index out of range: {index} (0..{len(samples)-1})")

        return samples[index].wav_path

    raise ValueError("Provide either --file OR (--from_manifest with --index/--random) OR (--utt_id with --from_manifest).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="Path to .wav/.flac (absolute or relative to project root)")
    ap.add_argument("--utt_id", default=None, help="utt_id like LA_D_1234567 (requires --from_manifest)")

    ap.add_argument("--from_manifest", default=None, help="train/dev/test (uses balanced manifests)")
    ap.add_argument("--index", type=int, default=None, help="Pick sample by index from manifest")
    ap.add_argument("--random", action="store_true", help="Pick a random sample from manifest")

    ap.add_argument("--model", default=str(CHECKPOINTS / "bimodal_v1" / "model.keras"))
    ap.add_argument("--stats", default=str(ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"))
    args = ap.parse_args()

    audio_path = resolve_audio_path(args.file, args.utt_id, args.from_manifest, args.index, args.random)

    det = DeepfakeDetector(model_path=Path(args.model), stats_path=Path(args.stats))
    out = det.predict_file_with_debug(audio_path)

    p = out["p_fake"]
    print(f"Audio: {audio_path}")
    print(f"mel_in shape: {out['mel_shape']}, seq_in shape: {out['seq_shape']}")
    print(f"p_fake: {p:.6f}")
    print("Prediction:", "SPOOF (fake)" if p >= 0.5 else "BONAFIDE (real)")


if __name__ == "__main__":
    main()
