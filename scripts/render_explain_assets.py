# main/scripts/render_explain_assets.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.explainability.render import (
    RenderConfig,
    render_mel_base,
    render_mel_overlay,
    render_time_importance,
    render_waveform_with_highlights,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample_dir",
        required=True,
        help="Directory like artifacts/explain/<utt_id> containing report.json and *.npy",
    )
    ap.add_argument("--out_dir", default=None, help="Default: <sample_dir>/viz")
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    report_path = sample_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json at {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        rep = json.load(f)

    wav_path = Path(rep["wav_path"])
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio not found: {wav_path}")

    mel_sal_path = sample_dir / "mel_saliency.npy"
    cam_path = sample_dir / "mel_gradcam.npy"
    time_path = sample_dir / "time_importance.npy"

    for p in [mel_sal_path, cam_path, time_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    mel_sal = np.load(mel_sal_path)        # [80,T]
    cam = np.load(cam_path)                # [80,T]
    time_imp = np.load(time_path)          # [T]

    out_dir = Path(args.out_dir) if args.out_dir else (sample_dir / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RenderConfig()

    # Base mel spectrogram (recomputed from audio for clean visuals)
    mel_png = out_dir / "mel.png"
    mel_db, _times = render_mel_base(wav_path, mel_png, cfg=cfg)

    # Overlays
    render_mel_overlay(
        mel_db=mel_db,
        overlay=mel_sal,
        out_png=out_dir / "mel_saliency_overlay.png",
        title="Mel + Saliency Overlay (What pushed p_fake up)",
        cfg=cfg,
    )

    render_mel_overlay(
        mel_db=mel_db,
        overlay=cam,
        out_png=out_dir / "mel_gradcam_overlay.png",
        title="Mel + Grad-CAM Overlay (CNN focus regions)",
        cfg=cfg,
    )

    # Time importance curve
    render_time_importance(
        time_importance=time_imp,
        out_png=out_dir / "time_importance.png",
        cfg=cfg,
        title="Time Importance (Suspiciousness over Time)",
    )

    # Waveform highlights
    render_waveform_with_highlights(
        wav_path=wav_path,
        time_importance=time_imp,
        out_png=out_dir / "waveform_highlight.png",
        cfg=cfg,
        title="Waveform with Highlighted Suspicious Regions",
    )

    print("Rendered visualization assets to:", out_dir)

if __name__ == "__main__":
    main()
