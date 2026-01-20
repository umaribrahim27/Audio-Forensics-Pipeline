# main/src/app/streamlit_app.py
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
import zipfile
from datetime import datetime

import numpy as np
import streamlit as st

from src.common.paths import ARTIFACTS, CHECKPOINTS
from src.pipeline.detector import DeepfakeDetector
from src.explainability.saliency import compute_saliency
from src.explainability.gradcam import gradcam_mel
from src.explainability.stats import compute_explain_stats
from src.explainability.render import (
    RenderConfig,
    render_mel_base,
    render_mel_overlay,
    render_time_importance,
    render_waveform_with_highlights,
)

# ============================================================
# BACKEND CONFIG (CHANGE HERE IF NEEDED)
# ============================================================
MODEL_PATH = CHECKPOINTS / "bimodal_v2_explain" / "model.keras"
STATS_PATH = ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"

# Default operating threshold. Lower => more sensitive (more spoof flagged).
DEFAULT_THRESHOLD = 0.50

# Where UI cases are stored
UI_CASES_DIR = ARTIFACTS / "explain_ui"
TMP_UPLOADS_DIR = ARTIFACTS / "tmp_uploads"
# ============================================================


def save_uploaded_file(uploaded) -> Path:
    """Save uploaded file into artifacts/tmp_uploads and return path."""
    TMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower()
    out_path = TMP_UPLOADS_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(uploaded.getbuffer())
    return out_path


def package_bundle(sample_dir: Path, include_audio: bool) -> bytes:
    """
    Create a ZIP bundle in memory from a sample_dir produced by this app.
    Includes:
      - report.json
      - *.npy maps
      - viz/*.png
      - optional audio/<original_filename>
    Returns bytes for st.download_button.
    """
    report_path = sample_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json in {sample_dir}")

    rep = json.loads(report_path.read_text(encoding="utf-8"))
    utt_id = rep.get("utt_id", sample_dir.name)

    required = ["report.json", "mel_saliency.npy", "mel_gradcam.npy", "time_importance.npy"]
    optional = ["seq_saliency.npy"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta = {
            "utt_id": utt_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "sample_dir": str(sample_dir),
            "includes_audio": bool(include_audio),
            "contents": {
                "report.json": "Model score + key forensic stats (top times/bins, entropy, concentration).",
                "*.npy": "Raw explanation arrays (saliency/gradcam/time-importance) for reproducibility.",
                "viz/*.png": "Presentation-ready figures for UI and reporting.",
                "audio/* (optional)": "The uploaded audio file (if include-audio is enabled).",
            },
        }
        zf.writestr("bundle_meta.json", json.dumps(meta, indent=2))

        for name in required:
            p = sample_dir / name
            if p.exists():
                zf.write(p, arcname=name)

        for name in optional:
            p = sample_dir / name
            if p.exists():
                zf.write(p, arcname=name)

        viz_dir = sample_dir / "viz"
        if viz_dir.exists():
            for p in sorted(viz_dir.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(sample_dir).as_posix()  # viz/...
                    zf.write(p, arcname=rel)

        if include_audio:
            audio_path_str = rep.get("wav_path")
            if audio_path_str:
                audio_path = Path(audio_path_str)
                if audio_path.exists():
                    zf.write(audio_path, arcname=f"audio/{audio_path.name}")

    return buf.getvalue()


def run_forensic_explain(detector: DeepfakeDetector, audio_path: Path, out_root: Path) -> Path:
    """
    Produces:
      out_root/
        report.json
        mel_saliency.npy
        seq_saliency.npy
        time_importance.npy
        mel_gradcam.npy
        viz/*.png
    """
    out_root.mkdir(parents=True, exist_ok=True)

    x = detector.preprocess_file(audio_path)
    sal = compute_saliency(detector.model, x)
    cam = gradcam_mel(detector.model, x, conv_layer_name="mel_conv_last")

    st_stats = compute_explain_stats(
        mel_importance=sal.mel_saliency,
        time_importance=sal.time_importance,
        hop_length=detector.cfg.hop_length,
        sr=detector.cfg.sr,
        top_k=10,
    )

    np.save(out_root / "mel_saliency.npy", sal.mel_saliency.astype(np.float32))
    np.save(out_root / "seq_saliency.npy", sal.seq_saliency.astype(np.float32))
    np.save(out_root / "time_importance.npy", sal.time_importance.astype(np.float32))
    np.save(out_root / "mel_gradcam.npy", cam.astype(np.float32))

    report = {
        "utt_id": out_root.name,
        "wav_path": str(audio_path),
        "p_fake": float(sal.p_fake),
        "top_time_frames": st_stats.top_time_frames,
        "top_time_seconds": st_stats.top_time_seconds,
        "time_entropy": st_stats.time_entropy,
        "time_concentration": st_stats.time_concentration,
        "top_mel_bins": st_stats.top_mel_bins,
        "freq_entropy": st_stats.freq_entropy,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    viz_dir = out_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    cfg = RenderConfig(
        sr=detector.cfg.sr,
        duration_s=detector.cfg.duration_s,
        hop_length=detector.cfg.hop_length,
        n_fft=detector.cfg.n_fft,
        n_mels=detector.cfg.n_mels,
    )

    mel_png = viz_dir / "mel.png"
    mel_db, _ = render_mel_base(audio_path, mel_png, cfg=cfg)

    render_mel_overlay(
        mel_db=mel_db,
        overlay=sal.mel_saliency,
        out_png=viz_dir / "mel_saliency_overlay.png",
        title="Mel + Saliency Overlay (What pushed p_fake up)",
        cfg=cfg,
    )

    render_mel_overlay(
        mel_db=mel_db,
        overlay=cam,
        out_png=viz_dir / "mel_gradcam_overlay.png",
        title="Mel + Grad-CAM Overlay (CNN focus regions)",
        cfg=cfg,
    )

    render_time_importance(
        time_importance=sal.time_importance,
        out_png=viz_dir / "time_importance.png",
        cfg=cfg,
        title="Time Importance (Suspiciousness over Time)",
    )

    render_waveform_with_highlights(
        wav_path=audio_path,
        time_importance=sal.time_importance,
        out_png=viz_dir / "waveform_highlight.png",
        cfg=cfg,
        title="Waveform with Highlighted Suspicious Regions",
    )

    return out_root


@st.cache_resource
def get_detector() -> DeepfakeDetector:
    return DeepfakeDetector(model_path=MODEL_PATH, stats_path=STATS_PATH)


def main():
    st.set_page_config(page_title="Deepfake Audio Forensics", layout="wide")
    st.title("Deepfake Audio Forensics")
    st.caption(
        "Upload a single audio file. If spoofing is detected, the app generates forensic explanations and visuals."
    )

    st.sidebar.header("Controls")

    threshold = st.sidebar.slider(
        "Spoof detection threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_THRESHOLD),
        step=0.01,
        help=(
            "If p_fake ≥ threshold → SPOOF. "
            "Lower threshold = more sensitive (flags more audio as spoof). "
            "Higher threshold = stricter (flags fewer)."
        ),
    )

    st.sidebar.markdown(
        "**Sensitivity note:** lowering the threshold makes the model *more sensitive* "
        "(it will flag spoofing more often). Raising it makes the model stricter."
    )

    include_audio = st.sidebar.checkbox("Include audio in ZIP bundle", value=False)

    uploaded = st.file_uploader("Drop a .flac or .wav file", type=["flac", "wav"])
    if uploaded is None:
        st.info("Upload an audio file to begin.")
        return

    audio_path = save_uploaded_file(uploaded)
    detector = get_detector()

    with st.spinner("Running inference..."):
        p_fake = detector.predict_file(audio_path)

    col1, col2 = st.columns([2, 3], vertical_alignment="center")
    with col1:
        st.subheader("Prediction")
        st.metric("p_fake", f"{p_fake:.6f}")
    with col2:
        if p_fake >= threshold:
            st.error("SPOOF detected: this audio likely contains spoofing artifacts.")
        else:
            st.success("No spoofing detected (BONAFIDE).")
            st.caption("Explainability is only generated when spoofing is detected.")
            return

    st.markdown("---")
    st.subheader("Forensic Explanations")

    case_id = f"ui_{uuid.uuid4().hex}"
    case_dir = UI_CASES_DIR / case_id

    with st.spinner("Generating explanations and rendering visuals..."):
        run_forensic_explain(detector, audio_path, case_dir)

    report = json.loads((case_dir / "report.json").read_text(encoding="utf-8"))
    viz_dir = case_dir / "viz"

    tabs = st.tabs(
        [
            "Summary",
            "Mel",
            "Saliency Overlay",
            "Grad-CAM Overlay",
            "Time Importance",
            "Waveform Highlight",
            "Download Bundle",
        ]
    )

    with tabs[0]:
        st.markdown("### Diagnostic Summary")
        st.write(
            {
                "p_fake": report["p_fake"],
                "top_time_seconds": report["top_time_seconds"],
                "top_mel_bins": report["top_mel_bins"],
                "time_entropy": report["time_entropy"],
                "time_concentration": report["time_concentration"],
                "freq_entropy": report["freq_entropy"],
            }
        )
        st.caption("Top timestamps indicate where suspicious artifacts are most concentrated.")

    with tabs[1]:
        st.image(str(viz_dir / "mel.png"), caption="Mel Spectrogram (dB)", use_container_width=True)

    with tabs[2]:
        st.image(str(viz_dir / "mel_saliency_overlay.png"), caption="Mel + Saliency Overlay", use_container_width=True)

    with tabs[3]:
        st.image(str(viz_dir / "mel_gradcam_overlay.png"), caption="Mel + Grad-CAM Overlay", use_container_width=True)

    with tabs[4]:
        st.image(str(viz_dir / "time_importance.png"), caption="Time Importance (Suspiciousness over Time)", use_container_width=True)

    with tabs[5]:
        st.image(str(viz_dir / "waveform_highlight.png"), caption="Waveform with Highlighted Suspicious Regions", use_container_width=True)

    with tabs[6]:
        st.markdown("### Download Forensic Bundle (ZIP)")

        with st.expander("What the ZIP contains"):
            st.markdown(
                """
- **report.json** — model score and key forensic stats (top suspicious times/bins, entropy, concentration)  
- **\*.npy** — raw explanation arrays (saliency, gradcam, time importance) for reproducibility  
- **viz/\*.png** — presentation-ready figures shown in the UI  
- **audio/\*** *(optional)* — your uploaded audio file if you enable “Include audio in ZIP bundle”
                """
            )

        with st.spinner("Packaging ZIP bundle..."):
            zip_bytes = package_bundle(case_dir, include_audio=include_audio)

        zip_name = f"{case_id}_forensic_bundle.zip"
        st.download_button(
            label="Download ZIP bundle",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
        )

        st.caption(f"Bundle source folder: {case_dir}")


if __name__ == "__main__":
    main()
