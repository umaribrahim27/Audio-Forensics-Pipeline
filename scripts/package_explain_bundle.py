# main/scripts/package_explain_bundle.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile
from datetime import datetime


REQUIRED = [
    "report.json",
    "mel_saliency.npy",
    "mel_gradcam.npy",
    "time_importance.npy",
]

OPTIONAL = [
    "seq_saliency.npy",
]

VIZ_DIR = "viz"


def add_file(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    if file_path.exists() and file_path.is_file():
        zf.write(file_path, arcname=arcname)


def add_dir(zf: zipfile.ZipFile, dir_path: Path, arc_prefix: str) -> None:
    if not dir_path.exists() or not dir_path.is_dir():
        return
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            rel = p.relative_to(dir_path).as_posix()
            zf.write(p, arcname=f"{arc_prefix}/{rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample_dir",
        required=True,
        help="Directory like artifacts/explain/<utt_id> containing report.json and *.npy (and optionally viz/)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output zip path. Default: <sample_dir>/<utt_id>_bundle.zip",
    )
    ap.add_argument(
        "--include_audio",
        action="store_true",
        help="If set, also include the original audio referenced by report.json",
    )
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(sample_dir)

    report_path = sample_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json in {sample_dir}")

    # Load report to get utt_id and wav_path
    with report_path.open("r", encoding="utf-8") as f:
        rep = json.load(f)

    utt_id = rep.get("utt_id", sample_dir.name)
    wav_path = Path(rep["wav_path"]) if "wav_path" in rep else None

    out_zip = Path(args.out) if args.out else (sample_dir / f"{utt_id}_bundle.zip")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # Validate required files (warn but don't crash if you want partial bundles)
    missing = [name for name in REQUIRED if not (sample_dir / name).exists()]
    if missing:
        print("WARNING: Missing required files (bundle will be partial):", missing)

    # Create zip
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Metadata file
        meta = {
            "utt_id": utt_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "sample_dir": str(sample_dir),
            "includes_audio": bool(args.include_audio),
        }
        zf.writestr("bundle_meta.json", json.dumps(meta, indent=2))

        # Core files
        add_file(zf, report_path, "report.json")

        for name in REQUIRED:
            add_file(zf, sample_dir / name, name)

        for name in OPTIONAL:
            add_file(zf, sample_dir / name, name)

        # Visualizations directory
        add_dir(zf, sample_dir / VIZ_DIR, "viz")

        # Optional: include original audio (only if requested and exists)
        if args.include_audio and wav_path is not None:
            if wav_path.exists():
                add_file(zf, wav_path, f"audio/{wav_path.name}")
            else:
                print("WARNING: Audio path in report.json not found:", wav_path)

    print("Saved bundle:", out_zip)


if __name__ == "__main__":
    main()
