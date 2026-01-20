# main/src/dataio/protocols.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class Sample:
    utt_id: str
    wav_path: Path
    label: int          # 0=bonafide, 1=spoof
    split: str          # train/dev/eval
    attack: Optional[str] = None

def _label_to_int(label: str) -> int:
    label = label.strip().lower()
    if label == "bonafide":
        return 0
    if label == "spoof":
        return 1
    raise ValueError(f"Unknown label: {label}")

def parse_asvspoof2019_la_cm_protocol(protocol_path: Path) -> List[dict]:
    """
    ASVspoof2019 LA CM protocol format (per line) typically:
      <speaker_id> <utt_id> <...> <attack_id> <label>
    We'll parse robustly by taking:
      utt_id = tokens[1]
      attack_id = tokens[-2]
      label = tokens[-1]
    """
    rows = []
    with protocol_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            utt_id = toks[1]
            label = toks[-1]
            attack = toks[-2] if toks[-2].upper().startswith("A") else None
            rows.append({"utt_id": utt_id, "label": _label_to_int(label), "attack": attack})
    return rows

def build_manifest_la(
    la_root: Path,
    split: str,
    protocol_dirname: str = "ASVspoof2019_LA_cm_protocols",
) -> List[Sample]:
    """
    la_root = main/data/LA
    split in {"train","dev","eval"}
    """
    split = split.lower()
    if split not in {"train", "dev", "eval"}:
        raise ValueError("split must be one of: train/dev/eval")

    # audio dir structure
    # ASVspoof2019_LA_train/flac/ or wav/ depending on your extracted set
    # Most provided sets have "flac"; some class setups have "wav".
    audio_base = la_root / f"ASVspoof2019_LA_{split}"
    # Try flac first, then wav
    if (audio_base / "flac").exists():
        audio_dir = audio_base / "flac"
        ext = ".flac"
    else:
        audio_dir = audio_base / "wav"
        ext = ".wav"

    proto_dir = la_root / protocol_dirname
    # Common filenames:
    # ASVspoof2019.LA.cm.train.trn.txt / dev.trl.txt / eval.trl.txt (varies by release)
    # We'll choose by searching for the split string in filename.
    candidates = sorted([p for p in proto_dir.glob("*.txt") if split in p.name.lower()])
    if not candidates:
        raise FileNotFoundError(f"No protocol .txt found for split='{split}' in {proto_dir}")
    protocol_path = candidates[0]

    rows = parse_asvspoof2019_la_cm_protocol(protocol_path)
    samples: List[Sample] = []
    for r in rows:
        utt_id = r["utt_id"]
        wav_path = audio_dir / f"{utt_id}{ext}"
        if not wav_path.exists():
            # don’t crash hard; you’ll see which ids are missing
            continue
        samples.append(Sample(
            utt_id=utt_id,
            wav_path=wav_path,
            label=r["label"],
            split=split,
            attack=r.get("attack"),
        ))
    return samples
