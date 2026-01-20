# main/src/dataio/manifests.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import List, Optional

from src.dataio.protocols import Sample


def stratified_sample(samples: List[Sample], n_total: int, seed: int = 7) -> List[Sample]:
    """
    Make a 50/50 (bonafide/spoof) subset of size n_total.
    If n_total is odd, the extra example goes to spoof (arbitrary but consistent).
    """
    rng = random.Random(seed)

    bona = [s for s in samples if s.label == 0]
    spoof = [s for s in samples if s.label == 1]

    rng.shuffle(bona)
    rng.shuffle(spoof)

    n_bona = n_total // 2
    n_spoof = n_total - n_bona

    if len(bona) < n_bona or len(spoof) < n_spoof:
        raise ValueError(
            f"Not enough samples for requested 50/50 subset: "
            f"need bona={n_bona}, spoof={n_spoof} but have bona={len(bona)}, spoof={len(spoof)}"
        )

    subset = bona[:n_bona] + spoof[:n_spoof]
    rng.shuffle(subset)
    return subset


def save_manifest_csv(samples: List[Sample], out_path: Path) -> None:
    """
    Save a manifest CSV with columns:
      utt_id,wav_path,label,split,attack

    label is saved as 0/1.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "wav_path", "label", "split", "attack"])
        w.writeheader()
        for s in samples:
            w.writerow(
                {
                    "utt_id": s.utt_id,
                    "wav_path": str(s.wav_path),
                    "label": int(s.label),
                    "split": s.split,
                    "attack": s.attack if s.attack is not None else "",
                }
            )


def _parse_label(x: Optional[str]) -> int:
    """
    Accepts:
      - "0"/"1"
      - 0/1
      - "bonafide"/"spoof"
    """
    if x is None:
        raise ValueError("label is None")
    x = str(x).strip().lower()
    if x in {"0", "1"}:
        return int(x)
    if x == "bonafide":
        return 0
    if x == "spoof":
        return 1
    if x == "":
        raise ValueError("empty label")
    raise ValueError(f"bad label value: '{x}'")


def load_manifest_csv(path: Path, strict: bool = True) -> List[Sample]:
    """
    Load a manifest CSV produced by save_manifest_csv.

    If strict=True (default), raises on the first bad row.
    If strict=False, skips bad rows and prints a message with the CSV line number.
    """
    samples: List[Sample] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"utt_id", "wav_path", "label"}
        fieldnames = set(r.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(f"Manifest missing required columns {required}. Found: {r.fieldnames}")

        for line_no, row in enumerate(r, start=2):  # header is line 1
            try:
                utt_id = (row.get("utt_id", "") or "").strip()
                wav_path = (row.get("wav_path", "") or "").strip()
                label = _parse_label(row.get("label", ""))

                if not utt_id:
                    raise ValueError("empty utt_id")
                if not wav_path:
                    raise ValueError("empty wav_path")

                samples.append(
                    Sample(
                        utt_id=utt_id,
                        wav_path=Path(wav_path),
                        label=label,
                        split=(row.get("split", "") or "").strip(),
                        attack=(row.get("attack", "") or "").strip() or None,
                    )
                )

            except Exception as e:
                msg = f"[load_manifest_csv] Bad row at line {line_no}: {e} | row={row}"
                if strict:
                    raise ValueError(msg) from e
                print(msg)
                continue

    if not samples:
        raise RuntimeError(f"No valid samples loaded from {path}")
    return samples
