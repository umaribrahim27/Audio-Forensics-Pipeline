from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

def format_top_bins(top_bins):
    return ", ".join([f"bin{d['bin']}({d['count']})" for d in top_bins])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to summary.json produced by explain_manifest_stats")
    ap.add_argument("--out", default=None, help="Output markdown file. Default: same folder/report.md")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        s = json.load(f)

    out_path = Path(args.out) if args.out else summary_path.parent / "report.md"

    def sec(name, block):
        lines = []
        lines.append(f"## {name}")
        lines.append(f"- Count: {block['count']}")
        lines.append(f"- p_fake mean/std: {block['p_fake_mean']:.4f} / {block['p_fake_std']:.4f}")
        lines.append(f"- Time entropy mean/std: {block['time_entropy_mean']:.4f} / {block['time_entropy_std']:.4f}")
        lines.append(f"- Time concentration mean/std: {block['time_concentration_mean']:.4f} / {block['time_concentration_std']:.4f}")
        lines.append(f"- Freq entropy mean/std: {block['freq_entropy_mean']:.4f} / {block['freq_entropy_std']:.4f}")
        lines.append(f"- Top mel bins (bin(count)): {format_top_bins(block['top_mel_bins'])}")
        lines.append(f"- Top time frames (frame(count)): " + ", ".join([f"t{d['frame']}({d['count']})" for d in block['top_time_frames']]))
        lines.append("")
        return "\n".join(lines)

    md = []
    md.append("# Explainability Forensics Report")
    md.append("")
    md.append("### Run Metadata")
    md.append(f"- Manifest: `{s['manifest']}`")
    md.append(f"- Limit analyzed: {s['limit']}")
    md.append(f"- Model: `{s['model']}`")
    md.append(f"- Stats: `{s['stats']}`")
    md.append("")
    md.append("### Interpretation Notes")
    md.append("- **Time entropy**: low means explanations are concentrated in specific moments; high means spread across the clip.")
    md.append("- **Time concentration**: fraction of importance mass contained in the top 10% frames.")
    md.append("- **Freq entropy**: low means specific bands dominate; high means importance is spread across bands.")
    md.append("")

    md.append(sec("ALL SAMPLES", s["all"]))
    md.append(sec("BONAFIDE (REAL)", s["bonafide"]))
    md.append(sec("SPOOF (FAKE)", s["spoof"]))

    # Add a small “difference insight” section based on mean profiles if available
    if s["bonafide"]["mel_importance_mean"] and s["spoof"]["mel_importance_mean"]:
        bona = np.array(s["bonafide"]["mel_importance_mean"], dtype=np.float32)
        spoof = np.array(s["spoof"]["mel_importance_mean"], dtype=np.float32)
        diff = spoof - bona
        top_diff = np.argsort(diff)[::-1][:10].tolist()

        md.append("## Spoof vs Bonafide — Most Discriminative Mel Bins (by mean importance difference)")
        md.append("Top bins where spoof importance exceeds bonafide:")
        md.append(", ".join([f"bin{b}(Δ={diff[b]:.4f})" for b in top_diff]))
        md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print("Saved markdown report to:", out_path)

if __name__ == "__main__":
    main()
