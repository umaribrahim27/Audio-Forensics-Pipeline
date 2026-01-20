# Deepfake Audio Forensics (ASVspoof LA)

End-to-end deepfake (spoof) audio detection + explainability + a Streamlit UI.

## What this project does
- Predicts whether an uploaded audio file is **BONAFIDE (real)** or **SPOOF (fake)**.
- If spoofing is detected, it generates **forensic evidence**:
  - Mel spectrogram
  - Saliency overlay (what pushed `p_fake` up)
  - Grad-CAM overlay (where the CNN focused)
  - Time-importance curve (when spoof artifacts happen)
  - Waveform highlight (suspicious regions)
- Lets the user download a **ZIP forensic bundle** containing the evidence.

---

## Project Structure (high level)

main/  
data/LA/ — ASVspoof2019 LA dataset (ignored by git)  
artifacts/ — generated outputs (ignored by git)  
checkpoints/ — trained models (ignored by git)  
scripts/ — CLI scripts (train/eval/explain/etc.)  
src/ — library code (features/models/pipeline/explainability/app)  
requirements.txt  
README.md  

---

## Setup

### 1) Create and activate a virtual environment (recommended)

python3 -m venv .venv  
source .venv/bin/activate  

### 2) Install dependencies

pip install -r requirements.txt  

---

## Dataset

This project expects ASVspoof2019 LA to be placed under:

main/data/LA/

Example (simplified):

data/LA/  
  ASVspoof2019_LA_train/  
    flac/  
    protocol/  
  ASVspoof2019_LA_dev/  
    flac/  
    protocol/  
  ASVspoof2019_LA_eval/  
    flac/  
    protocol/  

data/ is ignored in git. Each teammate should place the dataset locally.

---

## Feature 1 — Core Training Pipeline (balanced subsets → stats → training)

### A) Create balanced manifests (50/50)
Creates CSVs used across training and evaluation:

python3 -m scripts.make_balanced_subsets  

Outputs:
- artifacts/manifests/train_3000_balanced.csv
- artifacts/manifests/dev_800_balanced.csv
- artifacts/manifests/test_800_balanced.csv

### B) Fit feature normalization stats (fit on train only)

python3 -m scripts.fit_feature_stats --manifest artifacts/manifests/train_3000_balanced.csv  

Output:
- artifacts/feature_stats/stats_train3000_v1.npz

### C) Train bimodal model (Mel-CNN + BiLSTM)

python3 -m scripts.train_bimodal  

Output:
- checkpoints/<run_name>/model.keras

---

## Feature 1 — Inference & Evaluation

Single inference (manifest pick):

python3 -m scripts.infer_file --from_manifest test --random  

Test evaluation + best-threshold search:

python3 -m scripts.eval_test --find_best_threshold  

Batch inference on a manifest (writes predictions CSV):

python3 -m scripts.batch_infer_manifest --manifest artifacts/manifests/test_800_balanced.csv  

Output:
- artifacts/preds/test_800_balanced_preds.csv

---

## Feature 2 — Explainability & Forensics (single file)

Explain one file (generates .npy maps + report):

python3 -m scripts.explain_one --from_manifest test --random  

Output:
- artifacts/explain/<utt_id>/
  - report.json
  - mel_saliency.npy
  - seq_saliency.npy
  - time_importance.npy
  - mel_gradcam.npy

Render presentable PNGs (npy → viz/*.png):

python3 -m scripts.render_explain_assets --sample_dir artifacts/explain/<utt_id>  

Output:
- artifacts/explain/<utt_id>/viz/
  - mel.png
  - mel_saliency_overlay.png
  - mel_gradcam_overlay.png
  - time_importance.png
  - waveform_highlight.png

Package forensic bundle (ZIP):

python3 -m scripts.package_explain_bundle --sample_dir artifacts/explain/<utt_id>  

Optional: include audio in the bundle

python3 -m scripts.package_explain_bundle --sample_dir artifacts/explain/<utt_id> --include_audio  

---

## Feature 3 — Streamlit UI (single-file upload)

Run the app (from main/):

streamlit run src/app/streamlit_app.py  

If streamlit isn’t found:

python3 -m streamlit run src/app/streamlit_app.py  

### UI behavior
- Upload a single .flac or .wav
- App predicts p_fake
- If BONAFIDE → shows “No spoofing detected” and stops
- If SPOOF → shows forensic tabs + offers ZIP bundle download (optional include-audio)
