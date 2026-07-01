# Vision-Based Peristaltic Pump Anomaly Detection

This repository contains a reproducible system for camera-based anomaly detection of a peristaltic
pump using a Raspberry Pi 5 + Camera Module 3. The system extracts motion-based features from
short video clips and detects abnormal behaviour using an ensemble of one-class machine learning
models trained only on normal operation — no physical sensors on the pump itself.

It accompanies a conference paper and is released as that work's reproducibility artifact: the
full training/evaluation pipeline (PC) and the real-time inference pipeline (Raspberry Pi).

---

## Quick Start (PC only — no hardware required)

This runs the full training + evaluation pipeline on the feature data already included in this
repository (raw pump videos aren't shipped here — see [Repository structure](#repository-structure)
for why — so this works immediately, no recordings needed).

### 1. Clone the repository
```bash
git clone <repo-url>
cd <repo>
```

### 2. Create a Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements-pc.txt
```

### 4. Run the pipeline (from the repo root, in this order)
```bash
python PC/feature_selection.py
python PC/training.py
python PC/evaluate.py
python PC/compare.py
```

**Expected result:**
- Trained models in `models/*.joblib`
- Metrics in `evaluation/evaluation_summary.json`
- Plots (ROC, boxplot, heatmap, vote confusion matrix) in `evaluation/`

---

## System Requirements

### PC pipeline
- Python 3.10 or 3.11 (developed and tested on 3.11; exact versions in `requirements-pc.txt`)
- Linux / Windows / macOS
- No GPU required
- ~2 GB free disk space

### Raspberry Pi pipeline
- Raspberry Pi 5
- Raspberry Pi OS (Bookworm recommended)
- Camera Module 3, enabled and aimed at a fixed region of interest containing the pump
- Python 3 (as shipped with Raspberry Pi OS)

---

## System Overview

1. Camera observes pump operation (Pi)
2. Motion detection triggers a short video recording
3. Each clip is reduced to 23 numerical signal features (frequency, cycle shape,
   spatial-correlation, motion statistics)
4. Feature selection narrows this to the 14 features the deployed models use
5. Five one-class models (Mahalanobis distance, Local Outlier Factor, Isolation Forest,
   Elliptic Envelope, One-Class SVM), each trained only on normal behaviour, score the clip
6. Majority vote (≥3 of 5) decides whether the clip is anomalous
7. A separate cycle monitor checks activation timing/duration against a learned schedule

Thresholds for the 5 models are set via leave-one-out cross-validation (95th percentile of
out-of-fold scores on normal training data).

---

## Repository structure

```
PC/                      Training and evaluation pipeline (run on PC)
  feature_extractor.py     video -> 23-feature CSV (needs your own recordings; see note below)
  preprocess.py             feature CSV -> clipped/scaled training features
  feature_selection.py      permutation-importance feature selection (23 -> 14 features)
  training.py               fits the 5 one-class models, LOO-CV thresholds
  evaluate.py               scores trained models on held-out normal/abnormal clips
  compare.py                aggregates evaluate.py output into comparison figures

pi/pump_pipeline/        Real-time inference pipeline (runs on the Raspberry Pi 5)
  main.py                   entry point; wires up the 3 worker threads
  camera.py                 Thread 1 - PiCamera2 capture loop
  motion_save.py            Thread 2 - motion trigger + video clip writer
  anomaly_worker.py         Thread 3 - feature extraction + ensemble scoring + logging
  cycle_monitor.py         schedule-anomaly check (interval/duration/time-of-day)
  feature_extraction/       Pi-side copy of the 14-feature extractor used at inference time
  config.py                 single source of truth for all pipeline constants/paths (ROI, etc.)
  models/                   model artefacts loaded at runtime (see "Deploying to the Pi" below)

data/
  training/                 already-extracted + preprocessed training features and artefacts
  test/                     held-out normal/abnormal evaluation clips' features + diagnostic plots

models/                  trained model artefacts produced by PC/training.py
evaluation/              figures + metrics produced by PC/evaluate.py and PC/compare.py
time_cycle/              trains the pump activation schedule model (cycle_model.json)
other_figures/           standalone notebooks used to generate additional thesis/paper figures
```

**Note on raw video:** raw pump recordings are not included in this repository (large lab
footage). Everything downstream of feature extraction *is* included, so
`feature_selection.py -> training.py -> evaluate.py -> compare.py` is fully reproducible out of
the box. `feature_extractor.py` and `preprocess.py` are the two stages upstream of that — included
for reuse with your own recordings, but their inputs aren't shipped here.

---

## PC Pipeline (Training + Evaluation)

Run in this exact order, from the repository root:

```bash
python PC/feature_selection.py
python PC/training.py
python PC/evaluate.py
python PC/compare.py
```

| Step                    | Output                                                             |
|-------------------------|---------------------------------------------------------------------|
| `feature_selection.py`  | `data/training/feature_names_selected.joblib` (selected feature list) |
| `training.py`           | `models/*.joblib` (one file per model)                               |
| `evaluate.py`           | `evaluation/evaluation_summary.json`, `evaluation/evaluation_results.csv` |
| `compare.py`            | comparison plots (ROC, boxplot, heatmap, vote) in `evaluation/`      |

### (Optional) Full retraining from your own raw video

```bash
python PC/feature_extractor.py
python PC/preprocess.py
```

Input format and ROI configuration are documented in the docstring at the top of each script.

---

## Raspberry Pi Setup (step-by-step)

### 1. Enable the camera
```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
sudo reboot
```

### 2. Install system dependencies
```bash
sudo apt update
sudo apt install -y python3-picamera2   # picamera2 is not pip-installable
```

### 3. Create a Python environment
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements-pi.txt
```

### 5. Copy trained models from the PC

`pi/pump_pipeline/anomaly_worker.py` loads its model assets from `pi/pump_pipeline/models/` at
import time. After running the PC pipeline above, copy these files there:

| From (PC)                                                                              | To (Pi)                    |
|------------------------------------------------------------------------------------------|-------------------------------|
| `models/mahalanobis.joblib`, `lof.joblib`, `isoforest.joblib`, `elliptic_envelope.joblib`, `ocsvm.joblib` | `pi/pump_pipeline/models/` |
| `data/training/robust_scaler.joblib`                                                      | `pi/pump_pipeline/models/`    |
| `data/training/feature_names_selected.joblib`                                             | `pi/pump_pipeline/models/`    |
| `time_cycle/cycle_model.json` (output of `time_cycle/train_cycle_model.py`)                | `pi/pump_pipeline/models/`    |

### 6. Run the system
```bash
cd pi
python -m pump_pipeline.main
```

**What happens:** the camera starts continuously, motion detection triggers recording, each
completed clip is converted to features and scored by majority vote, the cycle monitor checks
timing, and results are logged to `pump_pipeline/models/pipeline_execution.log` and streamed to
stdout.

---

## Testing without hardware

You can exercise the feature extraction and inference logic on a PC/laptop using any local
`.mp4` file, without a camera. Run this from the `pi/` directory (so the `pump_pipeline` package
is importable), after step 5 above has populated `pi/pump_pipeline/models/`:

```bash
cd pi
python -c "from pump_pipeline.feature_extraction.features import extract_features; print(extract_features('test.mp4'))"
```

---

## Troubleshooting

**Camera not detected (Pi)**
- Confirm the camera is enabled: `sudo raspi-config` -> Interface Options -> Camera
- Reboot after enabling.

**`picamera2` import error**
- It must be installed via `apt`, not `pip`: `sudo apt install python3-picamera2`

**No features generated / `extract_features` returns `None`**
- Check the ROI configuration (`MOTION_ROI` / `PUMP_ROI`) in `pi/pump_pipeline/config.py` — it
  must actually contain the pump in frame.

**Model loading errors on the Pi**
- Confirm all files from the copy table in step 5 are present in `pi/pump_pipeline/models/`.

---

## Results

All evaluation outputs are in `evaluation/`, including ROC curves, score-distribution plots, the
per-model detection heatmap, the majority-vote confusion matrix
(`comparison_*.png`/`.pdf`), and per-model metrics in `evaluation/evaluation_summary.json`.

## Citation

See [`CITATION.cff`](CITATION.cff). The paper citation will be added once the conference
proceedings are published.

## License

MIT — see [`LICENSE`](LICENSE).
