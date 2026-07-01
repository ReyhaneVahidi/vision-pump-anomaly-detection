# Vision-Based Peristaltic Pump Anomaly Detection

Camera-based anomaly detection for a peristaltic pump using a Raspberry Pi 5 + Camera Module 3.
Short video clips are reduced to motion-based features and scored by an ensemble of one-class
models trained only on normal pump operation.

---

## System Overview

1. Camera observes pump operation (Pi)
2. Motion detection triggers a short video recording
3. Each clip → 23 numerical features (frequency, cycle shape, spatial correlation, motion)
4. Feature selection narrows this to the 14 features the deployed models use
5. Five one-class models (Mahalanobis, LOF, Isolation Forest, Elliptic Envelope, OC-SVM),
   each trained only on normal behaviour, score the clip
6. Majority vote (≥3 of 5) decides whether the clip is anomalous
7. A separate cycle monitor checks activation timing/duration against a learned schedule

Thresholds are set via leave-one-out cross-validation (95th percentile of out-of-fold scores
on normal training data).

The repository has two independent halves:

- **`PC/`** — offline pipeline that turns your videos into trained models (below)
- **`pi/`** — real-time inference pipeline that runs those trained models on a Raspberry Pi
  ([Section 5](#5-raspberry-pi-deployment))

---

## Repository structure

```
PC/                      Training and evaluation pipeline (run on PC)
  feature_extractor.py     video -> 23-feature CSV
  preprocess.py             feature CSV -> clipped/scaled training features
  feature_selection.py      permutation-importance selection (23 -> 14 features)
  training.py               fits the 5 one-class models, LOO-CV thresholds
  evaluate.py               scores trained models on held-out normal/abnormal clips
  compare.py                aggregates evaluate.py output into comparison figures

pi/pump_pipeline/        Real-time inference pipeline (runs on the Raspberry Pi 5)
  main.py                   entry point; wires up the 3 worker threads
  camera.py                 Thread 1 - PiCamera2 capture loop
  motion_save.py            Thread 2 - motion trigger + video clip writer
  anomaly_worker.py         Thread 3 - feature extraction + ensemble scoring + logging
  cycle_monitor.py          schedule-anomaly check (interval/duration/time-of-day)
  feature_extraction/       Pi-side copy of the 14-feature extractor used at inference time
  config.py                 single source of truth for pipeline constants/paths (ROI, etc.)
  models/                   model artefacts loaded at runtime (see Section 5, step 3)

data/
  training/                 training videos' extracted features (raw + preprocessed)
  test/                     held-out normal/abnormal evaluation clips' features + diagnostic plots

models/                  trained model artefacts produced by PC/training.py
evaluation/              figures + metrics produced by PC/evaluate.py and PC/compare.py
time_cycle/              trains the pump activation schedule model (cycle_model.json)
other_figures/           standalone notebooks used to generate additional thesis/paper figures
```

---

## System Requirements

**PC pipeline:** Python 3.10/3.11, Linux/Windows/macOS.

**Raspberry Pi pipeline:** Raspberry Pi 5, Raspberry Pi OS (Bookworm), Camera Module 3 aimed at
a fixed region of interest containing the pump, Python 3 as shipped with Raspberry Pi OS.

---

## 1. Setup

```bash
git clone <repo-url>
cd <repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-pc.txt
```

---

## 2. Record videos and define the ROI

You need two separate video sets, recorded the same way (same camera position, same pump):

- **Training set** — normal pump operation only. This is what the models learn "normal" from.
- **Evaluation set** — a mix of normal and known-anomalous clips, held out from training. Used
  to measure detection rate / false positive rate.

For every video, the region of interest (ROI) — the rectangle of frame the pump body occupies —
must be defined. Any bounding-box annotation tool works (e.g. [CVAT](https://www.cvat.ai/)); the
result just needs to end up as a single JSON file mapping filename → box:

```json
{
  "20260202_205111.jpg": [x, y, w, h],
  "20260202_205230.jpg": [x, y, w, h]
}
```

`x, y, w, h` are pixel coordinates (top-left corner + width/height) in the source video frame.
One entry per video, keyed by the video's filename with a `.jpg` extension. Keep a separate ROI
JSON for the training set and the evaluation set (or one combined file — `feature_extractor.py`
just looks up each video by filename).

---

## 3. Run the PC pipeline

Run in order. Steps 1–4 use the **training** videos only; step 5 switches to the **evaluation**
videos.

```bash
# 1. Raw video -> 23-feature CSV (training videos)
python PC/feature_extractor.py
#    set VIDEOS_DIR, ROIS_JSON, OUT_CSV at the top of the script first

# 2. Zero-variance / near-constant / domain removal + clip + scale
python PC/preprocess.py

# 3. Permutation-importance feature selection (23 -> 14), refits scaler on the subset
python PC/feature_selection.py

# 4. Fit the 5 one-class models + LOO-CV thresholds
python PC/training.py

# 5. Raw video -> feature CSV, now for the EVALUATION videos
python PC/feature_extractor.py
#    point VIDEOS_DIR / ROIS_JSON / OUT_CSV at your eval set + ROI file this time

# 6. Score the trained models on the eval features
#    evaluate.py clips + scales internally using the scaler saved in step 3 —
#    do NOT run preprocess.py or feature_selection.py on eval data.
python PC/evaluate.py

# 7. Comparison figures
python PC/compare.py
```

**Output:**
- `models/*.joblib` — one file per trained model
- `evaluation/evaluation_summary.json` — per-model metrics (detection rate, FPR, precision, etc.)
- `evaluation/comparison_*.png/.pdf` — ROC curves, boxplots, detection heatmap, vote confusion matrix

> **Just want to see the pipeline run first, without your own videos?** `data/training/` and
> `data/test/` already contain both the raw and preprocessed feature CSVs from the original
> study. You can start directly at step 3 (`feature_selection.py`) or step 2 (`preprocess.py`,
> if you want to see the cleaning step too) and skip straight to modelling.

---

## 4. Time-cycle model (optional)

`time_cycle/train_cycle_model.py` fits the schedule model (expected interval / duration /
time-of-day) consumed by `cycle_monitor.py` on the Pi. Run it once against your training videos'
timestamps; it outputs `cycle_model.json`.

---

## 5. Raspberry Pi Deployment

### 1. Enable the camera
```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
sudo reboot
```

### 2. Install dependencies
```bash
sudo apt update
sudo apt install -y python3-picamera2   # not pip-installable
python -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt
```

### 3. Copy trained models from the PC

`anomaly_worker.py` loads its model assets from `pi/pump_pipeline/models/` at import time.
After Section 3 (and step 4 above) have produced these files, copy them over:

| From (PC)                                                                                   | To (Pi)                    |
|-------------------------------------------------------------------------------------------------|--------------------------------|
| `models/mahalanobis.joblib`, `lof.joblib`, `isoforest.joblib`, `elliptic_envelope.joblib`, `ocsvm.joblib` | `pi/pump_pipeline/models/` |
| `data/training/robust_scaler.joblib`                                                             | `pi/pump_pipeline/models/`    |
| `data/training/feature_names_selected.joblib`                                                    | `pi/pump_pipeline/models/`    |
| `time_cycle/cycle_model.json`                                                                     | `pi/pump_pipeline/models/`    |

Also set `MOTION_ROI` / `PUMP_ROI` in `pi/pump_pipeline/config.py` to match where the pump sits
in the Pi camera's frame (same idea as the ROI JSON in Section 2, but fixed once here since the
Pi's camera position doesn't change between clips).

### 4. Run the system
```bash
cd pi
python -m pump_pipeline.main
```

The camera starts continuously; motion detection triggers recording; each completed clip is
converted to features and scored by majority vote; the cycle monitor checks timing; results are
logged to `pump_pipeline/models/pipeline_execution.log` and streamed to stdout.

### Testing without hardware

Once step 3 has populated `pi/pump_pipeline/models/`, you can run the same feature extraction
and scoring logic on any local `.mp4` — no camera needed:

```bash
cd pi
python -c "from pump_pipeline.feature_extraction.features import extract_features; print(extract_features('test.mp4'))"
```

---

## Troubleshooting

**Camera not detected (Pi)** — confirm enabled via `sudo raspi-config` → Interface Options →
Camera, then reboot.

**`picamera2` import error** — must be installed via `apt`, not `pip`.

**No features generated / `extract_features` returns `None`** — check `MOTION_ROI` / `PUMP_ROI`
in `pi/pump_pipeline/config.py` actually contains the pump in frame.

**Model loading errors on the Pi** — confirm all files from the Section 5, step 3 table are
present in `pi/pump_pipeline/models/`.

**`evaluate.py` fails on eval features** — make sure step 5 in Section 3 (running
`feature_extractor.py` on eval videos) has been done, and that you did **not** run
`preprocess.py` or `feature_selection.py` on the eval data.

---

## Results

`evaluation/` contains ROC curves, score-distribution plots, the per-model detection heatmap,
the majority-vote confusion matrix (`comparison_*.png`/`.pdf`), and per-model metrics in
`evaluation_summary.json`.

## Citation

See [`CITATION.cff`](CITATION.cff). Paper citation will be added once the conference proceedings
are published.

## License

MIT — see [`LICENSE`](LICENSE).
