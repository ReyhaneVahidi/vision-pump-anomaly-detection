# Vision-Based Peristaltic Pump Anomaly Detection

Camera-only anomaly detection for a peristaltic pump: a Raspberry Pi 5 with a Camera Module 3
watches the pump, extracts motion-based signal features from short video clips, and flags
anomalies with an ensemble of one-class models — no physical sensors on the pump itself.

This repository accompanies a conference paper and is released as the reproducibility artifact
for that work: the full training/evaluation pipeline (PC) and the real-time inference pipeline
(Raspberry Pi) used in the study.

## How it works, in one paragraph

A motion-detection thread on the Pi watches a fixed region of interest and records a short clip
whenever the pump activates. Each clip is reduced to 23 ROI-size-invariant signal features
(frequency, cycle shape, spatial-correlation, and motion statistics). A fixed preprocessing +
feature-selection pipeline (fit once, offline, on normal-operation clips) narrows this to 14
features and feeds them to five one-class models — Mahalanobis distance, Local Outlier Factor,
Isolation Forest, Elliptic Envelope, and One-Class SVM — trained only on normal pump behaviour.
Thresholds are set via leave-one-out cross-validation (95th percentile of out-of-fold scores).
A clip is flagged anomalous by majority vote (≥3 of 5 models). A separate schedule model
(`time_cycle/`) checks whether cycles occur at the expected time/duration/interval.

## Repository structure

```
PC/                   Training pipeline: run on a laptop/desktop, not the Pi
  preprocess.py          raw feature CSV -> clipped/scaled training features
  feature_selection.py   permutation-importance feature selection -> final feature list
  training.py             fits the 5 (+ optional ECOD) one-class models, LOO-CV thresholds
  evaluate.py             scores trained models on held-out normal/abnormal clips
  compare.py              aggregates evaluate.py output into comparison figures
  feature_extractor.py    raw pump video -> the 23-feature CSV (needs your own videos)

pi/pump_pipeline/     Real-time inference pipeline: runs on the Raspberry Pi 5
  main.py                entry point; wires up the 3 worker threads
  camera.py               Thread 1 - PiCamera2 capture loop
  motion_save.py          Thread 2 - motion trigger + video clip writer
  anomaly_worker.py        Thread 3 - feature extraction + ensemble scoring + logging
  cycle_monitor.py        schedule-anomaly check (interval/duration/time-of-day)
  feature_extraction/      Pi-side copy of the 14-feature extractor used at inference time
  config.py                single source of truth for all pipeline constants/paths
  models/                  model artefacts loaded at runtime (see "Deploying to the Pi" below)

data/
  training/                already-extracted + preprocessed training features and artefacts
  test/                     held-out normal/abnormal evaluation clips' features + diagnostic plots

models/                trained model artefacts produced by PC/training.py
evaluation/            figures + metrics produced by PC/evaluate.py and PC/compare.py
time_cycle/            trains the pump activation schedule model (cycle_model.json)
other_figures/          standalone notebooks used to generate additional thesis/paper figures
```

## What you can reproduce out of the box

Raw pump videos are not included in this repository (they're large lab recordings). What *is*
included is every artefact downstream of feature extraction, so the modelling side of the
pipeline is fully reproducible without any hardware:

```
data/training/features_scaled.csv  ->  feature_selection.py  ->  training.py  ->  evaluate.py  ->  compare.py
```

`feature_extractor.py` (raw video → feature CSV) and `preprocess.py` (feature CSV → clipped/scaled
training CSV) are included for completeness and reuse with your own recordings, but their inputs
aren't shipped here — they are the two stages upstream of what's already in `data/training/`.

## PC setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-pc.txt
```

Run from the repository root (all scripts use paths relative to the repo root, e.g. `data/training`,
`models/`, `evaluation/`):

```bash
python PC/feature_selection.py   # data/training/features_scaled.csv -> feature_names_selected.joblib
python PC/training.py            # -> models/*.joblib
python PC/evaluate.py            # -> evaluation/evaluation_results.csv, evaluation_summary.json
python PC/compare.py             # -> evaluation/comparison_*.png/.pdf
```

To extract features from your own pump videos and retrain from scratch, additionally run
`PC/feature_extractor.py` and `PC/preprocess.py` first (see the docstring at the top of each file
for the exact expected input format, including the ROI JSON schema).

## Raspberry Pi setup

Hardware: Raspberry Pi 5 + Camera Module 3, camera aimed at a fixed region of interest containing
the pump.

```bash
sudo apt install -y python3-picamera2   # picamera2 is not pip-installable
pip install -r requirements-pi.txt
```

### Deploying a trained model to the Pi

`pi/pump_pipeline/anomaly_worker.py` loads its model assets from `pi/pump_pipeline/models/` at
import time. After running the PC training pipeline, copy these files there:

| From (PC)                                              | To (Pi)                              |
|----------------------------------------------------------|----------------------------------------|
| `models/mahalanobis.joblib`, `lof.joblib`, `isoforest.joblib`, `elliptic_envelope.joblib`, `ocsvm.joblib` | `pi/pump_pipeline/models/` |
| `data/training/robust_scaler.joblib`                      | `pi/pump_pipeline/models/`              |
| `data/training/feature_names_selected.joblib`             | `pi/pump_pipeline/models/`              |
| `time_cycle/cycle_model.json` (output of `time_cycle/train_cycle_model.py`) | `pi/pump_pipeline/models/` |

### Running the pipeline

```bash
cd pi
python -m pump_pipeline.main
```

`main.py` starts the camera and three daemon threads: capture → motion-triggered clip saving →
feature extraction + ensemble scoring + cycle check. Results are logged to
`pump_pipeline/models/pipeline_execution.log` and streamed to stdout.

You don't need the Pi hardware to exercise the inference logic itself: `extract_features()` in
`pi/pump_pipeline/feature_extraction/features.py` can be called directly on any local `.mp4` file
to test feature extraction, and the scoring logic in `anomaly_worker.py` can be driven the same way
once the model assets above are in place.

## Results

See `evaluation/` for ROC curves, score-distribution plots, the per-model detection heatmap, and
the majority-vote confusion matrix (`comparison_*.png/.pdf`), plus per-model metrics in
`evaluation/evaluation_summary.json`.

## Citation

See [`CITATION.cff`](CITATION.cff). The paper citation will be added once the conference
proceedings are published.

## License

MIT — see [`LICENSE`](LICENSE).
