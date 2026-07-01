"""
anomaly_worker.py — Thread 3: anomaly detection pipeline.

    video path
        → extract_features()          step 1 — 14-feature extraction
        → NaN fill + RobustScaler     step 2 — scale features
        → Majority Vote Ensemble      step 3 — 5-model inference (majority 3/5)
        → log timing + CPU/thermal    step 4 — real-time performance logging
        → CycleMonitor.check_cycle()  step 5 — cycle schedule anomaly
        → result_queue                step 6 — pass result to main thread
"""
import time
import logging
import os
import numpy as np
import pandas as pd
from queue import Queue, Empty
from threading import Event

import joblib
import psutil

from pump_pipeline.feature_extraction.features import extract_features
from pump_pipeline.cycle_monitor import CycleMonitor
import pump_pipeline.config as cfg

log = logging.getLogger(__name__)

# ── Load ML assets once at import time ───────────────────────────────────────
_MODELS = {}
_SCALER = None
_PUMP_FEATURE_ORDER = []

try:
    _SCALER = joblib.load(cfg.SCALER_PATH)

    # Feature order is loaded from the saved joblib artefact produced by
    # feature_selection.py — single source of truth, no manual list to maintain.
    _PUMP_FEATURE_ORDER = joblib.load(cfg.FEATURE_NAMES_PATH)

    # Load the 5 ensemble models
    _model_names = ["mahalanobis", "lof", "isoforest", "elliptic_envelope", "ocsvm"]
    for name in _model_names:
        path = os.path.join(cfg.MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            _MODELS[name] = joblib.load(path)
        else:
            log.error("Missing critical model file: %s", path)

    log.info("Ensemble loaded | Models: %s | Features: %d",
             list(_MODELS.keys()), len(_PUMP_FEATURE_ORDER))

except Exception as exc:
    log.error("Failed to load pump model assets: %s", exc)

# ── Load cycle monitor ───────────────────────────────────────────────────────
try:
    _CYCLE_MONITOR = CycleMonitor(cfg.CYCLE_MODEL_PATH)
    log.info("Cycle monitor loaded")
except Exception as exc:
    log.error("Failed to load cycle monitor: %s", exc)
    _CYCLE_MONITOR = None


def _score_model(name: str, bundle: dict, X_scaled: pd.DataFrame) -> tuple[float, bool]:
    """Return (score, is_anomaly) for one model. Higher score = more anomalous."""
    threshold = bundle["threshold"]
    try:
        if name == "mahalanobis":
            from scipy.spatial.distance import mahalanobis
            score = float(mahalanobis(X_scaled.values[0], bundle["mean"], bundle["cov_inv"]))
        elif name == "lof":
            score = float(-bundle["model"].score_samples(X_scaled.values)[0])
        else:
            score = float(-bundle["model"].decision_function(X_scaled.values)[0])

        # bool() cast prevents np.bool_ JSON serialisation errors
        return score, bool(score > threshold)

    except Exception as exc:
        log.error("Scoring failed for %s: %s", name, exc)
        return 0.0, False


def _majority_vote(video_path: str, X_scaled: pd.DataFrame) -> dict:
    """Score all 5 models and return vote summary. Anomaly if >= 3/5 models agree."""
    votes: dict[str, bool] = {}
    scores: dict[str, float] = {}

    for name, bundle in _MODELS.items():
        score, flag = _score_model(name, bundle, X_scaled)
        scores[name] = score
        votes[name] = flag

    n_anomaly = int(sum(votes.values()))
    is_anomaly = bool(n_anomaly >= 3)

    log.info(
        "Vote | %s | %d/5 anomaly | %s",
        os.path.basename(video_path),
        n_anomaly,
        " ".join(f"{k}={'A' if v else 'N'}" for k, v in votes.items()),
    )

    return {
        "is_anomaly":    is_anomaly,
        "anomaly_votes": n_anomaly,
        "anomaly_scores": scores,
        "anomaly_flags": votes,
    }


def anomaly_loop(video_queue: Queue, result_queue: Queue, stop: Event) -> None:
    # Log RAM footprint after models are loaded into this thread's process
    try:
        _mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        log.info("Model RAM footprint at thread startup: %.1f MB", _mem_mb)
    except Exception as mem_err:
        log.error("Failed to read startup memory footprint: %s", mem_err)

    log.info("Anomaly worker started")

    while not (stop.is_set() and video_queue.empty()):
        try:
            video_path = video_queue.get(timeout=1.0)
        except Empty:
            continue

        file_name = os.path.basename(video_path)
        log.info("Processing: %s", file_name)

        t_start = time.perf_counter()

        # ── Step 1: Feature extraction ────────────────────────────────────────
        result = extract_features(video_path)
        if result is None:
            continue

        t_features = time.perf_counter()

        # Fallback timestamps — ensure timing log never crashes if steps are skipped
        t_scale = t_features
        t_vote  = t_features

        # ── Steps 2 + 3: Scale + ensemble inference ───────────────────────────
        if _SCALER is not None and _MODELS:
            try:
                # Fill missing features with 0.0; replace any NaN produced by
                # degenerate signals (e.g. zero-length cycle) with 0.0.
                vector_data = [
                    [0.0 if np.isnan(v := result.get(feat, 0.0)) else v
                     for feat in _PUMP_FEATURE_ORDER]
                ]
                df_raw    = pd.DataFrame(vector_data, columns=_PUMP_FEATURE_ORDER)
                df_scaled = pd.DataFrame(
                    _SCALER.transform(df_raw), columns=_PUMP_FEATURE_ORDER
                )
                t_scale = time.perf_counter()

                result.update(_majority_vote(video_path, df_scaled))
                t_vote = time.perf_counter()

            except Exception as exc:
                log.error("Ensemble inference failed for %s: %s", video_path, exc)

        t_total = time.perf_counter()

        # ── Step 4: Log timing and system metrics ─────────────────────────────
        log.info(
            "T3 timing | total=%.2fs  features=%.2fs  scale=%.3fs  vote=%.3fs",
            t_total    - t_start,
            t_features - t_start,
            t_scale    - t_features,
            t_vote     - t_scale,
        )

        try:
            if not os.path.exists(cfg.PERF_LOG_PATH):
                with open(cfg.PERF_LOG_PATH, "w") as f:
                    f.write("timestamp,file_name,total_s,features_s,"
                            "scale_s,vote_s,cpu_percent,temp_c\n")

            current_cpu = psutil.cpu_percent(interval=None)

            # Read CPU temperature from Pi 5 thermal zone sysfs node
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as tz:
                    current_temp = float(tz.read()) / 1000.0
            except Exception:
                current_temp = 0.0

            with open(cfg.PERF_LOG_PATH, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{file_name},"
                        f"{t_total - t_start:.3f},{t_features - t_start:.3f},"
                        f"{t_scale - t_features:.3f},{t_vote - t_scale:.3f},"
                        f"{current_cpu:.1f},{current_temp:.1f}\n")

        except Exception as csv_err:
            log.error("Failed to write performance log: %s", csv_err)

        # ── Step 5: Cycle anomaly detection ───────────────────────────────────
        if _CYCLE_MONITOR is not None:
            try:
                result["cycle_anomalies"] = _CYCLE_MONITOR.check_cycle(
                    file_name=file_name,
                    pump_duration=result.get("pump_duration_s", 0.0),
                )
            except Exception as exc:
                log.error("Cycle monitoring failed: %s", exc)

        # ── Step 6: Send result to main thread ────────────────────────────────
        try:
            result_queue.put(result, timeout=1.0)
        except Exception:
            log.error("result_queue full — dropping result for %s", video_path)

    log.info("Anomaly worker stopped")