"""
features.py — Pi-side feature extraction for anomaly detection.

Loads a saved pump video, detects the active pump segment, and computes
the 14 features used by the deployed anomaly detection ensemble.

Feature groups
--------------
    1. Frequency      — dominant pump speed, spectral power, entropy
    2. Cycle metrics  — stroke amplitude, rise/fall timing, interval regularity
    3. Regularity     — autocorrelation, cycle shape consistency, envelope
    4. Spatial        — top/bottom and left/right motion correlation, quadrant CV
    5. Motion         — overall motion coefficient of variation

Public API
----------
    extract_features(video_path) -> dict | None
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, hilbert, welch

from pump_pipeline.config import (
    FPS,
    PUMP_ROI,
    DIFF_THRESH_FACTOR,
    DIFF_THRESH_MIN,
    PUMP_PERSIST_FRAMES,
    PEAK_MIN_DIST_SEC,
    PEAK_PROM_FACTOR,
    FREQ_LO_HZ,
    FREQ_HI_HZ,
    FREQ_WIN_SEC,
    FREQ_STEP_SEC,
    ACTIVITY_MASK_FACTOR,
    SHARPNESS_MIN,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(video_path: str) -> Optional[dict]:
    """
    Extract the 14 anomaly-detection features from a pump video.

    Returns a dict of feature values plus pump timing metadata, or None if
    the video is rejected (too short, blurry, or pump not detected).
    """
    signals = _load_signals(video_path)
    if signals is None:
        return None
    return _compute_features(signals)


# ──────────────────────────────────────────────────────────────────────────────
# Signal extraction
# ──────────────────────────────────────────────────────────────────────────────

def _load_signals(video_path: str) -> Optional[dict]:
    """
    Decode a video, isolate the active pump segment, and return raw signals.

    Steps
    -----
    1. Read all frames; apply Gaussian blur and crop to PUMP_ROI.
    2. Reject blurry videos via Laplacian variance on raw (unblurred) ROI crops.
    3. Compute per-frame ROI motion to find pump start / end frame indices.
    4. Build the active-pixel mask from temporal variance of the pump segment.
    5. Compute diff_sig (normalised active-pixel motion) and spatial sub-signals.

    Returns None on any rejection condition.
    """
    x1, y1, x2, y2 = PUMP_ROI
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open: %s", video_path)
        return None

    # Use camera FPS from file metadata; fall back to config default if missing
    fps: float = float(cap.get(cv2.CAP_PROP_FPS)) or FPS

    roi_frames:      list[np.ndarray] = []
    raw_roi_samples: list[np.ndarray] = []
    i_frame = 0

    ret, frame = cap.read()
    while ret:
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g        = cv2.GaussianBlur(gray_raw, (5, 5), 0)
        roi_frames.append(g[y1:y2, x1:x2].copy())

        # Collect raw (unblurred) crops every ~10 s for sharpness measurement
        if i_frame % 300 == 0:
            raw_roi_samples.append(gray_raw[y1:y2, x1:x2].copy())

        i_frame += 1
        ret, frame = cap.read()
    cap.release()

    N = len(roi_frames)
    if N < 60:
        log.warning("Too few frames (%d): %s", N, video_path)
        return None

    H, W = roi_frames[0].shape

    # Sharpness gate — measured on raw ROI to avoid blur artefacts suppressing
    # Laplacian variance. Videos below SHARPNESS_MIN are rejected.
    sharpness = float(np.mean(
        [cv2.Laplacian(r, cv2.CV_64F).var() for r in raw_roi_samples]
    )) if raw_roi_samples else 999.0

    if sharpness < SHARPNESS_MIN:
        log.warning("REJECTED — blurry (sharpness=%.1f): %s", sharpness, video_path)
        return None

    # Adaptive diff threshold — scales with mean ROI brightness to handle
    # lighting variation across recording sessions.
    roi_sample  = np.stack(roi_frames[N // 4: N // 4 + 30]).mean(axis=0)
    diff_thresh = max(DIFF_THRESH_MIN, int(roi_sample.mean() * DIFF_THRESH_FACTOR))

    # Per-frame ROI motion (pixel count) used only for pump on/off detection.
    # Fixed threshold of 15 is intentionally coarse here — we only need a
    # binary active/inactive signal, not a precise measurement.
    roi_mot = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        d = cv2.absdiff(roi_frames[i], roi_frames[i - 1])
        _, bm    = cv2.threshold(d, 15, 255, cv2.THRESH_BINARY)
        roi_mot[i] = float(np.count_nonzero(cv2.medianBlur(bm, 5)))

    # Pump active threshold: 5% of ROI pixels must change per frame
    roi_mot_thresh = max(10, int(H * W * 0.05))

    # Pump start: first run of PUMP_PERSIST_FRAMES consecutive active frames
    ps:     Optional[int] = None
    streak: int           = 0
    for i, v in enumerate(roi_mot):
        if v >= roi_mot_thresh:
            streak += 1
            if streak >= PUMP_PERSIST_FRAMES and ps is None:
                ps = i - PUMP_PERSIST_FRAMES + 1
        else:
            streak = 0

    # Pump end: last active frame
    pe: Optional[int] = next(
        (i for i in range(N - 1, -1, -1) if roi_mot[i] >= roi_mot_thresh), None
    )

    if ps is None or pe is None or pe <= ps:
        log.warning("Pump not detected: %s", video_path)
        return None

    roi_seg  = roi_frames[ps: pe + 1]
    T        = len(roi_seg)
    frames_f = np.stack([f.astype(np.float32) for f in roi_seg])

    # Active-pixel mask — pixels whose temporal variance exceeds the ROI mean
    # scaled by ACTIVITY_MASK_FACTOR belong to the moving pump body.
    # Dividing diff_sig by active_count makes the signal ROI-size invariant.
    var_map      = frames_f.var(axis=0)
    active_mask  = var_map > var_map.mean() * ACTIVITY_MASK_FACTOR
    active_count = max(1, int(active_mask.sum()))

    # Primary motion signal and spatial sub-signals over the pump segment
    diff_sig = np.zeros(T, dtype=np.float32)
    top_d    = np.zeros(T, dtype=np.float32)
    bot_d    = np.zeros(T, dtype=np.float32)
    left_d   = np.zeros(T, dtype=np.float32)
    right_d  = np.zeros(T, dtype=np.float32)

    for i in range(1, T):
        d      = cv2.absdiff(roi_seg[i], roi_seg[i - 1])
        _, bm  = cv2.threshold(d, diff_thresh, 255, cv2.THRESH_BINARY)
        bm_med = cv2.medianBlur(bm, 5)

        diff_sig[i] = float(np.count_nonzero(bm_med[active_mask])) / active_count
        top_d[i]    = float(np.mean(d[: H // 2, :]))
        bot_d[i]    = float(np.mean(d[H // 2:,  :]))
        left_d[i]   = float(np.mean(d[:,  : W // 2]))
        right_d[i]  = float(np.mean(d[:, W // 2:]))

    return dict(
        fps=fps, T=T, H=H, W=W,
        pump_start_s=ps / fps,
        pump_end_s=pe / fps,
        pump_duration_s=(pe - ps) / fps,
        diff=diff_sig,
        top=top_d, bot=bot_d, left=left_d, right=right_d,
        frames=frames_f,
        active_mask=active_mask,
        active_count=active_count,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feature computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_features(sig: dict) -> dict:
    """
    Compute the 14 anomaly-detection features from the raw signal dict.

    All features are scalar floats. CV = coefficient of variation (std/mean).
    A 1e-9 epsilon guards against division by zero throughout.
    """
    fps           = sig["fps"]
    seg           = sig["diff"]
    top           = sig["top"]
    bot           = sig["bot"]
    left          = sig["left"]
    right         = sig["right"]
    frames        = sig["frames"]
    T, H, W       = frames.shape
    activity_mask = sig["active_mask"]

    f: dict = {}

    # ── 1. Frequency ──────────────────────────────────────────────────────────
    # Welch PSD gives a stable spectral estimate on the motion signal.
    fw, psd = welch(seg, fps, nperseg=min(512, len(seg) // 4))
    bw_mask = (fw >= FREQ_LO_HZ) & (fw <= FREQ_HI_HZ)

    if np.any(bw_mask):
        peak_idx = int(np.argmax(psd[bw_mask]))
        dom      = float(fw[bw_mask][peak_idx])
        # Half-power bandwidth around dominant peak, minimum 20% of dom freq
        half_max          = psd[bw_mask][peak_idx] / 2.0
        fw_band, psd_band = fw[bw_mask], psd[bw_mask]
        left_i  = next((i for i in range(peak_idx, -1, -1)        if psd_band[i] < half_max), 0)
        right_i = next((i for i in range(peak_idx, len(fw_band))  if psd_band[i] < half_max), len(fw_band) - 1)
        bandwidth = max(fw_band[right_i] - fw_band[left_i], dom * 0.20)
        lo, hi   = dom - bandwidth, dom + bandwidth
    else:
        dom, lo, hi = 0.0, 0.0, 0.0

    f["dominant_freq_hz"] = dom

    # FFT-based spectral features over the full pump segment
    s_full   = seg - seg.mean()
    yf       = np.abs(rfft(s_full))
    xf       = rfftfreq(len(s_full), 1.0 / fps)
    ab       = (xf >= FREQ_LO_HZ) & (xf <= FREQ_HI_HZ)   # analysis band
    pb       = (xf >= lo) & (xf <= hi)                     # peak band

    total_pw = float(np.sum(yf[ab] ** 2)) + 1e-9
    f["spectral_power_in_band"] = float(np.sum(yf[pb] ** 2)) / total_pw
    f["spectral_snr"]           = float(yf[ab].max() / (np.median(yf[ab]) + 1e-9)) if np.any(ab) else 0.0
    psd_norm                    = psd[bw_mask] / (psd[bw_mask].sum() + 1e-9)
    f["spectral_entropy"]       = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))

    # Frequency stability — std of dominant frequency across sliding windows.
    # A 4 s window gives 0.25 Hz resolution, preventing bin-hopping at stable
    # speeds. High std indicates pump speed drift or wobbling.
    wf2  = int(fps * FREQ_WIN_SEC)
    sf2  = int(fps * FREQ_STEP_SEC)
    tv_f: list[float] = []
    for start in range(0, len(seg) - wf2 + 1, sf2):
        c   = seg[start: start + wf2] - seg[start: start + wf2].mean()
        yfc = np.abs(rfft(c))
        xfc = rfftfreq(len(c), 1.0 / fps)
        ba  = (xfc >= FREQ_LO_HZ) & (xfc <= FREQ_HI_HZ)
        tv_f.append(float(xfc[ba][np.argmax(yfc[ba])]) if np.any(ba) else 0.0)
    f["freq_stability_std"] = float(np.std(tv_f)) if tv_f else 0.0

    # ── 2. Cycle metrics ──────────────────────────────────────────────────────
    peaks, _   = find_peaks(seg,
                             distance=max(1, int(PEAK_MIN_DIST_SEC * fps)),
                             prominence=seg.std() * PEAK_PROM_FACTOR)
    valleys, _ = find_peaks(-seg,
                             distance=max(1, int(PEAK_MIN_DIST_SEC * fps)))

    # Inter-stroke interval CV — high value indicates irregular pump cadence
    if len(peaks) > 1:
        ivls             = np.diff(peaks) / fps * 1000   # milliseconds
        f["interval_cv"] = float(ivls.std() / (ivls.mean() + 1e-9))
    else:
        f["interval_cv"] = 0.0

    amps:   list[float] = []
    rise_t: list[float] = []
    fall_t: list[float] = []

    for pk in peaks:
        lv = valleys[valleys < pk]
        rv = valleys[valleys > pk]
        if len(lv) and len(rv):
            amps.append(float(seg[pk] - min(seg[lv[-1]], seg[rv[0]])))
        if len(lv):
            rise_t.append(float(pk - lv[-1]) / fps * 1000)
        if len(rv):
            fall_t.append(float(rv[0] - pk) / fps * 1000)

    amps_arr           = np.array(amps) if amps else np.array([0.0])
    f["stroke_amp_cv"] = float(amps_arr.std() / (amps_arr.mean() + 1e-9))

    if rise_t and fall_t:
        f["rise_fall_ratio"] = float(np.mean(rise_t) / (np.mean(fall_t) + 1e-9))
        f["rise_time_cv"]    = float(np.std(rise_t)  / (np.mean(rise_t) + 1e-9))
        f["fall_time_cv"]    = float(np.std(fall_t)  / (np.mean(fall_t) + 1e-9))
    else:
        f["rise_fall_ratio"] = f["rise_time_cv"] = f["fall_time_cv"] = 0.0

    # ── 3. Regularity ─────────────────────────────────────────────────────────
    # lag_p = one period in frames, used as autocorrelation lag and cycle length
    lag_p = int(round(fps / dom)) if dom > 0 else 3

    f["autocorr_at_period"] = (
        float(np.corrcoef(seg[:-lag_p], seg[lag_p:])[0, 1])
        if len(seg) > lag_p else 0.0
    )

    # Cycle shape consistency — each cycle is resampled to lag_p points then
    # compared across all cycles. High CV or low corr_mean indicates irregular shape.
    if len(peaks) > 4 and dom > 0:
        shapes: list[np.ndarray] = []
        for i in range(len(peaks) - 1):
            cyc = seg[peaks[i]: peaks[i + 1]]
            if len(cyc) > 3:
                shapes.append(
                    np.interp(np.linspace(0, 1, lag_p),
                              np.linspace(0, 1, len(cyc)), cyc)
                )
        if len(shapes) > 3:
            ca                   = np.array(shapes)
            f["cycle_shape_cv"]  = float(ca.std(axis=0).mean() / (ca.mean(axis=0).mean() + 1e-9))
            cc                   = [float(np.corrcoef(ca[i], ca[i + 1])[0, 1]) for i in range(len(ca) - 1)]
            f["cycle_corr_mean"] = float(np.mean(cc))
        else:
            f["cycle_shape_cv"] = f["cycle_corr_mean"] = 0.0
    else:
        f["cycle_shape_cv"] = f["cycle_corr_mean"] = 0.0

    # Envelope CV — Hilbert envelope of the motion signal; high CV indicates
    # amplitude modulation (e.g. partial occlusion, wobbling)
    envelope         = np.abs(hilbert(seg - seg.mean()))
    f["envelope_cv"] = float(envelope.std() / (envelope.mean() + 1e-9))

    # ── 4. Spatial correlation & symmetry ────────────────────────────────────
    # top_bot_corr — correlation between top and bottom half motion signals;
    # low value indicates asymmetric or localised movement
    f["top_bot_corr"] = float(np.corrcoef(top[1:], bot[1:])[0, 1]) if len(top) > 1 else 0.0

    # left_right_corr — computed on active pixel columns where possible for
    # robustness; falls back to raw spatial signals if no active columns found
    active_cols_left  = activity_mask[:, : W // 2].any(axis=0)
    active_cols_right = activity_mask[:, W // 2:].any(axis=0)
    if active_cols_left.any() and active_cols_right.any():
        l_ts = frames[:, :, : W // 2][:, :, active_cols_left].mean(axis=(1, 2))
        r_ts = frames[:, :, W // 2:][:, :, active_cols_right].mean(axis=(1, 2))
        f["left_right_corr"] = float(np.corrcoef(l_ts, r_ts)[0, 1])
    else:
        f["left_right_corr"] = float(np.corrcoef(left[1:], right[1:])[0, 1]) if len(left) > 1 else 0.0

    # quadrant_cv_max — max CV of frame-difference across the 4 ROI quadrants;
    # captures spatially localised motion irregularities
    qcvs: list[float] = []
    for ys, ye, xs, xe in [(0, H // 2, 0, W // 2), (0, H // 2, W // 2, W),
                           (H // 2, H, 0, W // 2), (H // 2, H, W // 2, W)]:
        qd = np.array([
            float(np.mean(np.abs(frames[j, ys:ye, xs:xe] - frames[j - 1, ys:ye, xs:xe])))
            for j in range(1, T)
        ])
        qcvs.append(float(qd.std() / (qd.mean() + 1e-9)) if len(qd) > 0 else 0.0)
    f["quadrant_cv_max"] = float(max(qcvs)) if qcvs else 0.0

    f["active_roi_fraction"] = float(activity_mask.mean())

    # ── 5. Motion statistics ──────────────────────────────────────────────────
    # motion_cv — global CV of the diff_sig; captures overall motion regularity
    f["motion_cv"] = float(seg.std() / (seg.mean() + 1e-9))

    # Pump timing metadata — passed through for CycleMonitor and logging
    f["pump_duration_s"] = sig["pump_duration_s"]
    f["pump_start_s"]    = sig["pump_start_s"]
    f["pump_end_s"]      = sig["pump_end_s"]

    return f