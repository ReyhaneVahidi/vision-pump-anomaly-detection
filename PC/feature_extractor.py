"""
pump_extractor.py — Extract 23 ROI-size-invariant anomaly-detection features
from peristaltic pump videos and write diagnostics + a feature CSV.

Usage
-----
    python pump_extractor.py

ROI JSON format
---------------
    { "20260202_205111.jpg": [x, y, w, h], ... }

Output CSV columns
------------------
    file, pump_start_s, pump_end_s, pump_dur_s,
    roi_x1, roi_y1, roi_x2, roi_y2,
    <23 feature columns>

Dependencies
------------
    pip install opencv-python scipy numpy matplotlib pandas
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, hilbert, welch

# ══════════════════════════════════════════════════════════════════════════════
# IO CONFIG  
# ══════════════════════════════════════════════════════════════════════════════

VIDEOS_DIR: str = "data/videos"
ROIS_JSON:  str = "data/rois.json"
OUT_CSV:    str = "data/features.csv"
OUT_PLOTS:  str = "data/plots"

# ══════════════════════════════════════════════════════════════════════════════
# TUNING CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

FPS_DEFAULT:          float           = 30.0
DIFF_THRESH_FACTOR:   float           = 0.25
DIFF_THRESH_MIN:      int             = 20
PUMP_PERSIST_FRAMES:  int             = 6
PEAK_MIN_DIST_SEC:    float           = 0.07
PEAK_PROM_FACTOR:     float           = 0.35

# 4 seconds gives 0.25 Hz resolution: much more stable.
FREQ_WIN_SEC:         float           = 4.0   
FREQ_STEP_SEC:        float           = 0.5
FREQ_LO_HZ:           float           = 1.0
FREQ_HI_HZ:           float           = 20.0
FLOW_DOWNSAMPLE:      float           = 0.5
ACTIVITY_MASK_FACTOR: float           = 0.5
CLAHE_CLIP:           float           = 2.0
CLAHE_GRID:           tuple[int, int] = (4, 4)
# minimum Laplacian variance for the ROI to be considered sharp enough.
# Videos below this are rejected with status "BLURRY" rather than scored.
SHARPNESS_MIN:        float           = 25.0   # calibrated on RAW (unblurred) ROI frames

# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "dominant_freq_hz",
    "spectral_power_in_band",
    "spectral_snr",
    "spectral_entropy",
    "harmonic2_ratio",
    "freq_stability_std",
    "interval_cv",
    "stroke_amp_cv",
    "rise_fall_ratio",
    "rise_time_cv",
    "fall_time_cv",
    "autocorr_at_period",
    "cycle_shape_cv",
    "cycle_corr_mean",
    "phase_portrait_eccentricity",
    "envelope_cv",
    "top_bot_corr",
    "left_right_corr",
    "quadrant_cv_max",
    "active_roi_fraction",
    "motion_cv",
    "flow_dom_freq",
    #"flow_dom_freq_global",
    "freq_flow_diff",
]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_signals(video_path: str, roi: tuple[int, int, int, int]) -> dict | None:
    """Decode video, detect pump window, return raw signal dict; None on failure."""
    x1, y1, x2, y2 = roi
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open: %s", video_path)
        return None

    fps: float = float(cap.get(cv2.CAP_PROP_FPS)) or FPS_DEFAULT
    roi_frames:  list[np.ndarray] = []

    raw_roi_samples: list[np.ndarray] = []   # unblurred, for sharpness check
    i_frame: int = 0
    ret, frame = cap.read()
    while ret:
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g        = cv2.GaussianBlur(gray_raw, (5, 5), 0)
        roi_frames.append(g[y1:y2, x1:x2].copy())
        # collect a few raw (unblurred) ROI crops spread across the video
        if i_frame % 300 == 0:          # every 10 s at 30 fps
            raw_roi_samples.append(gray_raw[y1:y2, x1:x2].copy())
        i_frame += 1
        ret, frame = cap.read()
    cap.release()

    N: int = len(roi_frames)
    if N < 60:
        log.warning("Too few frames (%d): %s", N, video_path)
        return None

    H, W = roi_frames[0].shape

    # ── blurry video gate ──────────────────────────────────────────────
    # Measure sharpness on RAW (unblurred) ROI frames.

    if raw_roi_samples:
        sharpness_vals = [float(cv2.Laplacian(r, cv2.CV_64F).var()) for r in raw_roi_samples]
        sharpness: float = float(np.mean(sharpness_vals))
    else:
        sharpness = 999.0   # fallback: no samples collected, do not reject
    log.info("  sharpness (Laplacian var, raw ROI) = %.1f", sharpness)
    if sharpness < SHARPNESS_MIN:
        log.warning(
            "REJECTED — blurry video (sharpness=%.1f < %.1f): %s",
            sharpness, SHARPNESS_MIN, video_path,
        )
        return None
    # ─────────────────────────────────────────────────────────────────────────



    roi_sample: np.ndarray = np.stack(roi_frames[N // 4 : N // 4 + 30]).mean(axis=0)
    diff_thresh: int = max(DIFF_THRESH_MIN, int(roi_sample.mean() * DIFF_THRESH_FACTOR))
    log.info("  adaptive diff_thresh=%d (roi_mean=%.1f)", diff_thresh, roi_sample.mean())

    # ROI-based motion for pump on/off detection
    roi_mot: np.ndarray = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        d = cv2.absdiff(roi_frames[i], roi_frames[i - 1])
        _, bm = cv2.threshold(d, 15, 255, cv2.THRESH_BINARY)
        roi_mot[i] = float(np.count_nonzero(cv2.medianBlur(bm, 5)))

    # fraction-based threshold: 5% of ROI pixels must be active
    roi_mot_thresh: int = max(10, int(H * W * 0.05))
    log.info("  roi_mot_thresh=%d (5%% of %dx%d ROI)", roi_mot_thresh, W, H)

    # ── pump start: first run of PUMP_PERSIST_FRAMES consecutive active frames ─
    ps: int | None = None
    streak: int    = 0
    for i, v in enumerate(roi_mot):
        if v >= roi_mot_thresh:
            streak += 1
            if streak >= PUMP_PERSIST_FRAMES and ps is None:
                ps = i - PUMP_PERSIST_FRAMES + 1
        else:
            streak = 0

    # ── pump end: last active frame ───────────────────────────────────────────
    pe: int | None = next(
        (i for i in range(N - 1, -1, -1) if roi_mot[i] >= roi_mot_thresh), None
    )

    if ps is None or pe is None or pe <= ps:
        log.warning("Pump not detected: %s", video_path)
        return None

    log.info("  pump %.1fs → %.1fs (%.1fs)", ps / fps, pe / fps, (pe - ps) / fps)

    roi_seg: list[np.ndarray] = roi_frames[ps : pe + 1]
    T: int = len(roi_seg)
    frames_f: np.ndarray = np.stack([f.astype(np.float32) for f in roi_seg])

    # ── active pixel masking for diff_sig ──────────────────────────────
    # Compute per-pixel temporal variance on the pump segment.
    # Only pixels whose variance exceeds the mean * factor are "active" —
    # i.e. they belong to the moving pump body, not static background.
    # Dividing by active_count instead of H*W makes the signal invariant to
    # how much background the ROI contains (different zoom levels / angles).
    var_map_load: np.ndarray      = frames_f.var(axis=0)
    active_mask_load: np.ndarray  = var_map_load > var_map_load.mean() * ACTIVITY_MASK_FACTOR
    active_count: int             = max(1, int(active_mask_load.sum()))
    log.info(
        "  active pixels = %d / %d (%.1f%%)",
        active_count, H * W, 100 * active_count / (H * W),
    )
    # ─────────────────────────────────────────────────────────────────────────

    diff_sig = np.zeros(T, dtype=np.float32)
    top_d    = np.zeros(T, dtype=np.float32)
    bot_d    = np.zeros(T, dtype=np.float32)
    left_d   = np.zeros(T, dtype=np.float32)
    right_d  = np.zeros(T, dtype=np.float32)

    for i in range(1, T):
        d = cv2.absdiff(roi_seg[i], roi_seg[i - 1])
        _, bm = cv2.threshold(d, diff_thresh, 255, cv2.THRESH_BINARY)
        bm_med = cv2.medianBlur(bm, 5)

        # divide by active pixel count, not full ROI area
        diff_sig[i] = float(np.count_nonzero(bm_med[active_mask_load])) / active_count

        top_d[i]  = float(np.mean(d[: H // 2, :]))
        bot_d[i]  = float(np.mean(d[H // 2 :, :]))
        left_d[i] = float(np.mean(d[:, : W // 2]))
        right_d[i] = float(np.mean(d[:, W // 2 :]))

    # optical flow
    flow_sig = np.zeros(T, dtype=np.float32)
    ds   = FLOW_DOWNSAMPLE
    prev = cv2.resize(roi_seg[0], (max(8, int(W * ds)), max(8, int(H * ds))))
    diag = float(np.hypot(prev.shape[0], prev.shape[1]) + 1e-6)
    fb   = dict(pyr_scale=0.5, levels=3, winsize=20,
                iterations=3, poly_n=5, poly_sigma=1.7, flags=0)
    for i in range(1, T):
        cur = cv2.resize(roi_seg[i], prev.shape[::-1])
        fl  = cv2.calcOpticalFlowFarneback(prev, cur, None, **fb)
        mag, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        flow_sig[i] = float(np.mean(np.clip(mag, 0, np.percentile(mag, 99))) / diag)
        prev = cur

    return dict(
        fps=fps, T=T, H=H, W=W,
        pump_start_s=ps / fps,
        pump_end_s=pe / fps,
        pump_dur_s=(pe - ps) / fps,
        sharpness=sharpness,
        diff=diff_sig, flow=flow_sig,
        top=top_d, bot=bot_d, left=left_d, right=right_d,
        frames=frames_f,
        diff_thresh=diff_thresh,
        active_mask=active_mask_load,   # pass through so compute_features can reuse it
        active_count=active_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(sig: dict) -> dict:
    """Compute 23 ROI-size-invariant features."""
    fps    = sig["fps"]
    seg    = sig["diff"]
    flow   = sig["flow"]
    top    = sig["top"]
    bot    = sig["bot"]
    left   = sig["left"]
    right  = sig["right"]
    frames = sig["frames"]
    T, H, W = frames.shape
    f: dict = {}

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    frames_eq: np.ndarray = np.stack(
        [clahe.apply(frames[i].astype(np.uint8)) for i in range(T)]
    ).astype(np.float32)

    # Reuse the activity mask computed in load_signals (already calibrated to
    # this video's pump segment). Fall back to recomputing if not present.
    if "active_mask" in sig:
        activity_mask: np.ndarray = sig["active_mask"]
    else:
        var_map: np.ndarray = frames.var(axis=0)
        activity_mask = var_map > var_map.mean() * ACTIVITY_MASK_FACTOR

    # ── 1. FREQUENCY / PUMP SPEED ─────────────────────────────────────────────
    fw, psd = welch(seg, fps, nperseg=min(512, len(seg) // 4))
    bw_mask = (fw >= FREQ_LO_HZ) & (fw <= FREQ_HI_HZ)

    if np.any(bw_mask):
        peak_idx: int = int(np.argmax(psd[bw_mask]))
        dom: float    = float(fw[bw_mask][peak_idx])
        half_max      = psd[bw_mask][peak_idx] / 2.0
        fw_band       = fw[bw_mask]
        psd_band      = psd[bw_mask]
        left_i  = next((i for i in range(peak_idx, -1, -1) if psd_band[i] < half_max), 0)
        right_i = next((i for i in range(peak_idx, len(fw_band)) if psd_band[i] < half_max),
                       len(fw_band) - 1)
        bandwidth = max(fw_band[right_i] - fw_band[left_i], dom * 0.20)
        lo: float = dom - bandwidth
        hi: float = dom + bandwidth
    else:
        dom, lo, hi = 0.0, 0.0, 0.0

    f["dominant_freq_hz"] = dom

    s_full = seg - seg.mean()
    yf     = np.abs(rfft(s_full))
    xf     = rfftfreq(len(s_full), 1.0 / fps)
    ab     = (xf >= FREQ_LO_HZ) & (xf <= FREQ_HI_HZ)
    pb     = (xf >= lo) & (xf <= hi)
    ph2    = (xf >= dom * 1.8) & (xf <= dom * 2.2)

    total_pw: float = float(np.sum(yf[ab] ** 2)) + 1e-9
    f["spectral_power_in_band"] = float(np.sum(yf[pb] ** 2)) / total_pw
    f["spectral_snr"]           = (
        float(yf[ab].max() / (np.median(yf[ab]) + 1e-9)) if np.any(ab) else 0.0
    )
    psd_norm = psd[bw_mask] / (psd[bw_mask].sum() + 1e-9)
    f["spectral_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
    f["harmonic2_ratio"]  = (
        float(np.sum(yf[ph2] ** 2)) / (float(np.sum(yf[pb] ** 2)) + 1e-9)
    )

    # FREQ_WIN_SEC is now 4.0 (set in constants above).
    # A 4-second window at 30 fps gives 0.25 Hz frequency resolution,
    # so a pump running steadily at 10.8 Hz will not flip between 10 and 11 Hz.
    wf2 = int(fps * FREQ_WIN_SEC)
    sf2 = int(fps * FREQ_STEP_SEC)
    tv_f: list[float] = []
    for start in range(0, len(seg) - wf2 + 1, sf2):
        c   = seg[start : start + wf2] - seg[start : start + wf2].mean()
        yfc = np.abs(rfft(c))
        xfc = rfftfreq(len(c), 1.0 / fps)
        ba  = (xfc >= FREQ_LO_HZ) & (xfc <= FREQ_HI_HZ)
        tv_f.append(float(xfc[ba][np.argmax(yfc[ba])]) if np.any(ba) else 0.0)
    tv_arr = np.array(tv_f)
    f["freq_stability_std"] = float(tv_arr.std())

    # ── 2. STROKE CYCLE ───────────────────────────────────────────────────────
    peaks, _   = find_peaks(seg,
                             distance=max(1, int(PEAK_MIN_DIST_SEC * fps)),
                             prominence=seg.std() * PEAK_PROM_FACTOR)
    valleys, _ = find_peaks(-seg,
                             distance=max(1, int(PEAK_MIN_DIST_SEC * fps)))

    if len(peaks) > 1:
        ivls = np.diff(peaks) / fps * 1000
        f["interval_cv"] = float(ivls.std() / (ivls.mean() + 1e-9))
    else:
        ivls = np.array([0.0])
        f["interval_cv"] = 0.0

    amps: list[float] = []
    for pk in peaks:
        lv = valleys[valleys < pk]
        rv = valleys[valleys > pk]
        if len(lv) and len(rv):
            amps.append(float(seg[pk] - min(seg[lv[-1]], seg[rv[0]])))
    amps_arr = np.array(amps) if amps else np.array([0.0])
    f["stroke_amp_cv"] = float(amps_arr.std() / (amps_arr.mean() + 1e-9))

    rise_t: list[float] = []
    fall_t: list[float] = []
    for pk in peaks:
        lv = valleys[valleys < pk]
        rv = valleys[valleys > pk]
        if len(lv):
            rise_t.append(float(pk - lv[-1]) / fps * 1000)
        if len(rv):
            fall_t.append(float(rv[0] - pk) / fps * 1000)
    if rise_t and fall_t:
        f["rise_fall_ratio"] = float(np.mean(rise_t) / (np.mean(fall_t) + 1e-9))
        f["rise_time_cv"]    = float(np.std(rise_t)  / (np.mean(rise_t) + 1e-9))
        f["fall_time_cv"]    = float(np.std(fall_t)  / (np.mean(fall_t) + 1e-9))
    else:
        f["rise_fall_ratio"] = f["rise_time_cv"] = f["fall_time_cv"] = 0.0

    # ── 3. REGULARITY ─────────────────────────────────────────────────────────
    lag_p: int = int(round(fps / dom)) if dom > 0 else 3
    f["autocorr_at_period"] = (
        float(np.corrcoef(seg[:-lag_p], seg[lag_p:])[0, 1]) if len(seg) > lag_p else 0.0
    )

    if lag_p < len(seg):
        x_pp, y_pp = seg[:-lag_p], seg[lag_p:]
        eigs = np.linalg.eigvalsh(np.cov(x_pp, y_pp))
        f["phase_portrait_eccentricity"] = float(
            np.sqrt(1 - (eigs.min() / (eigs.max() + 1e-9)))
        )
    else:
        f["phase_portrait_eccentricity"] = 0.0

    if len(peaks) > 4 and dom > 0:
        shapes: list[np.ndarray] = []
        for i in range(len(peaks) - 1):
            cyc = seg[peaks[i] : peaks[i + 1]]
            if len(cyc) > 3:
                shapes.append(
                    np.interp(np.linspace(0, 1, lag_p),
                              np.linspace(0, 1, len(cyc)), cyc)
                )
        if len(shapes) > 3:
            ca = np.array(shapes)
            f["cycle_shape_cv"]  = float(
                ca.std(axis=0).mean() / (ca.mean(axis=0).mean() + 1e-9)
            )
            cc = [float(np.corrcoef(ca[i], ca[i + 1])[0, 1]) for i in range(len(ca) - 1)]
            f["cycle_corr_mean"] = float(np.mean(cc))
        else:
            f["cycle_shape_cv"] = f["cycle_corr_mean"] = 0.0
    else:
        f["cycle_shape_cv"] = f["cycle_corr_mean"] = 0.0

    envelope = np.abs(hilbert(seg - seg.mean()))
    f["envelope_cv"] = float(envelope.std() / (envelope.mean() + 1e-9))

    # ── 4. SPATIAL ────────────────────────────────────────────────────────────
    f["top_bot_corr"] = float(np.corrcoef(top[1:], bot[1:])[0, 1])

    active_cols_left  = activity_mask[:, : W // 2].any(axis=0)
    active_cols_right = activity_mask[:, W // 2 :].any(axis=0)
    if active_cols_left.any() and active_cols_right.any():
        l_ts = frames[:, :, : W // 2][:, :, active_cols_left].mean(axis=(1, 2))
        r_ts = frames[:, :, W // 2 :][:, :, active_cols_right].mean(axis=(1, 2))
        f["left_right_corr"] = float(np.corrcoef(l_ts, r_ts)[0, 1])
    else:
        f["left_right_corr"] = float(np.corrcoef(left[1:], right[1:])[0, 1])

    qcvs: list[float] = []
    for ys, ye, xs, xe in [
        (0, H // 2, 0, W // 2), (0, H // 2, W // 2, W),
        (H // 2, H, 0, W // 2), (H // 2, H, W // 2, W),
    ]:
        qd = np.array([
            float(np.mean(np.abs(frames[j, ys:ye, xs:xe] - frames[j - 1, ys:ye, xs:xe])))
            for j in range(1, T)
        ])
        qcvs.append(float(qd.std() / (qd.mean() + 1e-9)))
    f["quadrant_cv_max"] = float(max(qcvs))

    f["active_roi_fraction"] = float(activity_mask.mean())

    # ── 5. MOTION / FLOW ─────────────────────────────────────────────────────
    f["motion_cv"] = float(seg.std() / (seg.mean() + 1e-9))

    fw_f, psd_f = welch(flow[1:], fps, nperseg=min(512, len(flow) // 4))
    pump_band = (fw_f >= 7.0) & (fw_f <= 14.0)
    flow_dom: float = float(fw_f[pump_band][np.argmax(psd_f[pump_band])]) if np.any(pump_band) else 0.0
    flow_dom_global: float = float(fw_f[np.argmax(psd_f)])
    f["flow_dom_freq"]        = flow_dom
    #f["flow_dom_freq_global"] = flow_dom_global
    f["freq_flow_diff"]       = abs(dom - flow_dom)

    # private helpers for plotting
    f["_peaks"]   = peaks
    f["_valleys"] = valleys
    f["_tv_f"]    = tv_arr
    f["_dom"]     = dom
    f["_lo"]      = lo
    f["_hi"]      = hi
    f["_amps"]    = amps_arr
    f["_ivls"]    = ivls

    return f


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOT  
# ══════════════════════════════════════════════════════════════════════════════

def save_diagnostic(name: str, sig: dict, feat: dict, out_path: str) -> None:
    fps    = sig["fps"]
    seg    = sig["diff"]
    top    = sig["top"]
    bot    = sig["bot"]
    frames = sig["frames"]
    T, H, W = frames.shape
    t = np.arange(T) / fps

    peaks   = feat["_peaks"]
    valleys = feat["_valleys"]
    tv_f    = feat["_tv_f"]
    dom     = feat["_dom"]
    lo, hi  = feat["_lo"], feat["_hi"]
    amps    = feat["_amps"]
    ivls    = feat["_ivls"]

    C = "#2E7D32"
    fig = plt.figure(figsize=(20, 20))
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.52, wspace=0.35)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, seg, lw=0.6, color=C, alpha=0.9)
    ax.plot(t[peaks],   seg[peaks],   "v", color="k",    ms=2.5, alpha=0.4)
    ax.plot(t[valleys], seg[valleys], "^", color="gray", ms=2,   alpha=0.3)
    ax.set_title(
        f"{name}\n"
        f"dom={dom:.2f} Hz / {dom*60:.0f} RPM   "
        f"{len(peaks)} strokes   pump={sig['pump_dur_s']:.1f}s   "
        f"sharpness={sig.get('sharpness', 0):.0f}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ROI motion (active-masked)"); ax.set_xlim(0, t[-1])

    ax = fig.add_subplot(gs[0, 2])
    pv = frames.var(axis=0)
    im = ax.imshow(pv, cmap="hot", aspect="equal")
    ax.contour((pv > pv.mean() * 1.5).astype(float), levels=[0.5],
               colors="cyan", linewidths=1.5)
    plt.colorbar(im, ax=ax, label="Temporal variance")
    ax.set_title("Pixel variance map\ncyan = active region", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(gs[1, :2])
    fw2, psd2 = welch(seg, fps, nperseg=min(512, len(seg) // 4))
    bw2 = (fw2 >= 1) & (fw2 <= 20)
    ax.semilogy(fw2[bw2], psd2[bw2], color=C, lw=2)
    ax.axvspan(lo, hi, alpha=0.15, color=C, label=f"band {lo:.1f}–{hi:.1f} Hz")
    ax.axvline(dom, color="k", lw=1.2, ls=":")
    ax.set_title(
        f"Welch PSD\n"
        f"band_pwr={feat['spectral_power_in_band']:.3f}   "
        f"SNR={feat['spectral_snr']:.1f}   "
        f"entropy={feat['spectral_entropy']:.3f}   "
        f"h2={feat['harmonic2_ratio']:.3f}",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8); ax.set_xlabel("Hz"); ax.set_xlim(0, 22)

    ax = fig.add_subplot(gs[1, 2])
    tv_t = np.arange(len(tv_f)) * FREQ_STEP_SEC
    pt_c = [C if lo <= v <= hi else "#FF6F00" for v in tv_f]
    ax.scatter(tv_t, tv_f, c=pt_c, s=25, zorder=3)
    ax.plot(tv_t, tv_f, lw=0.6, color="#9E9E9E", zorder=2)
    ax.axhspan(lo, hi, alpha=0.12, color=C)
    ax.set_title(
        f"Freq stability (4s windows)\nstd={feat['freq_stability_std']:.3f} Hz",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz"); ax.set_ylim(0, 22)

    ax = fig.add_subplot(gs[2, 0])
    n_cyc = 10
    zoom  = min(int(n_cyc / dom * fps) if dom > 0 else int(fps * 3), len(seg))
    tz    = np.arange(zoom) / fps
    pkz, _ = find_peaks(seg[:zoom], distance=max(1, int(PEAK_MIN_DIST_SEC * fps)),
                        prominence=seg.std() * PEAK_PROM_FACTOR)
    vlz, _ = find_peaks(-seg[:zoom], distance=max(1, int(PEAK_MIN_DIST_SEC * fps)))
    ax.plot(tz, seg[:zoom], lw=1.5, color=C)
    ax.plot(tz[pkz], seg[:zoom][pkz], "v", color="k",    ms=6, zorder=5)
    ax.plot(tz[vlz], seg[:zoom][vlz], "^", color="gray", ms=5, zorder=5)
    ax.fill_between(tz, seg[:zoom], alpha=0.15, color=C)
    ax.set_title(
        (f"~{n_cyc} cycles zoom\nperiod ≈ {1/dom*1000:.0f} ms" if dom > 0 else "Cycle zoom"),
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Motion")

    ax = fig.add_subplot(gs[2, 1])
    if len(ivls) > 1:
        ax.hist(ivls, bins=40, color=C, alpha=0.75, edgecolor="white")
        ax.axvline(ivls.mean(), color="k", lw=2, label=f"mean={ivls.mean():.1f}ms")
        ax.axvline(ivls.mean() - 2 * ivls.std(), color="red", lw=1, ls="--")
        ax.axvline(ivls.mean() + 2 * ivls.std(), color="red", lw=1, ls="--")
        ax.legend(fontsize=8)
    ax.set_title(f"Inter-stroke interval\ncv={feat['interval_cv']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("ms")

    ax = fig.add_subplot(gs[2, 2])
    ax.hist(amps, bins=35, color=C, alpha=0.75, edgecolor="white")
    ax.axvline(amps.mean(), color="k", lw=2, label=f"mean={amps.mean():.4f}")
    ax.set_title(f"Stroke amplitude dist\ncv={feat['stroke_amp_cv']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.set_xlabel("Amplitude")

    ax = fig.add_subplot(gs[3, 0])
    lag_p = int(round(fps / dom)) if dom > 0 else 3
    shapes_plot: list[np.ndarray] = []
    for i in range(len(peaks) - 1):
        cyc = seg[peaks[i] : peaks[i + 1]]
        if len(cyc) > 3:
            shapes_plot.append(
                np.interp(np.linspace(0, 1, lag_p), np.linspace(0, 1, len(cyc)), cyc)
            )
    if shapes_plot:
        ca = np.array(shapes_plot)
        xc = np.linspace(0, 1, lag_p)
        for row in ca[:: max(1, len(ca) // 40)]:
            ax.plot(xc, row, color=C, lw=0.5, alpha=0.25)
        ax.plot(xc, ca.mean(axis=0), color="k", lw=2, label="mean")
        ax.fill_between(xc, ca.mean(axis=0) - ca.std(axis=0),
                            ca.mean(axis=0) + ca.std(axis=0),
                        alpha=0.2, color=C)
    ax.set_title(
        f"Cycle shape overlay\nshape_cv={feat['cycle_shape_cv']:.3f}   "
        f"cycle_corr={feat['cycle_corr_mean']:.3f}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Normalised position"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 1])
    lag_q = max(1, int(round(fps / dom / 4))) if dom > 0 else 3
    ax.scatter(seg[:-lag_q], seg[lag_q:], c=np.arange(len(seg) - lag_q),
               cmap="plasma", s=1, alpha=0.15, rasterized=True)
    ax.set_title(
        f"Phase portrait\neccentricity={feat['phase_portrait_eccentricity']:.3f}   "
        f"envelope_cv={feat['envelope_cv']:.3f}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("x[t]"); ax.set_ylabel(f"x[t+{lag_q}]")

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t, top / (top.max() + 1e-9), lw=0.8, color="#E91E63", alpha=0.8, label="Top")
    ax.plot(t, bot / (bot.max() + 1e-9), lw=0.8, color="#00897B", alpha=0.8, label="Bot")
    ax.set_title(
        f"Top vs Bottom motion\ncorr={feat['top_bot_corr']:.3f}   "
        f"left_right_corr={feat['left_right_corr']:.3f}   "
        f"active_frac={feat['active_roi_fraction']:.3f}",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8); ax.set_xlabel("Time (s)"); ax.set_xlim(0, t[-1])


    plt.suptitle(f"Pump Anomaly Analysis — {name}", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  plot saved: %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(
    videos_dir: str,
    rois_path:  str,
    out_csv:    str,
    out_dir:    str,
) -> pd.DataFrame:
    roi_db: dict = json.load(open(rois_path))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    mp4s: list[Path] = sorted(Path(videos_dir).glob("*.mp4"))
    log.info("Found %d .mp4 files", len(mp4s))

    rows:    list[dict] = []
    skipped: list[str]  = []

    for mp4 in mp4s:
        key = mp4.stem + ".jpg"
        if key not in roi_db:
            log.warning("No ROI for %s — skipping", mp4.name)
            skipped.append(mp4.name)
            continue

        log.info("Processing %s ...", mp4.name)
        x, y, w, h = roi_db[key]
        roi: tuple[int, int, int, int] = (int(x), int(y), int(x + w), int(y + h))

        signals = load_signals(str(mp4), roi)
        if signals is None:
            log.error("  FAILED: %s", mp4.name)
            skipped.append(mp4.name)
            continue

        feat = compute_features(signals)

        plot_path = str(Path(out_dir) / (mp4.stem + ".png"))
        save_diagnostic(mp4.stem, signals, feat, plot_path)

        row: dict = {
            "file":         mp4.name,
            "pump_start_s": round(signals["pump_start_s"], 2),
            "pump_end_s":   round(signals["pump_end_s"],   2),
            "pump_dur_s":   round(signals["pump_dur_s"],   2),
            "roi_x1": roi[0], "roi_y1": roi[1],
            "roi_x2": roi[2], "roi_y2": roi[3],
        }
        for col in FEATURE_COLS:
            row[col] = round(feat.get(col, 0.0), 6)
        rows.append(row)

        log.info(
            "  dom=%.2f Hz  band_pwr=%.3f  freq_std=%.3f  interval_cv=%.3f",
            feat["dominant_freq_hz"],
            feat["spectral_power_in_band"],
            feat["freq_stability_std"],
            feat["interval_cv"],
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info("Saved %d rows → %s", len(df), out_csv)
    if skipped:
        log.warning("Skipped %d videos: %s", len(skipped), skipped)
    return df


if __name__ == "__main__":
    run(
        videos_dir=VIDEOS_DIR,
        rois_path=ROIS_JSON,
        out_csv=OUT_CSV,
        out_dir=OUT_PLOTS,
    )