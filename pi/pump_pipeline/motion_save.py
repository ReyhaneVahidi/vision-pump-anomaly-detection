"""
motion_save.py — Thread 2: motion detection + video saving.

Consumes raw frames from frame_queue. When motion is detected,
opens a VideoWriter, flushes the pre-trigger buffer into it, and
records until the pump goes quiet. Completed video paths are pushed
onto video_queue for downstream processing.
"""

import os
import time
import logging
from collections import deque
from datetime import datetime
from queue import Queue, Empty
from threading import Event

import cv2

import pump_pipeline.config as cfg

log = logging.getLogger(__name__)


# ── VideoWriter helpers ────────────────────────────────────────────────────────

def _open_writer(frame) -> tuple:
    """Try MP4, fall back to AVI. Returns (VideoWriter, path) or (None, None)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = frame.shape[:2]

    for ext, fourcc_str in (("mp4", "mp4v"), ("avi", "MJPG")):
        path = os.path.join(cfg.SAVE_DIR, f"{ts}.{ext}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, cfg.FPS, (w, h))
        if writer.isOpened():
            return writer, path

    return None, None


def _close_event(
    writer, outfile: str, start_ts: float, frames_written: int, video_queue: Queue
) -> None:
    """Release writer, log timing report, push completed path to video_queue."""
    real_dur  = time.monotonic() - start_ts
    video_dur = frames_written / cfg.FPS
    post_dur  = video_dur - cfg.PRE_SECONDS
    loss      = real_dur - post_dur

    writer.release()
    log.info(
        "Saved: %s | real=%.1fs  video(post-pre)=%.1fs  lost=%.2fs",
        os.path.basename(outfile), real_dur, post_dur, loss,
    )

    try:
        video_queue.put(outfile, timeout=2.0)
        log.info("Queued for anomaly worker: %s", os.path.basename(outfile))
    except Exception:
        log.error("video_queue full — skipping: %s", outfile)


# ── ROI helper ────────────────────────────────────────────────────────────────

def _get_roi(img, roi):
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


# ── Thread entry point ────────────────────────────────────────────────────────

def motion_save_loop(frame_queue: Queue, video_queue: Queue, stop: Event) -> None:
    log.info("Motion/save thread started")

    prebuffer       = deque(maxlen=int(cfg.PRE_SECONDS * cfg.FPS))
    prev_gray       = None
    active_streak   = 0
    last_trigger_ts = 0.0
    recording       = False
    writer          = None
    outfile         = None
    last_motion_ts  = 0.0
    frames_written  = 0
    event_start_ts  = 0.0

    counter = 0
    t0      = time.monotonic()

    while not stop.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        counter += 1
        now = time.monotonic()
        if now - t0 >= 5.0:
            log.info("Processing FPS: %.1f  |  Q=%d", counter / (now - t0),
                     frame_queue.qsize())
            counter = 0
            t0 = now

        prebuffer.append(frame.copy())

        # ── Motion detection ──────────────────────────────────────────────────
        roi_img = _get_roi(frame, cfg.MOTION_ROI)
        gray    = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray    = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_now = False
        if prev_gray is None:
            prev_gray = gray
        else:
            diff = cv2.absdiff(prev_gray, gray)
            _, bm = cv2.threshold(diff, cfg.MOTION_ABS_DIFF_THRESH, 255, cv2.THRESH_BINARY)
            bm = cv2.medianBlur(bm, 5)

            if cv2.countNonZero(bm) >= cfg.MOTION_MIN_PIXELS:
                active_streak += 1
            else:
                active_streak = max(0, active_streak - 1)

            if active_streak >= cfg.PERSIST_FRAMES:
                motion_now     = True
                last_motion_ts = now
                active_streak  = cfg.PERSIST_FRAMES  # clamp

        prev_gray = gray

        # ── Trigger ───────────────────────────────────────────────────────────
        if motion_now and not recording and (now - last_trigger_ts) >= cfg.COOLDOWN_SECONDS:
            writer, outfile = _open_writer(frame)
            if writer is not None:
                frames_written = 0
                event_start_ts = now
                for f in prebuffer:
                    writer.write(f)
                    frames_written += 1
                recording = True
                log.info("Trigger → %s  (pre-buffer: %d frames)", outfile, frames_written)
            else:
                log.error("Could not open VideoWriter")
            last_trigger_ts = now

        # ── Record ────────────────────────────────────────────────────────────
        if recording and writer is not None:
            writer.write(frame)
            frames_written += 1

            tail_done   = last_motion_ts > 0 and (now - last_motion_ts) >= cfg.INACTIVE_TAIL_SECONDS
            cap_reached = (now - event_start_ts) >= cfg.MAX_EVENT_SECONDS

            if tail_done or cap_reached:
                _close_event(writer, outfile, event_start_ts, frames_written, video_queue)
                writer, outfile = None, None
                recording       = False
                active_streak   = 0
                last_motion_ts  = 0.0

    # Flush any open recording on shutdown
    if writer is not None:
        log.info("Flushing open writer on shutdown")
        _close_event(writer, outfile, event_start_ts, frames_written, video_queue)

    log.info("Motion/save thread stopped")
