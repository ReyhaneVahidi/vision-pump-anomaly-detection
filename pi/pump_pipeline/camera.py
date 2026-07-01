"""
camera.py — Thread 1: PiCamera2 capture loop.

Continuously pulls frames from the camera and pushes them onto
frame_queue. Drops frames silently when the queue is full so the
camera never stalls.
"""

import time
import logging
from queue import Queue
from threading import Event

log = logging.getLogger(__name__)


def capture_loop(picam2, frame_queue: Queue, stop: Event) -> None:
    log.info("Capture thread started")
    counter = 0
    t0 = time.monotonic()

    while not stop.is_set():
        frame = picam2.capture_array()
        counter += 1

        now = time.monotonic()
        if now - t0 >= 5.0:
            log.info("Capture FPS: %.1f", counter / (now - t0))
            counter = 0
            t0 = now

        try:
            frame_queue.put(frame, timeout=0.05)
        except Exception:
            pass  # queue full — frame dropped

    log.info("Capture thread stopped")
