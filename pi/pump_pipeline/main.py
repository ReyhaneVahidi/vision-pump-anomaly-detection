"""
main.py — Pipeline entry point.

Responsibilities:
    - Start the camera
    - Create the shared queues
    - Spin up the three worker threads
    - Consume results from the anomaly worker
    - Shut everything down cleanly on Ctrl-C

All logic lives in the modules below.
"""

import os
import sys
import time
import json
import logging
from queue import Queue, Empty
from threading import Event, Thread

import pump_pipeline.config as cfg
from pump_pipeline.camera import capture_loop
from pump_pipeline.motion_save import motion_save_loop
from pump_pipeline.anomaly_worker import anomaly_loop

# ── Logging Configuration ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(cfg.LOG_FILE),  # Saves EVERYTHING persistently to disk
        logging.StreamHandler(sys.stdout)    # Keeps displaying live text on screen
    ]
)
log = logging.getLogger("main")


def handle_result(result: dict) -> None:
    """
    Called in the main thread for every completed result dict.
    Plug your anomaly scoring / alerting logic here, or move it
    fully into anomaly_worker.py once the model is ready.
    """
    log.info("Result received for video anomaly validation loop.")
    
    # Render JSON block safely formatted using the log file engine
    cleaned_result = {k: round(v, 5) if isinstance(v, float) else v for k, v in result.items()}
    log.info("Detailed Payload:\n%s", json.dumps(cleaned_result, indent=2))


def main() -> None:
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    try:
        from picamera2 import Picamera2
    except ImportError:
        log.error("picamera2 not found — are you on a Pi?")
        sys.exit(1)

    # ── Camera ────────────────────────────────────────────────────────────────
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": cfg.RESOLUTION, "format": cfg.PIXEL_FORMAT},
        controls={"FrameRate": cfg.FPS},
    ))
    picam2.start()
    time.sleep(0.5)
    log.info("Camera ready  %dx%d @ %d fps", *cfg.RESOLUTION, cfg.FPS)

    # ── Queues ────────────────────────────────────────────────────────────────
    frame_queue  = Queue(maxsize=cfg.FRAME_QUEUE_SIZE)
    video_queue  = Queue(maxsize=cfg.VIDEO_QUEUE_SIZE)
    result_queue = Queue(maxsize=cfg.RESULT_QUEUE_SIZE)
    stop         = Event()

    # ── Threads ───────────────────────────────────────────────────────────────
    threads = [
        Thread(target=capture_loop,
               args=(picam2, frame_queue, stop),
               name="T1-Capture", daemon=True),
        Thread(target=motion_save_loop,
               args=(frame_queue, video_queue, stop),
               name="T2-MotionSave", daemon=True),
        Thread(target=anomaly_loop,
               args=(video_queue, result_queue, stop),
               name="T3-AnomalyWorker", daemon=True),
    ]
    for t in threads:
        t.start()

    log.info("Pipeline running — Ctrl-C to stop")

    # ── Main loop: drain results ───────────────────────────────────────────────
    try:
        while True:
            try:
                res_data = result_queue.get(timeout=1.0)
                if res_data is not None:
                    handle_result(res_data)
            except Empty:
                pass

    except KeyboardInterrupt:
        log.info("Shutting down…")
        stop.set()
        for t in threads:
            t.join(timeout=15)
            if t.is_alive():
                log.warning("%s did not exit cleanly", t.name)

    finally:
        picam2.stop()
        picam2.close()
        log.info("Exit.")


if __name__ == "__main__":
    main()