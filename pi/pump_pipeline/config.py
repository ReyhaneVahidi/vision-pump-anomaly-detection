"""
config.py — Single source of truth for the entire pump anomaly pipeline.

Sections:
    CAMERA / CAPTURE    — PiCamera2 settings
    STORAGE             — where videos are saved
    MOTION DETECTION    — trigger logic for the save thread
    FEATURE EXTRACTION  — ROI coords + signal parameters
    PIPELINE            — queue sizes, thread behaviour
"""


# ─────────────────────────────────────────────
#  CAMERA / CAPTURE
# ─────────────────────────────────────────────
FPS: int            = 30
RESOLUTION: tuple   = (1280, 720)
PIXEL_FORMAT: str   = "RGB888"     # picamera2 format string

# ─────────────────────────────────────────────
#  STORAGE
# ─────────────────────────────────────────────
SAVE_DIR: str = "/home/pi5/Desktop/captured/pipeline"
LOG_FILE: str = "pump_pipeline/models/pipeline_execution.log"

# ─────────────────────────────────────────────
#  MOTION DETECTION  (capture → save thread)
# ─────────────────────────────────────────────

# Region of interest used by the *live* motion trigger (x, y, w, h)
# This is the lightweight ROI checked in real-time on the Pi.
MOTION_ROI: tuple = (820, 320, 195, 195)       # (x, y, width, height)

MOTION_ABS_DIFF_THRESH: int  = 15     # absdiff pixel threshold
MOTION_MIN_PIXELS: int       = 60     # minimum changed pixels to count as motion
PERSIST_FRAMES: int          = 5      # consecutive active frames to confirm pump running (T3)
INACTIVE_TAIL_SECONDS: float = 4.0    # silence after last motion before closing clip
MAX_EVENT_SECONDS: float     = 180.0  # hard cap on clip length
COOLDOWN_SECONDS: float      = 1.0    # minimum gap between consecutive triggers

PRE_SECONDS: int  = 10                # seconds of pre-buffer written before trigger


# Frame preprocessing
GAUSSIAN_BLUR_K: tuple = (5, 5)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (post-save thread)
# ─────────────────────────────────────────────

# Fixed pump ROI for offline feature extraction (x1, y1, x2, y2) — pixel coords
_x, _y, _w, _h = MOTION_ROI
PUMP_ROI: tuple = (_x, _y, _x + _w, _y + _h)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION PARAMETERS (for the anomaly worker)
# ─────────────────────────────────────────────
FPS_DEFAULT:          float           = 30.0
DIFF_THRESH_FACTOR:   float           = 0.25
DIFF_THRESH_MIN:      int             = 15
PUMP_PERSIST_FRAMES:  int             = 5
PEAK_MIN_DIST_SEC:    float           = 0.07
PEAK_PROM_FACTOR:     float           = 0.35
FREQ_LO_HZ:           float           = 1.0
FREQ_HI_HZ:           float           = 20.0
ACTIVITY_MASK_FACTOR: float           = 0.5
SHARPNESS_MIN:        float           = 20.0
FREQ_WIN_SEC: float = 4.0
FREQ_STEP_SEC: float = 0.5

# ─────────────────────────────────────────────
#  MODELS / PATHS
# ─────────────────────────────────────────────
MODELS_DIR:         str = "pump_pipeline/models"
SCALER_PATH:        str = "pump_pipeline/models/robust_scaler.joblib"
FEATURE_NAMES_PATH: str = "pump_pipeline/models/feature_names_selected.joblib"
CYCLE_MODEL_PATH:   str = "pump_pipeline/models/cycle_model.json"
PERF_LOG_PATH:      str = "pump_pipeline/models/t3_performance_log.csv"

# ─────────────────────────────────────────────
#  PIPELINE  (queue sizes, worker behaviour)
# ─────────────────────────────────────────────

# Thread 1 → Thread 2  (raw frames, large to absorb jitter)
FRAME_QUEUE_SIZE: int = FPS * 10      # ~10 s of frames

# Thread 2 → Thread 3  (paths of completed video files)
VIDEO_QUEUE_SIZE: int = 32

# Thread 3 → caller    (extracted feature dicts)
RESULT_QUEUE_SIZE: int = 32
