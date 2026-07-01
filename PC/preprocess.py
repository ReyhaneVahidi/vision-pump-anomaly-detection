"""
preprocess_training.py

All statistics (variance checks, clipping thresholds, and scaling
parameters) are estimated exclusively from the training set to prevent
data leakage into later evaluation stages.
======================
Preprocessing pipeline for pump anomaly detection — training data only.

Applies a three-step quality filtering pipeline, clips outliers, and scales
the data for use by feature_selection.py. The scaler fitted here is
intermediate — feature_selection.py refits and saves the final scaler on
the selected feature subset only.

Pipeline steps (all fitted on training data only):
    1. Zero-variance removal     — features with std == 0
    2. Near-constant removal     — features where >95% of values are identical
    3. Domain-justified removal  — features identified as non-discriminative
    4. Outlier clipping          — per-feature 1st–99th percentile
    5. Robust scaling            — RobustScaler (median + IQR)

Why RobustScaler:
    Scaling is based on the median and interquartile range (IQR),
    making it less sensitive to outliers than StandardScaler.

Input:
    <CSV_PATH>   raw feature CSV from pump_extractor.py

Outputs (all to OUT_DIR):
    features_scaled.csv      scaled training features (used by feature_selection.py)
    features_raw.csv         clipped but unscaled (for inspection)
    feature_names.joblib     ordered feature list after preprocessing

Note: robust_scaler.joblib and clip_bounds.joblib are saved by
feature_selection.py on the final selected feature subset, not here.

Run:
    python preprocess_training.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH  = "data/features.csv"
OUT_DIR   = "data/training"

CLIP_LOW  = 1
CLIP_HIGH = 99

META_COLS = [
    "file", "pump_start_s", "pump_end_s", "pump_dur_s",
    "roi_x1", "roi_y1", "roi_x2", "roi_y2",
]

DOMAIN_EXCLUSIONS = [
    #"flow_dom_freq_global",
    #"freq_stability_std",
]

REMOVE_FILES: list[str] = []
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} recordings, {df.shape[1]} columns")

df = df[~df["file"].isin(REMOVE_FILES)].reset_index(drop=True)
if REMOVE_FILES:
    print(f"After removing {len(REMOVE_FILES)} bad files: {len(df)} samples")

feature_cols = [c for c in df.columns if c not in META_COLS]
X = df[feature_cols].copy()

if X.isna().sum().sum() > 0:
    print("[WARN] NaNs found — filling with column median")
    X = X.fillna(X.median())

print(f"\nStep 0 — features entering pipeline: {X.shape[1]}")
print(f"  {list(X.columns)}")

# ── 2. ZERO-VARIANCE REMOVAL ──────────────────────────────────────────────────
zero_var = X.columns[X.std() == 0].tolist()
print(f"\nStep 1 — zero-variance removal: {zero_var}")
X = X.drop(columns=zero_var)
print(f"  -> {X.shape[1]} features remain")

# ── 3. NEAR-CONSTANT REMOVAL ──────────────────────────────────────────────────
near_const = []
for c in X.columns:
    top_freq = X[c].value_counts(normalize=True).iloc[0]
    if top_freq > 0.95:
        near_const.append(c)
        print(f"  [near-constant] {c}: {top_freq*100:.1f}% identical values")

print(f"\nStep 2 — near-constant removal: {near_const}")
X = X.drop(columns=near_const)
print(f"  -> {X.shape[1]} features remain")

# ── 4. DOMAIN-JUSTIFIED REMOVAL ───────────────────────────────────────────────
missing_domain = [f for f in DOMAIN_EXCLUSIONS if f not in X.columns]
if missing_domain:
    print(f"[WARN] Domain exclusions not found in data: {missing_domain}")

present_domain = [f for f in DOMAIN_EXCLUSIONS if f in X.columns]
print(f"\nStep 3 — domain-justified removal: {present_domain}")
X = X.drop(columns=present_domain)
print(f"  -> {X.shape[1]} features remain")

print(f"\nFinal feature set ({X.shape[1]}):")
for i, c in enumerate(X.columns):
    print(f"  {i+1:2d}. {c}")

# ── 5. CLIP TO 1ST-99TH PERCENTILE ───────────────────────────────────────────
clip_bounds = {}
X_clipped = X.copy()
for col in X.columns:
    lo = float(np.percentile(X[col], CLIP_LOW))
    hi = float(np.percentile(X[col], CLIP_HIGH))
    clip_bounds[col] = (lo, hi)
    X_clipped[col] = X[col].clip(lo, hi)
print(f"\nStep 4 — clipping done (1st–99th percentile)")

# ── 6. ROBUST SCALING ─────────────────────────────────────────────────────────
scaler = RobustScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_clipped),
    columns=X.columns,
)
print(f"Step 5 — RobustScaler fitted")

low_var_after = X_scaled.std()[X_scaled.std() < 1e-3]
if len(low_var_after) > 0:
    print(f"[WARN] Near-zero variance after scaling: {low_var_after.index.tolist()}")

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
# Only feature_names.joblib and the scaled/raw CSVs are saved here.
# robust_scaler.joblib and clip_bounds.joblib are saved by feature_selection.py
# after refitting on the final selected feature subset.
joblib.dump(list(X.columns), os.path.join(OUT_DIR, "feature_names.joblib"))

df[["file"]].join(X_scaled).to_csv(
    os.path.join(OUT_DIR, "features_scaled.csv"), index=False
)
df[["file"]].join(X_clipped).to_csv(
    os.path.join(OUT_DIR, "features_raw.csv"), index=False
)

print(f"\nArtefacts saved to: {OUT_DIR}")
print(f"Note: robust_scaler.joblib and clip_bounds.joblib will be saved")
print(f"      by feature_selection.py on the final selected feature subset.")
print(f"\nDONE — {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features")