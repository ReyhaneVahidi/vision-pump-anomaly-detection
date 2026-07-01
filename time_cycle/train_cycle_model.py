"""
train_cycle_model.py
====================
Fit a statistical schedule model from a labelled training CSV of normal pump
cycles and save it as cycle_model.json for deployment in CycleMonitor.

The model captures three things:
    1. Inter-cycle interval statistics  → detect missed / extra cycles
    2. Pump duration statistics         → detect stall / short runs
    3. Per-slot expected hour-of-day    → detect cycles at wrong time

The JSON output is consumed verbatim by CycleMonitor.check_cycle().

Usage
-----
    python train_cycle_model.py                       # uses defaults below
    python train_cycle_model.py --csv data/cycles.csv --out models/cycle_model.json

Input CSV (from pump_extractor.py or feature_extractor.py)
----------------------------------------------------------
    Required columns:
        file           — filename encoded as YYYYMMDD_HHMMSS.mp4
        pump_duration_s — extracted pump segment duration in seconds

Output JSON (cycle_model.json)
------------------------------
    trained_at              ISO timestamp of when the model was trained
    n_training_days         number of complete days used
    n_training_cycles       total cycle count after filtering
    interval_mean           mean intra-day inter-cycle interval [s]
    interval_std            std  intra-day inter-cycle interval [s]
    duration_mean           mean pump duration [s]
    duration_std            std  pump duration [s]
    expected_cycles_per_day median cycles per complete day
    expected_hours          list of mean hour-of-day per slot index

Deployed to
-----------
    pi/models/cycle_model.json   (copy after training on PC)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = "time_cycle/cycle_training.csv"
DEFAULT_OUT = "pi/pump_pipeline/models/cycle_model.json"   # deploy path; copy to Pi after training
# ─────────────────────────────────────────────────────────────────────────────


def parse_timestamp(filename: str) -> datetime:
    """Parse YYYYMMDD_HHMMSS.mp4 filename into a datetime object."""
    ts = filename.replace(".mp4", "")
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")


def train_cycle_model(csv_file: str, model_out: str) -> dict:
    """
    Fit a cycle schedule model from a training CSV and save it as JSON.

    Parameters
    ----------
    csv_file  : path to raw feature CSV (must contain 'file' and 'pump_duration_s')
    model_out : output path for cycle_model.json

    Returns
    -------
    model dict (same content as the saved JSON)
    """
    df = pd.read_csv(csv_file)

    # ── Parse and sort by timestamp ────────────────────────────────────────
    df["timestamp"] = df["file"].apply(parse_timestamp)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour_of_day"] = (
        df["timestamp"].dt.hour +
        df["timestamp"].dt.minute / 60 +
        df["timestamp"].dt.second / 3600
    )
    df["date"] = df["timestamp"].dt.date

    # ── Filter incomplete days ─────────────────────────────────────────────
    # A "complete" day must have at least as many cycles as the median.
    # Using >= 4 would silently accept days with half the expected cycles.
    cycles_per_day = df.groupby("date").size()
    median_cpd     = int(cycles_per_day.median())
    complete_days  = cycles_per_day[cycles_per_day >= median_cpd].index
    df = df[df["date"].isin(complete_days)].reset_index(drop=True)

    print(f"Complete days : {len(complete_days)}  |  "
          f"Total cycles  : {len(df)}  |  "
          f"Expected/day  : {median_cpd}")

    # ── Intra-day intervals only ───────────────────────────────────────────
    # Overnight gaps (last cycle of day N → first cycle of day N+1) are much
    # longer than intra-day gaps and would inflate interval_std, weakening the
    # missed-cycle detector.  We keep only consecutive same-day pairs.
    intra_mask  = df["date"] == df["date"].shift(1)
    intervals   = df.loc[intra_mask, "timestamp"].diff().dt.total_seconds().dropna()
    print(f"Intra-day intervals : mean={intervals.mean():.1f}s  "
          f"std={intervals.std():.1f}s  n={len(intervals)}")

    # ── Durations ──────────────────────────────────────────────────────────
    durations = df["pump_duration_s"]
    print(f"Duration      : mean={durations.mean():.1f}s  std={durations.std():.1f}s")

    # ── Per-slot expected hour-of-day ──────────────────────────────────────
    # Within each complete day, assign each cycle to its positional slot
    # (0 = first cycle of the day, 1 = second, …).  Averaging per slot gives
    # the expected clock hour for CycleMonitor's time-of-day check.
    slot_hours: dict[int, list[float]] = {i: [] for i in range(median_cpd)}

    for _date, group in df.groupby("date"):
        day_sorted = group.sort_values("timestamp").reset_index(drop=True)
        for slot_idx, row in day_sorted.iterrows():
            if slot_idx < median_cpd:
                slot_hours[slot_idx].append(row["hour_of_day"])

    expected_hours = []
    for i in range(median_cpd):
        hours = slot_hours[i]
        if hours:
            mean_h = float(np.mean(hours))
            expected_hours.append(round(mean_h, 6))
            print(f"  Slot {i}: {mean_h:.3f}h ± {np.std(hours):.3f}h  (n={len(hours)})")

    # ── Assemble model dict ────────────────────────────────────────────────
    # Field names must match exactly what CycleMonitor reads from the JSON.
    model = {
        "trained_at":              datetime.now().isoformat(timespec="seconds"),
        "n_training_days":         len(complete_days),
        "n_training_cycles":       len(df),
        "interval_mean":           round(float(intervals.mean()), 3),
        "interval_std":            round(float(intervals.std()),  3),
        "duration_mean":           round(float(durations.mean()), 3),
        "duration_std":            round(float(durations.std()),  3),
        "expected_cycles_per_day": median_cpd,
        "expected_hours":          expected_hours,
    }

    # ── Save ───────────────────────────────────────────────────────────────
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "w") as f:
        json.dump(model, f, indent=4)

    print(f"\nSaved → {model_out}")
    print(json.dumps(model, indent=4))
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleMonitor schedule model.")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Input feature CSV (default: %(default)s)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output JSON path (default: %(default)s)")
    args = parser.parse_args()

    train_cycle_model(csv_file=args.csv, model_out=args.out)