"""
cycle_monitor.py — Stateful cycle schedule anomaly detector.

Checks each pump activation against a learned schedule model (JSON).
Detects three types of scheduling anomalies:
    - missed_cycle   : interval since last activation exceeds 3 std devs
    - extra_cycle    : interval is shorter than expected by 3 std devs
    - wrong_duration : pump ran for an unexpected duration
    - wrong_time     : activation occurred outside the expected daily time window

The model JSON must contain:
    interval_mean   — expected seconds between activations
    interval_std    — standard deviation of inter-activation intervals
    duration_mean   — expected pump run duration in seconds
    duration_std    — standard deviation of pump duration
    expected_hours  — list of expected activation times as fractional hours
"""

import json
from datetime import datetime
import numpy as np


class CycleMonitor:
    """
    Stateful monitor that checks each pump cycle against a learned schedule.
    State (last timestamp, daily cycle count) is updated on every call to
    check_cycle() and resets automatically at midnight.
    """

    def __init__(self, model_path: str) -> None:
        """Load the schedule model from a JSON file."""
        with open(model_path) as f:
            self.model = json.load(f)

        self.last_timestamp = None
        self.today_cycles   = 0
        self.current_day    = None

    def _parse_timestamp(self, filename: str) -> datetime:
        """Parse activation timestamp from video filename (YYYYMMDD_HHMMSS.mp4)."""
        return datetime.strptime(filename.replace(".mp4", ""), "%Y%m%d_%H%M%S")

    def check_cycle(self, file_name: str, pump_duration: float) -> list[str]:
        """
        Check one pump activation against the schedule model.

        Parameters
        ----------
        file_name     : video filename, used to extract the activation timestamp
        pump_duration : measured pump run duration in seconds

        Returns
        -------
        List of anomaly labels (empty if everything is normal).
        """
        anomalies = []
        ts   = self._parse_timestamp(file_name)
        hour = ts.hour + ts.minute / 60 + ts.second / 3600
        day  = ts.date()

        # Reset daily cycle counter at midnight
        if self.current_day != day:
            self.today_cycles = 0
            self.current_day  = day

        # Interval anomaly — compare gap since last activation to learned distribution
        if self.last_timestamp is not None:
            interval = (ts - self.last_timestamp).total_seconds()
            mean, std = self.model["interval_mean"], self.model["interval_std"]

            if abs(interval - mean) > 3 * std:
                anomalies.append("missed_cycle" if interval > mean else "extra_cycle")

        # Duration anomaly — pump ran too long or too short
        dur_mean, dur_std = self.model["duration_mean"], self.model["duration_std"]
        if abs(pump_duration - dur_mean) > 3 * dur_std:
            anomalies.append("wrong_duration")

        # Time-of-day anomaly — activation outside expected daily window (±30 min)
        expected_hours = self.model["expected_hours"]
        if self.today_cycles < len(expected_hours):
            if abs(hour - expected_hours[self.today_cycles]) > 0.5:
                anomalies.append("wrong_time")

        # Update state for next call
        self.today_cycles  += 1
        self.last_timestamp = ts

        return anomalies