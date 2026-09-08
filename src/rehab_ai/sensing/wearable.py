from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from rehab_ai.domain import WearableFeatures, WearableSample


def extract_wearable_features(samples: list[WearableSample]) -> WearableFeatures:
    t = np.array([s.t for s in samples], dtype=float)
    ax = np.array([s.ax for s in samples], dtype=float)
    ay = np.array([s.ay for s in samples], dtype=float)
    az = np.array([s.az for s in samples], dtype=float)
    duration = float(t[-1] - t[0])
    fs = (len(t) - 1) / duration
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    dynamic = magnitude - np.mean(magnitude)
    accel_rms = float(np.sqrt(np.mean(dynamic**2)))
    jerk = np.diff(dynamic) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk**2))) if len(jerk) else 0.0
    min_distance = max(1, int(fs * 0.35))
    peaks, _ = find_peaks(az, distance=min_distance, prominence=max(0.04, float(np.std(az) * 0.25)))
    step_frequency = len(peaks) / duration if duration > 0 else 0.0
    sway_index = float(np.std(ay) + 0.5 * np.std(ax))
    return WearableFeatures(
        duration_s=duration,
        sample_rate_hz=float(fs),
        accel_rms=accel_rms,
        jerk_rms=jerk_rms,
        step_frequency_hz=float(step_frequency),
        sway_index=sway_index,
    )
