from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MovementSignalQuality:
    rms_acceleration: float
    jerk_rms: float
    spectral_entropy: float
    periodicity_index: float


def movement_signal_quality(acceleration: np.ndarray, sample_rate_hz: float) -> MovementSignalQuality:
    x = np.asarray(acceleration, dtype=float)
    if x.ndim != 1 or len(x) < 16 or sample_rate_hz <= 0:
        raise ValueError("acceleration must be 1D with >=16 samples and positive sample rate")
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x*x)))
    jerk = np.diff(x) * sample_rate_hz
    jerk_rms = float(np.sqrt(np.mean(jerk*jerk)))
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    if spectrum.sum() <= 1e-12:
        entropy = 0.0
        periodicity = 0.0
    else:
        p = spectrum / spectrum.sum()
        nz = p[p > 0]
        entropy = float(-np.sum(nz * np.log(nz)) / np.log(len(p))) if len(p) > 1 else 0.0
        # Peak non-DC spectral mass over total non-DC mass.
        non_dc = spectrum[1:]
        periodicity = float(np.max(non_dc) / np.sum(non_dc)) if non_dc.sum() > 0 else 0.0
    return MovementSignalQuality(rms, jerk_rms, entropy, periodicity)
