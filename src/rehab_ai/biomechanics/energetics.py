from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JointEnergetics:
    positive_work_j: float
    negative_work_j: float
    net_work_j: float
    peak_abs_power_w: float
    mechanical_cost_j_per_kg: float


def joint_power(moment_nm: np.ndarray, angular_velocity_rad_s: np.ndarray) -> np.ndarray:
    m = np.asarray(moment_nm, dtype=float)
    w = np.asarray(angular_velocity_rad_s, dtype=float)
    if m.shape != w.shape or m.ndim != 1:
        raise ValueError("moment and angular velocity must be same-length 1D arrays")
    return m * w


def joint_energetics(moment_nm: np.ndarray, angular_velocity_rad_s: np.ndarray,
                     time_s: np.ndarray, body_mass_kg: float) -> JointEnergetics:
    p = joint_power(moment_nm, angular_velocity_rad_s)
    t = np.asarray(time_s, dtype=float)
    if t.shape != p.shape or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing and match signal length")
    if body_mass_kg <= 0:
        raise ValueError("body_mass_kg must be positive")
    pos = np.clip(p, 0, None)
    neg = np.clip(p, None, 0)
    positive = float(np.trapezoid(pos, t))
    negative = float(np.trapezoid(neg, t))
    net = float(np.trapezoid(p, t))
    return JointEnergetics(
        positive_work_j=positive,
        negative_work_j=negative,
        net_work_j=net,
        peak_abs_power_w=float(np.max(np.abs(p))),
        mechanical_cost_j_per_kg=float((positive + abs(negative)) / body_mass_kg),
    )
