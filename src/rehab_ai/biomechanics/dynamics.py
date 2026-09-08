from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Segment:
    mass_kg: float
    com_fraction: float
    length_m: float

@dataclass(frozen=True)
class InverseDynamicsResult:
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    gravitational_moment_nm: np.ndarray
    inertial_moment_nm: np.ndarray
    net_joint_moment_nm: np.ndarray


def segment_com(proximal_xy: np.ndarray, distal_xy: np.ndarray, com_fraction: float) -> np.ndarray:
    p = np.asarray(proximal_xy, dtype=float)
    d = np.asarray(distal_xy, dtype=float)
    if p.shape != d.shape or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('proximal/distal inputs must both have shape (n,2)')
    f = float(com_fraction)
    if not 0 <= f <= 1:
        raise ValueError('com_fraction must lie in [0,1]')
    return p + f * (d - p)


def whole_body_com(segment_positions: list[np.ndarray], segment_masses_kg: list[float]) -> np.ndarray:
    if len(segment_positions) != len(segment_masses_kg) or not segment_positions:
        raise ValueError('segment positions and masses must be nonempty and have equal length')
    arrays = [np.asarray(x, dtype=float) for x in segment_positions]
    n = arrays[0].shape[0]
    if any(a.shape != (n,2) for a in arrays):
        raise ValueError('all segment COM arrays must have shape (n,2)')
    masses = np.asarray(segment_masses_kg, dtype=float)
    if np.any(masses <= 0):
        raise ValueError('segment masses must be positive')
    stacked = np.stack(arrays, axis=0)
    return np.sum(stacked * masses[:,None,None], axis=0) / masses.sum()


def planar_joint_inverse_dynamics(
    angle_rad: np.ndarray,
    dt_s: float,
    segment: Segment,
    gravity: float = 9.80665,
) -> InverseDynamicsResult:
    q = np.asarray(angle_rad, dtype=float)
    if q.ndim != 1 or q.size < 5:
        raise ValueError('angle_rad must be a 1D series with at least 5 samples')
    if dt_s <= 0:
        raise ValueError('dt_s must be positive')
    omega = np.gradient(q, dt_s, edge_order=2)
    alpha = np.gradient(omega, dt_s, edge_order=2)
    r = segment.length_m * segment.com_fraction
    inertia = segment.mass_kg * (segment.length_m ** 2) / 12.0
    grav = segment.mass_kg * gravity * r * np.sin(q)
    inertial = inertia * alpha
    net = inertial + grav
    return InverseDynamicsResult(omega, alpha, grav, inertial, net)
