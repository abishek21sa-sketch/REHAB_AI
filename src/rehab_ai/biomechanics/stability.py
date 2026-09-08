from __future__ import annotations

from dataclasses import dataclass
import numpy as np

G = 9.80665


@dataclass(frozen=True)
class StabilityMetrics:
    com_position_m: float
    com_velocity_mps: float
    extrapolated_com_m: float
    bos_boundary_m: float
    margin_of_stability_m: float


def segmental_com(positions_m: np.ndarray, mass_fractions: np.ndarray) -> float:
    """Mass-weighted 1-D COM location. Fractions need not be pre-normalized."""
    p = np.asarray(positions_m, dtype=float)
    m = np.asarray(mass_fractions, dtype=float)
    if p.ndim != 1 or m.ndim != 1 or len(p) != len(m) or len(p) == 0:
        raise ValueError("positions and mass fractions must be equal non-empty vectors")
    if np.any(m < 0) or float(m.sum()) <= 0:
        raise ValueError("mass fractions must be non-negative and have positive total")
    return float(np.dot(p, m) / m.sum())


def extrapolated_com(com_position_m: float, com_velocity_mps: float, effective_leg_length_m: float) -> float:
    """Hof-style extrapolated COM under inverted-pendulum approximation."""
    if effective_leg_length_m <= 0:
        raise ValueError("effective_leg_length_m must be positive")
    omega0 = np.sqrt(G / effective_leg_length_m)
    return float(com_position_m + com_velocity_mps / omega0)


def margin_of_stability(
    com_position_m: float,
    com_velocity_mps: float,
    bos_boundary_m: float,
    effective_leg_length_m: float,
) -> StabilityMetrics:
    xcom = extrapolated_com(com_position_m, com_velocity_mps, effective_leg_length_m)
    return StabilityMetrics(
        com_position_m=float(com_position_m),
        com_velocity_mps=float(com_velocity_mps),
        extrapolated_com_m=xcom,
        bos_boundary_m=float(bos_boundary_m),
        margin_of_stability_m=float(bos_boundary_m - xcom),
    )


def symmetry_index(left: float, right: float, eps: float = 1e-9) -> float:
    """Absolute bilateral symmetry index: |L-R| / ((|L|+|R|)/2)."""
    denom = 0.5 * (abs(left) + abs(right))
    if denom <= eps:
        return 0.0
    return float(abs(left - right) / denom)
