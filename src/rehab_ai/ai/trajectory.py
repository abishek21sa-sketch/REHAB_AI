from __future__ import annotations

import math
from rehab_ai.domain import PatientState, Prediction


MODEL_VERSION = "transparent-trajectory-v2"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_gait_deterioration(state: PatientState, trend: dict[str, float] | None = None) -> Prediction:
    trend = trend or {}
    capacity_delta = trend.get("functional_capacity", 0.0)
    speed_delta = trend.get("gait_speed_mps", 0.0)
    x = (
        -2.35
        + 1.55 * state.gait_variability
        + 1.20 * state.trunk_instability
        + 0.90 * state.fatigue
        + 0.75 * state.pain
        - 1.25 * capacity_delta
        - 0.80 * speed_delta
    )
    value = _sigmoid(x)
    interval = 0.09 + 0.08 * state.gait_variability
    return Prediction(
        name="four_week_gait_deterioration_risk",
        value=value,
        lower=max(0.0, value - interval),
        upper=min(1.0, value + interval),
        confidence=max(0.45, 0.84 - 0.15 * state.gait_variability),
        model_version=MODEL_VERSION,
    )


def predict_adherence_risk(state: PatientState) -> Prediction:
    value = _sigmoid(-1.6 + 1.6 * state.fatigue + 1.15 * state.pain + 1.35 * (1.0 - state.adherence))
    return Prediction(
        name="four_week_nonadherence_risk",
        value=value,
        lower=max(0.0, value - 0.10),
        upper=min(1.0, value + 0.10),
        confidence=0.74,
        model_version=MODEL_VERSION,
    )


def predict_fatigue_exacerbation(state: PatientState) -> Prediction:
    value = _sigmoid(-2.0 + 2.15 * state.fatigue + 1.05 * state.pain + 0.65 * (1.0 - state.functional_capacity))
    return Prediction(
        name="two_week_fatigue_exacerbation_risk",
        value=value,
        lower=max(0.0, value - 0.09),
        upper=min(1.0, value + 0.09),
        confidence=0.76,
        model_version=MODEL_VERSION,
    )
