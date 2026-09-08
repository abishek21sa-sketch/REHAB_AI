from __future__ import annotations

import math
from rehab_ai.domain import PatientState, Prediction


MODEL_VERSION = "transparent-fall-risk-v1"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_fall_risk(state: PatientState) -> tuple[Prediction, dict[str, float]]:
    factors = {
        "slow_gait": max(0.0, 1.15 - state.gait_speed_mps) * 1.35,
        "gait_variability": state.gait_variability * 1.25,
        "sway": min(1.5, state.sway_index * 4.0) * 0.85,
        "asymmetry": min(1.5, state.knee_asymmetry_deg / 15.0) * 0.70,
        "trunk_instability": state.trunk_instability * 1.15,
        "fatigue": state.fatigue * 0.80,
        "functional_reserve": (1.0 - state.functional_capacity) * 1.20,
    }
    logit = -3.35 + sum(factors.values())
    risk = _sigmoid(logit)
    uncertainty = 0.08 + 0.10 * state.gait_variability
    # Primary inference is the trained synthetic-validation model when its versioned artifact is present.
    # The transparent score above is retained only as an explanation decomposition/fallback.
    try:
        from rehab_ai.ml.fall_risk_model import MODEL_PATH, predict_ml_fall_risk
        prediction = predict_ml_fall_risk(state) if MODEL_PATH.exists() else None
    except Exception:
        prediction = None
    if prediction is None:
        prediction = Prediction(name="six_week_fall_risk_index", value=risk, lower=max(0.0, risk-uncertainty), upper=min(1.0, risk+uncertainty), confidence=max(0.45,0.86-state.gait_variability*0.18), model_version=MODEL_VERSION)
    return prediction, factors
