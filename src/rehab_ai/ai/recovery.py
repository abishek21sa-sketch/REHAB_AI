from __future__ import annotations

from rehab_ai.domain import PatientState, Prediction, TherapyPlan

MODEL_VERSION = "transparent-recovery-v1"


def predict_recovery_gain(state: PatientState, plan: TherapyPlan | None = None) -> Prediction:
    weekly_dose = 75.0
    intensity = 2.5
    balance_fraction = 0.25
    if plan is not None:
        weekly_dose = plan.session_minutes * plan.sessions_per_week
        intensity = float(plan.intensity)
        balance_fraction = plan.balance_minutes / max(1, plan.session_minutes)

    dose_effect = min(0.22, weekly_dose / 900.0)
    intensity_effect = min(0.08, intensity * 0.015)
    balance_effect = min(0.05, balance_fraction * 0.08)
    reserve = 1.0 - state.functional_capacity
    expected = (
        0.025
        + 0.16 * reserve
        + dose_effect
        + intensity_effect
        + balance_effect
        + 0.08 * state.adherence
        - 0.09 * state.fatigue
        - 0.07 * state.pain
    )
    expected = max(-0.05, min(0.45, expected))
    interval = 0.055 + 0.05 * state.fatigue + 0.035 * (1 - state.adherence)
    return Prediction(
        name="six_week_functional_capacity_gain",
        value=expected,
        lower=max(-0.15, expected - interval),
        upper=min(0.60, expected + interval),
        confidence=max(0.42, 0.82 - interval),
        model_version=MODEL_VERSION,
    )
