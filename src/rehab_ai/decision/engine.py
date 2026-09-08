from __future__ import annotations

from rehab_ai.ai.explain import explain_plan, explain_risk_factors
from rehab_ai.ai.recovery import predict_recovery_gain
from rehab_ai.config import Settings
from rehab_ai.domain import PatientState, Prediction, Recommendation
from rehab_ai.optimization.therapy import optimize_therapy


def make_recommendation(
    state: PatientState,
    fall_risk: Prediction,
    fall_risk_factors: dict[str, float],
    settings: Settings,
) -> Recommendation:
    optimized = optimize_therapy(state, fall_risk.value, settings)
    recovery = predict_recovery_gain(state, optimized.selected.plan)
    rationale = explain_risk_factors(fall_risk_factors) + explain_plan(state, optimized.selected.plan, optimized.selected)
    return Recommendation(
        patient_id=state.patient_id,
        selected_plan=optimized.selected.plan,
        score=optimized.score,
        fall_risk=fall_risk,
        recovery_prediction=recovery,
        scenario=optimized.selected,
        rationale=rationale,
        alternatives=optimized.alternatives,
    )
