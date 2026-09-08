from rehab_ai.ai.risk import predict_fall_risk
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.config import Settings
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.optimization.therapy import clinical_feasible, optimize_therapy
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features


def test_optimizer_returns_feasible_plan():
    s = generate_synthetic_session(impairment=0.7)
    state = reconstruct_patient_state(s, extract_wearable_features(s.wearable), extract_movement_features(s.pose))
    risk, _ = predict_fall_risk(state)
    settings = Settings()
    result = optimize_therapy(state, risk.value, settings)
    assert result.feasible > 0
    assert result.evaluated >= result.feasible
    assert clinical_feasible(result.selected.plan, state, risk.value, settings)
    if risk.value >= settings.clinical.high_fall_risk_threshold:
        assert result.selected.plan.intensity <= 3
        assert result.selected.plan.balance_minutes >= settings.clinical.minimum_balance_minutes_if_unstable
