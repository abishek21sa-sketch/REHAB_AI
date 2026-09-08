from rehab_ai.ai.risk import predict_fall_risk
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.domain import TherapyPlan
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.twin.forecast import forecast_twin
from rehab_ai.vision.movement import extract_movement_features


def test_twin_forecast_has_complete_horizon_and_bounded_state():
    s = generate_synthetic_session(impairment=0.55)
    state = reconstruct_patient_state(s, extract_wearable_features(s.wearable), extract_movement_features(s.pose))
    risk, _ = predict_fall_risk(state)
    plan = TherapyPlan(balance_minutes=15, gait_minutes=20, strength_minutes=10, intensity=3, sessions_per_week=3)
    f = forecast_twin(state, plan, risk.value, horizon_weeks=6)
    assert len(f.trajectory) == 7
    assert f.trajectory[0].week == 0 and f.trajectory[-1].week == 6
    assert all(0 <= p.functional_capacity <= 1 for p in f.trajectory)
    assert f.projected_function_gain >= 0
