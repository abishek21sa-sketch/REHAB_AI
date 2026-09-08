from rehab_ai.ai.risk import predict_fall_risk
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.domain import TherapyPlan
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.simulation.engine import simulate_plan
from rehab_ai.vision.movement import extract_movement_features


def test_simulation_is_reproducible():
    s = generate_synthetic_session()
    state = reconstruct_patient_state(s, extract_wearable_features(s.wearable), extract_movement_features(s.pose))
    risk, _ = predict_fall_risk(state)
    plan = TherapyPlan(balance_minutes=15, gait_minutes=20, strength_minutes=10, intensity=3, sessions_per_week=3)
    a = simulate_plan(state, plan, risk.value, replications=200, seed=9)
    b = simulate_plan(state, plan, risk.value, replications=200, seed=9)
    assert a == b
    assert a.function_gain_p10 < a.expected_function_gain < a.function_gain_p90
