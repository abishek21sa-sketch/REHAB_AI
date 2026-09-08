from rehab_ai.ai.risk import predict_fall_risk
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features


def _state(impairment):
    session = generate_synthetic_session(impairment=impairment)
    return reconstruct_patient_state(session, extract_wearable_features(session.wearable), extract_movement_features(session.pose))


def test_fall_risk_increases_with_synthetic_impairment():
    low, _ = predict_fall_risk(_state(0.2))
    mid, _ = predict_fall_risk(_state(0.45))
    high, _ = predict_fall_risk(_state(0.75))
    assert low.value < mid.value < high.value
    assert 0 <= high.lower <= high.value <= high.upper <= 1
