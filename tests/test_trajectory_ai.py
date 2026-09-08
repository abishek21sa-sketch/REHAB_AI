from rehab_ai.ai.trajectory import predict_adherence_risk, predict_fatigue_exacerbation, predict_gait_deterioration
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features


def _state(impairment: float):
    s = generate_synthetic_session(impairment=impairment)
    return reconstruct_patient_state(s, extract_wearable_features(s.wearable), extract_movement_features(s.pose))


def test_risk_predictions_are_bounded_and_worsen_with_impairment():
    low = _state(0.2)
    high = _state(0.8)
    for fn in (predict_adherence_risk, predict_fatigue_exacerbation):
        a, b = fn(low), fn(high)
        assert 0 <= a.value <= 1 and 0 <= b.value <= 1
        assert a.value < b.value
    assert predict_gait_deterioration(low).value < predict_gait_deterioration(high).value
