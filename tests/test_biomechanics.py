from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features


def state_for(impairment):
    s = generate_synthetic_session(impairment=impairment)
    return reconstruct_patient_state(s, extract_wearable_features(s.wearable), extract_movement_features(s.pose))


def test_higher_impairment_reduces_functional_capacity():
    low = state_for(0.2)
    high = state_for(0.75)
    assert low.gait_speed_mps > high.gait_speed_mps
    assert low.functional_capacity > high.functional_capacity
