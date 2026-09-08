from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features


def test_feature_extractors_return_physical_ranges():
    session = generate_synthetic_session()
    w = extract_wearable_features(session.wearable)
    m = extract_movement_features(session.pose)
    assert 0.5 < w.step_frequency_hz < 3.5
    assert w.accel_rms > 0
    assert 5 < m.knee_rom_deg < 90
    assert m.knee_asymmetry_deg >= 0
