from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.fusion.state_fusion import assess_multimodal_fusion
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features


def test_multimodal_fusion_is_bounded_and_explainable():
    session = generate_synthetic_session(impairment=0.55)
    fusion = assess_multimodal_fusion(
        extract_wearable_features(session.wearable),
        extract_movement_features(session.pose),
    )
    assert 0 <= fusion.fused_stability_index <= 1
    assert 0 <= fusion.confidence <= 1
    assert 0 <= fusion.cross_modal_consistency <= 1
