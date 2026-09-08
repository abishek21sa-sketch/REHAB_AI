from rehab_ai.ml.fall_risk_model import train_and_validate, predict_ml_fall_risk
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features
from rehab_ai.biomechanics.gait import reconstruct_patient_state

def test_trained_model_beats_naive_brier_and_has_discrimination(tmp_path):
    m=train_and_validate(tmp_path/'m.joblib',tmp_path/'metrics.json')
    assert m['test_brier'] < m['baseline_test_brier']
    assert m['test_auc'] > .60
    assert m['data_status']=='SYNTHETIC VALIDATION'

def test_trained_model_inference_is_bounded():
    s=generate_synthetic_session(impairment=.55)
    state=reconstruct_patient_state(s,extract_wearable_features(s.wearable),extract_movement_features(s.pose))
    p=predict_ml_fall_risk(state)
    assert 0<=p.value<=1 and p.lower<=p.value<=p.upper
