from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.policy_studio import run_policy_studio


def test_policy_studio_connects_observation_estimation_ml_ie_or():
    r = run_policy_studio(generate_synthetic_session(seed=11, impairment=.45), horizon_weeks=3, seed=11)
    assert r.biomechanical['margin_of_stability_m'] < .2
    assert r.latent_state['estimator'] == 'Unscented Kalman Filter'
    assert r.treatment_response_validation['test_mae'] < r.treatment_response_validation['baseline_mae']
    assert r.control_policy['status'] in {'OPTIMAL_ENUMERATED','INFEASIBLE'}
    assert r.evidence_labels['clinical_validation'] == 'EXTERNAL VALIDATION PENDING'
