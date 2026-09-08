from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.apace_studio import run_apace_studio


def test_full_apace_studio_pipeline():
    r=run_apace_studio(generate_synthetic_session(impairment=.42),horizon=3,seed=9,beam_width=9,rollouts=32)
    d=r.to_dict()
    assert d['latent_state']['estimator']=='Unscented Kalman Filter'
    assert d['treatment_response_ml']['test_mae'] < d['treatment_response_ml']['baseline_mae']
    assert d['phenotype']['population_silhouette'] > .4
    assert d['apace_policy']['status'] in {'OPTIMAL_BEAM_APPROXIMATION','INFEASIBLE'}
    assert 'EXTERNAL VALIDATION PENDING' in d['evidence_labels']['clinical_validation']
