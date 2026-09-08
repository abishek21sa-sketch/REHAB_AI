from rehab_ai.apace.validation import build_apace_validation


def test_portfolio_validation_is_explicitly_nonclinical_and_safe():
    out=build_apace_validation(patients=4,horizon=2,seed=15)
    assert out['status'] in {'PASS','REVIEW'}
    assert out['claim_policy']['universal_dominance_claimed'] is False
    assert out['claim_policy']['clinical_validation']=='EXTERNAL_VALIDATION_PENDING'
    assert out['claim_policy']['production_write_allowed'] is False
    assert out['checks']['zero_unsafe_rate']
    assert out['checks']['oracle_zero_regret']
