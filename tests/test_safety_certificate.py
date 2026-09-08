from copy import deepcopy
from rehab_ai.apace.validation import build_apace_validation
from rehab_ai.governance import build_apace_safety_certificate, verify_apace_safety_certificate


def test_safety_certificate_is_review_only_and_tamper_evident():
    v=build_apace_validation(patients=100,horizon=2,seed=2026)
    c=build_apace_safety_certificate(v)
    assert c['decision_state']=='CLINICAL_REVIEW_REQUIRED'
    assert c['therapy_change_execution_allowed'] is False
    assert verify_apace_safety_certificate(c)['valid'] is True
    bad=deepcopy(c); bad['modeled_unsafe_rate_upper_bound']=0
    assert verify_apace_safety_certificate(bad)['valid'] is False
