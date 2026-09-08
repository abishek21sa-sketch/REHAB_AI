from rehab_ai.governance.audit import build_decision_audit


def test_decision_audit_is_deterministic_for_content():
    a = build_decision_audit("P1", {"x": 1}, {"review_status": "clinician_review_required", "y": 2}, {"m": "v1"})
    b = build_decision_audit("P1", {"x": 1}, {"y": 2, "review_status": "clinician_review_required"}, {"m": "v1"})
    assert a.decision_id == b.decision_id
    assert len(a.input_digest) == 64
    assert a.review_status == "clinician_review_required"
