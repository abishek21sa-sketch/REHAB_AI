"""Safety assurance certificate for APACE portfolio evidence.

This object certifies the *integrity and internal release checks* of the synthetic
policy benchmark. It is not a clinical safety certificate or medical-device approval.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json
from typing import Any


def _canonical(x: Any) -> bytes:
    return json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode('utf-8')


def build_apace_safety_certificate(validation: dict[str,Any]) -> dict[str,Any]:
    safety=dict(validation.get('finite_sample_safety') or {})
    claim=dict(validation.get('claim_policy') or {})
    checks={
        'portfolio_validation_passed':validation.get('status')=='PASS',
        'modeled_safety_bound_within_threshold':float(safety.get('wilson_upper_bound',1.0)) <= float(safety.get('release_threshold',0.0)),
        'external_clinical_validation_not_misrepresented':claim.get('clinical_validation')=='EXTERNAL_VALIDATION_PENDING',
        'production_write_blocked':claim.get('production_write_allowed') is False,
        'universal_dominance_not_claimed':claim.get('universal_dominance_claimed') is False,
    }
    state='CLINICAL_REVIEW_REQUIRED' if all(checks.values()) else 'HOLD'
    core={
        'certificate_type':'REHAB_AI_APACE_SAFETY_ASSURANCE_V1',
        'decision_state':state,
        'validation_release':validation.get('release'),
        'synthetic_trial_count':validation.get('patients'),
        'modeled_unsafe_events':safety.get('modeled_unsafe_events'),
        'modeled_unsafe_rate_upper_bound':safety.get('wilson_upper_bound'),
        'one_sided_confidence':safety.get('one_sided_confidence'),
        'release_threshold':safety.get('release_threshold'),
        'checks':checks,
        'validation_evidence_sha256':hashlib.sha256(_canonical(validation)).hexdigest(),
        'approval_authority':'LICENSED_REHABILITATION_CLINICIAN_OR_RESEARCH_PI',
        'required_external_reviews':['CLINICAL_PROTOCOL_REVIEW','PROSPECTIVE_VALIDATION','HUMAN_FACTORS','DATA_GOVERNANCE'],
        'therapy_change_execution_allowed':False,
        'claim_boundary':'Synthetic/model safety-envelope evidence only. This certificate is not clinical validation, medical advice, medical-device certification, or authorization to alter patient care.',
    }
    return {**core,'certificate_sha256':hashlib.sha256(_canonical(core)).hexdigest(),'generated_at_utc':datetime.now(timezone.utc).isoformat()}


def verify_apace_safety_certificate(certificate: dict[str,Any])->dict[str,Any]:
    stored=certificate.get('certificate_sha256')
    core={k:v for k,v in certificate.items() if k not in {'certificate_sha256','generated_at_utc'}}
    expected=hashlib.sha256(_canonical(core)).hexdigest()
    return {'valid':bool(stored and stored==expected),'stored_sha256':stored,'expected_sha256':expected,'decision_state':certificate.get('decision_state')}
