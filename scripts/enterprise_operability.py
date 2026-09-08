from __future__ import annotations
import json,time
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT/'src'):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
from rehab_ai.governance import build_apace_safety_certificate,verify_apace_safety_certificate

start=time.perf_counter()
validation=json.loads((ROOT/'artifacts'/'portfolio_validation.json').read_text())
cert=build_apace_safety_certificate(validation)
bad=deepcopy(cert); bad['therapy_change_execution_allowed']=True
tamper=verify_apace_safety_certificate(bad)
misrepresented=deepcopy(validation); misrepresented['claim_policy']['clinical_validation']='CLINICALLY_VALIDATED'
mis_cert=build_apace_safety_certificate(misrepresented)
elapsed=time.perf_counter()-start
checks={'certificate_valid':verify_apace_safety_certificate(cert)['valid'] is True,'therapy_change_blocked':cert['therapy_change_execution_allowed'] is False,'tampering_detected':tamper['valid'] is False,'false_clinical_validation_claim_forces_hold':mis_cert['decision_state']=='HOLD','modeled_safety_bound_within_threshold':cert['checks']['modeled_safety_bound_within_threshold'],'reference_runtime_under_5s':elapsed<5}
payload={'phase':'ENTERPRISE_OPERABILITY_V1','status':'PASS' if all(checks.values()) else 'HOLD','checks':checks,'runtime_seconds':elapsed,'decision_state':cert['decision_state'],'certificate_sha256':cert['certificate_sha256'],'claim_boundary':cert['claim_boundary']}
(ROOT/'artifacts'/'enterprise_operability.json').write_text(json.dumps(payload,indent=2,sort_keys=True,default=str))
print(json.dumps({'status':payload['status'],'checks':checks,'runtime_seconds':round(elapsed,3),'decision_state':cert['decision_state']},indent=2))
if payload['status']!='PASS': raise SystemExit('REHAB_AI_ENTERPRISE_OPERABILITY=HOLD')
print('REHAB_AI_ENTERPRISE_OPERABILITY=PASS')
