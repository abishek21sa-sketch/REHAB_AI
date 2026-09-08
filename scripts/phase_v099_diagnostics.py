from __future__ import annotations
import json
from pathlib import Path

from rehab_ai.clinical_workflow import build_clinical_episode
from rehab_ai.vision.motion_session import analyze_landmark_csv

root=Path(__file__).resolve().parents[1]
case=build_clinical_episode(impairment=.45,phenotype='balanced',weeks=5,seed=42).to_dict()
motion=analyze_landmark_csv((root/'sample_data'/'observed_landmark_session.csv').read_text(encoding='utf-8'))
checks={
    'closed_loop_five_week_episode': len(case['weeks'])==5 and case['learning']['replans']==5,
    'patient_uncertainty_contracts': case['learning']['final_uncertainty'] < case['learning']['initial_uncertainty'],
    'zero_reference_envelope_violations': case['learning']['envelope_violations']==0,
    'multi_protocol_assessment': set(case['assessments'])=={'gait','balance','mobility','endurance'},
    'evidence_language_complete': {'CALCULATED','ESTIMATED','PREDICTED','SIMULATED','OPTIMIZED'} <= {x['kind'] for x in case['evidence_ledger']},
    'motion_trace_reconstructed': len(motion['trace']['time_s'])>10,
    'missing_clinical_signals_not_fabricated': motion['readiness']['full_gait_assessment']=='INSUFFICIENT_SIGNALS',
}
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'learning':case['learning'],'motion_readiness':motion['readiness'],'next_control':case['next_control']}
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['status']=='PASS' else 1)
