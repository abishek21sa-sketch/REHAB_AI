from __future__ import annotations
import json
from pathlib import Path
import tempfile
import numpy as np

from rehab_ai.apace import APACEConfig, BeliefState, exact_apace_small, solve_apace
from rehab_ai.control.stochastic_mpc import RehabAction
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.ie.human_performance import HumanPerformanceState
from rehab_ai.ml.phenotypes import RehabilitationPhenotyper
from rehab_ai.ml.treatment_response import GaussianProcessTreatmentResponse
from rehab_ai.services.apace_studio import run_apace_studio
from rehab_ai.storage import read_sensor_session, write_sensor_session


def predictor(perf, action):
    load=action.as_load().functional_load
    gain=0.05*np.exp(-((load-(0.5*perf.capacity+0.08))/0.30)**2)*(1-0.4*perf.fatigue)
    return float(gain), float(0.008+0.006*action.instability_demand)


def main():
    checks={}
    gp=GaussianProcessTreatmentResponse(42); metrics=gp.fit_validate(n=120,seed=42)
    checks['gp_beats_mean_baseline']=metrics.test_mae < metrics.baseline_mae
    checks['gp_interval_coverage_reasonable']=0.80 <= metrics.coverage_95 <= 1.0
    ph=RehabilitationPhenotyper(42); sil=ph.fit_validate(60)
    checks['phenotype_separation']=sil > 0.45

    actions=(RehabAction('A','balance',.26,20,.15),RehabAction('B','gait',.34,24,.12),RehabAction('R','recovery',.12,18,.03))
    belief=BeliefState(HumanPerformanceState(.56,.25,.16),.05,parameter_std=.15,state_uncertainty=.07)
    cfg=APACEConfig(horizon=2,beam_width=9,rollouts=64,seed=11,enable_dominance_pruning=False)
    approx=solve_apace(belief,predictor,cfg,actions); oracle=exact_apace_small(belief,predictor,cfg,actions)
    checks['apace_matches_exact_small_oracle']=approx.sequence == oracle.sequence and abs(approx.score-oracle.score)<1e-12
    checks['apace_reduces_response_uncertainty']=approx.trajectory[-1].parameter_std < belief.parameter_std
    checks['apace_safe_trajectory']=all(s.fatigue<=cfg.fatigue_limit and s.pain<=cfg.pain_limit for s in approx.trajectory)

    studio=run_apace_studio(generate_synthetic_session(impairment=.45),horizon=3,seed=42,beam_width=9,rollouts=32)
    checks['integrated_policy_feasible']=studio.apace_policy['status'] != 'INFEASIBLE'
    checks['external_validation_honest']=studio.evidence_labels['clinical_validation']=='EXTERNAL VALIDATION PENDING'

    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'sensor.h5'; t=np.linspace(0,2,101)
        write_sensor_session(path,'P','S',t,{'ax':np.sin(t),'gy':np.cos(t)},{'scope':'diagnostic'})
        back=read_sensor_session(path,'P','S')
        checks['hdf5_roundtrip']=len(back['timestamps'])==101 and back['metadata']['scope']=='diagnostic'

    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'gp_metrics':metrics.__dict__,'phenotype_silhouette':sil,'apace_sequence':list(approx.sequence),'apace_score':approx.score}
    out=Path('artifacts/validation/apace_diagnostics.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)

if __name__=='__main__': main()
