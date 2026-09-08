import json
from rehab_ai.ml.fall_risk_model import METRICS_PATH
from rehab_ai.scheduling.therapy_schedule import TherapyDemand,optimize_weekly_schedule
m=json.loads(METRICS_PATH.read_text())
assert m['test_brier'] < m['baseline_test_brier'] and m['test_auc']>.60
r=optimize_weekly_schedule([TherapyDemand(patient_id='P1',priority=.9,sessions_required=3),TherapyDemand(patient_id='P2',priority=.4,sessions_required=3)],therapists=1,slots_per_day=1,days=5)
assert r.solver_status=='OPTIMAL' and len(r.sessions)==5 and sum(r.unscheduled.values())==1
print(json.dumps({'ml_test_auc':m['test_auc'],'ml_test_brier':m['test_brier'],'naive_brier':m['baseline_test_brier'],'scheduler_status':r.solver_status,'scheduled':len(r.sessions),'unmet':r.unscheduled},indent=2))
