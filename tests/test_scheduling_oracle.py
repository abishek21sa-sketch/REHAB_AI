from itertools import product
from rehab_ai.scheduling.therapy_schedule import TherapyDemand,optimize_weekly_schedule

def test_milp_small_instance_matches_enumerated_max_sessions():
    ds=[TherapyDemand(patient_id='A',priority=1,sessions_required=2),TherapyDemand(patient_id='B',priority=.5,sessions_required=2)]
    r=optimize_weekly_schedule(ds,therapists=1,slots_per_day=1,days=3)
    # capacity is exactly three; enumeration over binary patient/day choices proves max scheduled count is 3.
    best=max(sum(v) for v in product([0,1],repeat=6) if all(v[d]+v[3+d]<=1 for d in range(3)) and sum(v[:3])<=2 and sum(v[3:])<=2)
    assert len(r.sessions)==best==3
    assert r.solver_status=='OPTIMAL'
