from __future__ import annotations
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

class TherapyDemand(BaseModel):
    patient_id: str
    priority: float = Field(ge=0, le=1)
    sessions_required: int = Field(ge=1, le=7)
    preferred_days: list[int] = Field(default_factory=lambda: [0,1,2,3,4])
class ScheduledSession(BaseModel):
    patient_id: str; day: int; slot: int
class ScheduleResult(BaseModel):
    sessions: list[ScheduledSession]; unscheduled: dict[str,int]; utilization: float; objective: float
    solver_status: str; solver_message: str

def optimize_weekly_schedule(demands:list[TherapyDemand],therapists:int=2,slots_per_day:int=6,days:int=5)->ScheduleResult:
    if therapists<1 or slots_per_day<1 or days<1: raise ValueError('capacity dimensions must be positive')
    # Binary x[p,d,s] assigns at most one session per patient/day/slot; integer u[p] is unmet demand.
    P,D,S=len(demands),days,slots_per_day; nx=P*D*S; n=nx+P
    def ix(p,d,s): return (p*D+d)*S+s
    c=np.zeros(n)
    for p,q in enumerate(demands):
        pref=set(q.preferred_days)
        for d in range(D):
            for s in range(S): c[ix(p,d,s)]=-q.priority*(1.0 if d in pref else .82)
        c[nx+p]=2.0+3.0*q.priority  # strongly penalize unmet high-priority sessions
    rows=[]; lb=[]; ub=[]
    # demand equality: sum assignments + unmet = required
    for p,q in enumerate(demands):
        row={ix(p,d,s):1 for d in range(D) for s in range(S)}; row[nx+p]=1; rows.append(row); lb.append(q.sessions_required); ub.append(q.sessions_required)
    # no more than one session per patient per day
    for p in range(P):
        for d in range(D): rows.append({ix(p,d,s):1 for s in range(S)}); lb.append(-np.inf); ub.append(1)
    # therapist capacity per slot
    for d in range(D):
        for s in range(S): rows.append({ix(p,d,s):1 for p in range(P)}); lb.append(-np.inf); ub.append(therapists)
    A=lil_matrix((len(rows),n),dtype=float)
    for r,row in enumerate(rows):
        for j,v in row.items(): A[r,j]=v
    lower=np.zeros(n); upper=np.ones(n); upper[nx:]=[q.sessions_required for q in demands]
    integrality=np.ones(n,dtype=int)
    res=milp(c,integrality=integrality,bounds=Bounds(lower,upper),constraints=LinearConstraint(A.tocsr(),np.array(lb),np.array(ub)),options={'time_limit':10})
    if res.x is None: raise RuntimeError(f'scheduling MILP failed: {res.message}')
    sessions=[]
    for p,q in enumerate(demands):
        for d in range(D):
            for s in range(S):
                if res.x[ix(p,d,s)]>.5: sessions.append(ScheduledSession(patient_id=q.patient_id,day=d,slot=s))
    uns={q.patient_id:int(round(res.x[nx+p])) for p,q in enumerate(demands) if res.x[nx+p]>.5}
    total_capacity=therapists*S*D
    return ScheduleResult(sessions=sessions,unscheduled=uns,utilization=len(sessions)/total_capacity,objective=float(-res.fun),solver_status='OPTIMAL' if res.status==0 else 'FEASIBLE',solver_message=res.message)
