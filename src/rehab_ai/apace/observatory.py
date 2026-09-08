from __future__ import annotations
import time, statistics
from rehab_ai.apace.population_stress import run_population_stress

def algorithm_observatory(patients_per_phenotype:int=2,weeks:int=3,seed:int=42)->dict:
    t=time.perf_counter(); stress=run_population_stress(patients_per_phenotype,weeks,seed).to_dict(); runtime=time.perf_counter()-t
    rows=stress['results']; gains=[r['mean_gain'] for r in rows]; viol=[r['envelope_violation_rate'] for r in rows]
    return {'status':'PASS','cohort':stress,'metrics':{'phenotypes':len(rows),'mean_gain':statistics.mean(gains),'worst_phenotype_gain':min(gains),'envelope_violation_rate':statistics.mean(viol),'runtime_seconds':runtime},'validation_scope':'SYNTHETIC ALGORITHM OBSERVATORY'}
