from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from rehab_ai.apace.integrated import IntegratedAdaptiveModels
from rehab_ai.twin.adaptive_episode import run_adaptive_episode, PHENOTYPES

@dataclass(frozen=True)
class PhenotypeStressResult:
    phenotype:str; n:int; mean_final_capacity:float; mean_gain:float; envelope_violation_rate:float; mean_final_uncertainty:float
@dataclass(frozen=True)
class PopulationStressResult:
    patients:int; seed:int; results:tuple[PhenotypeStressResult,...]; validation_scope:str='SYNTHETIC POPULATION STRESS TEST'
    def to_dict(self): return asdict(self)

def run_population_stress(patients_per_phenotype=5,weeks=5,seed=42):
    models=IntegratedAdaptiveModels(seed=seed); rng=np.random.default_rng(seed); out=[]
    for ph in PHENOTYPES:
        rows=[]
        for i in range(patients_per_phenotype):
            c=float(rng.uniform(.35,.60)); f=float(rng.uniform(.12,.34)); p=float(rng.uniform(.05,.28)); s=float(rng.uniform(.025,.055))
            r=run_adaptive_episode(phenotype=ph,weeks=weeks,seed=seed+100*i+len(rows),initial_capacity=c,initial_fatigue=f,initial_pain=p,stability_margin_m=s,models=models)
            u=r.weeks[-1].parameter_std if r.weeks else .2
            rows.append((r.final_capacity,c,r.envelope_violations,u))
        a=np.asarray(rows,float)
        out.append(PhenotypeStressResult(ph,len(rows),float(a[:,0].mean()),float((a[:,0]-a[:,1]).mean()),float((a[:,2]>0).mean()),float(a[:,3].mean())))
    return PopulationStressResult(patients_per_phenotype*len(PHENOTYPES),seed,tuple(out))
