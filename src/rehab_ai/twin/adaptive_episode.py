from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from rehab_ai.apace.algorithm import APACEConfig, BeliefState
from rehab_ai.apace.integrated import IntegratedAdaptiveModels, solve_integrated_apace
from rehab_ai.control.stochastic_mpc import default_actions
from rehab_ai.ie.human_performance import HumanPerformanceState, transition_human_performance

@dataclass(frozen=True)
class EpisodeWeek:
    week:int; action:str; capacity:float; fatigue:float; pain:float; stability_margin_m:float
    safe_limit:float; applied_load:float; predicted_gain:float; observed_gain:float; parameter_std:float

@dataclass(frozen=True)
class AdaptiveEpisodeResult:
    status:str; phenotype:str; weeks:tuple[EpisodeWeek,...]; final_capacity:float; envelope_violations:int; replans:int
    validation_scope:str='SYNTHETIC ADAPTIVE EPISODE'
    def to_dict(self): return asdict(self)

PHENOTYPES={
 'fast_responder':(1.22,0.90,0.90),
 'slow_responder':(0.72,1.00,1.00),
 'fatigue_sensitive':(0.92,1.35,1.00),
 'pain_sensitive':(0.92,1.00,1.35),
 'balanced':(1.00,1.00,1.00),
}

def run_adaptive_episode(*, phenotype='balanced', weeks=6, seed=42, initial_capacity=.48, initial_fatigue=.24, initial_pain=.16, stability_margin_m=.035, models=None):
    if phenotype not in PHENOTYPES: raise ValueError('unknown phenotype')
    models=models or IntegratedAdaptiveModels(seed=seed)
    response_scale,fatigue_scale,pain_scale=PHENOTYPES[phenotype]
    rng=np.random.default_rng(seed)
    belief=BeliefState(HumanPerformanceState(initial_capacity,initial_fatigue,initial_pain),stability_margin_m,parameter_std=.20,state_uncertainty=.10,adherence=.9)
    actions={a.name:a for a in default_actions()}; records=[]; violations=0
    for w in range(1,weeks+1):
        sol=solve_integrated_apace(belief,models,APACEConfig(horizon=min(3,weeks-w+1),beam_width=14,rollouts=48,seed=seed+w))
        if not sol.sequence: return AdaptiveEpisodeResult('INFEASIBLE',phenotype,tuple(records),belief.performance.capacity,violations,w-1)
        action=actions[sol.sequence[0]]; load=action.as_load().functional_load; limit=models.safe_limit(belief,action)
        if load>limit+1e-9: violations+=1
        mean_gain,std=models.response(belief.performance,action)
        observed_gain=float(rng.normal(mean_gain*response_scale,std))
        base=transition_human_performance(belief.performance,action.as_load(),belief.stability_margin_m)
        next_perf=HumanPerformanceState(float(np.clip(base.capacity+observed_gain,0,1)),float(np.clip(base.fatigue*fatigue_scale,0,1)),float(np.clip(base.pain*pain_scale,0,1)),base.cumulative_dose)
        # Bayesian-like contraction after observing treatment response.
        new_pstd=max(.02,belief.parameter_std*.78)
        new_stab=max(0.001,belief.stability_margin_m+.009*observed_gain-.004*next_perf.fatigue)
        hist=(belief.history+((belief.performance.capacity,belief.performance.fatigue,belief.performance.pain,belief.adherence,load),))[-4:]
        records.append(EpisodeWeek(w,action.name,next_perf.capacity,next_perf.fatigue,next_perf.pain,new_stab,limit,load,mean_gain,observed_gain,new_pstd))
        belief=BeliefState(next_perf,new_stab,parameter_mean=belief.parameter_mean,parameter_std=new_pstd,state_uncertainty=max(.025,belief.state_uncertainty*.84),adherence=belief.adherence,recent_overload=float(max(0,load/max(limit,.05)-1)),history=hist)
    return AdaptiveEpisodeResult('COMPLETED',phenotype,tuple(records),belief.performance.capacity,violations,weeks)
