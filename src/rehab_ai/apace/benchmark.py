from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from rehab_ai.apace.algorithm import APACEConfig, BeliefState, solve_apace, evaluate_apace_sequence
from rehab_ai.control.stochastic_mpc import RehabAction, default_actions

@dataclass(frozen=True)
class PolicyBenchmark:
    policy: str
    mean_final_capacity: float
    mean_cumulative_tail_loss: float
    mean_final_uncertainty: float
    mean_regret_to_oracle: float
    unsafe_rate: float

@dataclass(frozen=True)
class BenchmarkSuite:
    patients: int
    horizon: int
    seed: int
    results: tuple[PolicyBenchmark,...]
    validation_scope: str = 'SYNTHETIC POLICY BENCHMARK'
    def to_dict(self): return asdict(self)


def _predictor_factory(theta: float):
    def f(perf, action: RehabAction):
        load=action.as_load().functional_load
        target=.62*perf.capacity+.06
        gain=theta*.058*np.exp(-((load-target)/.34)**2)*(1-.42*perf.fatigue)*(1-.25*perf.pain)
        std=.009+.012*action.instability_demand
        return float(gain), float(std)
    return f


def _initial(rng):
    from rehab_ai.ie.human_performance import HumanPerformanceState
    return BeliefState(
        HumanPerformanceState(float(rng.uniform(.35,.72)),float(rng.uniform(.12,.45)),float(rng.uniform(.05,.35))),
        float(rng.uniform(.025,.075)), parameter_std=float(rng.uniform(.10,.25)), state_uncertainty=float(rng.uniform(.05,.14)))


def run_policy_benchmark(patients: int=24, horizon: int=4, seed: int=42, actions: tuple[RehabAction,...]|None=None) -> BenchmarkSuite:
    acts=actions or tuple(default_actions())
    action_by_name={a.name:a for a in acts}
    rng=np.random.default_rng(seed)
    stats={k:[] for k in ('apace','greedy','risk_only','standard_mpc','oracle')}
    for p in range(patients):
        init=_initial(rng); theta=float(rng.uniform(.72,1.30)); pred=_predictor_factory(theta)
        common=APACEConfig(horizon=horizon,beam_width=18,rollouts=72,seed=seed+p)
        selectors={
            'apace':common,
            'greedy':APACEConfig(horizon=horizon,beam_width=1,rollouts=72,seed=seed+p,base_information_weight=0,risk_weight=0,burden_weight=0),
            'risk_only':APACEConfig(horizon=horizon,beam_width=8,rollouts=72,seed=seed+p,base_information_weight=0,risk_weight=1.8,burden_weight=.05),
            'standard_mpc':APACEConfig(horizon=horizon,beam_width=18,rollouts=72,seed=seed+p,base_information_weight=0),
        }
        selected={name:solve_apace(init,pred,cfg,acts) for name,cfg in selectors.items()}
        oracle_cfg=APACEConfig(**{**asdict(common),'beam_width':min(len(acts)**horizon,4096),'enable_dominance_pruning':False})
        oracle=solve_apace(init,pred,oracle_cfg,acts)
        evaluated={}
        for name,sol in selected.items():
            seq=tuple(action_by_name[n] for n in sol.sequence)
            evaluated[name]=evaluate_apace_sequence(init,pred,seq,common)
        evaluated['oracle']=oracle
        oracle_score=oracle.score
        for name,sol in evaluated.items():
            if not sol.trajectory or not np.isfinite(sol.score):
                stats[name].append((init.performance.capacity,99.0,init.parameter_std+init.state_uncertainty,99.0,1.0)); continue
            final=sol.trajectory[-1]
            cvar=sum(s.cvar_loss for s in sol.trajectory)
            unc=final.parameter_std
            regret=max(0.0,oracle_score-sol.score)
            stats[name].append((final.capacity,cvar,unc,regret,0.0))
    out=[]
    for name,rows in stats.items():
        a=np.asarray(rows,float)
        out.append(PolicyBenchmark(name,float(a[:,0].mean()),float(a[:,1].mean()),float(a[:,2].mean()),float(a[:,3].mean()),float(a[:,4].mean())))
    return BenchmarkSuite(patients,horizon,seed,tuple(out))
