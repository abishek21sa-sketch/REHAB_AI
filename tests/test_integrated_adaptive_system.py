from rehab_ai.apace.algorithm import APACEConfig, BeliefState
from rehab_ai.apace.integrated import IntegratedAdaptiveModels, solve_integrated_apace
from rehab_ai.ie.human_performance import HumanPerformanceState
from rehab_ai.twin.adaptive_episode import run_adaptive_episode
from rehab_ai.apace.population_stress import run_population_stress

def test_integrated_apace_uses_learned_envelope_and_temporal_model():
    m=IntegratedAdaptiveModels(seed=7)
    b=BeliefState(HumanPerformanceState(.52,.20,.12),.04,parameter_std=.15,state_uncertainty=.08,adherence=.9)
    sol=solve_integrated_apace(b,m,APACEConfig(horizon=2,beam_width=10,rollouts=32,seed=7))
    assert sol.status.startswith('OPTIMAL') and len(sol.sequence)==2
    assert m.evidence.envelope_test_mae < m.evidence.envelope_baseline_mae
    assert m.evidence.temporal_test_mae < m.evidence.temporal_persistence_mae

def test_adaptive_episode_replans_and_contracts_uncertainty():
    r=run_adaptive_episode(phenotype='balanced',weeks=4,seed=11)
    assert r.status=='COMPLETED' and r.replans==4
    assert r.weeks[-1].parameter_std < .20
    assert r.final_capacity >= .30
    assert r.envelope_violations==0

def test_population_stress_covers_all_phenotypes_and_is_safe():
    r=run_population_stress(patients_per_phenotype=2,weeks=3,seed=5)
    assert len(r.results)==5 and r.patients==10
    assert all(x.envelope_violation_rate <= .5 for x in r.results)
