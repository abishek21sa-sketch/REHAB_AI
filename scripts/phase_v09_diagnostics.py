import json
from rehab_ai.apace.integrated import IntegratedAdaptiveModels
from rehab_ai.twin.adaptive_episode import run_adaptive_episode
from rehab_ai.apace.population_stress import run_population_stress
m=IntegratedAdaptiveModels(seed=42)
e=run_adaptive_episode(phenotype='balanced',weeks=4,seed=42,models=m)
s=run_population_stress(patients_per_phenotype=2,weeks=3,seed=42)
checks={
 'safe_envelope_beats_baseline':m.evidence.envelope_test_mae<m.evidence.envelope_baseline_mae,
 'temporal_model_beats_persistence':m.evidence.temporal_test_mae<m.evidence.temporal_persistence_mae,
 'episode_replans_each_week':e.replans==4,
 'episode_no_envelope_violation':e.envelope_violations==0,
 'uncertainty_contracts':e.weeks[-1].parameter_std<e.weeks[0].parameter_std,
 'population_all_phenotypes':len(s.results)==5,
}
print(json.dumps({'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'episode':e.to_dict(),'stress':s.to_dict()},indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
