from __future__ import annotations
import json
from rehab_ai.biomechanics.stability import margin_of_stability
from rehab_ai.control.stochastic_mpc import solve_stochastic_mpc
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.ie.human_performance import HumanPerformanceState, TherapyLoad, transition_human_performance
from rehab_ai.ml.treatment_response import GaussianProcessTreatmentResponse
from rehab_ai.services.policy_studio import run_policy_studio

m = margin_of_stability(0.01, 0.05, 0.12, 0.9)
assert m.margin_of_stability_m > 0
low = transition_human_performance(HumanPerformanceState(.55,.25,.2), TherapyLoad(.25,20,.1), .05)
high = transition_human_performance(HumanPerformanceState(.55,.25,.2), TherapyLoad(1.0,55,.7), .05)
assert high.fatigue > low.fatigue
model = GaussianProcessTreatmentResponse(); metrics = model.fit_validate(n=90, seed=42)
assert metrics.test_mae < metrics.baseline_mae
assert 0.75 <= metrics.coverage_95 <= 1.0
sol = solve_stochastic_mpc(HumanPerformanceState(.50,.35,.25), horizon_weeks=3, stability_margin_m=.05)
assert sol.status == 'OPTIMAL_ENUMERATED' and sol.feasible_sequences > 0
r = run_policy_studio(generate_synthetic_session(), horizon_weeks=3)
assert r.control_policy['status'] == 'OPTIMAL_ENUMERATED'
print(json.dumps({
  'margin_of_stability_m': m.margin_of_stability_m,
  'fatigue_low_load': low.fatigue,
  'fatigue_high_load': high.fatigue,
  'gp_test_mae': metrics.test_mae,
  'gp_baseline_mae': metrics.baseline_mae,
  'gp_coverage_95': metrics.coverage_95,
  'mpc_sequences_evaluated': sol.sequences_evaluated,
  'mpc_feasible_sequences': sol.feasible_sequences,
  'policy_first_action': r.control_policy['sequence'][0] if r.control_policy['sequence'] else None,
}, indent=2))
