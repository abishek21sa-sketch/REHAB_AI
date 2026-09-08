import math

from rehab_ai.apace import APACEConfig, BeliefState, exact_apace_small, solve_apace
from rehab_ai.control.stochastic_mpc import RehabAction
from rehab_ai.ie.human_performance import HumanPerformanceState


def predictor(perf, action):
    # deterministic model surrogate used only as a mathematical oracle fixture
    target = 0.55 * perf.capacity + 0.08
    load = action.as_load().functional_load
    gain = 0.055 * math.exp(-((load-target)/0.28)**2) * (1-0.4*perf.fatigue)
    return gain, 0.010 + 0.008 * action.instability_demand


def initial():
    return BeliefState(HumanPerformanceState(0.55, 0.28, 0.18), 0.05, parameter_std=0.16, state_uncertainty=0.08)


def actions():
    return (
        RehabAction('Low balance','balance',0.28,22,0.20),
        RehabAction('Gait','gait',0.38,26,0.18),
        RehabAction('Recovery','recovery',0.12,18,0.04),
    )


def test_apace_returns_feasible_policy_and_reduces_parameter_uncertainty():
    cfg = APACEConfig(horizon=3, beam_width=9, rollouts=64, seed=7)
    sol = solve_apace(initial(), predictor, cfg, actions())
    assert sol.status == 'OPTIMAL_BEAM_APPROXIMATION'
    assert len(sol.sequence) == 3
    assert sol.trajectory[-1].parameter_std < initial().parameter_std
    assert all(s.fatigue <= cfg.fatigue_limit for s in sol.trajectory)
    assert all(s.pain <= cfg.pain_limit for s in sol.trajectory)


def test_apace_reproducible_for_seed():
    cfg = APACEConfig(horizon=3, beam_width=8, rollouts=48, seed=17)
    a = solve_apace(initial(), predictor, cfg, actions())
    b = solve_apace(initial(), predictor, cfg, actions())
    assert a.sequence == b.sequence
    assert a.score == b.score


def test_small_beam_matches_exact_oracle_on_reference_case():
    cfg = APACEConfig(horizon=2, beam_width=9, rollouts=80, seed=3, enable_dominance_pruning=False)
    oracle = exact_apace_small(initial(), predictor, cfg, actions())
    tested = solve_apace(initial(), predictor, cfg, actions())
    assert tested.sequence == oracle.sequence
    assert abs(tested.score - oracle.score) < 1e-12


def test_infeasible_when_stability_is_below_boundary():
    b = BeliefState(HumanPerformanceState(0.5,0.2,0.1), 0.0, parameter_std=.1, state_uncertainty=.05)
    sol = solve_apace(b, predictor, APACEConfig(horizon=2, rollouts=32), actions())
    assert sol.status == 'INFEASIBLE'


def test_information_weight_changes_dual_control_preference():
    # Use two actions: one higher-gain, one more informative through intensity/instability.
    acts=(
        RehabAction('Conservative','gait',0.22,20,0.05),
        RehabAction('Probe','balance',0.38,20,0.30),
    )
    low = solve_apace(initial(), predictor, APACEConfig(horizon=1, beam_width=5, rollouts=128, base_information_weight=0.0, risk_weight=.15, burden_weight=.05, seed=21), acts)
    high = solve_apace(initial(), predictor, APACEConfig(horizon=1, beam_width=5, rollouts=128, base_information_weight=1.0, risk_weight=.15, burden_weight=.05, seed=21), acts)
    assert high.trajectory[0].information_gain >= low.trajectory[0].information_gain - 1e-12
