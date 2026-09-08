from itertools import product
from rehab_ai.control.stochastic_mpc import RehabAction, solve_stochastic_mpc
from rehab_ai.ie.human_performance import HumanPerformanceState


def test_mpc_exact_enumeration_count_and_feasibility():
    actions = (
        RehabAction('rest','recovery',.10,15,.02),
        RehabAction('train','mixed',.35,25,.10),
    )
    sol = solve_stochastic_mpc(HumanPerformanceState(.65,.15,.10), horizon_weeks=3, actions=actions, stability_margin_m=.08, response_uncertainty=.005)
    assert sol.sequences_evaluated == len(list(product(actions, repeat=3))) == 8
    assert sol.status == 'OPTIMAL_ENUMERATED'
    assert len(sol.sequence) == 3
    for step in sol.trajectory:
        assert step.load <= step.safe_limit
        assert step.expected_fatigue <= .82
        assert step.expected_pain <= .78


def test_mpc_can_report_infeasible():
    actions = (RehabAction('unsafe','mixed',1.5,60,.8),)
    sol = solve_stochastic_mpc(HumanPerformanceState(.15,.8,.75), horizon_weeks=2, actions=actions, stability_margin_m=-.02)
    assert sol.status == 'INFEASIBLE'
