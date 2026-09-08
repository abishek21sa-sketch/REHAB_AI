from rehab_ai.domain import ScenarioResult, TherapyPlan
from rehab_ai.optimization.multiobjective import pareto_front


def _r(gain, risk, fatigue, adherence, minutes):
    return ScenarioResult(
        plan=TherapyPlan(balance_minutes=5, gait_minutes=minutes-10, strength_minutes=5, intensity=2, sessions_per_week=3),
        expected_function_gain=gain, function_gain_p10=gain-.02, function_gain_p90=gain+.02,
        end_fall_risk=risk, expected_fatigue=fatigue, expected_adherence=adherence,
    )


def test_pareto_front_removes_clearly_dominated_plan():
    strong = _r(.20, .20, .20, .9, 30)
    weak = _r(.10, .40, .40, .7, 40)
    front = pareto_front([strong, weak])
    assert len(front) == 1
    assert front[0].result == strong
