from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from rehab_ai.config import Settings
from rehab_ai.domain import PatientState, ScenarioResult, TherapyPlan
from rehab_ai.simulation.engine import simulate_plan


@dataclass(frozen=True)
class OptimizationResult:
    selected: ScenarioResult
    alternatives: list[ScenarioResult]
    score: float
    evaluated: int
    feasible: int


def clinical_feasible(plan: TherapyPlan, state: PatientState, baseline_fall_risk: float, settings: Settings) -> bool:
    c = settings.clinical
    if plan.session_minutes > c.max_session_minutes or plan.session_minutes < 20:
        return False
    if baseline_fall_risk >= c.high_fall_risk_threshold and plan.balance_minutes < c.minimum_balance_minutes_if_unstable:
        return False
    if baseline_fall_risk >= c.high_fall_risk_threshold and plan.intensity > 3:
        return False
    if state.fatigue >= c.high_fatigue_threshold and plan.intensity > 3:
        return False
    if state.fatigue >= 0.80 and plan.sessions_per_week > 4:
        return False
    if state.pain >= 0.75 and plan.intensity > 2:
        return False
    return True


def _score(result: ScenarioResult, settings: Settings) -> float:
    w = settings.optimization
    burden = result.plan.session_minutes * result.plan.sessions_per_week / 300.0
    return (
        w.weight_recovery * result.expected_function_gain
        - w.weight_safety * result.end_fall_risk
        - w.weight_fatigue * result.expected_fatigue
        - w.weight_burden * burden
        + 0.10 * result.expected_adherence
    )


def optimize_therapy(
    state: PatientState,
    baseline_fall_risk: float,
    settings: Settings,
) -> OptimizationResult:
    candidates: list[TherapyPlan] = []
    for balance, gait, strength, intensity, sessions in product(
        (5, 10, 15, 20),
        (10, 15, 20, 25),
        (5, 10, 15, 20),
        (2, 3, 4),
        (2, 3, 4, 5),
    ):
        for device in ((False, True) if baseline_fall_risk >= 0.55 else (False,)):
            plan = TherapyPlan(
                balance_minutes=balance,
                gait_minutes=gait,
                strength_minutes=strength,
                intensity=intensity,
                sessions_per_week=sessions,
                assistive_device=device,
            )
            candidates.append(plan)

    feasible = [p for p in candidates if clinical_feasible(p, state, baseline_fall_risk, settings)]
    if not feasible:
        raise RuntimeError("No clinically feasible plan found under configured constraints")

    scored: list[tuple[float, ScenarioResult]] = []
    for i, plan in enumerate(feasible):
        result = simulate_plan(
            state,
            plan,
            baseline_fall_risk,
            replications=settings.simulation.replications,
            seed=settings.random_seed + i,
        )
        scored.append((_score(result, settings), result))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    alternatives = [r for _, r in scored[1:4]]
    return OptimizationResult(
        selected=best,
        alternatives=alternatives,
        score=best_score,
        evaluated=len(candidates),
        feasible=len(feasible),
    )
