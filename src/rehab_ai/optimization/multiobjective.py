from __future__ import annotations

from dataclasses import dataclass
from rehab_ai.domain import ScenarioResult


@dataclass(frozen=True)
class ParetoPoint:
    result: ScenarioResult
    utility: float
    safety: float
    burden: float


def _metrics(r: ScenarioResult) -> tuple[float, float, float]:
    utility = r.expected_function_gain + 0.08 * r.expected_adherence
    safety = 1.0 - r.end_fall_risk - 0.35 * r.expected_fatigue
    burden = r.plan.session_minutes * r.plan.sessions_per_week / 300.0
    return utility, safety, burden


def pareto_front(results: list[ScenarioResult]) -> list[ParetoPoint]:
    points = [ParetoPoint(r, *_metrics(r)) for r in results]
    front: list[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_or_equal = q.utility >= p.utility and q.safety >= p.safety and q.burden <= p.burden
            strictly_better = q.utility > p.utility or q.safety > p.safety or q.burden < p.burden
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: (p.utility + p.safety - 0.20 * p.burden), reverse=True)
