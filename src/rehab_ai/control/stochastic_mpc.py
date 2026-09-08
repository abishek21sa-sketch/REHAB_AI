from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable
import math

from rehab_ai.ie.human_performance import HumanPerformanceState, TherapyLoad, safe_load_limit, transition_human_performance


@dataclass(frozen=True)
class RehabAction:
    name: str
    modality: str
    intensity: float
    duration_min: float
    instability_demand: float

    def as_load(self, recovery_hours: float = 24.0) -> TherapyLoad:
        return TherapyLoad(self.intensity, self.duration_min, self.instability_demand, 1.0, recovery_hours)


@dataclass(frozen=True)
class MPCStep:
    week: int
    action: RehabAction
    expected_capacity: float
    expected_fatigue: float
    expected_pain: float
    load: float
    safe_limit: float


@dataclass(frozen=True)
class MPCSolution:
    status: str
    objective: float
    sequence: tuple[RehabAction, ...]
    trajectory: tuple[MPCStep, ...]
    sequences_evaluated: int
    feasible_sequences: int


def default_actions() -> tuple[RehabAction, ...]:
    return (
        RehabAction("Recovery", "recovery", 0.12, 20, 0.05),
        RehabAction("Balance precision", "balance", 0.42, 30, 0.45),
        RehabAction("Gait progression", "gait", 0.52, 35, 0.32),
        RehabAction("Strength adaptation", "strength", 0.58, 32, 0.18),
        RehabAction("Integrated function", "mixed", 0.48, 40, 0.38),
    )


def _evaluate_sequence(
    initial: HumanPerformanceState,
    sequence: Iterable[RehabAction],
    stability_margin_m: float,
    fatigue_limit: float,
    pain_limit: float,
    uncertainty_z: float,
    response_uncertainty: float,
) -> tuple[bool, float, tuple[MPCStep, ...]]:
    state = initial
    trajectory: list[MPCStep] = []
    burden = 0.0
    for week, action in enumerate(sequence, start=1):
        load = action.as_load()
        L = load.functional_load
        limit = safe_load_limit(state.capacity, state.pain, stability_margin_m)
        # chance/robust surrogate: reserve headroom proportional to response uncertainty
        robust_load = L + uncertainty_z * response_uncertainty * (1.0 + action.instability_demand)
        if robust_load > limit:
            return False, math.inf, tuple(trajectory)
        nxt = transition_human_performance(state, load, stability_margin_m)
        if nxt.fatigue > fatigue_limit or nxt.pain > pain_limit:
            return False, math.inf, tuple(trajectory)
        trajectory.append(MPCStep(week, action, nxt.capacity, nxt.fatigue, nxt.pain, L, limit))
        burden += 0.0015 * action.duration_min + 0.018 * action.intensity
        state = nxt
    # minimize terminal deficit plus path safety/burden penalties
    objective = (1.0 - state.capacity) ** 2 + 0.55 * state.fatigue**2 + 0.45 * state.pain**2 + burden
    return True, objective, tuple(trajectory)


def solve_stochastic_mpc(
    initial: HumanPerformanceState,
    horizon_weeks: int = 4,
    actions: tuple[RehabAction, ...] | None = None,
    stability_margin_m: float = 0.045,
    fatigue_limit: float = 0.82,
    pain_limit: float = 0.78,
    confidence: float = 0.95,
    response_uncertainty: float = 0.025,
) -> MPCSolution:
    if horizon_weeks < 1 or horizon_weeks > 6:
        raise ValueError("horizon_weeks must be between 1 and 6")
    acts = actions or default_actions()
    # Normal z values sufficient for configured research scenarios.
    z = 1.645 if confidence <= 0.95 else 1.96 if confidence <= 0.975 else 2.326
    best: tuple[float, tuple[RehabAction, ...], tuple[MPCStep, ...]] | None = None
    evaluated = feasible = 0
    for seq in product(acts, repeat=horizon_weeks):
        evaluated += 1
        ok, obj, traj = _evaluate_sequence(initial, seq, stability_margin_m, fatigue_limit, pain_limit, z, response_uncertainty)
        if not ok:
            continue
        feasible += 1
        if best is None or obj < best[0] - 1e-12:
            best = (obj, tuple(seq), traj)
    if best is None:
        return MPCSolution("INFEASIBLE", math.inf, tuple(), tuple(), evaluated, feasible)
    return MPCSolution("OPTIMAL_ENUMERATED", best[0], best[1], best[2], evaluated, feasible)
