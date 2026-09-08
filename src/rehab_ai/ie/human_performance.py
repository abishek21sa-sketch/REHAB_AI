from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HumanPerformanceState:
    capacity: float
    fatigue: float
    pain: float
    cumulative_dose: float = 0.0


@dataclass(frozen=True)
class TherapyLoad:
    intensity: float
    duration_min: float
    instability_demand: float
    repetition_factor: float = 1.0
    recovery_hours: float = 24.0

    @property
    def functional_load(self) -> float:
        # normalized dose: intensity x duration x instability x repetition
        return max(0.0, self.intensity) * max(0.0, self.duration_min) / 60.0 * (1.0 + max(0.0, self.instability_demand)) * max(0.0, self.repetition_factor)


def safe_load_limit(capacity: float, pain: float, stability_margin_m: float) -> float:
    """Patient-specific upper functional load envelope (normalized load units)."""
    c = min(1.0, max(0.0, capacity))
    p = min(1.0, max(0.0, pain))
    stability_factor = min(1.15, max(0.25, 0.65 + 5.0 * stability_margin_m))
    return float(max(0.10, 1.35 * c * (1.0 - 0.55 * p) * stability_factor))


def adaptation_gain(load: float, capacity: float, fatigue: float) -> float:
    """Hormetic dose-response: underload gives little adaptation; overload reverses gain."""
    c = max(0.08, capacity)
    ratio = load / c
    # peak adaptation near ratio 0.85; overload becomes negative after ~1.6
    positive = 0.055 * math.exp(-((ratio - 0.85) / 0.48) ** 2)
    overload = 0.045 * max(0.0, ratio - 1.45) ** 2
    return float((positive - overload) * (1.0 - 0.65 * min(1.0, max(0.0, fatigue))))


def transition_human_performance(
    state: HumanPerformanceState,
    load: TherapyLoad,
    stability_margin_m: float,
) -> HumanPerformanceState:
    L = load.functional_load
    cap_limit = safe_load_limit(state.capacity, state.pain, stability_margin_m)
    overload = max(0.0, L - cap_limit)
    # fatigue carry-over with exponential recovery
    recovery = math.exp(-max(0.0, load.recovery_hours) / 36.0)
    fatigue = min(1.0, max(0.0, state.fatigue * recovery + 0.24 * L + 0.42 * overload))
    gain = adaptation_gain(L, state.capacity, fatigue)
    capacity = min(1.0, max(0.0, state.capacity + gain))
    pain = min(1.0, max(0.0, 0.86 * state.pain + 0.18 * overload + 0.025 * L))
    return HumanPerformanceState(
        capacity=capacity,
        fatigue=fatigue,
        pain=pain,
        cumulative_dose=state.cumulative_dose + L,
    )
