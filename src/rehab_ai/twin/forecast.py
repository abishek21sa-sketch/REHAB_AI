from __future__ import annotations

from pydantic import BaseModel, Field

from rehab_ai.domain import PatientState, TherapyPlan


class TwinTrajectoryPoint(BaseModel):
    week: int = Field(ge=0)
    functional_capacity: float = Field(ge=0, le=1)
    fall_risk: float = Field(ge=0, le=1)
    fatigue: float = Field(ge=0, le=1)
    adherence: float = Field(ge=0, le=1)


class TwinForecast(BaseModel):
    horizon_weeks: int
    trajectory: list[TwinTrajectoryPoint]
    projected_function_gain: float
    projected_risk_reduction: float
    safety_flags: list[str]


def forecast_twin(
    state: PatientState,
    plan: TherapyPlan,
    baseline_fall_risk: float,
    horizon_weeks: int = 6,
) -> TwinForecast:
    capacity = state.functional_capacity
    risk = baseline_fall_risk
    fatigue = state.fatigue
    adherence = state.adherence
    trajectory = [TwinTrajectoryPoint(week=0, functional_capacity=capacity, fall_risk=risk, fatigue=fatigue, adherence=adherence)]
    dose = plan.session_minutes * plan.sessions_per_week
    weekly_gain = min(0.055, 0.006 + dose / 6200.0 + plan.intensity / 850.0) * (0.65 + 0.35 * adherence)
    weekly_balance = min(0.05, plan.balance_minutes * plan.sessions_per_week / 1800.0)
    weekly_burden = min(0.055, dose * plan.intensity / 16500.0)

    flags: list[str] = []
    for week in range(1, horizon_weeks + 1):
        capacity = min(1.0, capacity + weekly_gain * (1.0 - 0.45 * fatigue))
        fatigue = min(1.0, max(0.0, fatigue * 0.82 + weekly_burden + 0.025 * state.pain))
        adherence = min(1.0, max(0.30, adherence - max(0.0, weekly_burden - 0.025) * 0.30 + 0.01 * (1.0 - fatigue)))
        risk = min(1.0, max(0.01, risk * (1.0 - weekly_balance - 0.025 * weekly_gain) + 0.018 * fatigue))
        if fatigue > 0.82:
            flags.append(f"Projected fatigue exceeds 0.82 at week {week}.")
        if adherence < 0.50:
            flags.append(f"Projected adherence falls below 0.50 at week {week}.")
        trajectory.append(TwinTrajectoryPoint(week=week, functional_capacity=capacity, fall_risk=risk, fatigue=fatigue, adherence=adherence))

    return TwinForecast(
        horizon_weeks=horizon_weeks,
        trajectory=trajectory,
        projected_function_gain=capacity - state.functional_capacity,
        projected_risk_reduction=baseline_fall_risk - risk,
        safety_flags=sorted(set(flags)),
    )
