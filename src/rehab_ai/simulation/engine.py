from __future__ import annotations

import numpy as np
from rehab_ai.ai.recovery import predict_recovery_gain
from rehab_ai.domain import PatientState, ScenarioResult, TherapyPlan


def simulate_plan(
    state: PatientState,
    plan: TherapyPlan,
    baseline_fall_risk: float,
    replications: int = 500,
    seed: int = 42,
) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    recovery = predict_recovery_gain(state, plan)
    mean = recovery.value
    sigma = max(0.025, (recovery.upper - recovery.lower) / 3.29 if recovery.upper is not None and recovery.lower is not None else 0.06)
    gains = rng.normal(mean, sigma, size=replications)

    dose = plan.session_minutes * plan.sessions_per_week
    safety_benefit = min(0.30, plan.balance_minutes * plan.sessions_per_week / 650.0)
    gait_benefit = min(0.18, plan.gait_minutes * plan.sessions_per_week / 850.0)
    device_benefit = 0.08 if plan.assistive_device else 0.0
    fatigue_pressure = min(0.30, dose * plan.intensity / 4500.0)

    end_fall_risk = baseline_fall_risk * (1 - safety_benefit - gait_benefit - device_benefit) + 0.12 * fatigue_pressure
    end_fall_risk = float(np.clip(end_fall_risk, 0.01, 0.99))
    expected_fatigue = float(np.clip(state.fatigue * 0.72 + fatigue_pressure + 0.08 * state.pain, 0.0, 1.0))
    expected_adherence = float(np.clip(state.adherence - max(0.0, dose - 180) / 900.0 - max(0, plan.intensity - 3) * 0.035, 0.35, 1.0))

    return ScenarioResult(
        plan=plan,
        expected_function_gain=float(np.mean(gains)),
        function_gain_p10=float(np.quantile(gains, 0.10)),
        function_gain_p90=float(np.quantile(gains, 0.90)),
        end_fall_risk=end_fall_risk,
        expected_fatigue=expected_fatigue,
        expected_adherence=expected_adherence,
    )
