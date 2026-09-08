from __future__ import annotations

from rehab_ai.domain import PatientState, ScenarioResult, TherapyPlan


def explain_risk_factors(factors: dict[str, float], top_k: int = 3) -> list[str]:
    labels = {
        "slow_gait": "reduced gait speed",
        "gait_variability": "elevated gait variability",
        "sway": "increased mediolateral sway",
        "asymmetry": "left-right knee excursion asymmetry",
        "trunk_instability": "trunk instability",
        "fatigue": "reported fatigue",
        "functional_reserve": "reduced functional reserve",
    }
    ranked = sorted(factors.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [f"Fall-risk estimate is most influenced by {labels[k]} (contribution {v:.2f})." for k, v in ranked]


def explain_plan(state: PatientState, plan: TherapyPlan, scenario: ScenarioResult) -> list[str]:
    reasons: list[str] = []
    if plan.balance_minutes >= 10:
        reasons.append("Balance work is retained because the current state indicates a stability-related safety burden.")
    if plan.intensity <= 3 and state.fatigue >= 0.65:
        reasons.append("Intensity is capped to limit fatigue escalation under the current fatigue state.")
    if plan.assistive_device:
        reasons.append("An assistive-device flag is included as a safety constraint for higher predicted fall risk; clinician selection is required.")
    reasons.append(
        f"The selected scenario projects a median functional-capacity gain of {scenario.expected_function_gain:.3f} over six weeks "
        f"with an end-horizon fall-risk index of {scenario.end_fall_risk:.3f}."
    )
    reasons.append("The recommendation is decision support only and requires clinician review before any therapy change.")
    return reasons
