from __future__ import annotations

from rehab_ai.ai.risk import MODEL_VERSION as FALL_RISK_VERSION
from rehab_ai.ai.recovery import MODEL_VERSION as RECOVERY_VERSION
from rehab_ai.ai.trajectory import MODEL_VERSION as TRAJECTORY_VERSION


def active_model_versions() -> dict[str, str]:
    return {
        "fall_risk": FALL_RISK_VERSION,
        "recovery": RECOVERY_VERSION,
        "trajectory": TRAJECTORY_VERSION,
    }
