from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class WearableSample(BaseModel):
    t: float = Field(ge=0)
    ax: float
    ay: float
    az: float
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0


class PoseFrame(BaseModel):
    t: float = Field(ge=0)
    left_hip_y: float
    right_hip_y: float
    left_knee_angle: float = Field(ge=0, le=180)
    right_knee_angle: float = Field(ge=0, le=180)
    trunk_lean_deg: float = Field(ge=-90, le=90)


class SessionInput(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64)
    age: int = Field(ge=18, le=100)
    wearable: list[WearableSample] = Field(min_length=20)
    pose: list[PoseFrame] = Field(min_length=10)
    six_minute_walk_m: float = Field(gt=0, le=1500)
    pain_score: float = Field(ge=0, le=10)
    perceived_fatigue: float = Field(ge=0, le=10)
    adherence_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def aligned_time_ranges(self):
        if self.wearable[-1].t <= self.wearable[0].t:
            raise ValueError("wearable timestamps must increase")
        if self.pose[-1].t <= self.pose[0].t:
            raise ValueError("pose timestamps must increase")
        return self


class WearableFeatures(BaseModel):
    duration_s: float
    sample_rate_hz: float
    accel_rms: float
    jerk_rms: float
    step_frequency_hz: float
    sway_index: float


class MovementFeatures(BaseModel):
    knee_rom_deg: float
    knee_asymmetry_deg: float
    trunk_lean_mean_deg: float
    trunk_lean_variability_deg: float
    pelvic_vertical_variability: float


class PatientState(BaseModel):
    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gait_speed_mps: float = Field(ge=0)
    cadence_spm: float = Field(ge=0)
    gait_variability: float = Field(ge=0)
    sway_index: float = Field(ge=0)
    knee_rom_deg: float = Field(ge=0)
    knee_asymmetry_deg: float = Field(ge=0)
    trunk_instability: float = Field(ge=0)
    fatigue: float = Field(ge=0, le=1)
    pain: float = Field(ge=0, le=1)
    adherence: float = Field(ge=0, le=1)
    functional_capacity: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    name: str
    value: float
    lower: float | None = None
    upper: float | None = None
    confidence: float = Field(ge=0, le=1)
    model_version: str


class TherapyPlan(BaseModel):
    balance_minutes: int = Field(ge=0, le=60)
    gait_minutes: int = Field(ge=0, le=60)
    strength_minutes: int = Field(ge=0, le=60)
    intensity: int = Field(ge=1, le=5)
    sessions_per_week: int = Field(ge=1, le=7)
    assistive_device: bool = False

    @property
    def session_minutes(self) -> int:
        return self.balance_minutes + self.gait_minutes + self.strength_minutes


class ScenarioResult(BaseModel):
    plan: TherapyPlan
    expected_function_gain: float
    function_gain_p10: float
    function_gain_p90: float
    end_fall_risk: float
    expected_fatigue: float
    expected_adherence: float


class Recommendation(BaseModel):
    patient_id: str
    selected_plan: TherapyPlan
    score: float
    fall_risk: Prediction
    recovery_prediction: Prediction
    scenario: ScenarioResult
    rationale: list[str]
    alternatives: list[ScenarioResult]
    review_status: Literal["clinician_review_required"] = "clinician_review_required"
    disclaimer: str = "Research decision support only; not validated for autonomous clinical care."
