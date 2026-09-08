from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from rehab_ai.domain import PoseFrame, WearableSample


class ClinicalObservation(BaseModel):
    patient_id: str
    code: str
    value: float
    unit: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "clinical"


def normalize_fhir_observation(resource: dict) -> ClinicalObservation:
    """Normalize the small FHIR Observation subset needed by the platform contract.

    This is an adapter contract, not a live EHR connector. It intentionally rejects
    incomplete resources rather than guessing clinical meaning.
    """
    if resource.get("resourceType") != "Observation":
        raise ValueError("resourceType must be Observation")
    subject = resource.get("subject", {}).get("reference", "")
    patient_id = subject.rsplit("/", 1)[-1] if subject else ""
    coding = resource.get("code", {}).get("coding", [])
    code = coding[0].get("code", "") if coding else resource.get("code", {}).get("text", "")
    q = resource.get("valueQuantity", {})
    if not patient_id or not code or "value" not in q or not q.get("unit"):
        raise ValueError("FHIR Observation missing patient, code, value, or unit")
    observed = resource.get("effectiveDateTime")
    return ClinicalObservation(
        patient_id=patient_id,
        code=code,
        value=float(q["value"]),
        unit=str(q["unit"]),
        observed_at=datetime.fromisoformat(observed.replace("Z", "+00:00")) if observed else datetime.now(timezone.utc),
        source="fhir-observation-adapter",
    )


class SensorStreamBuffer:
    """Bounded in-memory adapter for streaming acquisition before session validation."""

    def __init__(self, max_wearable: int = 10_000, max_pose: int = 5_000):
        if max_wearable < 20 or max_pose < 10:
            raise ValueError("stream capacities are below minimum session requirements")
        self.max_wearable = max_wearable
        self.max_pose = max_pose
        self.wearable: list[WearableSample] = []
        self.pose: list[PoseFrame] = []

    def add_wearable(self, sample: WearableSample) -> None:
        self.wearable.append(sample)
        if len(self.wearable) > self.max_wearable:
            self.wearable.pop(0)

    def add_pose(self, frame: PoseFrame) -> None:
        self.pose.append(frame)
        if len(self.pose) > self.max_pose:
            self.pose.pop(0)

    def ready(self) -> bool:
        return len(self.wearable) >= 20 and len(self.pose) >= 10
