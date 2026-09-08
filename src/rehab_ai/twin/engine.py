from __future__ import annotations

from dataclasses import dataclass
from rehab_ai.domain import PatientState
from rehab_ai.persistence.repository import StateRepository


@dataclass
class TwinUpdate:
    previous: PatientState | None
    current: PatientState
    deltas: dict[str, float]


class PatientDigitalTwin:
    """Persistent patient-state twin with explicit temporal deltas."""

    def __init__(self, repository: StateRepository):
        self.repository = repository

    def update(self, state: PatientState) -> TwinUpdate:
        previous = self.repository.latest_state(state.patient_id)
        deltas: dict[str, float] = {}
        if previous:
            for field in (
                "gait_speed_mps",
                "cadence_spm",
                "gait_variability",
                "sway_index",
                "knee_rom_deg",
                "fatigue",
                "functional_capacity",
            ):
                deltas[field] = float(getattr(state, field) - getattr(previous, field))
        self.repository.save_state(state)
        return TwinUpdate(previous=previous, current=state, deltas=deltas)
