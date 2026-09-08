from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pydantic import BaseModel


class DecisionAudit(BaseModel):
    decision_id: str
    timestamp: str
    patient_id: str
    input_digest: str
    recommendation_digest: str
    model_versions: dict[str, str]
    review_status: str


def _digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_decision_audit(patient_id: str, input_payload: dict, recommendation_payload: dict, model_versions: dict[str, str]) -> DecisionAudit:
    input_digest = _digest(input_payload)
    rec_digest = _digest(recommendation_payload)
    decision_id = hashlib.sha256(f"{patient_id}:{input_digest}:{rec_digest}".encode()).hexdigest()[:20]
    return DecisionAudit(
        decision_id=decision_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        patient_id=patient_id,
        input_digest=input_digest,
        recommendation_digest=rec_digest,
        model_versions=model_versions,
        review_status=recommendation_payload.get("review_status", "unknown"),
    )
