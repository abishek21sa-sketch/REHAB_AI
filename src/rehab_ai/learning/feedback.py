from __future__ import annotations

from pydantic import BaseModel, Field


class ClinicianFeedback(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64)
    accepted: bool
    rationale: str = Field(min_length=1, max_length=2000)
    modified_plan: dict | None = None


class FeedbackReceipt(BaseModel):
    stored: bool
    learning_status: str = "feedback_captured_no_automatic_retraining"
