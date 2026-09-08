from __future__ import annotations

from pydantic import BaseModel, Field

from rehab_ai.domain import MovementFeatures, WearableFeatures


class FusionAssessment(BaseModel):
    wearable_reliability: float = Field(ge=0, le=1)
    vision_reliability: float = Field(ge=0, le=1)
    cross_modal_consistency: float = Field(ge=0, le=1)
    fused_stability_index: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str]


def assess_multimodal_fusion(wearable: WearableFeatures, movement: MovementFeatures) -> FusionAssessment:
    sample_reliability = min(1.0, max(0.0, wearable.sample_rate_hz / 45.0))
    motion_penalty = min(0.45, wearable.jerk_rms / 18.0)
    wearable_reliability = max(0.35, sample_reliability - motion_penalty)

    asymmetry_penalty = min(0.35, movement.knee_asymmetry_deg / 45.0)
    trunk_penalty = min(0.35, movement.trunk_lean_variability_deg / 25.0)
    vision_reliability = max(0.35, 1.0 - 0.45 * asymmetry_penalty - 0.55 * trunk_penalty)

    wearable_instability = min(1.0, wearable.sway_index * 2.2)
    vision_instability = min(
        1.0,
        movement.trunk_lean_variability_deg / 12.0
        + movement.pelvic_vertical_variability * 3.5
        + movement.knee_asymmetry_deg / 55.0,
    )
    disagreement = abs(wearable_instability - vision_instability)
    consistency = max(0.0, 1.0 - disagreement)

    total_weight = wearable_reliability + vision_reliability
    fused = (
        wearable_instability * wearable_reliability
        + vision_instability * vision_reliability
    ) / max(total_weight, 1e-9)
    confidence = min(wearable_reliability, vision_reliability) * (0.65 + 0.35 * consistency)

    warnings: list[str] = []
    if consistency < 0.55:
        warnings.append("Wearable and vision stability estimates disagree; clinician review should prioritize raw-signal inspection.")
    if wearable_reliability < 0.60:
        warnings.append("Wearable-derived state has reduced reliability.")
    if vision_reliability < 0.60:
        warnings.append("Vision-derived state has reduced reliability.")

    return FusionAssessment(
        wearable_reliability=wearable_reliability,
        vision_reliability=vision_reliability,
        cross_modal_consistency=consistency,
        fused_stability_index=min(1.0, max(0.0, fused)),
        confidence=min(1.0, max(0.0, confidence)),
        warnings=warnings,
    )
