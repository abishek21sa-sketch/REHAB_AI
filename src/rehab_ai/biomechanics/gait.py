from __future__ import annotations

from rehab_ai.domain import MovementFeatures, PatientState, SessionInput, WearableFeatures


def reconstruct_patient_state(
    session: SessionInput,
    wearable: WearableFeatures,
    movement: MovementFeatures,
) -> PatientState:
    gait_speed = session.six_minute_walk_m / 360.0
    cadence = wearable.step_frequency_hz * 60.0
    cadence_penalty = min(1.0, abs(cadence - 105.0) / 105.0)
    gait_variability = min(1.0, wearable.jerk_rms / 12.0 + movement.knee_asymmetry_deg / 40.0)
    trunk_instability = min(1.0, movement.trunk_lean_variability_deg / 10.0 + movement.pelvic_vertical_variability * 8)
    fatigue = min(1.0, session.perceived_fatigue / 10.0)
    pain = min(1.0, session.pain_score / 10.0)
    speed_score = min(1.0, gait_speed / 1.4)
    rom_score = min(1.0, movement.knee_rom_deg / 60.0)
    stability_score = max(0.0, 1.0 - (wearable.sway_index * 2.5 + trunk_instability) / 2)
    functional_capacity = max(
        0.0,
        min(1.0, 0.42 * speed_score + 0.22 * rom_score + 0.20 * stability_score + 0.16 * (1 - cadence_penalty)),
    )
    return PatientState(
        patient_id=session.patient_id,
        gait_speed_mps=gait_speed,
        cadence_spm=cadence,
        gait_variability=gait_variability,
        sway_index=wearable.sway_index,
        knee_rom_deg=movement.knee_rom_deg,
        knee_asymmetry_deg=movement.knee_asymmetry_deg,
        trunk_instability=trunk_instability,
        fatigue=fatigue,
        pain=pain,
        adherence=session.adherence_ratio,
        functional_capacity=functional_capacity,
    )
