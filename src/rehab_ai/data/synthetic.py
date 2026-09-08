from __future__ import annotations

import math
import random
from rehab_ai.domain import PoseFrame, SessionInput, WearableSample


def generate_synthetic_session(
    patient_id: str = "DEMO-001",
    seed: int = 42,
    impairment: float = 0.45,
) -> SessionInput:
    """Generate a deterministic, non-PHI rehabilitation session for testing and demos."""
    rng = random.Random(seed)
    impairment = max(0.0, min(1.0, impairment))
    duration = 20.0
    fs = 50
    step_hz = 1.65 - 0.45 * impairment
    wearable: list[WearableSample] = []
    for i in range(int(duration * fs)):
        t = i / fs
        phase = 2 * math.pi * step_hz * t
        noise = lambda s: rng.gauss(0, s)
        wearable.append(
            WearableSample(
                t=t,
                ax=0.25 * math.sin(phase) + noise(0.03 + impairment * 0.02),
                ay=0.10 * math.sin(phase / 2) + noise(0.02),
                az=1.0 + (0.55 - 0.15 * impairment) * abs(math.sin(phase)) + noise(0.04),
                gx=0.12 * math.cos(phase) + noise(0.015),
                gy=0.08 * math.sin(phase) + noise(0.015),
                gz=0.06 * math.sin(phase / 2) + noise(0.015),
            )
        )

    pose_fps = 25
    pose: list[PoseFrame] = []
    asym = 3 + impairment * 10
    for i in range(int(duration * pose_fps)):
        t = i / pose_fps
        phase = 2 * math.pi * step_hz * t
        base_knee = 145 - 10 * impairment
        excursion = 30 - 8 * impairment
        lk = base_knee - excursion * max(0.0, math.sin(phase))
        rk = base_knee - (excursion - asym) * max(0.0, math.sin(phase + math.pi))
        pose.append(
            PoseFrame(
                t=t,
                left_hip_y=0.50 + 0.015 * math.sin(2 * phase) + rng.gauss(0, 0.003 + impairment * 0.002),
                right_hip_y=0.50 + 0.015 * math.sin(2 * phase + 0.1) + rng.gauss(0, 0.003 + impairment * 0.002),
                left_knee_angle=max(0, min(180, lk + rng.gauss(0, 1.3))),
                right_knee_angle=max(0, min(180, rk + rng.gauss(0, 1.3))),
                trunk_lean_deg=3 + impairment * 5 + 2.5 * math.sin(phase / 2) + rng.gauss(0, 0.8 + impairment),
            )
        )

    return SessionInput(
        patient_id=patient_id,
        age=58,
        wearable=wearable,
        pose=pose,
        six_minute_walk_m=510 - 220 * impairment,
        pain_score=2.0 + 4.0 * impairment,
        perceived_fatigue=2.5 + 4.5 * impairment,
        adherence_ratio=0.91 - 0.20 * impairment,
    )
