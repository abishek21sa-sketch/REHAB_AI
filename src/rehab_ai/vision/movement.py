from __future__ import annotations

import numpy as np
from rehab_ai.domain import MovementFeatures, PoseFrame


def extract_movement_features(frames: list[PoseFrame]) -> MovementFeatures:
    left = np.array([f.left_knee_angle for f in frames], dtype=float)
    right = np.array([f.right_knee_angle for f in frames], dtype=float)
    trunk = np.array([f.trunk_lean_deg for f in frames], dtype=float)
    pelvis = np.array([(f.left_hip_y + f.right_hip_y) / 2 for f in frames], dtype=float)
    left_rom = float(np.percentile(left, 95) - np.percentile(left, 5))
    right_rom = float(np.percentile(right, 95) - np.percentile(right, 5))
    return MovementFeatures(
        knee_rom_deg=(left_rom + right_rom) / 2,
        knee_asymmetry_deg=abs(left_rom - right_rom),
        trunk_lean_mean_deg=float(np.mean(np.abs(trunk))),
        trunk_lean_variability_deg=float(np.std(trunk)),
        pelvic_vertical_variability=float(np.std(pelvis)),
    )
