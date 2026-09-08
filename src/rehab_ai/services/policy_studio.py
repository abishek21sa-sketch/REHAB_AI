from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.biomechanics.stability import margin_of_stability, symmetry_index
from rehab_ai.control.stochastic_mpc import MPCSolution, solve_stochastic_mpc
from rehab_ai.data.validation import validate_session
from rehab_ai.domain import SessionInput
from rehab_ai.ie.human_performance import HumanPerformanceState
from rehab_ai.ml.treatment_response import GaussianProcessTreatmentResponse, TreatmentResponseMetrics
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.state_estimation.ukf import UKFEstimate, UnscentedKalmanFilter
from rehab_ai.vision.movement import extract_movement_features


@dataclass(frozen=True)
class PolicyStudioResult:
    patient_id: str
    data_quality: dict
    observed: dict
    biomechanical: dict
    latent_state: dict
    treatment_response_validation: dict
    control_policy: dict
    evidence_labels: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_latent_state(capacity: float, fatigue: float, pain: float, stability: float) -> UKFEstimate:
    # latent state = [motor capacity, fatigue burden, pain burden, stability reserve]
    mean0 = np.array([0.50, 0.40, 0.30, 0.50], dtype=float)
    cov0 = np.diag([0.12, 0.10, 0.10, 0.12]) ** 2
    ukf = UnscentedKalmanFilter(4, np.diag([0.010, 0.015, 0.012, 0.015]) ** 2)
    pred = ukf.predict(
        UKFEstimate(mean0, cov0),
        lambda x: np.clip(np.array([x[0] + 0.02 * (1 - x[1]), 0.92 * x[1], 0.94 * x[2], x[3] + 0.01 * x[0]]), 0, 1),
    )
    obs = np.array([capacity, fatigue, pain, stability], dtype=float)
    upd = ukf.update(pred, obs, lambda x: x, np.diag([0.05, 0.06, 0.06, 0.07]) ** 2)
    upd.mean[:] = np.clip(upd.mean, 0, 1)
    return upd


def run_policy_studio(session: SessionInput, *, horizon_weeks: int = 4, seed: int = 42) -> PolicyStudioResult:
    quality = validate_session(session)
    wearable = extract_wearable_features(session.wearable)
    movement = extract_movement_features(session.pose)
    state = reconstruct_patient_state(session, wearable, movement)

    # Approximate AP COM excursion from trunk lean and dynamic sway for the current engineering prototype.
    trunk_rad = np.deg2rad(movement.trunk_lean_mean_deg)
    com_pos = float(0.025 * np.sin(trunk_rad))
    com_vel = float(min(0.30, wearable.sway_index * 1.8))
    bos_boundary = 0.105
    mos = margin_of_stability(com_pos, com_vel, bos_boundary, effective_leg_length_m=0.90)
    stability_norm = float(np.clip((mos.margin_of_stability_m + 0.02) / 0.12, 0, 1))
    bilateral_si = symmetry_index(
        max(1e-6, movement.knee_rom_deg + movement.knee_asymmetry_deg / 2),
        max(1e-6, movement.knee_rom_deg - movement.knee_asymmetry_deg / 2),
    )

    latent = _estimate_latent_state(state.functional_capacity, state.fatigue, state.pain, stability_norm)

    # The GP is intentionally trained from a deterministic synthetic response benchmark in V0.x.
    # It supplies uncertainty to the robust MPC; this is not a clinical treatment-effect claim.
    response = GaussianProcessTreatmentResponse(random_state=seed)
    metrics = response.fit_validate(n=90, seed=seed)
    reference_load = 0.42
    rp = response.predict(float(latent.mean[0]), float(latent.mean[1]), float(latent.mean[2]), reference_load, 1 - stability_norm)

    hp = HumanPerformanceState(
        capacity=float(latent.mean[0]),
        fatigue=float(latent.mean[1]),
        pain=float(latent.mean[2]),
    )
    policy: MPCSolution = solve_stochastic_mpc(
        hp,
        horizon_weeks=horizon_weeks,
        stability_margin_m=mos.margin_of_stability_m,
        response_uncertainty=max(0.006, min(0.05, rp.std_capacity_gain)),
    )

    return PolicyStudioResult(
        patient_id=session.patient_id,
        data_quality=quality,
        observed={
            "gait_speed_mps": state.gait_speed_mps,
            "cadence_spm": state.cadence_spm,
            "knee_rom_deg": state.knee_rom_deg,
            "fatigue": state.fatigue,
            "pain": state.pain,
        },
        biomechanical={
            "margin_of_stability_m": mos.margin_of_stability_m,
            "extrapolated_com_m": mos.extrapolated_com_m,
            "bilateral_symmetry_index": bilateral_si,
            "stability_reserve_normalized": stability_norm,
        },
        latent_state={
            "motor_capacity": float(latent.mean[0]),
            "fatigue_burden": float(latent.mean[1]),
            "pain_burden": float(latent.mean[2]),
            "stability_reserve": float(latent.mean[3]),
            "std": np.sqrt(np.diag(latent.covariance)).tolist(),
            "estimator": "Unscented Kalman Filter",
        },
        treatment_response_validation={
            **asdict(metrics),
            "reference_expected_gain": rp.expected_capacity_gain,
            "reference_std_gain": rp.std_capacity_gain,
            "model": "GaussianProcessRegressor/Matern",
        },
        control_policy={
            "status": policy.status,
            "objective": policy.objective,
            "sequence": [a.name for a in policy.sequence],
            "sequences_evaluated": policy.sequences_evaluated,
            "feasible_sequences": policy.feasible_sequences,
            "trajectory": [
                {
                    "week": s.week,
                    "action": s.action.name,
                    "capacity": s.expected_capacity,
                    "fatigue": s.expected_fatigue,
                    "pain": s.expected_pain,
                    "load": s.load,
                    "safe_limit": s.safe_limit,
                }
                for s in policy.trajectory
            ],
            "method": "finite-horizon chance-buffered stochastic MPC via exact finite action enumeration",
        },
        evidence_labels={
            "sensor_features": "CALCULATED FROM SYNTHETIC DEMO INPUT" if session.patient_id.startswith("DEMO") else "CALCULATED",
            "latent_state": "ESTIMATED",
            "treatment_response": "PREDICTED; SYNTHETIC MODEL VALIDATION",
            "policy": "OPTIMIZED",
            "future_trajectory": "SIMULATED / MODELED",
            "clinical_validation": "EXTERNAL VALIDATION PENDING",
        },
    )
