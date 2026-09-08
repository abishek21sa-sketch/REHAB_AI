from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from rehab_ai.apace import APACEConfig, BeliefState
from rehab_ai.apace.integrated import get_integrated_models, solve_integrated_apace
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.biomechanics.stability import margin_of_stability, symmetry_index
from rehab_ai.data.validation import validate_session
from rehab_ai.domain import SessionInput
from rehab_ai.ie.human_performance import HumanPerformanceState
from rehab_ai.ml.phenotypes import RehabilitationPhenotyper
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.state_estimation.ukf import UKFEstimate, UnscentedKalmanFilter
from rehab_ai.vision.movement import extract_movement_features


@dataclass(frozen=True)
class APACEStudioResult:
    patient_id: str
    evidence: dict
    biomechanics: dict
    latent_state: dict
    phenotype: dict
    treatment_response_ml: dict
    apace_policy: dict
    evidence_labels: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _ukf_state(capacity: float, fatigue: float, pain: float, stability: float) -> UKFEstimate:
    initial = UKFEstimate(
        np.array([0.50, 0.40, 0.30, 0.50], dtype=float),
        np.diag([0.13, 0.11, 0.10, 0.13]) ** 2,
    )
    ukf = UnscentedKalmanFilter(4, np.diag([0.010, 0.014, 0.012, 0.014]) ** 2)
    predicted = ukf.predict(
        initial,
        lambda x: np.clip(np.array([
            x[0] + 0.018 * (1 - x[1]),
            0.925 * x[1],
            0.945 * x[2],
            x[3] + 0.012 * x[0],
        ]), 0, 1),
    )
    measurement = np.array([capacity, fatigue, pain, stability], dtype=float)
    updated = ukf.update(predicted, measurement, lambda x: x, np.diag([0.05, 0.06, 0.06, 0.07]) ** 2)
    updated.mean[:] = np.clip(updated.mean, 0, 1)
    return updated


def run_apace_studio(session: SessionInput, *, horizon: int = 5, seed: int = 42, beam_width: int = 18, rollouts: int = 96) -> APACEStudioResult:
    quality = validate_session(session)
    wearable = extract_wearable_features(session.wearable)
    movement = extract_movement_features(session.pose)
    state = reconstruct_patient_state(session, wearable, movement)

    trunk_rad = np.deg2rad(movement.trunk_lean_mean_deg)
    com_pos = float(0.025 * np.sin(trunk_rad))
    com_vel = float(min(0.30, wearable.sway_index * 1.8))
    mos = margin_of_stability(com_pos, com_vel, 0.105, effective_leg_length_m=0.90)
    stability_norm = float(np.clip((mos.margin_of_stability_m + 0.02) / 0.12, 0, 1))
    bilateral_si = symmetry_index(
        max(1e-6, movement.knee_rom_deg + movement.knee_asymmetry_deg / 2),
        max(1e-6, movement.knee_rom_deg - movement.knee_asymmetry_deg / 2),
    )

    latent = _ukf_state(state.functional_capacity, state.fatigue, state.pain, stability_norm)
    latent_std = np.sqrt(np.diag(latent.covariance))

    phenotyper = RehabilitationPhenotyper(seed=seed)
    silhouette = phenotyper.fit_validate(n_per=45)
    phenotype = phenotyper.predict(np.array([
        latent.mean[0], latent.mean[1], latent.mean[2], latent.mean[3], max(0.0, 1.0 - bilateral_si)
    ]))

    integrated_models = get_integrated_models(seed)

    prior_std = float(np.clip(0.12 + 0.7 * np.mean(latent_std), 0.08, 0.28))
    belief = BeliefState(
        performance=HumanPerformanceState(float(latent.mean[0]), float(latent.mean[1]), float(latent.mean[2])),
        stability_margin_m=mos.margin_of_stability_m,
        parameter_mean=1.0,
        parameter_std=prior_std,
        state_uncertainty=float(np.mean(latent_std)),
    )
    cfg = APACEConfig(horizon=horizon, beam_width=beam_width, rollouts=rollouts, seed=seed)
    policy = solve_integrated_apace(belief, integrated_models, cfg)

    return APACEStudioResult(
        patient_id=session.patient_id,
        evidence={
            "quality": quality,
            "gait_speed_mps": state.gait_speed_mps,
            "cadence_spm": state.cadence_spm,
            "knee_rom_deg": state.knee_rom_deg,
            "pain": state.pain,
            "fatigue": state.fatigue,
        },
        biomechanics={
            "com_position_m": com_pos,
            "com_velocity_mps": com_vel,
            "extrapolated_com_m": mos.extrapolated_com_m,
            "margin_of_stability_m": mos.margin_of_stability_m,
            "bilateral_symmetry_index": bilateral_si,
            "stability_reserve_normalized": stability_norm,
        },
        latent_state={
            "motor_capacity": float(latent.mean[0]),
            "fatigue_burden": float(latent.mean[1]),
            "pain_burden": float(latent.mean[2]),
            "stability_reserve": float(latent.mean[3]),
            "std": latent_std.tolist(),
            "estimator": "Unscented Kalman Filter",
        },
        phenotype={**phenotype.to_dict(), "model": "KMeans", "population_silhouette": silhouette},
        treatment_response_ml={**asdict(integrated_models.evidence), "test_mae": integrated_models.evidence.treatment_gp_test_mae, "baseline_mae": integrated_models.evidence.treatment_gp_baseline_mae, "coverage_95": integrated_models.evidence.treatment_gp_coverage_95, "model": "GP response + temporal recovery + conformal safe envelope", "integration": "ALL THREE MODELS FEED APACE"},
        apace_policy=policy.to_dict(),
        evidence_labels={
            "observations": "CALCULATED FROM SYNTHETIC DEMO INPUT" if session.patient_id.startswith("DEMO") else "CALCULATED",
            "biomechanics": "CALCULATED",
            "latent_state": "ESTIMATED",
            "phenotype": "ML-INFERRED; SYNTHETIC VALIDATION",
            "response_model": "PREDICTED; SYNTHETIC VALIDATION; COUPLED INTO APACE",
            "policy": "OPTIMIZED APPROXIMATION; STOCHASTIC DIGITAL-TWIN ROLLOUT",
            "clinical_validation": "EXTERNAL VALIDATION PENDING",
        },
    )
