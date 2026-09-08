from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.apace_studio import run_apace_studio
from rehab_ai.twin.adaptive_episode import run_adaptive_episode
from rehab_ai.biomechanics.assessment_protocols import assessment_suite


@dataclass(frozen=True)
class LearningSummary:
    weeks: int
    initial_uncertainty: float
    final_uncertainty: float
    uncertainty_reduction_pct: float
    mean_abs_prediction_error: float
    replans: int
    envelope_violations: int


@dataclass(frozen=True)
class ClinicalEpisodeSummary:
    patient_id: str
    phenotype: str
    validation_scope: str
    baseline: dict[str, Any]
    assessments: dict[str, Any]
    weeks: list[dict[str, Any]]
    learning: dict[str, Any]
    next_control: dict[str, Any]
    evidence_ledger: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_clinical_episode(*, impairment: float = 0.45, phenotype: str = "balanced", weeks: int = 5, seed: int = 42) -> ClinicalEpisodeSummary:
    session = generate_synthetic_session(
        patient_id=f"CASE-{int(impairment * 100):02d}", seed=seed, impairment=impairment
    )
    studio = run_apace_studio(session, horizon=min(6, max(3, weeks)), seed=seed).to_dict()
    ep = run_adaptive_episode(phenotype=phenotype, weeks=weeks, seed=seed).to_dict()
    e = studio["evidence"]
    b = studio["biomechanics"]
    latent = studio["latent_state"]
    suite = assessment_suite(
        gait_speed_mps=e["gait_speed_mps"],
        cadence_spm=e["cadence_spm"],
        knee_rom_deg=e["knee_rom_deg"],
        symmetry_index=b["bilateral_symmetry_index"],
        margin_of_stability_m=b["margin_of_stability_m"],
        fatigue=float(latent.get("fatigue_burden", 0.0)),
        pain=float(latent.get("pain_burden", 0.0)),
        motor_capacity=float(latent.get("motor_capacity", 0.0)),
    )
    week_rows = ep["weeks"]
    errors = [abs(float(w["observed_gain"]) - float(w["predicted_gain"])) for w in week_rows]
    initial_uncertainty = 0.20
    final_uncertainty = float(week_rows[-1]["parameter_std"]) if week_rows else initial_uncertainty
    learning = LearningSummary(
        weeks=len(week_rows),
        initial_uncertainty=initial_uncertainty,
        final_uncertainty=final_uncertainty,
        uncertainty_reduction_pct=100.0 * (initial_uncertainty - final_uncertainty) / initial_uncertainty,
        mean_abs_prediction_error=sum(errors) / max(1, len(errors)),
        replans=int(ep["replans"]),
        envelope_violations=int(ep["envelope_violations"]),
    )
    trajectory = studio.get("apace_policy", {}).get("trajectory", [])
    next_row = trajectory[0] if trajectory else {}
    next_control = {
        "action": next_row.get("action", "NOT AVAILABLE"),
        "expected_gain": next_row.get("expected_gain"),
        "tail_risk": next_row.get("cvar_loss"),
        "status": studio.get("apace_policy", {}).get("status", "UNKNOWN"),
        "evidence_language": "OPTIMIZED DECISION / SYNTHETIC DIGITAL TWIN",
    }
    evidence_ledger = [
        {"label": "gait speed", "kind": "OBSERVED / SYNTHETIC FIXTURE", "value": f"{e['gait_speed_mps']:.2f} m/s"},
        {"label": "margin of stability", "kind": "CALCULATED", "value": f"{b['margin_of_stability_m']:.3f} m"},
        {"label": "motor capacity", "kind": "ESTIMATED", "value": f"{latent['motor_capacity']:.3f}"},
        {"label": "next treatment gain", "kind": "PREDICTED", "value": str(next_row.get('expected_gain', '—'))},
        {"label": "future trajectory", "kind": "SIMULATED", "value": f"{len(trajectory)} planned weeks"},
        {"label": "next intervention", "kind": "OPTIMIZED", "value": str(next_row.get('action', '—'))},
    ]
    return ClinicalEpisodeSummary(
        patient_id=studio["patient_id"],
        phenotype=phenotype,
        validation_scope="SYNTHETIC CLINICAL WORKFLOW / EXTERNAL VALIDATION PENDING",
        baseline={
            "motor_capacity": latent["motor_capacity"],
            "fatigue": latent["fatigue_burden"],
            "pain": latent["pain_burden"],
            "stability_reserve": latent["stability_reserve"],
        },
        assessments={k: v.to_dict() for k, v in suite.items()},
        weeks=week_rows,
        learning=asdict(learning),
        next_control=next_control,
        evidence_ledger=evidence_ledger,
    )


def episode_markdown_report(summary: ClinicalEpisodeSummary) -> str:
    d = summary.to_dict()
    lines = [
        f"# REHAB AI — Episode Engineering Report ({d['patient_id']})",
        "",
        f"**Validation scope:** {d['validation_scope']}",
        f"**Phenotype:** {d['phenotype']}",
        "",
        "## Evidence Ledger",
    ]
    for row in d["evidence_ledger"]:
        lines.append(f"- **{row['kind']} — {row['label']}:** {row['value']}")
    lines += ["", "## Assessment Suite"]
    for name, finding in d["assessments"].items():
        lines.append(f"- **{name}:** score {finding['score']:.3f}; primary limitation: {finding['primary_limitation']}")
    lines += ["", "## Adaptive Learning"]
    l = d["learning"]
    lines.append(f"- Uncertainty: {l['initial_uncertainty']:.3f} → {l['final_uncertainty']:.3f} ({l['uncertainty_reduction_pct']:.1f}% reduction)")
    lines.append(f"- Mean absolute prediction error: {l['mean_abs_prediction_error']:.4f}")
    lines.append(f"- Replans: {l['replans']}; envelope violations: {l['envelope_violations']}")
    lines += ["", "## Weekly Decision Ledger"]
    for w in d["weeks"]:
        lines.append(
            f"- Week {w['week']}: {w['action']} | load {w['applied_load']:.3f}/{w['safe_limit']:.3f} safe limit | predicted gain {w['predicted_gain']:.4f} | observed gain {w['observed_gain']:.4f}"
        )
    lines += ["", "## Next Optimized Control", f"**{d['next_control']['action']}** — {d['next_control']['evidence_language']}"]
    return "\n".join(lines) + "\n"
