from __future__ import annotations

from dataclasses import dataclass

from rehab_ai.ai.risk import predict_fall_risk
from rehab_ai.ai.trajectory import predict_adherence_risk, predict_fatigue_exacerbation, predict_gait_deterioration
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.config import Settings, load_settings
from rehab_ai.data.validation import validate_session
from rehab_ai.decision.engine import make_recommendation
from rehab_ai.domain import MovementFeatures, PatientState, Prediction, Recommendation, SessionInput, WearableFeatures
from rehab_ai.experiments.tracker import JsonlExperimentTracker
from rehab_ai.fusion.state_fusion import FusionAssessment, assess_multimodal_fusion
from rehab_ai.governance.audit import DecisionAudit, build_decision_audit
from rehab_ai.observability.metrics import metrics
from rehab_ai.optimization.multiobjective import pareto_front
from rehab_ai.persistence.repository import StateRepository
from rehab_ai.registry.models import active_model_versions
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.twin.engine import PatientDigitalTwin, TwinUpdate
from rehab_ai.twin.forecast import TwinForecast, forecast_twin
from rehab_ai.vision.movement import extract_movement_features


@dataclass
class PipelineResult:
    data_quality: dict[str, float | int]
    wearable_features: WearableFeatures
    movement_features: MovementFeatures
    fusion: FusionAssessment
    state: PatientState
    twin: TwinUpdate
    gait_deterioration: Prediction
    adherence_risk: Prediction
    fatigue_risk: Prediction
    twin_forecast: TwinForecast
    pareto_alternatives: list[dict]
    audit: DecisionAudit
    recommendation: Recommendation

    def to_dict(self) -> dict:
        return {
            "data_quality": self.data_quality,
            "wearable_features": self.wearable_features.model_dump(),
            "movement_features": self.movement_features.model_dump(),
            "fusion": self.fusion.model_dump(),
            "state": self.state.model_dump(mode="json"),
            "twin": {
                "previous": self.twin.previous.model_dump(mode="json") if self.twin.previous else None,
                "deltas": self.twin.deltas,
            },
            "predictions": {
                "gait_deterioration": self.gait_deterioration.model_dump(),
                "adherence_risk": self.adherence_risk.model_dump(),
                "fatigue_risk": self.fatigue_risk.model_dump(),
            },
            "twin_forecast": self.twin_forecast.model_dump(),
            "pareto_alternatives": self.pareto_alternatives,
            "audit": self.audit.model_dump(),
            "recommendation": self.recommendation.model_dump(mode="json"),
        }


class RehabDecisionPipeline:
    def __init__(self, settings: Settings | None = None, database_path: str | None = None):
        self.settings = settings or load_settings()
        db = database_path or self.settings.database_path
        self.repository = StateRepository(db)
        self.twin = PatientDigitalTwin(self.repository)
        self.tracker = JsonlExperimentTracker()

    def run(self, session: SessionInput) -> PipelineResult:
        metrics.inc("pipeline_runs")
        quality = validate_session(session)
        wearable = extract_wearable_features(session.wearable)
        movement = extract_movement_features(session.pose)
        fusion = assess_multimodal_fusion(wearable, movement)
        state = reconstruct_patient_state(session, wearable, movement)
        twin_update = self.twin.update(state)
        fall_risk, factors = predict_fall_risk(state)
        recommendation = make_recommendation(state, fall_risk, factors, self.settings)

        gait_deterioration = predict_gait_deterioration(state, twin_update.deltas)
        adherence_risk = predict_adherence_risk(state)
        fatigue_risk = predict_fatigue_exacerbation(state)
        twin_forecast = forecast_twin(
            state,
            recommendation.selected_plan,
            fall_risk.value,
            horizon_weeks=self.settings.simulation.horizon_weeks,
        )
        all_scenarios = [recommendation.scenario, *recommendation.alternatives]
        pareto = pareto_front(all_scenarios)
        pareto_payload = [
            {
                "plan": p.result.plan.model_dump(),
                "utility": p.utility,
                "safety": p.safety,
                "burden": p.burden,
            }
            for p in pareto[:5]
        ]
        audit = build_decision_audit(
            state.patient_id,
            session.model_dump(mode="json"),
            recommendation.model_dump(mode="json"),
            active_model_versions(),
        )

        self.repository.save_recommendation(recommendation)
        self.repository.save_audit(audit.model_dump_json())
        self.tracker.log("decision_completed", {
            "patient_id": state.patient_id,
            "decision_id": audit.decision_id,
            "fall_risk": recommendation.fall_risk.value,
            "gait_deterioration": gait_deterioration.value,
            "functional_capacity": state.functional_capacity,
            "fusion_confidence": fusion.confidence,
            "selected_plan": recommendation.selected_plan.model_dump(),
            "review_status": recommendation.review_status,
        })
        metrics.inc("recommendations_generated")
        return PipelineResult(
            data_quality=quality,
            wearable_features=wearable,
            movement_features=movement,
            fusion=fusion,
            state=state,
            twin=twin_update,
            gait_deterioration=gait_deterioration,
            adherence_risk=adherence_risk,
            fatigue_risk=fatigue_risk,
            twin_forecast=twin_forecast,
            pareto_alternatives=pareto_payload,
            audit=audit,
            recommendation=recommendation,
        )
