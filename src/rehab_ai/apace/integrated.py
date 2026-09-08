from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
from functools import lru_cache
import numpy as np

from rehab_ai.apace.algorithm import APACEConfig, BeliefState, solve_apace, APACESolution
from rehab_ai.control.stochastic_mpc import RehabAction, default_actions
from rehab_ai.ie.human_performance import HumanPerformanceState, transition_human_performance
from rehab_ai.ml.safe_envelope import PatientSafeEnvelopeLearner
from rehab_ai.ml.temporal_recovery import TemporalRecoveryModel
from rehab_ai.ml.treatment_response import GaussianProcessTreatmentResponse

@dataclass(frozen=True)
class IntegratedModelEvidence:
    envelope_test_mae: float
    envelope_baseline_mae: float
    envelope_coverage: float
    temporal_test_mae: float
    temporal_persistence_mae: float
    treatment_gp_test_mae: float
    treatment_gp_baseline_mae: float
    treatment_gp_coverage_95: float
    validation_scope: str = "SYNTHETIC INTEGRATION VALIDATION"

class IntegratedAdaptiveModels:
    """Fitted research models used together by the adaptive controller.

    The safe-envelope model defines feasibility. The temporal model modifies the
    future-capacity forecast. The GP supplies action-specific response and uncertainty.
    """
    def __init__(self, seed: int = 42):
        self.seed=seed
        self.envelope=PatientSafeEnvelopeLearner(random_state=seed)
        em=self.envelope.fit_validate(n=900, seed=seed)
        self.temporal=TemporalRecoveryModel(random_state=seed)
        tm=self.temporal.fit_validate(seed=seed)
        self.gp=GaussianProcessTreatmentResponse(random_state=seed)
        gm=self.gp.fit_validate(n=120, seed=seed)
        self.evidence=IntegratedModelEvidence(em.test_mae, em.baseline_mae, em.conservative_coverage, tm.test_mae, tm.persistence_mae, gm.test_mae, gm.baseline_mae, gm.coverage_95)

    def response(self, perf: HumanPerformanceState, action: RehabAction) -> tuple[float,float]:
        p=self.gp.predict(perf.capacity, perf.fatigue, perf.pain, action.as_load().functional_load, action.instability_demand)
        return p.expected_capacity_gain, p.std_capacity_gain

    def safe_limit(self, belief: BeliefState, action: RehabAction) -> float:
        adherence=getattr(belief,"adherence",0.90)
        overload=getattr(belief,"recent_overload",0.0)
        p=self.envelope.predict(belief.performance.capacity, belief.performance.fatigue, belief.performance.pain, belief.stability_margin_m, overload, adherence)
        return p.conservative_tolerated_load

    def temporal_capacity(self, belief: BeliefState, action: RehabAction, base_next: HumanPerformanceState) -> tuple[float,float]:
        hist=list(getattr(belief,"history",()) or ())
        dose=action.as_load().functional_load
        current=(belief.performance.capacity, belief.performance.fatigue, belief.performance.pain, getattr(belief,"adherence",0.90), dose)
        hist=(hist+[current])[-self.temporal.lags:]
        if len(hist)<self.temporal.lags:
            return base_next.capacity, 0.035
        pred=self.temporal.predict_next_capacity(np.asarray(hist,float))
        # Uncertainty proxy is deliberately tied to held-out temporal RMSE, not presented as a calibrated CI.
        std=max(0.006, self.temporal.metrics.test_rmse if self.temporal.metrics else 0.025)
        return pred, std


def solve_integrated_apace(initial: BeliefState, models: IntegratedAdaptiveModels, config: APACEConfig|None=None, actions: Sequence[RehabAction]|None=None) -> APACESolution:
    return solve_apace(initial, models.response, config=config, actions=actions or default_actions(), safe_limit_predictor=models.safe_limit, temporal_capacity_predictor=models.temporal_capacity)


@lru_cache(maxsize=8)
def get_integrated_models(seed:int=42) -> IntegratedAdaptiveModels:
    return IntegratedAdaptiveModels(seed=seed)
