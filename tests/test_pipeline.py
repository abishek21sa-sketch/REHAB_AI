from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.pipeline import RehabDecisionPipeline


def test_pipeline_end_to_end(tmp_path):
    pipe = RehabDecisionPipeline(database_path=str(tmp_path / "e2e.db"))
    result = pipe.run(generate_synthetic_session())
    assert result.recommendation.review_status == "clinician_review_required"
    assert result.recommendation.rationale
    assert result.recommendation.alternatives
    assert 0 <= result.recommendation.fall_risk.value <= 1
    assert result.recommendation.selected_plan.session_minutes <= pipe.settings.clinical.max_session_minutes
