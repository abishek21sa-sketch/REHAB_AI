from rehab_ai.ml.treatment_response import GaussianProcessTreatmentResponse


def test_gp_response_model_beats_mean_baseline_and_reports_uncertainty():
    model = GaussianProcessTreatmentResponse(random_state=42)
    m = model.fit_validate(n=90, seed=42)
    assert m.test_mae < m.baseline_mae
    assert m.test_rmse > 0
    assert .75 <= m.coverage_95 <= 1.0
    pred = model.predict(.55,.25,.15,.45,.20)
    assert pred.std_capacity_gain > 0
