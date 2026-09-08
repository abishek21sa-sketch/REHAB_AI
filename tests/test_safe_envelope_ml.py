from rehab_ai.ml.safe_envelope import PatientSafeEnvelopeLearner


def test_safe_envelope_beats_baseline_and_is_conservative():
    m = PatientSafeEnvelopeLearner()
    metrics = m.fit_validate(n=1100, seed=9)
    assert metrics.test_mae < metrics.baseline_mae * 0.55
    assert metrics.conservative_coverage >= 0.93
    pred = m.predict(0.65, 0.25, 0.15, 0.025, 0.05, 0.9)
    assert pred.conservative_tolerated_load < pred.nominal_tolerated_load
    assert pred.conservative_tolerated_load > 0


def test_safe_envelope_shrinks_with_pain_and_fatigue():
    m = PatientSafeEnvelopeLearner(); m.fit_validate(n=900, seed=11)
    low = m.predict(0.65, 0.15, 0.10, 0.025).conservative_tolerated_load
    high = m.predict(0.65, 0.75, 0.65, 0.025).conservative_tolerated_load
    assert high < low
