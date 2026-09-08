from rehab_ai.ml.temporal_recovery import TemporalRecoveryModel


def test_temporal_model_uses_time_aware_split_and_beats_persistence():
    m = TemporalRecoveryModel(random_state=7)
    metrics = m.fit_validate(seed=7)
    assert metrics.test_mae < metrics.persistence_mae
    assert metrics.n_train_windows > metrics.n_test_windows
    assert "LONGITUDINAL" in metrics.validation_scope
