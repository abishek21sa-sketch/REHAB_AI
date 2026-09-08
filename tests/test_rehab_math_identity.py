import numpy as np
from rehab_ai.biomechanics.stability import segmental_com, extrapolated_com, margin_of_stability, symmetry_index
from rehab_ai.ie.human_performance import HumanPerformanceState, TherapyLoad, safe_load_limit, transition_human_performance
from rehab_ai.state_estimation.ukf import UKFEstimate, UnscentedKalmanFilter


def test_segmental_com_reference_case():
    assert segmental_com(np.array([0.0, 1.0]), np.array([1.0, 3.0])) == 0.75


def test_margin_of_stability_reference_equation():
    xcom = extrapolated_com(0.02, 0.0, 1.0)
    assert abs(xcom - 0.02) < 1e-12
    m = margin_of_stability(0.02, 0.0, 0.10, 1.0)
    assert abs(m.margin_of_stability_m - 0.08) < 1e-12


def test_symmetry_index_boundary():
    assert symmetry_index(10, 10) == 0
    assert abs(symmetry_index(12, 8) - 0.4) < 1e-12


def test_human_performance_overload_increases_fatigue():
    state = HumanPerformanceState(.55, .20, .15)
    low = transition_human_performance(state, TherapyLoad(.25, 20, .1), .06)
    high = transition_human_performance(state, TherapyLoad(1.0, 55, .7), .06)
    assert high.fatigue > low.fatigue
    assert safe_load_limit(.8,.1,.06) > safe_load_limit(.4,.1,.06)


def test_ukf_observation_reduces_uncertainty():
    ukf = UnscentedKalmanFilter(2, np.diag([.01,.01])**2)
    est = UKFEstimate(np.array([.5,.5]), np.diag([.2,.2])**2)
    pred = ukf.predict(est, lambda x: x)
    upd = ukf.update(pred, np.array([.6,.4]), lambda x: x, np.diag([.05,.05])**2)
    assert np.trace(upd.covariance) < np.trace(pred.covariance)
    assert upd.mean[0] > .5 and upd.mean[1] < .5
