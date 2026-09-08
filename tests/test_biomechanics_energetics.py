import numpy as np
from rehab_ai.biomechanics.energetics import joint_power, joint_energetics


def test_joint_power_and_work_reference_case():
    t = np.array([0., 1., 2.])
    moment = np.array([2., 2., 2.])
    omega = np.array([1., 1., 1.])
    p = joint_power(moment, omega)
    assert np.allclose(p, 2.0)
    e = joint_energetics(moment, omega, t, body_mass_kg=50)
    assert abs(e.positive_work_j - 4.0) < 1e-12
    assert abs(e.net_work_j - 4.0) < 1e-12
    assert e.negative_work_j == 0.0
    assert abs(e.mechanical_cost_j_per_kg - 0.08) < 1e-12
