import numpy as np
from rehab_ai.biomechanics.signal_quality import movement_signal_quality


def test_periodic_signal_has_more_periodicity_than_noise():
    fs = 100.0
    t = np.arange(0, 4, 1/fs)
    sine = np.sin(2*np.pi*2*t)
    rng = np.random.default_rng(3)
    noise = rng.normal(size=len(t))
    a = movement_signal_quality(sine, fs)
    b = movement_signal_quality(noise, fs)
    assert a.periodicity_index > b.periodicity_index
    assert a.spectral_entropy < b.spectral_entropy
