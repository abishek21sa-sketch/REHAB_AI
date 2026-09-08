from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import numpy as np

from rehab_ai.ml.safe_envelope import PatientSafeEnvelopeLearner
from rehab_ai.ml.temporal_recovery import TemporalRecoveryModel
from rehab_ai.biomechanics.energetics import joint_energetics
from rehab_ai.biomechanics.signal_quality import movement_signal_quality


def main():
    envelope = PatientSafeEnvelopeLearner()
    em = envelope.fit_validate(n=1100, seed=17)
    temporal = TemporalRecoveryModel(random_state=17)
    tm = temporal.fit_validate(seed=17)

    fs = 100.0
    t = np.arange(0, 4, 1/fs)
    periodic = movement_signal_quality(np.sin(2*np.pi*1.8*t), fs)
    noise = movement_signal_quality(np.random.default_rng(17).normal(size=len(t)), fs)

    tt = np.array([0., .5, 1.0])
    e = joint_energetics(np.array([10., 10., 10.]), np.array([1., 1., 1.]), tt, 70.0)

    checks = {
        'safe_envelope_beats_baseline': em.test_mae < em.baseline_mae * 0.55,
        'safe_envelope_conservative_coverage': em.conservative_coverage >= 0.93,
        'temporal_ml_beats_persistence': tm.test_mae < tm.persistence_mae,
        'periodic_signal_detected': periodic.periodicity_index > noise.periodicity_index,
        'spectral_entropy_orders_noise': periodic.spectral_entropy < noise.spectral_entropy,
        'joint_work_reference': abs(e.positive_work_j - 10.0) < 1e-9,
    }
    out = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'safe_envelope_metrics': em.__dict__,
        'temporal_recovery_metrics': tm.__dict__,
        'signal_quality': {'periodic': periodic.__dict__, 'noise': noise.__dict__},
        'joint_energetics_reference': e.__dict__,
    }
    print(json.dumps(out, indent=2))
    if out['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
