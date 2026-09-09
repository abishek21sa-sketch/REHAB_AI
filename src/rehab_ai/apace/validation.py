"""Portfolio-grade APACE validation.

The validator deliberately separates objective agreement, safety, capacity, tail risk,
and uncertainty. It does not claim APACE universally dominates every baseline on every
metric, and it does not turn synthetic evidence into clinical validation.
"""
from __future__ import annotations
from dataclasses import asdict
from statistics import NormalDist
from math import sqrt
from .benchmark import run_policy_benchmark



def _wilson_upper_bound(events: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Wilson upper bound for the modeled policy-violation rate."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(confidence)
    phat = events / n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = z*sqrt((phat*(1-phat) + z*z/(4*n))/n) / denom
    return min(1.0, center + margin)


def build_apace_validation(*, patients: int = 10, horizon: int = 2, seed: int = 2026) -> dict:
    suite = run_policy_benchmark(patients=patients, horizon=horizon, seed=seed)
    rows = {r.policy: asdict(r) for r in suite.results}
    apace = rows['apace']
    oracle = rows['oracle']
    standard = rows['standard_mpc']
    unsafe_events = int(round(apace['unsafe_rate'] * patients))
    modeled_unsafe_rate_upper_95 = _wilson_upper_bound(unsafe_events, patients, .95)
    checks = {
        'zero_unsafe_rate': apace['unsafe_rate'] == 0.0,
        'modeled_unsafe_rate_upper_95_le_5pct': modeled_unsafe_rate_upper_95 <= 0.05,
        'nonnegative_oracle_regret': apace['mean_regret_to_oracle'] >= -1e-12,
        'uncertainty_not_worse_than_standard_mpc': apace['mean_final_uncertainty'] <= standard['mean_final_uncertainty'] + 1e-9,
        'oracle_zero_regret': abs(oracle['mean_regret_to_oracle']) <= 1e-12,
    }
    return {
        'release': 'REHAB_AI_PORTFOLIO_RELEASE',
        'validation_scope': suite.validation_scope,
        'patients': patients,
        'horizon': horizon,
        'seed': seed,
        'policies': rows,
        'finite_sample_safety': {
            'modeled_unsafe_events': unsafe_events,
            'trials': patients,
            'one_sided_confidence': 0.95,
            'wilson_upper_bound': modeled_unsafe_rate_upper_95,
            'release_threshold': 0.05,
            'interpretation': 'Bound applies only to modeled policy-envelope violations in the synthetic benchmark, not clinical adverse-event probability.',
        },
        'checks': checks,
        'status': 'PASS' if all(checks.values()) else 'REVIEW',
        'claim_policy': {
            'universal_dominance_claimed': False,
            'allowed_claim': 'APACE is evaluated on its governed multi-objective objective and safety envelope; capacity, tail risk and uncertainty are reported separately.',
            'clinical_validation': 'EXTERNAL_VALIDATION_PENDING',
            'production_write_allowed': False,
        },
        'evidence_boundary': 'Synthetic rehabilitation-policy benchmark only; not a clinical effectiveness, safety, or medical-device validation claim.',
    }
