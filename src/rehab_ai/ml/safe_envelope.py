from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SafeEnvelopePrediction:
    nominal_tolerated_load: float
    conservative_tolerated_load: float
    conformal_radius: float


@dataclass(frozen=True)
class SafeEnvelopeMetrics:
    test_mae: float
    baseline_mae: float
    conservative_coverage: float
    conformal_radius: float
    validation_scope: str = "SYNTHETIC VALIDATION"


class PatientSafeEnvelopeLearner:
    """Learns a patient-specific tolerated rehabilitation load with split-conformal safety margin.

    Features: capacity, fatigue, pain, stability margin, recent overload fraction, adherence.
    Target: maximum tolerated functional load before an adverse fatigue/pain/stability response.
    """

    def __init__(self, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=180, random_state=random_state)
        self.radius_: float | None = None
        self.metrics: SafeEnvelopeMetrics | None = None

    @staticmethod
    def synthetic_dataset(n: int = 1400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        capacity = rng.uniform(0.18, 0.95, n)
        fatigue = rng.uniform(0.02, 0.92, n)
        pain = rng.uniform(0.0, 0.88, n)
        stability = rng.uniform(-0.005, 0.055, n)
        recent_overload = rng.uniform(0.0, 0.55, n)
        adherence = rng.uniform(0.45, 1.0, n)

        stability_factor = np.clip(0.62 + 7.5 * stability, 0.22, 1.18)
        tolerated = 1.42 * capacity * (1 - 0.58 * pain) * (1 - 0.36 * fatigue) * stability_factor
        tolerated *= (1 - 0.42 * recent_overload) * (0.86 + 0.14 * adherence)
        tolerated += 0.05 * np.sin(3.0 * capacity) - 0.025 * pain * fatigue
        tolerated += rng.normal(0, 0.035, n)
        tolerated = np.clip(tolerated, 0.08, 1.35)
        X = np.c_[capacity, fatigue, pain, stability, recent_overload, adherence]
        return X, tolerated

    def fit_validate(self, n: int = 1400, seed: int = 42, alpha: float = 0.05) -> SafeEnvelopeMetrics:
        X, y = self.synthetic_dataset(n=n, seed=seed)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=seed)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed + 1)
        self.model.fit(X_train, y_train)

        cal_pred = self.model.predict(X_cal)
        residuals = np.abs(y_cal - cal_pred)
        q = min(1.0, np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
        radius = float(np.quantile(residuals, q, method="higher"))
        self.radius_ = radius

        pred = self.model.predict(X_test)
        conservative = pred - radius
        baseline = np.full_like(y_test, np.mean(y_train))
        # Conservative lower bound should be <= true tolerated load with ~1-alpha probability.
        coverage = float(np.mean(conservative <= y_test))
        metrics = SafeEnvelopeMetrics(
            test_mae=float(mean_absolute_error(y_test, pred)),
            baseline_mae=float(mean_absolute_error(y_test, baseline)),
            conservative_coverage=coverage,
            conformal_radius=radius,
        )
        self.metrics = metrics
        return metrics

    def predict(self, capacity: float, fatigue: float, pain: float, stability_margin_m: float,
                recent_overload_fraction: float = 0.0, adherence: float = 0.9) -> SafeEnvelopePrediction:
        if self.radius_ is None:
            raise RuntimeError("fit_validate must be called before predict")
        X = np.array([[capacity, fatigue, pain, stability_margin_m, recent_overload_fraction, adherence]], dtype=float)
        nominal = float(self.model.predict(X)[0])
        conservative = max(0.05, nominal - self.radius_)
        return SafeEnvelopePrediction(nominal, conservative, self.radius_)
