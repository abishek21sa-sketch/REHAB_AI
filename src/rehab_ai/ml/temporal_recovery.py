from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class TemporalRecoveryMetrics:
    test_mae: float
    test_rmse: float
    persistence_mae: float
    n_train_windows: int
    n_test_windows: int
    validation_scope: str = "SYNTHETIC LONGITUDINAL VALIDATION"


class TemporalRecoveryModel:
    """Time-aware next-assessment recovery model using lagged patient trajectories.

    No random future leakage: early synthetic episodes train, later held-out episodes test.
    """

    def __init__(self, random_state: int = 42, lags: int = 4):
        self.lags = lags
        self.model = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2, random_state=random_state, n_jobs=1)
        self.metrics: TemporalRecoveryMetrics | None = None

    @staticmethod
    def _episodes(n_patients: int = 180, weeks: int = 14, seed: int = 42):
        rng = np.random.default_rng(seed)
        episodes = []
        for _ in range(n_patients):
            phenotype = rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20])
            cap = rng.uniform(0.22, 0.62)
            fatigue = rng.uniform(0.12, 0.55)
            pain = rng.uniform(0.05, 0.55)
            adherence = rng.uniform(0.55, 0.98)
            arr = []
            for week in range(weeks):
                dose = np.clip(rng.normal(0.55 + 0.12 * phenotype, 0.12), 0.18, 0.95)
                response = (0.035 + 0.012 * phenotype) * dose * adherence * (1 - 0.55 * fatigue) * (1 - 0.35 * pain)
                response += 0.008 * np.sin(week / 2.0) - 0.012 * max(0, dose - cap - 0.20)
                cap = np.clip(cap + response + rng.normal(0, 0.006), 0.0, 1.0)
                fatigue = np.clip(0.68 * fatigue + 0.30 * dose - 0.10 * adherence + rng.normal(0, 0.015), 0, 1)
                pain = np.clip(0.78 * pain + 0.07 * dose + 0.04 * fatigue + rng.normal(0, 0.012), 0, 1)
                arr.append([cap, fatigue, pain, adherence, dose])
            episodes.append(np.asarray(arr, float))
        return episodes

    def _windows(self, episodes):
        X, y, baseline = [], [], []
        for ep in episodes:
            for t in range(self.lags, len(ep)):
                hist = ep[t-self.lags:t]
                X.append(hist.reshape(-1))
                y.append(ep[t, 0])
                baseline.append(ep[t-1, 0])
        return np.asarray(X), np.asarray(y), np.asarray(baseline)

    def fit_validate(self, seed: int = 42) -> TemporalRecoveryMetrics:
        episodes = self._episodes(seed=seed)
        split = int(0.75 * len(episodes))
        train_eps, test_eps = episodes[:split], episodes[split:]
        Xtr, ytr, _ = self._windows(train_eps)
        Xte, yte, persistence = self._windows(test_eps)
        self.model.fit(Xtr, ytr)
        pred = self.model.predict(Xte)
        metrics = TemporalRecoveryMetrics(
            test_mae=float(mean_absolute_error(yte, pred)),
            test_rmse=float(mean_squared_error(yte, pred) ** 0.5),
            persistence_mae=float(mean_absolute_error(yte, persistence)),
            n_train_windows=len(ytr),
            n_test_windows=len(yte),
        )
        self.metrics = metrics
        return metrics

    def predict_next_capacity(self, history: np.ndarray) -> float:
        history = np.asarray(history, float)
        if history.shape != (self.lags, 5):
            raise ValueError(f"history must have shape ({self.lags}, 5)")
        return float(np.clip(self.model.predict(history.reshape(1, -1))[0], 0.0, 1.0))
