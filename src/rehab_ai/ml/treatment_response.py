from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ResponsePrediction:
    expected_capacity_gain: float
    std_capacity_gain: float


@dataclass(frozen=True)
class TreatmentResponseMetrics:
    test_mae: float
    test_rmse: float
    baseline_mae: float
    coverage_95: float
    n_train: int
    n_test: int
    validation_scope: str = "SYNTHETIC VALIDATION"


class GaussianProcessTreatmentResponse:
    """Uncertainty-aware response model for capacity gain after a therapy dose."""

    def __init__(self, random_state: int = 42):
        kernel = ConstantKernel(1.0, (0.1, 100.0)) * Matern(length_scale=np.ones(5), nu=1.5) + WhiteKernel(noise_level=0.01)
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state, optimizer=None)
        self.metrics: TreatmentResponseMetrics | None = None
        self.uncertainty_scale = 1.25

    @staticmethod
    def synthetic_dataset(n: int = 420, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        capacity = rng.uniform(0.2, 0.9, n)
        fatigue = rng.uniform(0.05, 0.85, n)
        pain = rng.uniform(0.0, 0.8, n)
        load = rng.uniform(0.05, 1.2, n)
        instability = rng.uniform(0.0, 0.8, n)
        optimal = 0.75 * capacity + 0.08
        distance = (load - optimal) / 0.42
        gain = 0.06 * np.exp(-(distance**2)) * (1 - 0.55 * fatigue) * (1 - 0.35 * pain)
        gain -= 0.05 * np.maximum(0, load - 1.35 * capacity) ** 2
        gain -= 0.012 * instability * load
        gain += rng.normal(0, 0.0045, n)
        X = np.c_[capacity, fatigue, pain, load, instability]
        return X, gain

    def fit_validate(self, n: int = 420, seed: int = 42) -> TreatmentResponseMetrics:
        X, y = self.synthetic_dataset(n=n, seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
        self.model.fit(Xtr, ytr)
        pred, std = self.model.predict(Xte, return_std=True)
        std = std * self.uncertainty_scale
        baseline = np.full_like(yte, float(np.mean(ytr)))
        lower, upper = pred - 1.96 * std, pred + 1.96 * std
        metrics = TreatmentResponseMetrics(
            test_mae=float(mean_absolute_error(yte, pred)),
            test_rmse=float(mean_squared_error(yte, pred) ** 0.5),
            baseline_mae=float(mean_absolute_error(yte, baseline)),
            coverage_95=float(np.mean((yte >= lower) & (yte <= upper))),
            n_train=len(ytr),
            n_test=len(yte),
        )
        self.metrics = metrics
        return metrics

    def predict(self, capacity: float, fatigue: float, pain: float, load: float, instability: float) -> ResponsePrediction:
        x = np.array([[capacity, fatigue, pain, load, instability]], dtype=float)
        mean, std = self.model.predict(x, return_std=True)
        return ResponsePrediction(float(mean[0]), float(std[0] * self.uncertainty_scale))
