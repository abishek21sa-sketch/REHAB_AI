from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray


@dataclass
class UKFEstimate:
    mean: Array
    covariance: Array


class UnscentedKalmanFilter:
    """Small deterministic UKF for nonlinear rehabilitation state estimation."""

    def __init__(self, dim_x: int, process_noise: Array, alpha: float = 0.25, beta: float = 2.0, kappa: float = 0.0):
        if dim_x < 1:
            raise ValueError("dim_x must be positive")
        self.n = dim_x
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lam = alpha**2 * (dim_x + kappa) - dim_x
        self.Q = np.asarray(process_noise, dtype=float)
        if self.Q.shape != (dim_x, dim_x):
            raise ValueError("process_noise shape mismatch")
        c = dim_x + self.lam
        self.wm = np.full(2 * dim_x + 1, 1.0 / (2.0 * c))
        self.wc = self.wm.copy()
        self.wm[0] = self.lam / c
        self.wc[0] = self.wm[0] + (1.0 - alpha**2 + beta)

    def sigma_points(self, mean: Array, covariance: Array) -> Array:
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if mean.shape != (self.n,) or covariance.shape != (self.n, self.n):
            raise ValueError("state shape mismatch")
        jitter = 1e-10 * np.eye(self.n)
        root = np.linalg.cholesky((self.n + self.lam) * (covariance + jitter))
        pts = [mean]
        pts.extend(mean + root[:, i] for i in range(self.n))
        pts.extend(mean - root[:, i] for i in range(self.n))
        return np.asarray(pts)

    def predict(self, estimate: UKFEstimate, transition: Callable[[Array], Array]) -> UKFEstimate:
        sig = self.sigma_points(estimate.mean, estimate.covariance)
        propagated = np.asarray([transition(s) for s in sig], dtype=float)
        mean = np.sum(self.wm[:, None] * propagated, axis=0)
        cov = self.Q.copy()
        for w, s in zip(self.wc, propagated):
            d = s - mean
            cov += w * np.outer(d, d)
        return UKFEstimate(mean=mean, covariance=cov)

    def update(
        self,
        estimate: UKFEstimate,
        observation: Array,
        observe: Callable[[Array], Array],
        observation_noise: Array,
    ) -> UKFEstimate:
        sig = self.sigma_points(estimate.mean, estimate.covariance)
        zsig = np.asarray([observe(s) for s in sig], dtype=float)
        zmean = np.sum(self.wm[:, None] * zsig, axis=0)
        R = np.asarray(observation_noise, dtype=float)
        S = R.copy()
        Cxz = np.zeros((self.n, len(zmean)))
        for w, x, z in zip(self.wc, sig, zsig):
            dx, dz = x - estimate.mean, z - zmean
            S += w * np.outer(dz, dz)
            Cxz += w * np.outer(dx, dz)
        K = Cxz @ np.linalg.inv(S)
        innovation = np.asarray(observation, dtype=float) - zmean
        mean = estimate.mean + K @ innovation
        covariance = estimate.covariance - K @ S @ K.T
        covariance = 0.5 * (covariance + covariance.T)
        return UKFEstimate(mean=mean, covariance=covariance)
