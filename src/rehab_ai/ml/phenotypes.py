from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PhenotypeResult:
    phenotype: str
    cluster: int
    silhouette: float
    distances: tuple[float, ...]
    validation_scope: str = "SYNTHETIC VALIDATION"

    def to_dict(self) -> dict:
        return asdict(self)


class RehabilitationPhenotyper:
    labels = {
        0: "balance-limited",
        1: "fatigue-sensitive",
        2: "mobility-limited",
        3: "high-functioning",
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=4, random_state=seed, n_init=20)
        self.silhouette_: float | None = None
        self.cluster_labels_: dict[int, str] = {}

    @staticmethod
    def synthetic_population(n_per: int = 100, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        centers = np.array([
            [0.48, 0.38, 0.28, 0.24, 0.42],
            [0.52, 0.74, 0.35, 0.48, 0.50],
            [0.31, 0.48, 0.42, 0.36, 0.30],
            [0.79, 0.22, 0.18, 0.73, 0.77],
        ])
        blocks = [np.clip(rng.normal(c, 0.055, size=(n_per, 5)), 0, 1) for c in centers]
        return np.vstack(blocks)

    def fit_validate(self, n_per: int = 80) -> float:
        X = self.synthetic_population(n_per=n_per, seed=self.seed)
        Z = self.scaler.fit_transform(X)
        clusters = self.model.fit_predict(Z)
        self.silhouette_ = float(silhouette_score(Z, clusters))
        # Assign semantic label by nearest known synthetic archetype in original feature space.
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        anchors = self.synthetic_population(1, self.seed)
        for k, center in enumerate(centers):
            idx = int(np.argmin(np.linalg.norm(anchors - center, axis=1)))
            self.cluster_labels_[k] = self.labels[idx]
        return self.silhouette_

    def predict(self, features: np.ndarray) -> PhenotypeResult:
        if self.silhouette_ is None:
            self.fit_validate()
        x = np.asarray(features, dtype=float).reshape(1, -1)
        z = self.scaler.transform(x)
        cluster = int(self.model.predict(z)[0])
        d = tuple(float(v) for v in self.model.transform(z)[0])
        return PhenotypeResult(self.cluster_labels_[cluster], cluster, float(self.silhouette_), d)
