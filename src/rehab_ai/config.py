from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml


@dataclass(frozen=True)
class ClinicalConfig:
    max_session_minutes: int = 60
    high_fall_risk_threshold: float = 0.65
    high_fatigue_threshold: float = 0.70
    minimum_balance_minutes_if_unstable: int = 10


@dataclass(frozen=True)
class SimulationConfig:
    horizon_weeks: int = 6
    replications: int = 500


@dataclass(frozen=True)
class OptimizationConfig:
    weight_recovery: float = 1.0
    weight_safety: float = 1.35
    weight_fatigue: float = 0.70
    weight_burden: float = 0.25


@dataclass(frozen=True)
class Settings:
    app_name: str = "REHAB AI"
    environment: str = "development"
    database_path: str = "data/rehab_ai.db"
    random_seed: int = 42
    clinical: ClinicalConfig = ClinicalConfig()
    simulation: SimulationConfig = SimulationConfig()
    optimization: OptimizationConfig = OptimizationConfig()


def load_settings(path: str | None = None) -> Settings:
    config_path = Path(path or os.getenv("REHAB_AI_CONFIG", "config/default.yaml"))
    if not config_path.exists():
        return Settings()
    payload = yaml.safe_load(config_path.read_text()) or {}
    app = payload.get("app", {})
    return Settings(
        app_name=app.get("name", "REHAB AI"),
        environment=app.get("environment", "development"),
        database_path=app.get("database_path", "data/rehab_ai.db"),
        random_seed=int(app.get("random_seed", 42)),
        clinical=ClinicalConfig(**payload.get("clinical", {})),
        simulation=SimulationConfig(**payload.get("simulation", {})),
        optimization=OptimizationConfig(**payload.get("optimization", {})),
    )
