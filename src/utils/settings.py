"""
Core application settings using Pydantic BaseSettings.
All configuration is environment-driven for 12-factor app compliance.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="supply_chain")
    user: str = Field(default="scplatform")
    password: str = Field(default="changeme")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=10)
    pool_timeout: int = Field(default=30)
    echo: bool = Field(default=False)

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class MLflowSettings(BaseSettings):
    """MLflow tracking server configuration."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_prefix: str = Field(default="supply_chain")
    artifact_root: str = Field(default="s3://mlflow-artifacts")
    registry_uri: str = Field(default="")
    s3_endpoint_url: str = Field(default="")


class AirflowSettings(BaseSettings):
    """Airflow configuration."""

    model_config = SettingsConfigDict(env_prefix="AIRFLOW_")

    host: str = Field(default="localhost")
    port: int = Field(default=8080)
    username: str = Field(default="airflow")
    password: str = Field(default="airflow")


class ModelSettings(BaseSettings):
    """ML model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    forecast_horizon: int = Field(default=28, description="Days to forecast ahead")
    min_train_days: int = Field(default=365, description="Minimum training data days")
    cv_splits: int = Field(default=5)
    validation_window_days: int = Field(default=28)
    feature_version: int = Field(default=1)

    # Performance thresholds for retraining triggers
    mape_threshold: float = Field(default=0.20, description="Trigger retraining if MAPE > threshold")
    wape_threshold: float = Field(default=0.15)
    bias_threshold: float = Field(default=0.10)

    # Quantile levels for probabilistic forecasts
    quantiles: list[float] = Field(default=[0.10, 0.25, 0.50, 0.75, 0.90])

    # Hierarchical forecasting
    hierarchy_levels: list[str] = Field(
        default=["item_store", "dept_store", "cat_store", "item_state", "cat_state", "total"]
    )


class InventorySettings(BaseSettings):
    """Inventory optimization configuration."""

    model_config = SettingsConfigDict(env_prefix="INVENTORY_")

    default_service_level: float = Field(default=0.95)
    default_lead_time_days: int = Field(default=7)
    default_ordering_cost: float = Field(default=50.0)
    holding_cost_rate: float = Field(default=0.25, description="Annual holding cost as % of unit cost")
    stockout_cost_multiplier: float = Field(default=3.0, description="Multiplier on unit cost for stockouts")
    review_period_days: int = Field(default=7)


class MonitoringSettings(BaseSettings):
    """Monitoring and alerting configuration."""

    model_config = SettingsConfigDict(env_prefix="MONITOR_")

    drift_significance_level: float = Field(default=0.05)
    drift_check_frequency: str = Field(default="daily")
    performance_check_frequency: str = Field(default="daily")
    alert_email: str = Field(default="alerts@scplatform.io")
    slack_webhook: str = Field(default="")
    prometheus_port: int = Field(default=8001)


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = Field(default="Supply Chain Intelligence Platform")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")  # development, staging, production
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Paths
    base_dir: Path = Field(default=Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default=Path("data"))
    model_dir: Path = Field(default=Path("models"))
    logs_dir: Path = Field(default=Path("logs"))

    # Security
    api_secret_key: str = Field(default="dev-secret-key-change-in-production")
    api_key_header: str = Field(default="X-API-Key")

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    airflow: AirflowSettings = Field(default_factory=AirflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    inventory: InventorySettings = Field(default_factory=InventorySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached settings instance — import this everywhere."""
    return AppSettings()


# Convenience alias
settings = get_settings()
