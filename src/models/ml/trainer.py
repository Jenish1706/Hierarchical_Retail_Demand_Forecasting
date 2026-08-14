"""
Demand Forecasting ML Trainer.

Implements a complete model training pipeline:
  1. Baseline models (Moving Average, Linear Regression)
  2. Tree-based ML models (RF, XGBoost, LightGBM, CatBoost)
  3. Time-series cross-validation
  4. Optuna hyperparameter tuning
  5. MLflow experiment tracking
  6. Model selection and registration

Design decisions:
  - TimeSeriesSplit for proper temporal validation (no data leakage)
  - Optuna for efficient hyperparameter search (TPE sampler)
  - MLflow for full reproducibility and artifact storage
  - Champion/Challenger model comparison before promotion
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger
from src.utils.settings import settings

warnings.filterwarnings("ignore")
log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all forecasting metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)  # Demand is non-negative

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE: skip zeros in denominator
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100 if mask.sum() > 0 else np.nan

    # WAPE: Weighted Absolute Percentage Error
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100 if y_true.sum() > 0 else np.nan

    # Bias
    bias = float(np.mean(y_pred - y_true))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else -1.0,
        "wape": float(wape) if not np.isnan(wape) else -1.0,
        "bias": bias,
    }


# ---------------------------------------------------------------------------
# Model Registry (definitions, not MLflow)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {
    "linear_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "ridge": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    ),
    "xgboost": lambda: XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        eval_metric="mae",
    ),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    ),
    "catboost": lambda: CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------------
# Hyperparameter Search Spaces
# ---------------------------------------------------------------------------

def get_xgboost_trial_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def get_lgbm_trial_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    target_col: str = "sales"
    forecast_horizon: int = 28
    cv_splits: int = 3
    tune_trials: int = 50
    tune_models: list[str] = None  # Which models to tune (None = skip tuning)
    feature_cols: list[str] = None
    experiment_name: str = "demand_forecasting"

    def __post_init__(self):
        if self.tune_models is None:
            self.tune_models = ["xgboost", "lightgbm"]


class DemandForecastingTrainer:
    """
    Full model training pipeline with MLflow tracking.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._log = get_logger("DemandForecastingTrainer")

        # Configure MLflow
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(
            f"{settings.mlflow.experiment_prefix}/{config.experiment_name}"
        )

    def fit(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, float]]:
        """
        Train all models, track with MLflow, return results.

        Args:
            df: Feature DataFrame with target column

        Returns:
            {model_name: {metric: value}} for all models
        """
        X, y = self._prepare_data(df)
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)

        all_results = {}

        for model_name, model_factory in MODEL_REGISTRY.items():
            self._log.info("training_model", model=model_name)

            with mlflow.start_run(run_name=model_name) as run:
                # Hyperparameter tuning for select models
                if model_name in self.config.tune_models:
                    best_params = self._tune_hyperparameters(
                        model_name, X, y, tscv
                    )
                    mlflow.log_params({"tuned": True, **best_params})
                    if model_name == "xgboost":
                        model = XGBRegressor(**best_params, n_jobs=-1, random_state=42)
                    elif model_name == "lightgbm":
                        model = LGBMRegressor(**best_params, n_jobs=-1, random_state=42, verbose=-1)
                    else:
                        model = model_factory()
                else:
                    model = model_factory()
                    mlflow.log_params({"tuned": False})

                # Cross-validation
                cv_metrics = self._cross_validate(model, X, y, tscv)
                mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})

                # Final fit on full data
                t0 = time.time()
                model.fit(X, y)
                train_time = time.time() - t0

                # Training metrics
                y_pred_train = model.predict(X).clip(0)
                train_metrics = compute_metrics(y.values, y_pred_train)
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metric("train_time_seconds", train_time)

                # Log model
                self._log_model(run, model, model_name, X.columns.tolist())

                # Feature importance
                self._log_feature_importance(model, model_name, X.columns.tolist())

                all_results[model_name] = {
                    "run_id": run.info.run_id,
                    "cv_wape": cv_metrics["wape"],
                    "cv_mae": cv_metrics["mae"],
                    "train_wape": train_metrics["wape"],
                    "train_time": train_time,
                }
                self._log.info(
                    "model_trained",
                    model=model_name,
                    cv_wape=f"{cv_metrics['wape']:.2f}%",
                    run_id=run.info.run_id,
                )

        # Select champion model
        champion = self._select_champion(all_results)
        self._log.info("champion_selected", model=champion, wape=all_results[champion]["cv_wape"])

        return all_results

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Select and prepare features and target."""
        if self.config.feature_cols:
            feature_cols = [c for c in self.config.feature_cols if c in df.columns]
        else:
            # Auto-detect: exclude ID/target/metadata columns
            exclude_cols = {
                "item_id", "store_id", "date", "dept_id", "cat_id", "state_id",
                "id", self.config.target_col, "feature_version", "holiday_name",
            }
            feature_cols = [c for c in df.columns if c not in exclude_cols
                           and df[c].dtype != "object"]

        # Drop rows where target is null
        df_clean = df[feature_cols + [self.config.target_col]].dropna(
            subset=[self.config.target_col]
        )

        X = df_clean[feature_cols].fillna(0)  # Impute lags with 0 (acceptable for tree models)
        y = df_clean[self.config.target_col].clip(lower=0)

        self._log.info(
            "data_prepared",
            rows=len(X),
            features=len(feature_cols),
        )
        return X, y

    def _cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        tscv: TimeSeriesSplit,
    ) -> dict[str, float]:
        """Time-series cross-validation with proper temporal ordering."""
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val).clip(0)
            fold_m = compute_metrics(y_val.values, y_pred)
            fold_metrics.append(fold_m)

        # Average across folds
        avg_metrics = {
            k: float(np.mean([m[k] for m in fold_metrics if m[k] >= 0]))
            for k in fold_metrics[0]
        }
        return avg_metrics

    def _tune_hyperparameters(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        tscv: TimeSeriesSplit,
    ) -> dict:
        """Optuna hyperparameter tuning with TimeSeriesSplit."""

        def objective(trial: optuna.Trial) -> float:
            if model_name == "xgboost":
                params = get_xgboost_trial_params(trial)
                model = XGBRegressor(**params, n_jobs=-1, random_state=42)
            elif model_name == "lightgbm":
                params = get_lgbm_trial_params(trial)
                model = LGBMRegressor(**params, n_jobs=-1, random_state=42, verbose=-1)
            else:
                return 0.0

            # Use last fold only for speed
            *_, (train_idx, val_idx) = tscv.split(X)
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val).clip(0)
            metrics = compute_metrics(y_val.values, y_pred)
            return metrics["wape"]

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(
            objective,
            n_trials=self.config.tune_trials,
            show_progress_bar=False,
        )

        self._log.info(
            "tuning_complete",
            model=model_name,
            best_wape=study.best_value,
            n_trials=len(study.trials),
        )
        return study.best_params

    def _log_model(
        self,
        run: mlflow.ActiveRun,
        model: Any,
        model_name: str,
        feature_names: list[str],
    ) -> None:
        """Log model artifact to MLflow with signature."""
        from mlflow.models.signature import infer_signature
        import pandas as pd

        # Create sample input for signature
        sample_input = pd.DataFrame(
            np.zeros((1, len(feature_names))),
            columns=feature_names,
        )
        signature = infer_signature(sample_input, np.array([0.0]))

        if "xgboost" in model_name:
            mlflow.xgboost.log_model(model, "model", signature=signature)
        elif "lightgbm" in model_name:
            mlflow.lightgbm.log_model(model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)

        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("feature_count", len(feature_names))

    def _log_feature_importance(
        self, model: Any, model_name: str, feature_names: list[str]
    ) -> None:
        """Log feature importance as artifact."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = pd.DataFrame({
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)

                # Log as CSV artifact
                tmp_path = f"/tmp/feature_importance_{model_name}.csv"
                importances.to_csv(tmp_path, index=False)
                mlflow.log_artifact(tmp_path, "feature_importance")
        except Exception as e:
            self._log.warning("feature_importance_failed", model=model_name, error=str(e))

    def _select_champion(
        self, results: dict[str, dict]
    ) -> str:
        """Select the champion model based on CV WAPE."""
        valid_results = {k: v for k, v in results.items() if v["cv_wape"] > 0}
        return min(valid_results, key=lambda k: valid_results[k]["cv_wape"])
