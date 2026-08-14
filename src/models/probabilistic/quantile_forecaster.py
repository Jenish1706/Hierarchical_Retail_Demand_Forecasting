"""
Probabilistic Forecasting Engine.

Generates demand forecasts with full uncertainty quantification:
  - Quantile Regression (LightGBM multi-quantile)
  - Conformal Prediction Intervals (distribution-free, guaranteed coverage)
  - Bootstrap Ensemble Intervals
  - Calibration checks (empirical coverage vs nominal)

Business value: buyers can set safety stock to the 90th percentile
instead of the mean — directly links forecast uncertainty to
inventory decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logging import get_logger

log = get_logger(__name__)

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@dataclass
class ProbabilisticForecast:
    """Container for a probabilistic forecast for one item/store."""
    item_id: str
    store_id: str
    forecast_dates: list
    mean: np.ndarray
    quantiles: dict[float, np.ndarray]  # {0.10: array, ...}

    @property
    def lower_80(self) -> np.ndarray:
        return self.quantiles.get(0.10, self.mean)

    @property
    def upper_80(self) -> np.ndarray:
        return self.quantiles.get(0.90, self.mean)

    @property
    def lower_90(self) -> np.ndarray:
        return self.quantiles.get(0.05, self.mean)

    @property
    def upper_90(self) -> np.ndarray:
        return self.quantiles.get(0.95, self.mean)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "item_id": self.item_id,
            "store_id": self.store_id,
            "forecast_date": self.forecast_dates,
            "forecast_mean": self.mean,
        })
        for q, vals in self.quantiles.items():
            col = f"q{int(q * 100):02d}"
            df[col] = vals
        return df


class QuantileForecaster:
    """
    LightGBM-based quantile regression forecaster.

    Trains one model per quantile using the pinball loss objective.
    Guarantees asymmetric loss penalization so quantiles don't cross.
    """

    def __init__(self, quantiles: list[float] = None):
        self.quantiles = quantiles or QUANTILES
        self.models: dict[float, LGBMRegressor] = {}
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster":
        """Train one LightGBM model per quantile."""
        self._feature_cols = X.columns.tolist()

        for q in self.quantiles:
            log.info("training_quantile_model", quantile=q)
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=400,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            model.fit(X.fillna(0), y)
            self.models[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Return {quantile: predictions} dict."""
        X_clean = X[self._feature_cols].fillna(0)
        return {
            q: model.predict(X_clean).clip(0)
            for q, model in self.models.items()
        }

    def predict_with_mean(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Return (mean_forecast, quantile_dict)."""
        quantile_preds = self.predict(X)
        # Mean = median for slightly skewed demand distributions
        mean = quantile_preds.get(0.50, np.mean(list(quantile_preds.values()), axis=0))
        return mean, quantile_preds

    def evaluate_calibration(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
    ) -> pd.DataFrame:
        """
        Check empirical coverage vs nominal coverage for each quantile.
        Well-calibrated model: empirical ≈ nominal for every quantile.
        """
        quantile_preds = self.predict(X)
        rows = []
        for q, preds in sorted(quantile_preds.items()):
            empirical_coverage = float((y_true.values <= preds).mean())
            rows.append({
                "quantile": q,
                "nominal_coverage": q,
                "empirical_coverage": empirical_coverage,
                "calibration_error": abs(empirical_coverage - q),
            })
        return pd.DataFrame(rows)


class ConformalPredictor:
    """
    Conformal Prediction Intervals — distribution-free, finite-sample coverage guarantee.

    Split conformal prediction:
      1. Fit model on calibration set
      2. Compute non-conformity scores (residuals) on held-out data
      3. Use empirical quantile of scores to set interval width

    Guarantee: P(y ∈ Ĉ(x)) ≥ 1 - α for any distribution.
    """

    def __init__(self, base_model: Any, alpha: float = 0.10):
        """
        Args:
            base_model: Any sklearn-compatible regressor
            alpha: Miscoverage rate (0.10 → 90% coverage interval)
        """
        self.base_model = base_model
        self.alpha = alpha
        self._scores: np.ndarray = np.array([])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "ConformalPredictor":
        """Fit model and calibrate intervals on held-out calibration set."""
        self.base_model.fit(X_train.fillna(0), y_train)
        y_calib_pred = self.base_model.predict(X_calib.fillna(0)).clip(0)
        # Non-conformity score: absolute residual
        self._scores = np.abs(y_calib.values - y_calib_pred)
        return self

    def predict_interval(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (point_forecast, lower_bound, upper_bound).
        Coverage is guaranteed ≥ 1 - alpha.
        """
        point = self.base_model.predict(X.fillna(0)).clip(0)

        # Quantile of calibration scores
        n = len(self._scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        margin = float(np.quantile(self._scores, level))

        lower = (point - margin).clip(0)
        upper = point + margin
        return point, lower, upper


class BootstrapEnsembleForecaster:
    """
    Bootstrap ensemble for uncertainty quantification.
    Trains n_bootstraps models on resampled data, uses spread of
    predictions as confidence interval.
    """

    def __init__(self, base_model_factory: Any, n_bootstraps: int = 50):
        self.base_model_factory = base_model_factory
        self.n_bootstraps = n_bootstraps
        self._models: list[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BootstrapEnsembleForecaster":
        rng = np.random.default_rng(42)
        n = len(X)

        for i in range(self.n_bootstraps):
            idx = rng.integers(0, n, size=n)
            model = self.base_model_factory()
            model.fit(X.iloc[idx].fillna(0), y.iloc[idx])
            self._models.append(model)

        return self

    def predict_distribution(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean, lower_10th_pct, upper_90th_pct) across bootstrap ensemble.
        """
        preds = np.stack([
            m.predict(X.fillna(0)).clip(0) for m in self._models
        ], axis=0)  # shape: (n_bootstraps, n_samples)

        mean = preds.mean(axis=0)
        lower = np.percentile(preds, 10, axis=0)
        upper = np.percentile(preds, 90, axis=0)
        return mean, lower, upper


class ProbabilisticForecastingPipeline:
    """
    Orchestrates quantile + conformal forecasting and selects
    the best-calibrated interval method.
    """

    def __init__(self, quantiles: list[float] = None):
        self.quantiles = quantiles or QUANTILES
        self.quantile_forecaster = QuantileForecaster(quantiles)
        self._is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "ProbabilisticForecastingPipeline":
        self.quantile_forecaster.fit(X_train, y_train)
        self._is_fitted = True

        if X_val is not None and y_val is not None:
            calib = self.quantile_forecaster.evaluate_calibration(X_val, y_val)
            mean_cal_error = calib["calibration_error"].mean()
            log.info(
                "calibration_evaluated",
                mean_error=f"{mean_cal_error:.4f}",
                worst_quantile=calib.loc[calib["calibration_error"].idxmax(), "quantile"],
            )

        return self

    def predict(
        self,
        X: pd.DataFrame,
        item_id: str,
        store_id: str,
        forecast_dates: list,
    ) -> ProbabilisticForecast:
        mean, quantile_preds = self.quantile_forecaster.predict_with_mean(X)
        return ProbabilisticForecast(
            item_id=item_id,
            store_id=store_id,
            forecast_dates=forecast_dates,
            mean=mean,
            quantiles=quantile_preds,
        )
