"""Quantile and conformal forecasting with explicit time-series caveats.

The module separates three distinct ideas that are often conflated:

* independently fitted quantiles describe conditional demand levels;
* monotonic rearrangement makes those levels a valid predictive distribution;
* conformal intervals calibrate a point or quantile model on a held-out,
  chronological calibration window.

All interval results must be evaluated with pinball/CRPS *and* paired
coverage-sharpness metrics from :mod:`src.evaluation.probabilistic_metrics`.
Conformal guarantees assume exchangeable calibration and test observations;
time series violate that assumption. EnbPI is included because it is designed
for sequential dependence, but it is not presented as a magic finite-sample
guarantee for M5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.evaluation.probabilistic_metrics import enforce_monotonic_quantiles

DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
# Backwards-compatible public name used in early notebooks.
QUANTILES = list(DEFAULT_QUANTILES)


def _validate_quantiles(quantiles: Sequence[float] | None) -> tuple[float, ...]:
    levels = tuple(sorted(float(q) for q in (quantiles or DEFAULT_QUANTILES)))
    if not levels or len(set(levels)) != len(levels) or any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError("quantiles must be distinct values strictly between 0 and 1")
    return levels


def _feature_frame(X: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return finite numeric features in a stable fitted-column order."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named feature columns")
    out = X.copy() if columns is None else X.reindex(columns=list(columns))
    if out.isna().all(axis=None):
        raise ValueError("all feature values are missing")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal order statistic, conservatively rounded up."""
    values = np.sort(np.asarray(scores, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("calibration scores contain no finite values")
    rank = min(int(np.ceil((values.size + 1) * (1.0 - alpha))), values.size)
    return float(values[rank - 1])


@dataclass
class ProbabilisticForecast:
    """Forecast distribution for one series, including rearrangement diagnostics."""

    item_id: str
    store_id: str
    forecast_dates: list[Any]
    mean: np.ndarray
    quantiles: dict[float, np.ndarray]
    crossing_rate_before: float = 0.0

    @property
    def lower_80(self) -> np.ndarray:
        return self.quantiles.get(0.10, self.mean)

    @property
    def upper_80(self) -> np.ndarray:
        return self.quantiles.get(0.90, self.mean)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "item_id": self.item_id,
                "store_id": self.store_id,
                "forecast_date": self.forecast_dates,
                "forecast_point": self.mean,
            }
        )
        for q, values in sorted(self.quantiles.items()):
            df[f"q{q:g}"] = values
        df["crossing_rate_before"] = self.crossing_rate_before
        return df


class QuantileForecaster:
    """LightGBM quantile regression with mandatory monotonic rearrangement.

    Each quantile is fitted under LightGBM's pinball objective. Independent
    fits can cross, so :meth:`predict` sorts the predicted levels per row using
    Chernozhukov, Fernandez-Val & Galichon (2010). The raw crossing rate is
    retained for the results report rather than hidden by the repair.
    """

    def __init__(
        self,
        quantiles: Sequence[float] | None = None,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        self.quantiles = _validate_quantiles(quantiles)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.models: dict[float, LGBMRegressor] = {}
        self.feature_names_: list[str] = []
        self.last_crossing_rate_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> QuantileForecaster:
        X_clean = _feature_frame(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(X_clean) != len(target):
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(target).all():
            raise ValueError("y must contain finite unit counts")

        self.feature_names_ = X_clean.columns.tolist()
        self.models.clear()
        for q in self.quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=7,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                # Keep process-level parallelism explicit. The project uses
                # n_jobs=1 by default to remain reproducible on Windows.
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=-1,
            )
            model.fit(X_clean, target)
            self.models[q] = model
        return self

    def predict_raw(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Return independently fitted levels before the crossing repair."""
        if not self.models:
            raise RuntimeError("call fit before predict")
        X_clean = _feature_frame(X, self.feature_names_)
        return {
            q: np.clip(model.predict(X_clean), 0.0, None)
            for q, model in sorted(self.models.items())
        }

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        raw = self.predict_raw(X)
        repaired, crossing_rate = enforce_monotonic_quantiles(raw)
        self.last_crossing_rate_ = crossing_rate
        return repaired

    def predict_with_point(self, X: pd.DataFrame) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Return the median when available, otherwise the central quantile average."""
        quantile_preds = self.predict(X)
        point = quantile_preds.get(0.50)
        if point is None:
            point = np.mean(np.column_stack(list(quantile_preds.values())), axis=1)
        return np.asarray(point, dtype=float), quantile_preds

    # Kept for notebooks written before ``point`` replaced the misleading
    # ``mean`` label: a quantile median is not generally a distributional mean.
    predict_with_mean = predict_with_point

    def calibration_curve(self, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> pd.DataFrame:
        predictions = self.predict(X)
        actual = np.asarray(y_true, dtype=float).reshape(-1)
        if len(actual) != len(next(iter(predictions.values()))):
            raise ValueError("X and y_true must have the same number of rows")
        return pd.DataFrame(
            {
                "quantile": list(predictions),
                "nominal_coverage": list(predictions),
                "empirical_coverage": [
                    float(np.mean(actual <= pred)) for pred in predictions.values()
                ],
            }
        ).assign(
            calibration_error=lambda frame: (
                frame["empirical_coverage"] - frame["nominal_coverage"]
            ).abs()
        )

    evaluate_calibration = calibration_curve


class LinearQuantileForecaster:
    """Statsmodels QuantReg baseline using the same levels and rearrangement."""

    def __init__(self, quantiles: Sequence[float] | None = None, max_iter: int = 5_000) -> None:
        self.quantiles = _validate_quantiles(quantiles)
        self.max_iter = int(max_iter)
        self.models: dict[float, Any] = {}
        self.feature_names_: list[str] = []
        self.last_crossing_rate_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> LinearQuantileForecaster:
        import statsmodels.api as sm

        X_clean = _feature_frame(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(X_clean) != len(target):
            raise ValueError("X and y must have the same number of rows")
        self.feature_names_ = X_clean.columns.tolist()
        design = sm.add_constant(X_clean, has_constant="add")
        self.models = {
            q: sm.QuantReg(target, design).fit(q=q, max_iter=self.max_iter) for q in self.quantiles
        }
        return self

    def predict_raw(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        if not self.models:
            raise RuntimeError("call fit before predict")
        import statsmodels.api as sm

        design = sm.add_constant(_feature_frame(X, self.feature_names_), has_constant="add")
        return {
            q: np.clip(np.asarray(model.predict(design), dtype=float), 0.0, None)
            for q, model in sorted(self.models.items())
        }

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        repaired, crossing_rate = enforce_monotonic_quantiles(self.predict_raw(X))
        self.last_crossing_rate_ = crossing_rate
        return repaired


class SplitConformalPredictor:
    """Chronological split conformal intervals around any sklearn-like regressor.

    The usual finite-sample coverage statement relies on exchangeability. M5
    observations are serially and hierarchically dependent, so report observed
    coverage and sharpness rather than claiming that statement holds exactly.
    """

    def __init__(self, base_model: Any, alpha: float = 0.10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.base_model = base_model
        self.alpha = float(alpha)
        self.margin_: float | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_calib: pd.DataFrame,
        y_calib: pd.Series | np.ndarray,
    ) -> SplitConformalPredictor:
        self.base_model.fit(_feature_frame(X_train), np.asarray(y_train, dtype=float))
        calib_pred = np.clip(self.base_model.predict(_feature_frame(X_calib)), 0.0, None)
        scores = np.abs(np.asarray(y_calib, dtype=float) - calib_pred)
        self.margin_ = _conformal_margin(scores, self.alpha)
        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.margin_ is None:
            raise RuntimeError("call fit before predict_interval")
        point = np.clip(
            np.asarray(self.base_model.predict(_feature_frame(X)), dtype=float), 0.0, None
        )
        return point, np.clip(point - self.margin_, 0.0, None), point + self.margin_


# Compatibility for early notebooks. The name now describes the method more
# precisely, while preserving imports written before the refactor.
ConformalPredictor = SplitConformalPredictor


class ConformalizedQuantileRegressor:
    """CQR: calibrate a lower/upper quantile pair on a chronological holdout."""

    def __init__(
        self,
        quantile_model: QuantileForecaster | LinearQuantileForecaster,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
    ) -> None:
        if not 0.0 < lower_quantile < upper_quantile < 1.0:
            raise ValueError("require 0 < lower_quantile < upper_quantile < 1")
        self.quantile_model = quantile_model
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.margin_: float | None = None

    @property
    def alpha(self) -> float:
        return self.lower_quantile + (1.0 - self.upper_quantile)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_calib: pd.DataFrame,
        y_calib: pd.Series | np.ndarray,
    ) -> ConformalizedQuantileRegressor:
        if (
            self.lower_quantile not in self.quantile_model.quantiles
            or self.upper_quantile not in self.quantile_model.quantiles
        ):
            raise ValueError("quantile model must include CQR lower and upper quantiles")
        self.quantile_model.fit(X_train, y_train)
        pred = self.quantile_model.predict(X_calib)
        actual = np.asarray(y_calib, dtype=float)
        # Romano, Patterson & Candes (2019): a point outside either fitted
        # bound receives exactly the excess needed to make the interval cover it.
        scores = np.maximum(
            0.0,
            np.maximum(pred[self.lower_quantile] - actual, actual - pred[self.upper_quantile]),
        )
        self.margin_ = _conformal_margin(scores, self.alpha)
        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.margin_ is None:
            raise RuntimeError("call fit before predict_interval")
        predictions = self.quantile_model.predict(X)
        point = predictions.get(
            0.50, (predictions[self.lower_quantile] + predictions[self.upper_quantile]) / 2
        )
        lower = np.clip(predictions[self.lower_quantile] - self.margin_, 0.0, None)
        upper = predictions[self.upper_quantile] + self.margin_
        return np.asarray(point), lower, upper


class EnbPIPredictor:
    """MAPIE EnbPI adapter for sequentially dependent regression residuals."""

    def __init__(
        self, base_model: Any, alpha: float = 0.10, cv: int = 5, random_state: int = 42
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.base_model = base_model
        self.alpha = float(alpha)
        self.cv = int(cv)
        self.random_state = int(random_state)
        self.model_: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> EnbPIPredictor:
        try:
            from mapie.regression import TimeSeriesRegressor
        except ImportError as exc:  # pragma: no cover - declared project dependency
            raise ImportError("install mapie to use EnbPIPredictor") from exc

        self.model_ = TimeSeriesRegressor(
            estimator=self.base_model,
            method="enbpi",
            cv=self.cv,
            n_jobs=1,
            random_state=self.random_state,
        )
        self.model_.fit(_feature_frame(X), np.asarray(y, dtype=float))
        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("call fit before predict_interval")
        point, intervals = self.model_.predict(_feature_frame(X), confidence_level=1.0 - self.alpha)
        bounds = np.asarray(intervals, dtype=float)
        # MAPIE returns (n, 2, n_confidence_levels); normalize the one-level
        # output while accepting earlier (n, 2) releases for reproducibility.
        if bounds.ndim == 3:
            bounds = bounds[:, :, 0]
        if bounds.shape != (len(point), 2):
            raise RuntimeError(f"unexpected MAPIE interval shape {bounds.shape}")
        return (
            np.clip(np.asarray(point, dtype=float), 0.0, None),
            np.clip(bounds[:, 0], 0.0, None),
            np.clip(bounds[:, 1], 0.0, None),
        )


class BootstrapEnsembleForecaster:
    """Bootstrap base-model ensemble; prediction spread is an empirical interval."""

    def __init__(
        self, base_model_factory: Callable[[], Any], n_bootstraps: int = 50, random_state: int = 42
    ) -> None:
        if n_bootstraps < 2:
            raise ValueError("n_bootstraps must be at least 2")
        self.base_model_factory = base_model_factory
        self.n_bootstraps = int(n_bootstraps)
        self.random_state = int(random_state)
        self.models_: list[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BootstrapEnsembleForecaster:
        X_clean = _feature_frame(X).reset_index(drop=True)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(X_clean) != len(target):
            raise ValueError("X and y must have the same number of rows")
        rng = np.random.default_rng(self.random_state)
        self.models_.clear()
        for _ in range(self.n_bootstraps):
            indices = rng.integers(0, len(X_clean), size=len(X_clean))
            model = self.base_model_factory()
            model.fit(X_clean.iloc[indices], target[indices])
            self.models_.append(model)
        return self

    def predict_distribution(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.models_:
            raise RuntimeError("call fit before predict_distribution")
        X_clean = _feature_frame(X)
        samples = np.stack(
            [
                np.clip(np.asarray(model.predict(X_clean), dtype=float), 0.0, None)
                for model in self.models_
            ]
        )
        return (
            samples.mean(axis=0),
            np.quantile(samples, 0.10, axis=0),
            np.quantile(samples, 0.90, axis=0),
        )


class ProbabilisticForecastingPipeline:
    """LightGBM quantiles plus a reportable crossing and calibration diagnostic."""

    def __init__(self, quantiles: Sequence[float] | None = None, **forecaster_kwargs: Any) -> None:
        self.quantile_forecaster = QuantileForecaster(quantiles, **forecaster_kwargs)
        self.calibration_report_: pd.DataFrame | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> ProbabilisticForecastingPipeline:
        self.quantile_forecaster.fit(X_train, y_train)
        if X_val is not None and y_val is not None:
            self.calibration_report_ = self.quantile_forecaster.calibration_curve(X_val, y_val)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        item_id: str,
        store_id: str,
        forecast_dates: Sequence[Any],
    ) -> ProbabilisticForecast:
        point, quantiles = self.quantile_forecaster.predict_with_point(X)
        if len(point) != len(forecast_dates):
            raise ValueError("forecast_dates must have one value per feature row")
        return ProbabilisticForecast(
            item_id=item_id,
            store_id=store_id,
            forecast_dates=list(forecast_dates),
            mean=point,
            quantiles=quantiles,
            crossing_rate_before=float(self.quantile_forecaster.last_crossing_rate_ or 0.0),
        )
