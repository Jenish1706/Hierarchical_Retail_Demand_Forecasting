"""
Baseline forecasters.

Why baselines are not optional
------------------------------
RMSSE is defined relative to a naive forecast, so a model that cannot beat
the naive benchmark has an RMSSE above 1 and is, literally, worse than doing
nothing. Reporting LightGBM's WAPE without that reference point tells the
reader nothing about whether the machine learning earned its complexity.

M5 specifically punishes skipping this step. A large fraction of the 30,490
series are low-volume and intermittent, and on those a seasonal naive is a
genuinely strong competitor. The segment router (see `segment_router.py`)
exists precisely because that turns out to be true for part of the catalogue.

All forecasters here share one interface so the trainer, backtest and router
can treat them interchangeably:

    model.fit(y)             -> self      (y = 1-D history, chronological)
    model.predict(h)         -> ndarray   (h-step-ahead point forecast)

They operate per-series on raw history, unlike the global tree models which
consume the engineered feature matrix. That difference is intentional: these
are reference points, not candidates for the feature pipeline.

References
----------
Hyndman & Athanasopoulos (2021), "Forecasting: Principles and Practice", 3rd
    ed., ch. 5.2 -- simple forecasting methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# M5 covers 5.25 years of daily data with a dominant day-of-week cycle.
# A 364-day lag (52 x 7) is used rather than 365 because it preserves the
# weekday alignment: 365 days earlier is a *different* weekday, which for
# retail demand is the stronger of the two seasonalities to match.
WEEKLY_SEASONALITY = 7
ANNUAL_SEASONALITY_ALIGNED = 364


class BaselineForecaster(ABC):
    """Common interface for all per-series baselines."""

    name: str = "baseline"

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._fitted: bool = False

    @abstractmethod
    def _forecast(self, h: int) -> np.ndarray:
        """Produce h future values. Called only after `fit`."""

    def fit(self, y: np.ndarray) -> "BaselineForecaster":
        """Store the chronologically ordered history for this series."""
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError(f"{self.name}: cannot fit on an empty series")
        self._y = y
        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        h-step-ahead forecast, clipped at zero.

        Clipping is safe for every method here: unit demand cannot be
        negative, so a negative point forecast is never actionable.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        return np.clip(self._forecast(h), 0.0, None)

    def fit_predict(self, y: np.ndarray, h: int) -> np.ndarray:
        return self.fit(y).predict(h)


class NaiveForecaster(BaselineForecaster):
    """
    Random walk: every future value equals the last observed value.

    This is the benchmark RMSSE scales against (with seasonality=1), so it is
    the single most important row in any accuracy table.
    """

    name = "naive"

    def _forecast(self, h: int) -> np.ndarray:
        return np.full(h, self._y[-1], dtype=float)


class SeasonalNaiveForecaster(BaselineForecaster):
    """
    Repeat the value observed one full season ago.

        yhat_{T+k} = y_{T + k - m * ceil(k / m)}

    Default m = 364 (52 weeks), which keeps the weekday aligned. For daily
    retail demand this is a deceptively strong baseline: it captures both the
    weekly shape and the annual level in one lag with zero parameters.

    Falls back to a shorter weekly season, then to the series mean, when
    history is too short -- rather than raising. Roughly 5% of M5 items are
    introduced mid-panel and genuinely lack a year of history.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = ANNUAL_SEASONALITY_ALIGNED) -> None:
        super().__init__()
        self.season_length = season_length

    def _forecast(self, h: int) -> np.ndarray:
        y = self._y
        m = self.season_length

        if y.size < m:
            # Not enough history for the requested season. Degrade to the
            # weekly cycle, which needs only 7 observations.
            m = WEEKLY_SEASONALITY
        if y.size < m:
            return np.full(h, float(y.mean()))

        season = y[-m:]
        # np.resize tiles the seasonal block to the requested horizon, which
        # is exactly the repeating pattern the method specifies.
        return np.resize(season, h).astype(float)


class MovingAverageForecaster(BaselineForecaster):
    """
    Flat forecast equal to the mean of the last `window` observations.

    Window defaults to 28 to match the forecast horizon: averaging over the
    same span being predicted is the natural pairing, and 28 days covers four
    complete weekly cycles so the day-of-week effect averages out instead of
    biasing the level.
    """

    name = "moving_average"

    def __init__(self, window: int = 28) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window

    def _forecast(self, h: int) -> np.ndarray:
        w = min(self.window, self._y.size)
        return np.full(h, float(self._y[-w:].mean()))


class SimpleExponentialSmoothing(BaselineForecaster):
    """
    Simple exponential smoothing (SES) with an optionally fitted alpha.

        l_t = alpha * y_t + (1 - alpha) * l_{t-1}
        yhat_{T+k} = l_T     for all k

    SES has no trend or seasonal term, so its forecast function is flat. That
    is appropriate here: it is a baseline for the *level*, and the seasonal
    naive already covers the seasonal reference.

    When `alpha is None`, alpha is chosen by minimising in-sample SSE over a
    coarse grid. A grid search rather than an optimiser is deliberate -- the
    objective is one-dimensional and bounded, the surface is often flat near
    the optimum, and a fixed grid keeps the baseline exactly reproducible.
    """

    name = "ses"

    def __init__(self, alpha: float | None = None) -> None:
        super().__init__()
        if alpha is not None and not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.fitted_alpha_: float | None = None

    @staticmethod
    def _smooth(y: np.ndarray, alpha: float) -> float:
        """Return the final smoothed level l_T."""
        level = y[0]
        for value in y[1:]:
            level = alpha * value + (1.0 - alpha) * level
        return float(level)

    @staticmethod
    def _sse(y: np.ndarray, alpha: float) -> float:
        """In-sample sum of squared one-step-ahead errors."""
        level = y[0]
        sse = 0.0
        for value in y[1:]:
            sse += (value - level) ** 2      # forecast for t is l_{t-1}
            level = alpha * value + (1.0 - alpha) * level
        return sse

    def fit(self, y: np.ndarray) -> "SimpleExponentialSmoothing":
        super().fit(y)

        if self.alpha is not None:
            self.fitted_alpha_ = self.alpha
        else:
            grid = np.arange(0.05, 1.0001, 0.05)
            sses = [self._sse(self._y, a) for a in grid]
            self.fitted_alpha_ = float(grid[int(np.argmin(sses))])

        self._level_ = self._smooth(self._y, self.fitted_alpha_)
        return self

    def _forecast(self, h: int) -> np.ndarray:
        return np.full(h, self._level_, dtype=float)


class SeasonalNaiveWeeklyForecaster(SeasonalNaiveForecaster):
    """
    Seasonal naive on a 7-day cycle.

    Separate from the 364-day variant because on M5 the two behave very
    differently: the weekly version tracks recent level shifts (promotions,
    trend) while the annual version tracks calendar position. Reporting both
    tells you which of the two signals a learned model is actually adding to.
    """

    name = "seasonal_naive_weekly"

    def __init__(self) -> None:
        super().__init__(season_length=WEEKLY_SEASONALITY)


# Registry consumed by the trainer and the backtest harness. Keys are the
# model names that appear in accuracy tables and in `ml.predictions.model_name`.
BASELINE_REGISTRY: dict[str, type[BaselineForecaster]] = {
    NaiveForecaster.name: NaiveForecaster,
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    SeasonalNaiveWeeklyForecaster.name: SeasonalNaiveWeeklyForecaster,
    MovingAverageForecaster.name: MovingAverageForecaster,
    SimpleExponentialSmoothing.name: SimpleExponentialSmoothing,
}


def build_baseline(name: str, **kwargs) -> BaselineForecaster:
    """Instantiate a baseline by registry name."""
    if name not in BASELINE_REGISTRY:
        raise KeyError(
            f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[name](**kwargs)
