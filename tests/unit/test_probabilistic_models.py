"""Behavioural tests for quantile, conformal, and distributional components."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.models.probabilistic.distributions import fit_count_distributions
from src.models.probabilistic.quantile_forecaster import (
    ConformalizedQuantileRegressor,
    QuantileForecaster,
    SplitConformalPredictor,
)


def _toy_regression() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(21)
    X = pd.DataFrame({"trend": np.arange(80), "weekday": np.arange(80) % 7})
    y = np.clip(2.0 + 0.1 * X["trend"].to_numpy() + rng.normal(0, 1, len(X)), 0, None)
    return X, y


class TestQuantileForecasters:
    def test_lightgbm_quantiles_are_rearranged_and_nonnegative(self):
        X, y = _toy_regression()
        model = QuantileForecaster(n_estimators=8, n_jobs=1).fit(X.iloc[:60], y[:60])
        predictions = model.predict(X.iloc[60:])
        stacked = np.column_stack([predictions[q] for q in sorted(predictions)])

        assert set(predictions) == {0.1, 0.25, 0.5, 0.75, 0.9}
        assert (stacked >= 0).all()
        assert (np.diff(stacked, axis=1) >= 0).all()
        assert 0.0 <= float(model.last_crossing_rate_) <= 1.0

    def test_split_conformal_uses_a_chronological_calibration_block(self):
        X, y = _toy_regression()
        predictor = SplitConformalPredictor(DummyRegressor(strategy="mean"), alpha=0.1)
        predictor.fit(X.iloc[:40], y[:40], X.iloc[40:60], y[40:60])
        point, lower, upper = predictor.predict_interval(X.iloc[60:])

        assert len(point) == len(lower) == len(upper) == 20
        assert (lower <= point).all()
        assert (point <= upper).all()

    def test_cqr_never_returns_crossed_interval_bounds(self):
        X, y = _toy_regression()
        quantiles = QuantileForecaster(n_estimators=6, n_jobs=1)
        cqr = ConformalizedQuantileRegressor(quantiles).fit(
            X.iloc[:40], y[:40], X.iloc[40:60], y[40:60]
        )
        _, lower, upper = cqr.predict_interval(X.iloc[60:])
        assert (lower <= upper).all()


class TestDistributionScreen:
    def test_distribution_screen_compares_eligible_likelihood_models(self):
        rng = np.random.default_rng(22)
        demand = rng.negative_binomial(2, 0.25, size=500)
        comparison = fit_count_distributions(demand)

        assert comparison.best_aic_distribution in {
            "poisson", "negative_binomial", "zero_inflated_poisson", "zero_inflated_negative_binomial"
        }
        assert {"poisson", "negative_binomial", "tweedie"} <= set(comparison.table["distribution"])
        tweedie = comparison.table.loc[comparison.table["distribution"] == "tweedie"].iloc[0]
        assert not tweedie["selection_eligible"]
