"""
Unit tests for the evaluation harness.

These are known-answer tests, not smoke tests. Every assertion checks a
mathematical property with an analytically known value, so a regression in the
metrics -- which would silently corrupt every result the project produces --
fails here rather than in a leaderboard nobody can sanity-check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.point_metrics import (
    bias,
    compute_metrics,
    mae,
    mape_nonzero,
    mase,
    metrics_by_group,
    naive_scale,
    rmse,
    rmsse,
    wape,
)
from src.evaluation.probabilistic_metrics import (
    crps_from_samples,
    empirical_coverage,
    enforce_monotonic_quantiles,
    pinball_loss,
    reliability_curve,
    winkler_score,
)

# ===========================================================================
# Point metrics
# ===========================================================================

class TestPointMetrics:
    def test_perfect_forecast_scores_zero(self):
        y = np.array([3.0, 0.0, 7.0, 2.0])
        assert wape(y, y) == pytest.approx(0.0)
        assert mae(y, y) == pytest.approx(0.0)
        assert rmse(y, y) == pytest.approx(0.0)
        assert bias(y, y) == pytest.approx(0.0)

    def test_wape_is_hand_computable(self):
        y = np.array([10.0, 20.0, 30.0])      # sum = 60
        p = np.array([12.0, 18.0, 33.0])      # abs errors 2 + 2 + 3 = 7
        assert wape(y, p) == pytest.approx(100.0 * 7 / 60)

    def test_wape_undefined_when_actuals_all_zero(self):
        # Genuinely undefined -- must be NaN, never a convenient 0 or 100.
        assert np.isnan(wape(np.zeros(5), np.ones(5)))

    def test_bias_is_signed(self):
        y = np.array([5.0, 5.0])
        assert bias(y, np.array([7.0, 7.0])) == pytest.approx(2.0)    # over
        assert bias(y, np.array([3.0, 3.0])) == pytest.approx(-2.0)   # under

    def test_bias_detects_what_wape_hides(self):
        # Two forecasts with identical WAPE but opposite direction. This is
        # the exact case bias exists to expose.
        y = np.array([10.0, 10.0, 10.0])
        over, under = np.array([12.0] * 3), np.array([8.0] * 3)
        assert wape(y, over) == pytest.approx(wape(y, under))
        assert bias(y, over) > 0 > bias(y, under)

    def test_mape_skips_zeros(self):
        y = np.array([0.0, 10.0])
        p = np.array([5.0, 11.0])
        # Only the non-zero element contributes: |10-11|/10 = 10%.
        assert mape_nonzero(y, p) == pytest.approx(10.0)


    def test_rmsse_per_series_averaging_matches_m5_rule(self):
        # M5 rule: calculate RMSSE_i = sqrt(MSE_i / scale_i) for each series,
        # then take the mean across series (unweighted).
        # Series 1: MSE = 4, scale = 4 -> RMSSE_1 = 1.0
        # Series 2: MSE = 9, scale = 1 -> RMSSE_2 = 3.0
        # Mean RMSSE = (1.0 + 3.0) / 2 = 2.0
        df = pd.DataFrame({
            "series_id": ["s1", "s1", "s2", "s2"],
            "model": ["test_model"] * 4,
            "y_true": [0.0, 0.0, 0.0, 0.0],
            "y_pred": [2.0, 2.0, 3.0, 3.0],  # s1 sq_err = 4, s2 sq_err = 9
            "scale": [4.0, 4.0, 1.0, 1.0],
        })
        res = metrics_by_group(df, ["model"], scale_col="scale")
        assert res.iloc[0]["rmsse"] == pytest.approx(2.0)

    def test_naive_scale_trims_pre_launch_zeros(self):
        # Leading zeros prior to first sale must not dilute the random-walk scale
        y_active = np.array([1.0, 3.0, 6.0])  # diffs 2, 3 -> mean sq = 6.5
        y_padded = np.concatenate([np.zeros(50), y_active])
        assert naive_scale(y_padded, seasonality=1) == pytest.approx(6.5)

    def test_rmsse_zero_for_perfect_forecast(self):
        rng = np.random.default_rng(0)
        ins = rng.poisson(4, 300).astype(float)
        y = rng.poisson(4, 28).astype(float)
        assert rmsse(y, y, ins) == pytest.approx(0.0)

    def test_rmsse_near_one_for_naive_forecast(self):
        # A random-walk forecast should score close to 1 by construction --
        # that is the entire interpretive value of RMSSE.
        rng = np.random.default_rng(1)
        ins = rng.poisson(5, 2000).astype(float)
        y = rng.poisson(5, 500).astype(float)
        naive = np.full(500, ins[-1])
        assert 0.6 < rmsse(y, naive, ins) < 1.6

    def test_rmsse_nan_when_series_constant(self):
        # Constant in-sample -> zero scale -> genuinely undefined.
        assert np.isnan(rmsse(np.ones(5), np.ones(5), np.zeros(50)))

    def test_mase_matches_definition(self):
        ins = np.array([1.0, 3.0, 6.0])        # abs diffs 2, 3 -> mean 2.5
        y = np.array([10.0, 10.0])
        p = np.array([15.0, 15.0])             # MAE 5
        assert mase(y, p, ins) == pytest.approx(5.0 / 2.5)

    def test_bundle_reports_both_scaled_metrics(self):
        rng = np.random.default_rng(2)
        ins = rng.poisson(3, 200).astype(float)
        y, p = rng.poisson(3, 28).astype(float), rng.poisson(3, 28).astype(float)
        out = compute_metrics(y, p, y_insample=ins)
        assert {"wape", "rmsse", "mase", "bias", "mae", "rmse"} <= set(out)

    def test_negative_predictions_are_clipped(self):
        # Demand is a non-negative count; a negative order quantity is never
        # actionable, so clipping is correct rather than a metric trick.
        y = np.array([5.0, 5.0])
        out = compute_metrics(y, np.array([-10.0, 5.0]))
        assert out["mae"] == pytest.approx(2.5)   # clipped to 0 -> |5-0|=5, |5-5|=0


# ===========================================================================
# Probabilistic metrics
# ===========================================================================

class TestProbabilisticMetrics:
    def test_pinball_minimised_at_true_quantile(self):
        rng = np.random.default_rng(3)
        y = rng.normal(10, 2, 40_000)
        at_median = pinball_loss(y, np.full(y.size, 10.0), 0.5)
        for wrong in (8.0, 12.0):
            assert pinball_loss(y, np.full(y.size, wrong), 0.5) > at_median

    def test_pinball_is_asymmetric(self):
        # At q=0.9 under-prediction must cost more than over-prediction.
        y = np.array([10.0])
        under = pinball_loss(y, np.array([8.0]), 0.9)
        over = pinball_loss(y, np.array([12.0]), 0.9)
        assert under > over

    def test_pinball_rejects_invalid_quantile(self):
        with pytest.raises(ValueError, match="quantile must be in"):
            pinball_loss(np.array([1.0]), np.array([1.0]), 1.5)

    def test_crps_equals_mae_for_point_forecast(self):
        # CRPS reduces to MAE when the predictive distribution is a point
        # mass. This is what makes it comparable across point and
        # probabilistic models.
        rng = np.random.default_rng(4)
        y = rng.normal(0, 1, 5_000)
        point = np.zeros((y.size, 1))
        assert crps_from_samples(y, point) == pytest.approx(
            float(np.mean(np.abs(y))), rel=1e-9
        )

    def test_coverage_of_known_interval(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        assert empirical_coverage(y, np.zeros(5), np.full(5, 10.0)) == pytest.approx(80.0)

    def test_winkler_penalises_narrow_wrong_intervals(self):
        # The central property: sharpness cannot be bought at the cost of
        # coverage. A too-narrow interval must score WORSE despite being
        # narrower.
        rng = np.random.default_rng(5)
        y = rng.normal(10, 2, 10_000)
        wide_lo, wide_hi = np.full(y.size, 10 - 3.29), np.full(y.size, 10 + 3.29)
        narrow_lo, narrow_hi = np.full(y.size, 9.5), np.full(y.size, 10.5)

        assert (narrow_hi - narrow_lo).mean() < (wide_hi - wide_lo).mean()
        assert winkler_score(y, narrow_lo, narrow_hi, 0.10) > winkler_score(
            y, wide_lo, wide_hi, 0.10
        )

    def test_reliability_curve_flat_for_calibrated_model(self):
        from scipy import stats

        rng = np.random.default_rng(6)
        y = rng.normal(0, 1, 40_000)
        preds = {q: np.full(y.size, stats.norm.ppf(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        curve = reliability_curve(y, preds)
        assert curve["calibration_error"].abs().max() < 0.02


class TestQuantileCrossing:
    def test_rearrangement_removes_all_crossings(self):
        # q90 below q50 is not a distribution; sorting per row is the
        # Chernozhukov et al. (2010) repair.
        preds = {
            0.1: np.array([5.0, 1.0]),
            0.5: np.array([3.0, 2.0]),      # row 0 crosses
            0.9: np.array([4.0, 3.0]),
        }
        fixed, rate = enforce_monotonic_quantiles(preds)
        assert rate == pytest.approx(0.5)   # one of two rows crossed
        stacked = np.column_stack([fixed[q] for q in sorted(fixed)])
        assert (np.diff(stacked, axis=1) >= 0).all()

    def test_rearrangement_is_noop_when_already_ordered(self):
        preds = {0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}
        fixed, rate = enforce_monotonic_quantiles(preds)
        assert rate == pytest.approx(0.0)
        for q in preds:
            assert fixed[q] == pytest.approx(preds[q])
