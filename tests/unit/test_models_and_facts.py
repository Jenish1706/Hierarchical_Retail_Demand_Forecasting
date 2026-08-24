"""
Unit tests for M5 facts, splits, statistical models, and significance tests.

The M5-fact tests are the project's tripwire: if an ingest or a refactor ever
changes a cardinality, these fail immediately rather than letting a corrupted
panel flow into training.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config.settings import M5
from src.evaluation.significance import (
    adjust_pvalues, diebold_mariano, friedman_test, mcb_test,
    nemenyi_critical_difference, nemenyi_test,
)
from src.evaluation.splits import (
    assert_no_temporal_leakage, canonical_splits, rolling_origins,
)
from src.models.statistical.intermittent import (
    ADIDAForecaster, CrostonForecaster, DemandPattern, classify_demand,
)
from src.models.statistical.naive import BASELINE_REGISTRY, build_baseline
from src.models.router import DEFAULT_ROUTING, LITERATURE_ROUTING


# ===========================================================================
# M5 dataset facts
# ===========================================================================

class TestM5Facts:
    def test_series_cardinality_is_consistent(self):
        assert M5.N_ITEMS * M5.N_STORES == M5.N_SERIES == 30_490

    def test_categorical_domains_match_counts(self):
        assert len(M5.STORES) == M5.N_STORES
        assert len(M5.STATES) == M5.N_STATES
        assert len(M5.CATEGORIES) == M5.N_CATEGORIES
        assert len(M5.DEPARTMENTS) == M5.N_DEPARTMENTS

    def test_sales_window_ends_where_the_data_ends(self):
        # Verified against the workbook. The project brief said 2016-06-19,
        # which is the CALENDAR end, 56 days later.
        assert M5.END_DATE == date(2016, 4, 24)
        assert M5.CALENDAR_END_DATE == date(2016, 6, 19)
        assert (M5.END_DATE - M5.START_DATE).days + 1 == M5.N_DAYS

    def test_d_index_round_trips(self):
        assert M5.d_to_date("d_1") == M5.START_DATE
        assert M5.d_to_date("d_1913") == M5.END_DATE
        assert M5.date_to_d(M5.END_DATE) == "d_1913"

    def test_specified_test_window_has_no_actuals(self):
        # The brief's test split (2016-04-25..2016-05-22) is d_1914..d_1941,
        # past the end of sales data. This asserts the discrepancy so nobody
        # silently reintroduces it.
        assert not M5.has_actuals(date(2016, 4, 25))
        assert M5.has_actuals(M5.TEST_END)

    def test_store_and_dept_prefixes_decode(self):
        assert M5.store_to_state("CA_3") == "CA"
        assert M5.dept_to_category("FOODS_3") == "FOODS"
        for store in M5.STORES:
            assert M5.store_to_state(store) in M5.STATES
        for dept in M5.DEPARTMENTS:
            assert M5.dept_to_category(dept) in M5.CATEGORIES

    def test_price_coverage_is_recorded_as_partial(self):
        # The supplied workbook was truncated at Excel's row limit.
        assert set(M5.STORES_WITH_PRICE_DATA) < set(M5.STORES)
        assert M5.has_price_data("CA_1")
        assert not M5.has_price_data("TX_1")

    def test_cardinality_validator_flags_wrong_counts(self):
        assert M5.validate_cardinalities({"n_items": M5.N_ITEMS}) == []
        assert M5.validate_cardinalities({"n_items": 3_048})


# ===========================================================================
# Splits
# ===========================================================================

class TestSplits:
    def test_canonical_splits_are_chronological_and_disjoint(self):
        splits = canonical_splits()
        val, test = splits["validation"], splits["test"]
        assert val.train_end < val.test_start
        assert test.train_end < test.test_start
        assert val.horizon == test.horizon == M5.HORIZON_DAYS
        # Validation must end before the test window opens.
        assert val.test_end < test.test_start

    def test_rolling_origins_do_not_overlap(self):
        origins = rolling_origins(n_origins=5, horizon=28, last_date=M5.END_DATE)
        assert len(origins) == 5
        for earlier, later in zip(origins, origins[1:]):
            assert earlier.test_end < later.test_start
        for o in origins:
            assert o.train_end < o.test_start
            assert o.horizon == 28

    def test_leakage_assertion_catches_overlap(self):
        dates = pd.date_range("2016-01-01", periods=10, freq="D")
        train = pd.DataFrame({"date": dates[:6]})
        good = pd.DataFrame({"date": dates[6:]})
        bad = pd.DataFrame({"date": dates[4:]})       # overlaps training

        assert_no_temporal_leakage(train, good)       # must not raise
        with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE"):
            assert_no_temporal_leakage(train, bad)


# ===========================================================================
# Statistical models
# ===========================================================================

class TestBaselines:
    def test_seasonal_naive_reproduces_the_cycle_exactly(self):
        cycle = [5.0, 8.0, 3.0, 4.0, 9.0, 12.0, 15.0]
        y = np.tile(cycle, 60)
        out = build_baseline("seasonal_naive_weekly").fit_predict(y, 7)
        assert out == pytest.approx(cycle)

    def test_naive_repeats_last_value(self):
        y = np.array([1.0, 2.0, 7.0])
        assert build_baseline("naive").fit_predict(y, 4) == pytest.approx([7.0] * 4)

    def test_moving_average_uses_the_window(self):
        y = np.arange(1, 11, dtype=float)          # last 4 = 7,8,9,10 -> 8.5
        out = build_baseline("moving_average", window=4).fit_predict(y, 2)
        assert out == pytest.approx([8.5, 8.5])

    def test_all_baselines_are_non_negative(self):
        rng = np.random.default_rng(0)
        y = rng.poisson(2, 400).astype(float)
        for name in BASELINE_REGISTRY:
            assert (build_baseline(name).fit_predict(y, 28) >= 0).all()

    def test_short_history_does_not_raise(self):
        # ~5% of M5 items are introduced mid-panel and lack a full year.
        y = np.array([1.0, 0.0, 2.0])
        for name in BASELINE_REGISTRY:
            assert len(build_baseline(name).fit_predict(y, 28)) == 28

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            build_baseline("naive").predict(7)


class TestIntermittentClassification:
    def test_empirical_router_only_assigns_the_supported_quadrant(self):
        # The project evidence supports ADIDA only for INTERMITTENT. LUMPY's
        # marginal p_adj=0.052 is inconclusive, so it remains on the default.
        assert DEFAULT_ROUTING[DemandPattern.INTERMITTENT] == "adida"
        assert DEFAULT_ROUTING[DemandPattern.LUMPY] == "global"
        assert LITERATURE_ROUTING[DemandPattern.INTERMITTENT] == "croston_sba"

    def test_quadrants_are_assigned_correctly(self):
        rng = np.random.default_rng(1)
        dense_stable = rng.poisson(30, 600).astype(float)
        assert classify_demand(dense_stable).pattern == DemandPattern.SMOOTH

        sparse_stable = (rng.random(600) < 0.2) * 5.0
        assert classify_demand(sparse_stable).pattern == DemandPattern.INTERMITTENT

        assert classify_demand(np.zeros(100)).pattern == DemandPattern.NO_DEMAND

    def test_cv2_uses_non_zero_observations_only(self):
        # Including zeros would conflate the two SBT axes: every sparse series
        # would look size-variable and the quadrants would collapse.
        y = np.zeros(200)
        y[::10] = 5.0                              # constant size, sparse
        cls = classify_demand(y)
        assert cls.cv2 == pytest.approx(0.0, abs=1e-9)
        assert cls.adi > 1.0

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            classify_demand(np.array([]))


class TestIntermittentForecasters:
    def test_croston_recovers_a_known_rate(self):
        rng = np.random.default_rng(2)
        demand = (rng.random(4000) < 0.2) * 10.0    # true rate = 2.0/period
        rate = CrostonForecaster(variant="sba").fit_predict(demand, 1)[0]
        assert 1.4 < rate < 2.6

    def test_sba_is_below_classical_croston(self):
        # SBA multiplies by (1 - alpha/2) to correct Croston's positive bias,
        # so it must be strictly smaller for the same alpha.
        rng = np.random.default_rng(3)
        demand = (rng.random(2000) < 0.15) * 8.0
        plain = CrostonForecaster(variant="croston", alpha=0.1).fit_predict(demand, 1)[0]
        sba = CrostonForecaster(variant="sba", alpha=0.1).fit_predict(demand, 1)[0]
        assert sba < plain

    def test_zero_history_forecasts_zero(self):
        assert CrostonForecaster().fit_predict(np.zeros(100), 5) == pytest.approx(np.zeros(5))
        assert ADIDAForecaster().fit_predict(np.zeros(100), 5) == pytest.approx(np.zeros(5))

    def test_adida_recovers_a_known_rate(self):
        rng = np.random.default_rng(4)
        demand = (rng.random(3000) < 0.25) * 4.0    # true rate = 1.0
        assert 0.6 < ADIDAForecaster().fit_predict(demand, 1)[0] < 1.5

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            CrostonForecaster(alpha=0.0)


# ===========================================================================
# Significance tests
# ===========================================================================

class TestSignificance:
    def test_dm_detects_a_genuine_difference(self):
        rng = np.random.default_rng(5)
        better = rng.normal(0, 1.0, 500)
        worse = rng.normal(0, 2.0, 500)
        result = diebold_mariano(better, worse, horizon=28, model_a="A", model_b="B")
        assert result.significant
        assert result.better == "A"

    def test_dm_reports_identical_models_as_indistinguishable(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0, 1.0, 500)
        b = a + rng.normal(0, 1e-6, 500)
        assert diebold_mariano(a, b, horizon=28).better == "indistinguishable"

    def test_dm_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            diebold_mariano(np.zeros(10), np.zeros(11))

    def test_holm_is_more_conservative_than_bh(self):
        p = np.array([0.001, 0.01, 0.03, 0.049])
        holm, bh = adjust_pvalues(p, "holm"), adjust_pvalues(p, "bh")
        assert (holm >= bh - 1e-12).all()
        assert (holm >= p - 1e-12).all()          # never decreases a p-value

    def test_friedman_and_nemenyi_identify_the_tied_pair(self):
        rng = np.random.default_rng(7)
        scores = pd.DataFrame(
            {
                "good": rng.normal(20, 3, 300),
                "mid_a": rng.normal(25, 3, 300),
                "mid_b": rng.normal(25, 3, 300),   # same distribution as mid_a
                "bad": rng.normal(32, 3, 300),
            }
        )
        friedman = friedman_test(scores)
        assert friedman.significant

        nem = nemenyi_test(scores)
        # The only indistinguishable clique should be the two tied models.
        flat = {frozenset(c) for c in nem["cliques"]}
        assert frozenset({"mid_a", "mid_b"}) in flat

    def test_critical_difference_shrinks_with_more_series(self):
        # CD ~ sqrt(1/N): more series -> finer resolution.
        assert nemenyi_critical_difference(5, 1000) < nemenyi_critical_difference(5, 100)

    def test_mcb_marks_exactly_one_best(self):
        rng = np.random.default_rng(8)
        scores = pd.DataFrame(
            {"a": rng.normal(10, 2, 200), "b": rng.normal(20, 2, 200), "c": rng.normal(30, 2, 200)}
        )
        out = mcb_test(scores)
        assert int(out["is_best"].sum()) == 1
        assert out.loc[out["is_best"], "model"].iloc[0] == "a"
