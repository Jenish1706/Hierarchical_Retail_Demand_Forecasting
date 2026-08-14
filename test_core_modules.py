"""
Unit tests for core platform modules.

Test philosophy:
  - Fast: no DB, no network, pure function tests
  - Deterministic: fixed seeds everywhere
  - Boundary conditions: zero demand, missing data, extreme values
  - Business rules: EOQ >= 1, safety stock >= 0, ROP >= SS
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from src.feature_store.pipeline import FeatureEngineeringPipeline
from src.optimization.inventory.optimizer import (
    InventoryOptimizer,
    InventoryPolicy,
    ABCXYZClassifier,
)
from src.validation.data_validator import SalesDataValidator, PriceDataValidator
from src.models.ml.trainer import compute_metrics
from src.pricing.elasticity import DynamicPricingOptimizer, PricingConstraints


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Minimal valid sales DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    items = ["FOODS_1_001", "FOODS_1_002"]
    stores = ["CA_1", "TX_1"]

    records = []
    for item in items:
        for store in stores:
            for d in dates:
                records.append({
                    "item_id": item,
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": store,
                    "state_id": "CA" if "CA" in store else "TX",
                    "date": d,
                    "sales": max(0.0, rng.normal(10, 3)),
                })
    return pd.DataFrame(records)


@pytest.fixture
def sample_prices_df(sample_sales_df) -> pd.DataFrame:
    """Corresponding price data."""
    rng = np.random.default_rng(7)
    records = []
    for item in sample_sales_df["item_id"].unique():
        for store in sample_sales_df["store_id"].unique():
            for wk in range(1, 54):
                records.append({
                    "item_id": item,
                    "store_id": store,
                    "wm_yr_wk": 11100 + wk,
                    "sell_price": round(rng.uniform(1.5, 8.0), 2),
                })
    return pd.DataFrame(records)


@pytest.fixture
def sample_calendar_df() -> pd.DataFrame:
    """Calendar DataFrame covering 2022."""
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": [11100 + (d.isocalendar()[1]) for d in dates],
        "weekday": [d.strftime("%A") for d in dates],
        "wday": [d.weekday() for d in dates],
        "month": dates.month,
        "year": dates.year,
        "event_name_1": None,
        "event_type_1": None,
        "event_name_2": None,
        "event_type_2": None,
        "snap_CA": 0,
        "snap_TX": 0,
        "snap_WI": 0,
        "d": [f"d_{i+1}" for i in range(365)],
    })


# ============================================================================
# Feature Engineering Tests
# ============================================================================

class TestFeatureEngineeringPipeline:

    @pytest.mark.unit
    def test_time_features_generated(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        """All time feature columns must be present after transform."""
        pipeline = FeatureEngineeringPipeline(feature_version=1)
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)

        expected_cols = [
            "day_of_week", "month", "quarter", "year",
            "is_weekend", "is_holiday",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
        ]
        for col in expected_cols:
            assert col in df_out.columns, f"Missing expected column: {col}"

    @pytest.mark.unit
    def test_lag_features_generated(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        """Lag features must be present for all lag windows."""
        pipeline = FeatureEngineeringPipeline()
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)

        for lag in [1, 7, 14, 28, 90]:
            assert f"lag_{lag}" in df_out.columns

    @pytest.mark.unit
    def test_rolling_features_non_negative(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        """Rolling means over non-negative sales must be non-negative."""
        pipeline = FeatureEngineeringPipeline()
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)

        for w in [7, 28, 90]:
            col = f"rolling_mean_{w}"
            if col in df_out.columns:
                vals = df_out[col].dropna()
                assert (vals >= 0).all(), f"{col} has negative values"

    @pytest.mark.unit
    def test_no_future_leakage_in_lags(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        """lag_1 must equal previous day's sales (chronological shift)."""
        pipeline = FeatureEngineeringPipeline()
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)
        df_item = df_out[
            (df_out["item_id"] == "FOODS_1_001") & (df_out["store_id"] == "CA_1")
        ].sort_values("date").reset_index(drop=True)

        if "lag_1" in df_item.columns and len(df_item) > 2:
            # lag_1[i] should equal sales[i-1]
            lag_matches = (
                df_item["lag_1"].iloc[2:].values ==
                df_item["sales"].iloc[1:-1].values
            )
            assert lag_matches.mean() > 0.99, "lag_1 does not match shifted sales"

    @pytest.mark.unit
    def test_cyclical_encoding_range(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        """Cyclical sin/cos features must be in [-1, 1]."""
        pipeline = FeatureEngineeringPipeline()
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)

        for col in ["dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert df_out[col].between(-1.01, 1.01).all(), f"{col} out of [-1, 1]"

    @pytest.mark.unit
    def test_feature_version_set(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        pipeline = FeatureEngineeringPipeline(feature_version=3)
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)
        assert (df_out["feature_version"] == 3).all()

    @pytest.mark.unit
    def test_transform_row_count_preserved(self, sample_sales_df, sample_prices_df, sample_calendar_df):
        """Output should have same number of rows as input sales."""
        pipeline = FeatureEngineeringPipeline()
        df_out = pipeline.transform(sample_sales_df, sample_prices_df, sample_calendar_df)
        assert len(df_out) == len(sample_sales_df)


# ============================================================================
# Inventory Optimizer Tests
# ============================================================================

class TestInventoryOptimizer:

    def _make_policy(self, **overrides) -> InventoryPolicy:
        defaults = dict(
            item_id="TEST_001",
            store_id="CA_1",
            mean_daily_demand=15.0,
            std_daily_demand=4.5,
            annual_demand=15.0 * 365,
            mean_lead_time_days=7.0,
            std_lead_time_days=1.5,
            unit_cost=10.0,
            ordering_cost=50.0,
            holding_cost_rate=0.25,
            service_level=0.95,
        )
        defaults.update(overrides)
        return InventoryPolicy(**defaults)

    @pytest.mark.unit
    def test_eoq_positive(self):
        """EOQ must always be strictly positive."""
        optimizer = InventoryOptimizer()
        policy = self._make_policy()
        result = optimizer.optimize(policy)
        assert result.eoq > 0

    @pytest.mark.unit
    def test_eoq_formula(self):
        """EOQ = sqrt(2DS/H) — validate against known formula."""
        D = 15.0 * 365   # annual demand
        S = 50.0          # ordering cost
        H = 10.0 * 0.25  # holding cost per unit per year
        expected_eoq = np.sqrt(2 * D * S / H)

        optimizer = InventoryOptimizer()
        policy = self._make_policy()
        result = optimizer.optimize(policy)
        assert abs(result.eoq - expected_eoq) < 1.0, (
            f"EOQ {result.eoq:.2f} differs from formula {expected_eoq:.2f}"
        )

    @pytest.mark.unit
    def test_safety_stock_non_negative(self):
        """Safety stock must be >= 0 for any valid input."""
        optimizer = InventoryOptimizer()
        for sl in [0.80, 0.90, 0.95, 0.99]:
            policy = self._make_policy(service_level=sl)
            result = optimizer.optimize(policy)
            assert result.safety_stock >= 0, f"Negative SS at service level {sl}"

    @pytest.mark.unit
    def test_rop_geq_safety_stock(self):
        """Reorder point must always be >= safety stock."""
        optimizer = InventoryOptimizer()
        policy = self._make_policy()
        result = optimizer.optimize(policy)
        assert result.reorder_point >= result.safety_stock

    @pytest.mark.unit
    def test_higher_service_level_increases_safety_stock(self):
        """Increasing service level should increase safety stock."""
        optimizer = InventoryOptimizer()
        ss_values = []
        for sl in [0.80, 0.90, 0.95, 0.99]:
            p = self._make_policy(service_level=sl)
            r = optimizer.optimize(p)
            ss_values.append(r.safety_stock)
        assert ss_values == sorted(ss_values), "Safety stock not monotone in service level"

    @pytest.mark.unit
    def test_zero_demand_handled(self):
        """Should not crash or produce NaN with near-zero demand."""
        optimizer = InventoryOptimizer()
        policy = self._make_policy(
            mean_daily_demand=0.001,
            std_daily_demand=0.001,
            annual_demand=0.365,
        )
        result = optimizer.optimize(policy)
        assert result.eoq >= 1.0
        assert not np.isnan(result.safety_stock)

    @pytest.mark.unit
    def test_total_cost_equals_sum_of_components(self):
        """Total cost == ordering cost + holding cost."""
        optimizer = InventoryOptimizer()
        policy = self._make_policy()
        result = optimizer.optimize(policy)
        expected_total = result.expected_ordering_cost + result.expected_holding_cost
        assert abs(result.expected_total_cost - expected_total) < 0.01

    @pytest.mark.unit
    def test_inventory_turnover_positive(self):
        """Inventory turnover must be positive for positive demand."""
        optimizer = InventoryOptimizer()
        policy = self._make_policy()
        result = optimizer.optimize(policy)
        assert result.inventory_turnover > 0


# ============================================================================
# ABC-XYZ Classifier Tests
# ============================================================================

class TestABCXYZClassifier:

    @pytest.mark.unit
    def test_all_classes_assigned(self):
        """Every row should receive an ABC and XYZ class."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "item_id": [f"ITEM_{i}" for i in range(n)],
            "annual_revenue": rng.exponential(1000, n),
            "mean_daily_demand": rng.exponential(10, n),
            "std_daily_demand": rng.exponential(5, n),
        })
        classifier = ABCXYZClassifier()
        result = classifier.classify(df)
        assert result["abc_class"].notna().all()
        assert result["xyz_class"].notna().all()

    @pytest.mark.unit
    def test_abc_revenue_shares(self):
        """A-class items should cover ~70% of cumulative revenue."""
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "item_id": [f"ITEM_{i}" for i in range(n)],
            "annual_revenue": rng.exponential(500, n),
            "mean_daily_demand": rng.exponential(5, n),
            "std_daily_demand": rng.exponential(3, n),
        })
        classifier = ABCXYZClassifier()
        result = classifier.classify(df)
        a_revenue_pct = (
            result[result["abc_class"] == "A"]["annual_revenue"].sum()
            / result["annual_revenue"].sum()
        )
        assert 0.65 <= a_revenue_pct <= 0.75, f"A class revenue share = {a_revenue_pct:.2%}"


# ============================================================================
# Data Validator Tests
# ============================================================================

class TestSalesDataValidator:

    @pytest.mark.unit
    def test_valid_data_passes(self, sample_sales_df):
        validator = SalesDataValidator()
        result = validator.validate(sample_sales_df)
        assert result.passed
        assert result.error_count == 0

    @pytest.mark.unit
    def test_negative_sales_fails(self):
        df = pd.DataFrame({
            "item_id": ["X"] * 10,
            "store_id": ["CA_1"] * 10,
            "date": pd.date_range("2022-01-01", periods=10),
            "sales": [-1.0, 2.0, 3.0, -0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        validator = SalesDataValidator()
        result = validator.validate(df)
        assert not result.passed
        assert any("negative" in e.lower() for e in result.errors)

    @pytest.mark.unit
    def test_missing_required_column_fails(self):
        df = pd.DataFrame({"item_id": ["X"], "store_id": ["CA_1"], "sales": [1.0]})
        # Missing "date" column
        validator = SalesDataValidator()
        result = validator.validate(df)
        assert not result.passed
        assert any("date" in e.lower() for e in result.errors)

    @pytest.mark.unit
    def test_stats_computed(self, sample_sales_df):
        validator = SalesDataValidator()
        result = validator.validate(sample_sales_df)
        assert "sales_mean" in result.stats
        assert "zero_sale_pct" in result.stats
        assert result.stats["sales_mean"] > 0


class TestPriceDataValidator:

    @pytest.mark.unit
    def test_zero_price_fails(self):
        df = pd.DataFrame({
            "item_id": ["X"] * 5,
            "store_id": ["CA_1"] * 5,
            "wm_yr_wk": [11101] * 5,
            "sell_price": [0.0, 1.99, 2.49, 3.00, 0.0],
        })
        validator = PriceDataValidator()
        result = validator.validate(df)
        assert not result.passed

    @pytest.mark.unit
    def test_valid_prices_pass(self):
        df = pd.DataFrame({
            "item_id": ["X"] * 5,
            "store_id": ["CA_1"] * 5,
            "wm_yr_wk": [11101] * 5,
            "sell_price": [1.99, 2.49, 3.00, 4.50, 5.99],
        })
        validator = PriceDataValidator()
        result = validator.validate(df)
        assert result.passed


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:

    @pytest.mark.unit
    def test_perfect_forecast_zero_error(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["wape"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_metrics_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0, 100, 200)
        y_pred = rng.uniform(0, 100, 200)
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] >= 0
        assert m["rmse"] >= 0

    @pytest.mark.unit
    def test_negative_predictions_clipped(self):
        """Negative demand predictions should be clipped to 0."""
        y_true = np.array([5.0, 10.0, 3.0])
        y_pred = np.array([-1.0, 8.0, 4.0])  # First is negative
        m = compute_metrics(y_true, y_pred)
        # After clipping: pred = [0, 8, 4]
        expected_mae = np.mean([5.0, 2.0, 1.0])
        assert m["mae"] == pytest.approx(expected_mae, rel=0.01)


# ============================================================================
# Pricing Tests
# ============================================================================

class TestDynamicPricingOptimizer:

    @pytest.mark.unit
    def test_profit_maximizing_price_above_cost(self):
        """Recommended price must always exceed unit cost."""
        optimizer = DynamicPricingOptimizer(strategy="maximize_profit")
        rec = optimizer.recommend_price(
            item_id="X", store_id="CA_1",
            current_price=5.00, unit_cost=2.00,
            elasticity=-1.8, current_demand=100.0,
        )
        assert rec["recommended_price"] >= 2.00 * 1.05

    @pytest.mark.unit
    def test_constraint_respected_min_margin(self):
        """Price must not go below cost + min_margin_pct."""
        optimizer = DynamicPricingOptimizer(strategy="maximize_revenue")
        constraints = PricingConstraints(min_margin_pct=0.20)
        rec = optimizer.recommend_price(
            item_id="X", store_id="CA_1",
            current_price=3.00, unit_cost=2.00,
            elasticity=-3.0, current_demand=50.0,
            constraints=constraints,
        )
        min_allowed = 2.00 * 1.20
        assert rec["recommended_price"] >= min_allowed - 0.01

    @pytest.mark.unit
    def test_inelastic_demand_price_increase(self):
        """Inelastic demand (|ε| < 1) → price should increase."""
        optimizer = DynamicPricingOptimizer(strategy="maximize_revenue")
        rec = optimizer.recommend_price(
            item_id="X", store_id="CA_1",
            current_price=5.00, unit_cost=2.00,
            elasticity=-0.5,  # inelastic
            current_demand=100.0,
        )
        assert rec["recommended_price"] >= 5.00

    @pytest.mark.unit
    def test_price_change_within_bounds(self):
        """Price change must respect max increase/decrease constraints."""
        optimizer = DynamicPricingOptimizer(strategy="clear_inventory")
        constraints = PricingConstraints(
            max_price_increase_pct=0.10,
            max_price_decrease_pct=0.25,
        )
        rec = optimizer.recommend_price(
            item_id="X", store_id="CA_1",
            current_price=10.00, unit_cost=4.00,
            elasticity=-2.0, current_demand=50.0,
            constraints=constraints,
        )
        assert rec["recommended_price"] >= 10.00 * 0.75 - 0.01
        assert rec["recommended_price"] <= 10.00 * 1.10 + 0.01
