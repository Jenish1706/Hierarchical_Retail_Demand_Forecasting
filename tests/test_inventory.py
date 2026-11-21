"""Unit tests for the inventory optimization layer.

Covers:
- ABC thresholds sum to 100% of revenue
- XYZ classification boundaries are correct
- Safety stock increases monotonically with service level
- Reorder point >= safety stock for all SKUs
- Working capital released is non-negative for policy downgrades
- Cross-reference with SBC demand pattern classification
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.inventory.abc_xyz import ABCXYZClassifier
from src.inventory.safety_stock import SafetyStockCalculator
from src.inventory.service_level import ServiceLevelAnalyzer
from src.inventory.working_capital import WorkingCapitalAnalyzer
from src.models.statistical.intermittent import DemandPattern


@pytest.fixture
def synthetic_panel_and_prices():
    """Create a minimal panel and price table for 4 items over 100 days."""
    dates = [date(2016, 1, 1) + timedelta(days=i) for i in range(100)]
    stores = ["CA_1"]
    items = ["ITEM_A_X", "ITEM_B_Y", "ITEM_C_Z1", "ITEM_C_Z2"]

    rows = []
    rng = np.random.default_rng(42)

    for item in items:
        for store in stores:
            for d in dates:
                # ITEM_A_X: high volume, low variance (X: CV < 0.5)
                if "A_X" in item:
                    sales = int(rng.normal(loc=50, scale=5))
                # ITEM_B_Y: medium volume, medium variance (Y: 0.5 <= CV < 1.0)
                elif "B_Y" in item:
                    sales = int(rng.exponential(scale=10))
                # ITEM_C_Z: low volume, intermittent/erratic (Z: CV >= 1.0)
                else:
                    sales = int(rng.choice([0, 0, 0, 1, 5], p=[0.7, 0.1, 0.1, 0.05, 0.05]))

                sales = max(0, sales)
                wm_yr_wk = 11601 + (d.day // 7)
                rows.append({
                    "item_id": item,
                    "store_id": store,
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "date": pd.to_datetime(d),
                    "sales": sales,
                    "wm_yr_wk": wm_yr_wk,
                })

    panel = pd.DataFrame(rows)

    # Prices: high price for A, lower for others so A dominates revenue
    price_rows = [
        {"store_id": "CA_1", "item_id": "ITEM_A_X", "wm_yr_wk": w, "sell_price": 20.0}
        for w in panel["wm_yr_wk"].unique()
    ] + [
        {"store_id": "CA_1", "item_id": "ITEM_B_Y", "wm_yr_wk": w, "sell_price": 5.0}
        for w in panel["wm_yr_wk"].unique()
    ] + [
        {"store_id": "CA_1", "item_id": "ITEM_C_Z1", "wm_yr_wk": w, "sell_price": 1.0}
        for w in panel["wm_yr_wk"].unique()
    ] + [
        {"store_id": "CA_1", "item_id": "ITEM_C_Z2", "wm_yr_wk": w, "sell_price": 1.0}
        for w in panel["wm_yr_wk"].unique()
    ]

    prices = pd.DataFrame(price_rows)
    return panel, prices


class TestABCXYZClassifier:
    def test_abc_thresholds_sum_to_100_percent_revenue(self, synthetic_panel_and_prices):
        panel, prices = synthetic_panel_and_prices
        clf = ABCXYZClassifier(abc_thresholds=(0.80, 0.95))
        result = clf.classify(panel, prices=prices)

        assert "abc_class" in result.columns
        assert "revenue_share" in result.columns

        # Revenue share must sum to approximately 1.0 (100%)
        total_share = result["revenue_share"].sum()
        assert np.isclose(total_share, 1.0, atol=1e-5)

        # Classes present should be subset of A, B, C
        classes = set(result["abc_class"].unique())
        assert classes.issubset({"A", "B", "C"})

        # High-revenue item A_X should be class A
        a_items = result[result["abc_class"] == "A"]["item_id"].tolist()
        assert "ITEM_A_X" in a_items

    def test_xyz_classification_boundaries_are_correct(self):
        dates = [date(2016, 1, 1) + timedelta(days=i) for i in range(60)]

        # Series 1: Constant demand -> CV = 0 -> X
        # Series 2: Mean 10, std ~2 -> CV ~ 0.2 -> X (< 0.5)
        # Series 3: Mean 10, std ~7 -> CV ~ 0.7 -> Y (0.5 <= CV < 1.0)
        # Series 4: Mean 1, std ~2 -> CV ~ 2.0 -> Z (>= 1.0)
        rows = []
        for d in dates:
            rows.append({"item_id": "I_X1", "store_id": "S1", "date": pd.to_datetime(d), "sales": 10})
            rows.append({"item_id": "I_X2", "store_id": "S1", "date": pd.to_datetime(d), "sales": 10 + (d.day % 3)})
            # alternating 3 and 17: mean 10, std ~7, CV ~ 0.7 -> Y (0.5 <= CV < 1.0)
            rows.append({"item_id": "I_Y", "store_id": "S1", "date": pd.to_datetime(d), "sales": 3 if d.day % 2 == 0 else 17})
            rows.append({"item_id": "I_Z", "store_id": "S1", "date": pd.to_datetime(d), "sales": 0 if d.day % 5 != 0 else 10})

        panel = pd.DataFrame(rows)
        clf = ABCXYZClassifier(xyz_thresholds=(0.50, 1.00), cv_window_days=60)
        result = clf.classify(panel)

        lookup = dict(zip(result["item_id"], result["xyz_class"]))
        assert lookup["I_X1"] == "X"
        assert lookup["I_X2"] == "X"
        assert lookup["I_Y"] == "Y"
        assert lookup["I_Z"] == "Z"

    def test_cross_reference_sbc(self, synthetic_panel_and_prices):
        panel, prices = synthetic_panel_and_prices
        clf = ABCXYZClassifier()
        abc_xyz_df = clf.classify(panel, prices=prices)

        # Build mock SBC classifications
        sbc_rows = [
            {"item_id": "ITEM_A_X", "store_id": "CA_1", "demand_pattern": DemandPattern.SMOOTH.value},
            {"item_id": "ITEM_B_Y", "store_id": "CA_1", "demand_pattern": DemandPattern.ERRATIC.value},
            {"item_id": "ITEM_C_Z1", "store_id": "CA_1", "demand_pattern": DemandPattern.INTERMITTENT.value},
            {"item_id": "ITEM_C_Z2", "store_id": "CA_1", "demand_pattern": DemandPattern.LUMPY.value},
        ]
        sbc_df = pd.DataFrame(sbc_rows)
        contingency = ABCXYZClassifier.cross_reference_sbc(abc_xyz_df, sbc_df)

        assert isinstance(contingency, pd.DataFrame)
        assert not contingency.empty
        assert "All" in contingency.columns


class TestSafetyStockCalculator:
    def test_safety_stock_increases_monotonically_with_service_level(self, synthetic_panel_and_prices):
        panel, _ = synthetic_panel_and_prices
        calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.90, 0.95, 0.98))

        # Synthetic forecasts and actuals with non-zero residuals
        dates = panel["date"].unique()[-28:]
        test_panel = panel[panel["date"].isin(dates)].copy()

        forecasts = test_panel[["item_id", "store_id", "date"]].copy()
        forecasts["forecast"] = test_panel["sales"] * 0.8 + 2.0  # non-perfect forecast
        actuals = test_panel[["item_id", "store_id", "date", "sales"]].copy()

        ss_df = calc.from_forecast_errors(forecasts, actuals, panel)

        # For each SKU, check monotonic increase across 0.90, 0.95, 0.98
        for (_item, _store), group in ss_df.groupby(["item_id", "store_id"]):
            sorted_group = group.sort_values("service_level")
            ss_vals = sorted_group["safety_stock_units"].tolist()
            assert ss_vals[0] <= ss_vals[1] <= ss_vals[2]
            # When forecast error RMSE > 0, it should be strictly increasing
            if sorted_group["forecast_error_rmse"].iloc[0] > 0.01:
                assert ss_vals[0] < ss_vals[1] < ss_vals[2]

    def test_reorder_point_ge_safety_stock_for_all_skus(self, synthetic_panel_and_prices):
        panel, _ = synthetic_panel_and_prices
        calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.90, 0.95, 0.98))

        dates = panel["date"].unique()[-28:]
        test_panel = panel[panel["date"].isin(dates)].copy()
        forecasts = test_panel[["item_id", "store_id", "date"]].copy()
        forecasts["forecast"] = test_panel["sales"] * 0.9
        actuals = test_panel[["item_id", "store_id", "date", "sales"]].copy()

        ss_df = calc.from_forecast_errors(forecasts, actuals, panel)

        # ROP = mean_demand * L + SS >= SS (since mean_demand >= 0 and L >= 0)
        assert (ss_df["reorder_point"] >= ss_df["safety_stock_units"]).all()


class TestServiceLevelAndWorkingCapital:
    def test_working_capital_released_is_non_negative_for_downgrades(self, synthetic_panel_and_prices):
        panel, prices = synthetic_panel_and_prices

        # 1. ABC/XYZ
        clf = ABCXYZClassifier()
        abc_xyz = clf.classify(panel, prices=prices)

        # 2. Safety Stock
        calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.90, 0.95, 0.98))
        dates = panel["date"].unique()[-28:]
        test_panel = panel[panel["date"].isin(dates)].copy()
        forecasts = test_panel[["item_id", "store_id", "date"]].copy()
        forecasts["forecast"] = test_panel["sales"] * 0.85
        actuals = test_panel[["item_id", "store_id", "date", "sales"]].copy()
        ss_df = calc.from_forecast_errors(forecasts, actuals, panel)

        # 3. Service Level Analysis
        sla = ServiceLevelAnalyzer()
        scenarios = sla.analyze(ss_df, abc_xyz, prices=prices, panel=panel)

        # 4. Working Capital
        wca = WorkingCapitalAnalyzer()
        kpis = wca.compute_kpis(scenarios)
        recs = wca.recommend_policy(kpis, current_sl=0.95)

        assert not recs.empty
        assert "capital_released" in recs.columns

        # For downgrades (e.g. CZ moving from 95% to 90%), capital released must be >= 0
        downgrades = recs[recs["policy_change"] == "downgrade"]
        if not downgrades.empty:
            valid_releases = downgrades["capital_released"].dropna()
            assert (valid_releases >= 0).all()

        # For maintained items (service level unchanged), capital released should be 0.0
        maintained = recs[recs["policy_change"] == "maintain"]
        if not maintained.empty:
            valid_maintained = maintained["capital_released"].dropna()
            assert np.isclose(valid_maintained, 0.0, atol=1e-2).all()

    def test_marginal_analysis_computes_incremental_cost(self, synthetic_panel_and_prices):
        panel, prices = synthetic_panel_and_prices
        clf = ABCXYZClassifier()
        abc_xyz = clf.classify(panel, prices=prices)

        calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.90, 0.95, 0.98))
        dates = panel["date"].unique()[-28:]
        test_panel = panel[panel["date"].isin(dates)].copy()
        forecasts = test_panel[["item_id", "store_id", "date"]].copy()
        forecasts["forecast"] = test_panel["sales"] * 0.9
        actuals = test_panel[["item_id", "store_id", "date", "sales"]].copy()
        ss_df = calc.from_forecast_errors(forecasts, actuals, panel)

        sla = ServiceLevelAnalyzer()
        scenarios = sla.analyze(ss_df, abc_xyz, prices=prices, panel=panel)
        marginals = sla.marginal_analysis(scenarios)

        assert not marginals.empty
        assert "delta_safety_stock" in marginals.columns
        assert "delta_holding_cost" in marginals.columns
        assert (marginals["delta_safety_stock"] >= 0).all()

    def test_marginal_stockout_cost_delta_zero_when_safety_stock_unchanged(self):
        """When safety stock is 0 across service levels (e.g. zero forecast error RMSE),
        stockout cost delta must be 0 rather than falsely reporting savings."""
        sla = ServiceLevelAnalyzer()
        ss_df = pd.DataFrame([
            {
                "item_id": "ITEM_ZERO_VAR",
                "store_id": "CA_1",
                "service_level": 0.90,
                "lead_time_days": 7,
                "avg_daily_demand": 5.0,
                "safety_stock_units": 0.0,
                "forecast_error_rmse": 0.0,
                "avg_sell_price": 2.0,
                "cat_id": "FOODS",
            },
            {
                "item_id": "ITEM_ZERO_VAR",
                "store_id": "CA_1",
                "service_level": 0.95,
                "lead_time_days": 7,
                "avg_daily_demand": 5.0,
                "safety_stock_units": 0.0,
                "forecast_error_rmse": 0.0,
                "avg_sell_price": 2.0,
                "cat_id": "FOODS",
            },
        ])
        abc_xyz = pd.DataFrame([{
            "item_id": "ITEM_ZERO_VAR", "store_id": "CA_1", "abc_xyz_class": "CZ",
            "abc_class": "C", "xyz_class": "Z"
        }])
        analyzed = sla.analyze(ss_df, abc_xyz)
        marginals = sla.marginal_analysis(analyzed)
        assert len(marginals) == 1
        assert marginals.iloc[0]["delta_safety_stock"] == 0.0
        assert marginals.iloc[0]["delta_stockout_cost"] == 0.0
