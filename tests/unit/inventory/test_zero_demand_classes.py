"""M5-09: items with no demand must not be classed "stable" (X) or costed as zero risk.

Before the fix, ``cv = std / (mean + eps)`` gave an item with no sales in the CV
window CV = 0 and therefore class X, and a series with zero forecast-error spread
got ``fillna(0.0)`` safety stock. These tests pin the repaired behaviour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inventory.abc_xyz import ABCXYZClassifier
from src.inventory.safety_stock import MIN_ERROR_OBS, SafetyStockCalculator
from src.inventory.service_level import ServiceLevelAnalyzer
from src.inventory.working_capital import WorkingCapitalAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_DIR = PROJECT_ROOT / "reports" / "inventory"
README_PATH = PROJECT_ROOT / "README.md"

WINDOW = 84
DAYS = 120


def _panel() -> pd.DataFrame:
    """Four series over 120 days (store CA_1):

    ZERO     never sells
    DORMANT  sells 5/day for the first 20 days, then nothing (positive history, empty window)
    CONST    sells exactly 5/day throughout (genuinely stable: CV = 0)
    VARIED   alternates 3 and 17 (CV about 0.7)
    """
    dates = pd.date_range("2016-01-01", periods=DAYS)
    series = {
        "ZERO": np.zeros(DAYS),
        "DORMANT": np.r_[np.full(20, 5.0), np.zeros(DAYS - 20)],
        "CONST": np.full(DAYS, 5.0),
        "VARIED": np.where(np.arange(DAYS) % 2 == 0, 3.0, 17.0),
    }
    rows = []
    for item, sales in series.items():
        for i, d in enumerate(dates):
            rows.append(
                {
                    "item_id": item,
                    "store_id": "CA_1",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "date": d,
                    "sales": float(sales[i]),
                    "wm_yr_wk": 11601 + i // 7,
                }
            )
    return pd.DataFrame(rows)


def _prices(panel: pd.DataFrame, price_by_item: dict[str, float]) -> pd.DataFrame:
    weeks = panel["wm_yr_wk"].unique()
    return pd.DataFrame(
        [
            {"store_id": "CA_1", "item_id": item, "wm_yr_wk": w, "sell_price": p}
            for item, p in price_by_item.items()
            for w in weeks
        ]
    )


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------


def test_zero_demand_items_are_class_n_not_x():
    result = ABCXYZClassifier(cv_window_days=WINDOW).classify(_panel()).set_index("item_id")

    # No sales in the window: not "stable", and CV is undefined rather than 0.
    for item in ("ZERO", "DORMANT"):
        assert result.loc[item, "xyz_class"] == "N", item
        assert result.loc[item, "abc_xyz_class"] == "?N", item
        assert result.loc[item, "mean_demand"] == 0.0, item
        assert np.isnan(result.loc[item, "cv"]), item

    # A truly constant positive series *is* stable.
    assert result.loc["CONST", "xyz_class"] == "X"
    assert result.loc["CONST", "cv"] == pytest.approx(0.0)
    assert result.loc["VARIED", "xyz_class"] == "Y"


def test_no_x_class_item_has_zero_mean_demand():
    result = ABCXYZClassifier(cv_window_days=WINDOW).classify(_panel())
    x_items = result[result["xyz_class"] == "X"]
    assert len(x_items) > 0
    assert (x_items["mean_demand"] > 0).all()


def test_priced_zero_demand_items_keep_their_abc_letter_with_n():
    panel = _panel()
    prices = _prices(panel, {"ZERO": 1.0, "DORMANT": 100.0, "CONST": 1.0, "VARIED": 1.0})
    result = ABCXYZClassifier(cv_window_days=WINDOW).classify(panel, prices=prices).set_index("item_id")
    # DORMANT earned most of the full-history revenue, so it is an "A" item with no recent demand.
    assert result.loc["DORMANT", "abc_xyz_class"] == "AN"
    assert result.loc["ZERO", "abc_xyz_class"].endswith("N")
    assert result.loc["CONST", "xyz_class"] == "X"


def test_cv_window_is_exactly_cv_window_days():
    """The window must be the last N days, not N+1, so it can match the backtest window."""
    result = ABCXYZClassifier(cv_window_days=WINDOW).classify(_panel())
    assert (result["n_days"] == WINDOW).all()


# --------------------------------------------------------------------------
# Safety stock: a degenerate error estimate is "missing", never "zero risk"
# --------------------------------------------------------------------------


def _backtest_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = 84
    dates = pd.date_range("2016-02-01", periods=n)
    parts = []

    def add(series_id: str, y_true, y_pred, length: int = n):
        parts.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "date": dates[:length],
                    "y_true": np.asarray(y_true, dtype=float)[:length],
                    "y_pred": np.asarray(y_pred, dtype=float)[:length],
                    "model": "adida",
                    "dept_id": "FOODS_1",
                }
            )
        )

    add("CONST_CA_1", np.full(n, 5.0), np.full(n, 5.0))  # demand 5, forecast exact: RMSE 0
    add("ZERO_CA_1", np.zeros(n), np.zeros(n))  # no demand, forecast 0
    add("SHORT_CA_1", np.full(n, 3.0), np.full(n, 2.0), length=10)  # only 10 residuals
    add("NORMAL_CA_1", np.full(n, 5.0) + np.tile([1.0, -1.0], n // 2), np.full(n, 5.0))  # RMSE 1
    preds = pd.concat(parts, ignore_index=True)

    window_panel = (
        preds[["series_id", "date", "y_true"]]
        .rename(columns={"y_true": "sales"})
        .assign(
            item_id=lambda d: d["series_id"].str.rsplit("_", n=2).str[0],
            store_id=lambda d: d["series_id"].str.rsplit("_", n=2).str[1]
            + "_"
            + d["series_id"].str.rsplit("_", n=2).str[2],
            dept_id="FOODS_1",
        )
    )
    return preds, window_panel


def test_degenerate_error_estimates_are_missing_not_zero():
    preds, window_panel = _backtest_frame()
    calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.95,))
    out = calc.from_forecast_errors(preds, panel=window_panel, model="adida").set_index("item_id")

    # Exactly-zero error spread on positive demand tells us nothing about risk.
    assert out.loc["CONST", "error_estimate_status"] == "zero_error_with_demand"
    assert np.isnan(out.loc["CONST", "safety_stock_units"])
    assert np.isnan(out.loc["CONST", "reorder_point"])

    # Fewer than MIN_ERROR_OBS residuals is too few to estimate sigma_e.
    assert MIN_ERROR_OBS == 28
    assert out.loc["SHORT", "error_estimate_status"] == "insufficient_obs"
    assert np.isnan(out.loc["SHORT", "safety_stock_units"])

    # No demand and a zero forecast is a legitimate zero, not a missing estimate.
    assert out.loc["ZERO", "error_estimate_status"] == "ok"
    assert out.loc["ZERO", "safety_stock_units"] == 0.0
    assert out.loc["ZERO", "avg_daily_demand"] == 0.0

    # A normal series is unaffected.
    assert out.loc["NORMAL", "error_estimate_status"] == "ok"
    assert out.loc["NORMAL", "safety_stock_units"] > 0


def test_numerically_zero_forecast_on_zero_sales_is_not_a_missing_estimate():
    """A dormant item whose forecast has decayed to ~1e-13 and which sold nothing has no demand;
    floating-point residue must not make it look like 'zero error on positive demand'."""
    n = 84
    dates = pd.date_range("2016-02-01", periods=n)
    preds = pd.DataFrame(
        {
            "series_id": "DORMANT_CA_1",
            "date": dates,
            "y_true": np.zeros(n),
            "y_pred": np.full(n, 3e-13),
            "model": "adida",
            "dept_id": "FOODS_1",
        }
    )
    out = SafetyStockCalculator(lead_time_days=7, service_levels=(0.95,)).from_forecast_errors(preds, model="adida")
    assert out.iloc[0]["error_estimate_status"] == "ok"
    assert out.iloc[0]["safety_stock_units"] == pytest.approx(0.0, abs=1e-6)
    assert not np.isnan(out.iloc[0]["safety_stock_units"])


def test_explicit_panel_still_sets_demand_for_benchmarks_and_scoring():
    """An explicit actuals panel is still honoured (benchmarks, scoring); planning passes None
    and uses the forecast instead (see test_m5_10_cost_model)."""
    preds, window_panel = _backtest_frame()
    calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.95,))
    out = calc.from_forecast_errors(preds, panel=window_panel, model="adida").set_index("item_id")
    assert out.loc["NORMAL", "avg_daily_demand"] == pytest.approx(5.0)
    assert out.loc["CONST", "avg_daily_demand"] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Downstream: excluded SKUs are counted, N-class SKUs are held, nothing leaks into totals
# --------------------------------------------------------------------------


def _scenario_inputs():
    rows = []
    for item, ss, status, rmse in (
        ("OK_BZ", 4.0, "ok", 1.0),
        ("MISSING_BZ", np.nan, "insufficient_obs", np.nan),
        ("DORMANT_CN", 0.0, "ok", 0.0),
    ):
        for sl, z in ((0.90, 1.2816), (0.95, 1.6449), (0.98, 2.0537)):
            rows.append(
                {
                    "item_id": item,
                    "store_id": "CA_1",
                    "service_level": sl,
                    "lead_time_days": 7,
                    "avg_daily_demand": 0.0 if item == "DORMANT_CN" else 3.0,
                    "safety_stock_units": ss * z / 1.6449 if pd.notna(ss) else np.nan,
                    "forecast_error_rmse": rmse,
                    "error_estimate_status": status,
                    "avg_sell_price": 2.0,
                    "cat_id": "FOODS",
                }
            )
    abc = pd.DataFrame(
        [
            {"item_id": "OK_BZ", "store_id": "CA_1", "abc_xyz_class": "BZ", "abc_class": "B", "xyz_class": "Z"},
            {"item_id": "MISSING_BZ", "store_id": "CA_1", "abc_xyz_class": "BZ", "abc_class": "B", "xyz_class": "Z"},
            {"item_id": "DORMANT_CN", "store_id": "CA_1", "abc_xyz_class": "CN", "abc_class": "C", "xyz_class": "N"},
        ]
    )
    return pd.DataFrame(rows), abc


def test_missing_sigma_e_is_excluded_and_counted_downstream():
    ss_df, abc = _scenario_inputs()
    scen = ServiceLevelAnalyzer().analyze(ss_df, abc)
    kpis = WorkingCapitalAnalyzer().compute_kpis(scen)
    recs = WorkingCapitalAnalyzer().recommend_policy(kpis, current_sl=0.95).set_index("item_id")

    # No stockout cost is invented for an SKU whose stock could not be sized.
    missing_rows = scen[scen["item_id"] == "MISSING_BZ"]
    assert missing_rows["stockout_cost_annual"].isna().all()
    assert missing_rows["total_relevant_cost"].isna().all()

    assert recs.loc["MISSING_BZ", "policy_change"] == "excluded"
    assert np.isnan(recs.loc["MISSING_BZ", "capital_released"])
    assert np.isnan(recs.loc["MISSING_BZ", "stockout_risk_delta"])
    assert recs.loc["OK_BZ", "policy_change"] == "downgrade"  # BZ: 95% -> 90%

    port = WorkingCapitalAnalyzer().portfolio_summary(recs.reset_index())
    assert "excluded" in port.columns
    assert int(port["excluded"].sum()) == 1


def test_no_recent_demand_class_is_held_at_current_service_level():
    ss_df, abc = _scenario_inputs()
    scen = ServiceLevelAnalyzer().analyze(ss_df, abc)
    kpis = WorkingCapitalAnalyzer().compute_kpis(scen)
    recs = WorkingCapitalAnalyzer().recommend_policy(kpis, current_sl=0.95).set_index("item_id")
    assert recs.loc["DORMANT_CN", "abc_xyz_class"] == "CN"
    assert recs.loc["DORMANT_CN", "policy_change"] == "maintain"
    assert recs.loc["DORMANT_CN", "recommended_service_level"] == 0.95


# --------------------------------------------------------------------------
# Committed artifacts
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def abc_csv() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_DIR / "abc_xyz_classification.csv")


def test_committed_classification_has_no_zero_demand_x_items(abc_csv):
    x_items = abc_csv[abc_csv["xyz_class"] == "X"]
    assert len(x_items) > 0
    assert (x_items["mean_demand"] > 0).all()
    n_items = abc_csv[abc_csv["xyz_class"] == "N"]
    assert len(n_items) > 0
    assert (n_items["mean_demand"] == 0).all()
    assert (abc_csv["n_days"] == WINDOW).all()


def test_committed_no_demand_check_uses_days_before_the_scored_days(abc_csv):
    """M5-10 replaced M5-09's "one window" with a stricter split: classes (including N) are
    decided on the 84 days *before* the first test day; the test days are for scoring only."""
    path = INVENTORY_DIR / "ca_predictions.parquet"
    if not path.exists():
        pytest.skip("ca_predictions.parquet not present")
    first_scored_day = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"]).min()
    window_end = pd.to_datetime(abc_csv["xyz_window_end"].unique())
    assert len(window_end) == 1
    assert window_end[0] == first_scored_day - pd.Timedelta(days=1)
    assert (abc_csv["xyz_class"] == "N").sum() > 0


def test_committed_decision_window_and_scoring_window_are_separate(abc_csv):
    """Classes come from the decision window; planning demand is the forecast on the scored days.
    The two are different quantities, so they must not be forced equal (the old one-window rule)."""
    ss = pd.read_csv(INVENTORY_DIR / "safety_stock.csv")
    ss95 = ss[ss["service_level"] == 0.95]
    merged = ss95.merge(abc_csv[["item_id", "store_id", "mean_demand"]], on=["item_id", "store_id"])
    assert len(merged) == len(abc_csv)
    assert not np.allclose(merged["avg_daily_demand"], merged["mean_demand"], atol=1e-4)
    assert "error_estimate_status" in ss.columns
    assert ss["error_estimate_status"].isin(["ok", "insufficient_obs", "zero_error_with_demand"]).all()


def test_readme_has_a_no_recent_demand_row_that_matches_the_portfolio_csv():
    port = pd.read_csv(INVENTORY_DIR / "portfolio_summary.csv")
    priced_n = port[port["abc_xyz_class"].isin(["AN", "BN", "CN"])]["n_skus"].sum()
    assert priced_n > 0
    content = README_PATH.read_text(encoding="utf-8")
    rows = [line for line in content.splitlines() if "No recent demand" in line and line.startswith("|")]
    assert len(rows) == 1
    assert f"| {int(priced_n):,} |" in rows[0]
    assert "84-day" in content
