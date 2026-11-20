"""M5-12b: safety stock must be sized from forecast error that includes bias.

A standard deviation subtracts each model's mean error, so a forecast that is
consistently too low looks as precise as an unbiased one. These tests pin the
fix: sigma_e is the RMSE (error about zero), so a constant under-forecast must
raise the required stock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_inventory_analysis import compute_safety_stock_by_model
from src.inventory.safety_stock import (
    SafetyStockCalculator,
    _z_score,
    summarise_forecast_errors,
)
from src.inventory.service_level import DEFAULT_COST_MARGINS, ServiceLevelAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = PROJECT_ROOT / "reports" / "inventory"
README_PATH = PROJECT_ROOT / "README.md"

LEAD_TIME = 7
Z95 = _z_score(0.95)
SQRT_L = np.sqrt(LEAD_TIME)


def _preds(model: str, residuals, series_id: str = "FOODS_1_001_CA_1") -> pd.DataFrame:
    """Backtest-style predictions where ``y_true - y_pred == residuals`` exactly."""
    residuals = np.asarray(residuals, dtype=float)
    base = 50.0
    return pd.DataFrame(
        {
            "series_id": series_id,
            "date": pd.date_range("2016-01-01", periods=len(residuals)),
            "y_true": base,
            "y_pred": base - residuals,
            "model": model,
        }
    )


def _prices(series_ids=("FOODS_1_001_CA_1",)) -> pd.DataFrame:
    rows = []
    for sid in series_ids:
        parts = sid.rsplit("_", 2)
        rows.append({"item_id": parts[0], "store_id": f"{parts[1]}_{parts[2]}", "sell_price": 10.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The core requirement: a forecast that is always too low must raise stock
# --------------------------------------------------------------------------


def test_constant_under_forecast_raises_required_stock_by_model():
    """Forecast always 2 units too low has std 0, but must not need 0 safety stock."""
    preds = pd.concat(
        [_preds("perfect", np.zeros(28)), _preds("always_low", np.full(28, 2.0))],
        ignore_index=True,
    )
    summary = compute_safety_stock_by_model(preds, _prices(), lead_time=LEAD_TIME, csl=0.95)
    by_model = summary.set_index("model")

    assert by_model.loc["perfect", "total_ss_units"] == 0.0
    assert by_model.loc["always_low", "total_ss_units"] > 0.0
    assert by_model.loc["always_low", "total_ss_capital"] > by_model.loc["perfect", "total_ss_capital"]
    # RMSE of a constant error of 2 is 2, so SS = z * 2 * sqrt(L).
    assert by_model.loc["always_low", "total_ss_units"] == pytest.approx(Z95 * 2.0 * SQRT_L, abs=0.01)
    # The std column must stay visible and show why a std-based method was blind.
    assert by_model.loc["always_low", "mean_std"] == pytest.approx(0.0, abs=1e-9)
    assert by_model.loc["always_low", "mean_bias"] == pytest.approx(2.0)


def test_constant_under_forecast_raises_required_stock_in_calculator():
    """The per-SKU calculator that feeds safety_stock.csv must size on the same error."""
    calc = SafetyStockCalculator(lead_time_days=LEAD_TIME, service_levels=(0.90, 0.95, 0.98))
    low = calc.from_forecast_errors(_preds("always_low", np.full(28, 2.0)), model="always_low")
    perfect = calc.from_forecast_errors(_preds("perfect", np.zeros(28)), model="perfect")

    assert (low["safety_stock_units"] > 0).all()
    # M5-09: a forecast with exactly zero error on positive demand is "missing", not "needs
    # 0 stock" (a flat window says nothing about future risk), so the control is NaN here.
    assert (perfect["error_estimate_status"] == "zero_error_with_demand").all()
    assert perfect["safety_stock_units"].isna().all()
    at95 = low[low["service_level"] == 0.95].iloc[0]
    assert at95["safety_stock_units"] == pytest.approx(Z95 * 2.0 * SQRT_L, abs=0.01)
    assert at95["forecast_error_rmse"] == pytest.approx(2.0)
    assert at95["forecast_error_bias"] == pytest.approx(2.0)
    assert at95["forecast_error_std"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bias", [0.0, 0.5, 1.0, 2.0])
def test_equal_spread_more_bias_needs_more_stock(bias):
    """Same residual spread, larger mean error: required stock grows as sqrt(1 + bias^2)."""
    spread = np.tile([1.0, -1.0], 14)  # mean 0, RMSE 1.0
    preds = _preds("m", spread + bias)
    summary = compute_safety_stock_by_model(preds, _prices(), lead_time=LEAD_TIME, csl=0.95)
    row = summary.iloc[0]
    assert row["mean_rmse"] == pytest.approx(np.sqrt(1.0 + bias**2), abs=1e-4)
    assert row["total_ss_units"] == pytest.approx(Z95 * np.sqrt(1.0 + bias**2) * SQRT_L, abs=0.01)
    # Std is blind to bias: it is the same for every parametrised case.
    assert row["mean_std"] == pytest.approx(np.sqrt(28 / 27), abs=1e-4)


def test_stock_is_strictly_increasing_in_bias_at_equal_spread():
    spread = np.tile([1.0, -1.0], 14)
    preds = pd.concat(
        [_preds(f"bias_{b}", spread + b) for b in (0.0, 0.5, 1.0, 2.0)], ignore_index=True
    )
    summary = compute_safety_stock_by_model(preds, _prices(), lead_time=LEAD_TIME, csl=0.95)
    ordered = summary.set_index("model").loc[[f"bias_{b}" for b in (0.0, 0.5, 1.0, 2.0)]]
    assert ordered["total_ss_units"].is_monotonic_increasing
    assert ordered["total_ss_units"].is_unique


def test_lower_variance_model_still_needs_less_stock():
    """Fix must not break the intended ordering: smoother (smaller error) wins."""
    noisy = np.tile([3.0, -3.0], 14)
    calm = np.tile([1.0, -1.0], 14)
    preds = pd.concat([_preds("naive", noisy), _preds("adida", calm)], ignore_index=True)
    summary = compute_safety_stock_by_model(preds, _prices(), lead_time=LEAD_TIME, csl=0.95)
    by_model = summary.set_index("model")
    assert by_model.loc["adida", "total_ss_capital"] < by_model.loc["naive", "total_ss_capital"]
    assert by_model.loc["adida", "capital_reduction_vs_naive"] > 0


# --------------------------------------------------------------------------
# The shared error definition
# --------------------------------------------------------------------------


def test_summarise_forecast_errors_rmse_decomposes_into_bias_and_spread():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sku": np.repeat(["a", "b"], 60), "residual": rng.normal(0.7, 2.0, 120)})
    out = summarise_forecast_errors(df, by=["sku"]).set_index("sku")
    for sku, g in df.groupby("sku"):
        e = g["residual"].to_numpy()
        n = len(e)
        assert out.loc[sku, "forecast_error_rmse"] == pytest.approx(np.sqrt(np.mean(e**2)))
        assert out.loc[sku, "forecast_error_bias"] == pytest.approx(e.mean())
        assert out.loc[sku, "forecast_error_std"] == pytest.approx(e.std(ddof=1))
        assert out.loc[sku, "n_obs"] == n
        # RMSE^2 = bias^2 + population variance
        assert out.loc[sku, "forecast_error_rmse"] ** 2 == pytest.approx(
            out.loc[sku, "forecast_error_bias"] ** 2 + e.var(ddof=0)
        )
        assert out.loc[sku, "forecast_error_rmse"] >= out.loc[sku, "forecast_error_std"] * np.sqrt((n - 1) / n) - 1e-12


# --------------------------------------------------------------------------
# Downstream: the service-level layer must read the same sigma_e
# --------------------------------------------------------------------------


def test_service_level_layer_uses_rmse_not_std_for_stockout_risk():
    """Bias-only error (std 0) must not be costed as zero stockout risk."""
    z = _z_score(0.95)
    ss = round(z * 2.0 * SQRT_L, 2)
    ss_df = pd.DataFrame(
        [
            {
                "item_id": "ITEM_BIASED",
                "store_id": "CA_1",
                "service_level": 0.95,
                "lead_time_days": LEAD_TIME,
                "avg_daily_demand": 5.0,
                "safety_stock_units": ss,
                "forecast_error_rmse": 2.0,
                "forecast_error_std": 0.0,  # bias-free: what the old code (wrongly) read
                "forecast_error_bias": 2.0,
                "avg_sell_price": 2.0,
                "cat_id": "FOODS",
            }
        ]
    )
    abc_xyz = pd.DataFrame(
        [{"item_id": "ITEM_BIASED", "store_id": "CA_1", "abc_xyz_class": "AX", "abc_class": "A", "xyz_class": "X"}]
    )
    out = ServiceLevelAnalyzer().analyze(ss_df, abc_xyz)
    # SS was sized for 95% on RMSE, so the stockout probability is 5%, not 0%.
    unit_margin = 2.0 * DEFAULT_COST_MARGINS["FOODS"]
    expected = 0.05 * 5.0 * 365.0 * unit_margin
    assert out.iloc[0]["stockout_cost_annual"] == pytest.approx(expected, rel=1e-2)
    assert out.iloc[0]["stockout_cost_annual"] > 0.0


# --------------------------------------------------------------------------
# Committed artifacts
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed_rmse_by_model() -> pd.DataFrame:
    path = INVENTORY_DIR / "ca_predictions.parquet"
    if not path.exists():
        pytest.skip("reports/inventory/ca_predictions.parquet not present")
    df = pd.read_parquet(path, columns=["model", "series_id", "y_true", "y_pred"])
    sid = df["series_id"].astype(str)
    df["item_id"] = sid.str.rsplit("_", n=2).str[0]
    df["store_id"] = sid.str.rsplit("_", n=2).str[1] + "_" + sid.str.rsplit("_", n=2).str[2]
    df["residual"] = df["y_true"].astype(float) - df["y_pred"].astype(float)
    per_sku = summarise_forecast_errors(df, by=["model", "item_id", "store_id"])
    return per_sku.groupby("model", observed=True)[
        ["forecast_error_rmse", "forecast_error_std", "forecast_error_bias"]
    ].mean()


def test_committed_by_model_table_is_sized_on_rmse(committed_rmse_by_model):
    df = pd.read_csv(INVENTORY_DIR / "safety_stock_by_model.csv").set_index("model")
    assert "mean_sigma_e" not in df.columns, "ambiguous legacy column: it was a bias-free std"
    for col in ("mean_rmse", "mean_std", "mean_bias"):
        assert col in df.columns
    expected = committed_rmse_by_model
    for model in df.index:
        assert df.loc[model, "mean_rmse"] == pytest.approx(expected.loc[model, "forecast_error_rmse"], abs=1e-3)
        assert df.loc[model, "mean_bias"] == pytest.approx(expected.loc[model, "forecast_error_bias"], abs=1e-3)
    # Required stock tracks RMSE: it can never sit below the bias-free std figure.
    assert (df["mean_rmse"] >= df["mean_std"] * 0.99).all()


def test_historical_mean_is_not_cheapest_once_bias_is_counted(committed_rmse_by_model):
    """The visible symptom of the bug: historical mean (+0.29/day under-forecast) ranked 1st."""
    df = pd.read_csv(INVENTORY_DIR / "safety_stock_by_model.csv").set_index("model")
    assert df.loc["historical_mean", "mean_bias"] > 0.25
    assert df.index[0] != "historical_mean"
    assert df.loc["historical_mean", "total_ss_capital"] > df.loc["adida", "total_ss_capital"]


def test_champion_safety_stock_csv_is_sized_on_rmse():
    ss = pd.read_csv(INVENTORY_DIR / "safety_stock.csv")
    for col in ("forecast_error_rmse", "forecast_error_std", "forecast_error_bias", "model"):
        assert col in ss.columns
    expected = np.array([_z_score(sl) for sl in ss["service_level"]]) * ss["forecast_error_rmse"] * np.sqrt(
        ss["lead_time_days"]
    )
    assert np.allclose(ss["safety_stock_units"], expected.round(2), atol=0.01)


def _readme_inventory_section() -> str:
    content = README_PATH.read_text(encoding="utf-8")
    return content[content.index("## Inventory Decision Layer"):]


def test_readme_prose_quotes_only_numbers_from_the_by_model_csv():
    """The hand-written sentences must agree with the artifact (the render check only covers tables)."""
    section = _readme_inventory_section()
    df = pd.read_csv(INVENTORY_DIR / "safety_stock_by_model.csv").set_index("model")
    money = lambda v: f"${v:,.2f}"  # noqa: E731

    assert money(df.loc["naive", "total_ss_capital"]) in section
    assert money(df.loc["adida", "total_ss_capital"]) in section
    assert f"{money(df.loc['adida', 'capital_reduction_vs_naive'])} ({df.loc['adida', 'pct_reduction_vs_naive']:.2f}%)" in section
    smoothers = df.loc[["moving_average", "adida", "ses", "croston_sba"], "total_ss_capital"]
    assert money(smoothers.min()) in section
    assert money(smoothers.max()) in section
    assert f"{df.loc['moving_average', 'wape_pct']:.2f}%" in section
    assert f"{df.loc['adida', 'wape_pct']:.2f}%" in section
    assert f"{df.loc['historical_mean', 'mean_bias']:.2f} units per SKU per day" in section

    portfolio = pd.read_csv(INVENTORY_DIR / "portfolio_summary.csv")
    assert f"+{money(portfolio['capital_released_total'].sum())} in net working capital" in section

    # The stale pre-M5-12b headline must be gone.
    assert "13,453" not in section
    assert "22.09%" not in section


def test_readme_claims_hold_in_the_data_and_are_worded_honestly():
    section = _readme_inventory_section()
    df = pd.read_csv(INVENTORY_DIR / "safety_stock_by_model.csv").set_index("model")

    # "within 3% of each other" and "smoothing versus naive, not ADIDA specifically"
    smoothers = df.loc[["moving_average", "adida", "ses", "croston_sba"], "total_ss_capital"]
    assert (smoothers.max() - smoothers.min()) / smoothers.min() < 0.03
    assert "within 3% of each other" in section
    assert "smoothing versus naive, not from ADIDA specifically" in section
    # "ranks fifth" for the historical mean
    assert list(df["total_ss_capital"].sort_values().index).index("historical_mean") == 4
    assert "ranks fifth" in section
    # Champion caveat: ADIDA via the main benchmark; moving average has the lower WAPE on CA alone.
    assert df.loc["moving_average", "wape_pct"] < df.loc["adida", "wape_pct"]
    assert "main 3,000-series benchmark" in section
    assert "lower WAPE" in section
    gap = abs(df.loc["moving_average", "total_ss_capital"] - df.loc["adida", "total_ss_capital"])
    assert gap / df.loc["adida", "total_ss_capital"] < 0.001
    # Sizing rule is stated, and the 30% is qualified by the bias-free alternative.
    assert "**RMSE**" in section
    assert "Mean bias" in section
    assert "Caveat on the 30%" in section
    assert "bias-free standard deviation" in section
