"""M5-10: an honest inventory cost model.

Pins four repairs:
1. The shortage term is expected units short per cycle (sigma_L * G(z), the normal loss
   function) x cycles per year x unit margin, not "stockout probability x the whole
   lead-time demand".
2. Planning demand (cycle stock) is the champion's mean *forecast*; actuals are scoring-only.
3. Decisions (segmentation) use only data before the first test day; the test days are
   for scoring.
4. The README never shows capital released without the annual total-cost change beside it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from scripts.run_inventory_analysis import EVAL_WINDOW_DAYS, split_panel_for_decisions
from src.inventory.abc_xyz import ABCXYZClassifier
from src.inventory.safety_stock import SafetyStockCalculator, _z_score
from src.inventory.service_level import (
    DEFAULT_COST_MARGINS,
    ServiceLevelAnalyzer,
    normal_loss,
)
from src.inventory.working_capital import WorkingCapitalAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = PROJECT_ROOT / "reports" / "inventory"
README_PATH = PROJECT_ROOT / "README.md"

LT = 7
SQRT_L = np.sqrt(LT)
SLS = (0.90, 0.95, 0.98)


# --------------------------------------------------------------------------
# 1. The loss function and the new shortage term
# --------------------------------------------------------------------------


def test_normal_loss_function_matches_hand_values():
    assert normal_loss(1.2816) == pytest.approx(0.0473, abs=5e-4)  # G(z90), the brief's hand value
    assert normal_loss(0.0) == pytest.approx(0.398942, abs=1e-6)  # phi(0)
    assert normal_loss(3.0) == pytest.approx(0.000382, abs=1e-6)
    z = np.linspace(-3.0, 3.0, 25)
    assert np.all(np.diff(normal_loss(z)) < 0)  # strictly decreasing in z
    assert normal_loss(-1.2) == pytest.approx(normal_loss(1.2) + 1.2)  # G(-z) = G(z) + z


def _frames(specs, analyzer=None):
    """specs: (item_id, abc_xyz_class, cat_id, rmse, avg_demand, price). One store, 3 SLs."""
    rows, abc = [], []
    for item, cls, cat, rmse, dem, price in specs:
        for sl in SLS:
            rows.append(
                {
                    "item_id": item,
                    "store_id": "CA_1",
                    "service_level": sl,
                    "lead_time_days": LT,
                    "avg_daily_demand": dem,
                    "safety_stock_units": round(_z_score(sl) * rmse * SQRT_L, 2),
                    "forecast_error_rmse": rmse,
                    "avg_sell_price": price,
                    "cat_id": cat,
                }
            )
        abc.append({"item_id": item, "store_id": "CA_1", "abc_xyz_class": cls, "abc_class": cls[0], "xyz_class": cls[1]})
    analyzer = analyzer or ServiceLevelAnalyzer()
    scen = analyzer.analyze(pd.DataFrame(rows), pd.DataFrame(abc))
    return scen, WorkingCapitalAnalyzer().compute_kpis(scen)


def test_stockout_cost_is_units_short_per_cycle_times_cycles_times_margin():
    rmse, dem, price = 1.0594, 0.7857, 2.17
    scen, _ = _frames([("FOODS_1_001", "BZ", "FOODS", rmse, dem, price)])
    row = scen[scen["service_level"] == 0.90].iloc[0]

    z = row["safety_stock_units"] / (rmse * SQRT_L)
    expected_short = rmse * SQRT_L * (norm.pdf(z) - z * norm.sf(z))
    assert row["expected_units_short_per_cycle"] == pytest.approx(expected_short, rel=1e-3)
    assert row["cycles_per_year"] == pytest.approx(365.0 / LT)
    unit_margin = price * DEFAULT_COST_MARGINS["FOODS"]
    assert row["stockout_cost_annual"] == pytest.approx(expected_short * (365.0 / LT) * unit_margin, rel=1e-2)

    # The replaced formula charged (1 - SL) x the whole lead-time demand per cycle.
    old_units_short_per_cycle = (1 - 0.90) * dem * LT
    assert old_units_short_per_cycle / row["expected_units_short_per_cycle"] > 3.0


def test_shortage_cost_does_not_scale_with_the_demand_level():
    """sigma_L * G(z) depends on error size and stock, not on the mean: doubling the demand
    level changes holding cost (cycle stock) but not the expected shortage."""
    lo, _ = _frames([("A", "BZ", "FOODS", 1.5, 2.0, 3.0)])
    hi, _ = _frames([("A", "BZ", "FOODS", 1.5, 4.0, 3.0)])
    assert np.allclose(lo["stockout_cost_annual"], hi["stockout_cost_annual"])
    assert (hi["holding_cost_annual"] > lo["holding_cost_annual"]).all()


def test_more_safety_stock_means_fewer_units_short():
    scen, _ = _frames([("A", "BZ", "FOODS", 2.0, 3.0, 3.0)])
    s = scen.sort_values("service_level")
    assert s["expected_units_short_per_cycle"].is_monotonic_decreasing
    assert s["stockout_cost_annual"].is_monotonic_decreasing
    assert s["holding_cost_annual"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Policy deltas, cost-minimising levels and the side-by-side comparison
# --------------------------------------------------------------------------

CHEAP_STOCKOUT = {"FOODS": 0.60, "TOYS": 0.01}  # TOYS: a unit short costs almost nothing


def _policy_frames():
    specs = [
        ("AX1", "AX", "FOODS", 1.5, 3.0, 2.0),
        ("AX2", "AX", "FOODS", 2.5, 5.0, 2.0),
        ("CZ1", "CZ", "TOYS", 1.5, 3.0, 2.0),
        ("CZ2", "CZ", "TOYS", 2.0, 2.0, 2.0),
        ("AY1", "AY", "FOODS", 0.0, 1.0, 2.0),  # zero error: every service level costs the same
        ("CN1", "CN", "TOYS", 1.5, 0.0, 2.0),  # no recent demand: held, not optimised
    ]
    return _frames(specs, ServiceLevelAnalyzer(cost_margin_assumptions=CHEAP_STOCKOUT))


def test_cost_minimising_level_follows_total_cost_per_segment():
    _, kpis = _policy_frames()
    levels = WorkingCapitalAnalyzer().cost_minimising_levels(kpis, current_sl=0.95)
    assert levels["AX"] == 0.98  # margin >> holding: more protection pays
    assert levels["CZ"] == 0.90  # shortage almost free: holding dominates
    assert levels["AY"] == 0.95  # a tie keeps the current level
    assert levels["CN"] == 0.95  # class N is held, never optimised


def test_segment_cost_optimum_table_reports_costs_and_closed_form_level():
    _, kpis = _policy_frames()
    table = WorkingCapitalAnalyzer().segment_cost_optimum(kpis, current_sl=0.95).set_index("abc_xyz_class")
    assert {"cost_minimising_sl", "capital_release_sl", "total_cost_90", "total_cost_95", "total_cost_98",
            "closed_form_sl_star", "n_skus"} <= set(table.columns)
    assert table.loc["AX", "total_cost_98"] < table.loc["AX", "total_cost_90"]
    assert table.loc["CZ", "total_cost_90"] < table.loc["CZ", "total_cost_98"]
    # Closed form: SL* = 1 - h * unit_cost * L / (365 * unit_margin); FOODS: unit_cost/margin = 2/3.
    assert table.loc["AX", "closed_form_sl_star"] == pytest.approx(1 - 0.25 * (2 / 3) * LT / 365.0, abs=1e-3)
    assert table.loc["CZ", "closed_form_sl_star"] < 0.90  # below the menu: 90% is the best on offer


def test_recommend_policy_carries_holding_stockout_and_total_cost_deltas():
    _, kpis = _policy_frames()
    recs = WorkingCapitalAnalyzer().recommend_policy(kpis, current_sl=0.95)
    for col in ("holding_cost_delta", "stockout_risk_delta", "total_cost_delta"):
        assert col in recs.columns
    valid = recs.dropna(subset=["holding_cost_delta", "stockout_risk_delta", "total_cost_delta"])
    assert np.allclose(valid["holding_cost_delta"] + valid["stockout_risk_delta"], valid["total_cost_delta"], atol=0.02)
    port = WorkingCapitalAnalyzer().portfolio_summary(recs)
    assert {"holding_cost_delta_total", "total_cost_delta_total"} <= set(port.columns)


def test_capital_release_rule_saves_capital_but_costs_more_each_year():
    """The M5-10 headline in miniature: release capital by cutting BZ to 90% and total cost rises."""
    _, kpis = _frames([("B1", "BZ", "FOODS", 2.0, 3.0, 2.0), ("B2", "BZ", "FOODS", 1.0, 4.0, 3.0)])
    cmp = WorkingCapitalAnalyzer().compare_policies(kpis, current_sl=0.95).set_index("policy")
    assert list(cmp.index) == ["uniform_current", "capital_release_rule", "cost_minimising"]

    assert cmp.loc["uniform_current", "total_cost_change"] == 0.0
    assert cmp.loc["capital_release_rule", "wc_change_vs_current"] < 0  # capital released ...
    assert cmp.loc["capital_release_rule", "total_cost_change"] > 0  # ... at a higher annual cost
    assert cmp.loc["cost_minimising", "total_cost_change"] < 0
    assert cmp.loc["cost_minimising", "total_cost_change"] < cmp.loc["capital_release_rule", "total_cost_change"]
    row = cmp.loc["capital_release_rule"]
    assert row["holding_cost_change"] + row["stockout_cost_change"] == pytest.approx(row["total_cost_change"], abs=0.05)
    assert row["total_cost_annual"] == pytest.approx(row["holding_cost_annual"] + row["stockout_cost_annual"], abs=0.05)


def test_demand_basis_comparison_separates_planning_from_reference_bases():
    _, kpis = _frames([("A", "BZ", "FOODS", 1.0, 3.0, 2.0), ("B", "BZ", "FOODS", 1.0, 3.0, 2.0)])
    key = ["item_id", "store_id"]
    skus = kpis[kpis["service_level"] == 0.95][key].drop_duplicates()
    bases = {
        "forecast": skus.assign(demand=3.0),
        "actual": skus.assign(demand=4.5),
        "full_history": skus.assign(demand=2.0),
    }
    out = WorkingCapitalAnalyzer().demand_basis_comparison(
        kpis, bases, current_sl=0.95, planning_basis="forecast"
    ).set_index("basis")
    assert out.loc["forecast", "used_for"] == "planning"
    assert out.loc["actual", "used_for"] == "reference only"
    # Safety-stock capital is identical across bases; only cycle stock moves with demand.
    assert out["safety_stock_capital"].nunique() == 1
    assert out.loc["actual", "working_capital"] > out.loc["forecast", "working_capital"] > out.loc["full_history", "working_capital"]
    assert out.loc["forecast", "working_capital"] == pytest.approx(
        kpis[kpis["service_level"] == 0.95]["working_capital_tied"].sum(), abs=0.05
    )


# --------------------------------------------------------------------------
# 2. Planning demand is the forecast, never the actuals
# --------------------------------------------------------------------------


def test_planning_demand_is_the_mean_forecast_not_the_mean_actual():
    n = 84
    preds = pd.DataFrame(
        {
            "series_id": "FOODS_1_001_CA_1",
            "date": pd.date_range("2016-02-01", periods=n),
            "y_true": np.full(n, 10.0),  # what actually sold
            "y_pred": np.full(n, 4.0),  # what the planner could know
            "model": "adida",
            "dept_id": "FOODS_1",
        }
    )
    out = SafetyStockCalculator(lead_time_days=LT, service_levels=(0.95,)).from_forecast_errors(preds, model="adida")
    assert out.iloc[0]["avg_daily_demand"] == pytest.approx(4.0)
    assert out.iloc[0]["forecast_error_bias"] == pytest.approx(6.0)  # actuals still score the error
    assert out.iloc[0]["reorder_point"] == pytest.approx(4.0 * LT + out.iloc[0]["safety_stock_units"], abs=0.02)


# --------------------------------------------------------------------------
# 3. Decisions use only pre-evaluation data
# --------------------------------------------------------------------------


def _long_panel(days: int = 3 * EVAL_WINDOW_DAYS) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=days)
    early_end = days - EVAL_WINDOW_DAYS  # first scoring day index
    sales = {
        # sold only inside the 84 days before the scoring window
        "PAST": np.where((np.arange(days) >= early_end - EVAL_WINDOW_DAYS) & (np.arange(days) < early_end), 5.0, 0.0),
        # sold only on the scoring days
        "FUTURE": np.where(np.arange(days) >= early_end, 5.0, 0.0),
        "ALWAYS": np.full(days, 5.0),
    }
    rows = [
        {"item_id": k, "store_id": "CA_1", "date": d, "sales": float(v[i])}
        for k, v in sales.items()
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


def _window_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    last = sorted(panel["date"].unique())[-EVAL_WINDOW_DAYS:]
    return pd.DataFrame({"series_id": "PAST_CA_1", "date": last, "y_true": 0.0, "y_pred": 0.0})


def test_split_puts_every_decision_day_before_every_scoring_day():
    panel = _long_panel()
    decision, scoring = split_panel_for_decisions(panel, _window_predictions(panel))
    assert decision["date"].max() < scoring["date"].min()
    assert scoring["date"].nunique() == EVAL_WINDOW_DAYS
    assert set(decision["date"]).isdisjoint(set(scoring["date"]))
    assert len(decision) + len(scoring) == len(panel)


def test_split_refuses_predictions_that_are_not_the_last_evaluation_days():
    panel = _long_panel()
    wrong = _window_predictions(panel)
    wrong["date"] = wrong["date"] - pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="evaluation window"):
        split_panel_for_decisions(panel, wrong)


def test_classes_are_decided_without_seeing_the_scored_days():
    panel = _long_panel()
    decision, _ = split_panel_for_decisions(panel, _window_predictions(panel))
    clf = ABCXYZClassifier(cv_window_days=EVAL_WINDOW_DAYS)
    result = clf.classify(decision).set_index("item_id")

    # FUTURE only sells on the scored days: decided on the earlier days it has no demand yet.
    assert result.loc["FUTURE", "xyz_class"] == "N"
    # PAST sold before the scoring window, so it is not N even though it sells nothing after.
    assert result.loc["PAST", "xyz_class"] != "N"
    # The classification states which days it looked at.
    assert (result["xyz_window_end"] == decision["date"].max().date().isoformat()).all()
    assert (pd.to_datetime(result["xyz_window_end"]) < panel["date"].max() - pd.Timedelta(days=EVAL_WINDOW_DAYS - 1)).all()
    # Classifying on the whole panel would have looked ahead and called FUTURE stable.
    leaked = clf.classify(panel).set_index("item_id")
    assert leaked.loc["FUTURE", "xyz_class"] != "N"


# --------------------------------------------------------------------------
# Committed artifacts and README
# --------------------------------------------------------------------------


def _first_prediction_date() -> pd.Timestamp:
    path = INVENTORY_DIR / "ca_predictions.parquet"
    if not path.exists():
        pytest.skip("ca_predictions.parquet not present")
    return pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"]).min()


def test_committed_segmentation_ends_the_day_before_the_first_test_day():
    abc = pd.read_csv(INVENTORY_DIR / "abc_xyz_classification.csv")
    first_test = _first_prediction_date()
    ends = pd.to_datetime(abc["xyz_window_end"].unique())
    assert len(ends) == 1
    assert ends[0] == first_test - pd.Timedelta(days=1)  # 2016-01-31
    assert (abc["n_days"] == EVAL_WINDOW_DAYS).all()
    assert (pd.to_datetime(abc["xyz_window_start"]) < ends[0]).all()


def test_committed_planning_demand_is_the_adida_mean_forecast():
    path = INVENTORY_DIR / "ca_predictions.parquet"
    if not path.exists():
        pytest.skip("ca_predictions.parquet not present")
    p = pd.read_parquet(path, columns=["model", "series_id", "y_pred"])
    p = p[p["model"] == "adida"]
    sid = p["series_id"].astype(str)
    p["item_id"] = sid.str.rsplit("_", n=2).str[0]
    p["store_id"] = sid.str.rsplit("_", n=2).str[1] + "_" + sid.str.rsplit("_", n=2).str[2]
    fc = p.groupby(["item_id", "store_id"], observed=True)["y_pred"].mean().rename("fc").reset_index()
    ss = pd.read_csv(INVENTORY_DIR / "safety_stock.csv")
    ss = ss[ss["service_level"] == 0.95].merge(fc, on=["item_id", "store_id"])
    assert len(ss) == 6098
    assert np.allclose(ss["avg_daily_demand"], ss["fc"], atol=1e-4)


@pytest.fixture(scope="module")
def policy_cmp() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_DIR / "policy_comparison.csv").set_index("policy")


def test_committed_policy_comparison_reconciles_with_the_recommendations(policy_cmp):
    assert list(policy_cmp.index) == ["uniform_current", "capital_release_rule", "cost_minimising"]
    recs = pd.read_csv(INVENTORY_DIR / "policy_recommendations.csv")
    priced = recs.dropna(subset=["total_cost_delta"])
    row = policy_cmp.loc["capital_release_rule"]
    assert row["total_cost_change"] == pytest.approx(priced["total_cost_delta"].sum(), abs=2.0)
    assert row["wc_change_vs_current"] == pytest.approx(-priced["capital_released"].sum(), abs=2.0)
    assert row["n_skus"] == len(priced)
    assert policy_cmp.loc["cost_minimising", "total_cost_change"] <= row["total_cost_change"]


def test_committed_segment_optimum_picks_from_the_evaluated_menu():
    seg = pd.read_csv(INVENTORY_DIR / "segment_cost_optimum.csv")
    assert set(seg["cost_minimising_sl"]) <= {0.90, 0.95, 0.98}
    assert (seg["closed_form_sl_star"].between(0.0, 1.0)).all()


def _money(x: float) -> str:
    return f"+${x:,.0f}" if x > 0 else (f"-${abs(x):,.0f}" if x < 0 else "$0")


def test_readme_never_shows_capital_released_without_the_annual_cost_change(policy_cmp):
    text = README_PATH.read_text(encoding="utf-8")
    section = text[text.index("## Inventory Decision Layer"):]
    release = policy_cmp.loc["capital_release_rule"]

    flagged = [ln for ln in section.splitlines() if "Net Working Capital Released" in ln]
    assert flagged, "headline row missing"
    for line in flagged:
        assert re.search(r"/yr", line), f"capital released shown without annual cost: {line[:120]}"
        assert _money(release["total_cost_change"]) in line

    # The side-by-side table is rendered from policy_comparison.csv.
    assert "<!-- BEGIN_POLICY_COMPARISON_TABLE -->" in section
    assert "<!-- BEGIN_DEMAND_BASIS_TABLE -->" in section
    assert _money(release["total_cost_change"]) in section
    assert _money(policy_cmp.loc["cost_minimising", "total_cost_change"]) in section


def test_readme_explains_the_demand_bases_and_labels_sigma_e_honestly():
    text = README_PATH.read_text(encoding="utf-8")
    section = text[text.index("## Inventory Decision Layer"):]
    assert "forecast-error size measured on the backtest days" in section
    assert "full-history mean" in section
    assert "about 4% weight" in section
    assert "look-ahead" in section
    assert "2016-01-31" in section


def test_readme_prose_numbers_match_the_artifacts(policy_cmp):
    """The render check covers the tables; every number quoted in the sentences is tied here."""
    text = README_PATH.read_text(encoding="utf-8")
    section = text[text.index("## Inventory Decision Layer"):]
    rule, cm = policy_cmp.loc["capital_release_rule"], policy_cmp.loc["cost_minimising"]

    assert f"frees ${-rule['wc_change_vs_current']:,.0f} of working capital" in section
    assert f"**{_money(rule['total_cost_change'])}/yr**" in section
    assert f"Holding cost falls by ${-rule['holding_cost_change']:,.0f}/yr" in section
    assert f"expected stockout cost rises by ${rule['stockout_cost_change']:,.0f}/yr" in section
    assert f"**{_money(cm['total_cost_change'])}/yr**" in section
    assert f"**{_money(cm['wc_change_vs_current'])}** of working capital" in section

    seg = pd.read_csv(INVENTORY_DIR / "segment_cost_optimum.csv")
    optimised = seg[~seg["abc_xyz_class"].str.endswith("N")]
    assert set(optimised["cost_minimising_sl"]) == {0.98}
    assert "98% in every optimised segment" in section
    lo, hi = optimised["closed_form_sl_star"].min() * 100, optimised["closed_form_sl_star"].max() * 100
    assert f"is {lo:.1f}% to {hi:.1f}%" in section

    b = pd.read_csv(INVENTORY_DIR / "working_capital_demand_bases.csv").set_index("basis")
    wc = b["working_capital"]
    assert f"Current working capital is ${wc['forecast']:,.0f}" in section
    assert f"The earlier **${wc['full_history_mean']:,.0f}**" in section
    assert f"would give ${wc['actual_backtest_days']:,.0f}" in section
    assert f"moves capital by only ${wc['actual_backtest_days'] - wc['forecast']:,.0f}" in section
    d = b["mean_daily_demand_total"]
    assert f"({d['full_history_mean']:,.0f} to {d['forecast']:,.0f} units a day" in section
    assert f"Safety-stock capital is identical (${b['safety_stock_capital'].iloc[0]:,.0f})" in section
    assert abs(d["actual_backtest_days"] / d["forecast"] - 1) < 0.002  # "within about 0.1%"
    assert f"## Why working capital moved from ${wc['full_history_mean']:,.0f} to ${wc['forecast']:,.0f}" in section.replace("####", "##")


def test_guides_quote_the_current_headline_and_the_worked_example_matches_the_rows(policy_cmp):
    guide = PROJECT_ROOT / "docs" / "guide"
    rule = policy_cmp.loc["capital_release_rule"]
    for name in ("00_START_HERE.md", "02_ARCHITECTURE.md", "04_METHODOLOGY.md", "09_RESULTS_AND_LIMITATIONS.md"):
        text = (guide / name).read_text(encoding="utf-8")
        assert "3,223" not in text, f"{name} still quotes the retired headline"
        assert f"{_money(rule['total_cost_change'])}/yr" in text, name
        assert f"${-rule['wc_change_vs_current']:,.0f}" in text, name

    # The architecture guide's worked example is a trace of FOODS_1_001_CA_1: tie its numbers.
    arch = (guide / "02_ARCHITECTURE.md").read_text(encoding="utf-8")
    def key(d: pd.DataFrame) -> pd.DataFrame:
        return d[(d["item_id"] == "FOODS_1_001") & (d["store_id"] == "CA_1")]

    abc = key(pd.read_csv(INVENTORY_DIR / "abc_xyz_classification.csv")).iloc[0]
    ss = key(pd.read_csv(INVENTORY_DIR / "safety_stock.csv"))
    ss95 = ss[ss["service_level"] == 0.95].iloc[0]
    k = key(pd.read_csv(INVENTORY_DIR / "inventory_kpis.csv")).set_index("service_level")
    assert abc["abc_xyz_class"] == "BZ"
    assert f"CV = {abc['cv']:.2f}" in arch
    assert f"{abc['xyz_window_start']} to {abc['xyz_window_end']}" in arch
    assert f"{ss95['avg_daily_demand']:.4f} units per day" in arch
    assert f"RMSE $\\sigma_e = {ss95['forecast_error_rmse']:.4f}" in arch
    assert f"\\approx {ss95['reorder_point']:.2f} \\text{{ units}}" in arch
    assert f"\\approx {ss95['safety_stock_units']:.2f} \\text{{ units}}" in arch
    assert f"\\${k.loc[0.95, 'stockout_cost_annual']:.2f}$ a year" in arch
    assert f"shortage rises to \\${k.loc[0.90, 'stockout_cost_annual']:.2f}" in arch
    tot = k["holding_cost_annual"] + k["stockout_cost_annual"]
    assert f"from \\${tot[0.95]:.2f} to \\${tot[0.90]:.2f} a year" in arch


def test_readme_n_class_claim_matches_the_predictions():
    path = INVENTORY_DIR / "ca_predictions.parquet"
    if not path.exists():
        pytest.skip("ca_predictions.parquet not present")
    abc = pd.read_csv(INVENTORY_DIR / "abc_xyz_classification.csv")
    p = pd.read_parquet(path, columns=["model", "series_id", "y_true"])
    p = p[p["model"] == "adida"]
    sid = p["series_id"].astype(str)
    p["item_id"] = sid.str.rsplit("_", n=2).str[0]
    p["store_id"] = sid.str.rsplit("_", n=2).str[1] + "_" + sid.str.rsplit("_", n=2).str[2]
    sold = p.groupby(["item_id", "store_id"], observed=True)["y_true"].sum().gt(0).rename("sold").reset_index()
    n_items = abc[abc["xyz_class"] == "N"].merge(sold, on=["item_id", "store_id"])
    text = README_PATH.read_text(encoding="utf-8")
    assert f"{len(n_items):,} SKUs are class N, and {int(n_items['sold'].sum()):,} of them sold during the scored days" in text


def test_demand_basis_artifact_shows_the_jump_is_the_demand_level_not_look_ahead():
    bases = pd.read_csv(INVENTORY_DIR / "working_capital_demand_bases.csv").set_index("basis")
    assert set(bases["used_for"]) == {"planning", "reference only"}
    planning = bases[bases["used_for"] == "planning"]
    assert len(planning) == 1
    # ADIDA's portfolio-level forecast tracks the recent level, so planning and actuals agree closely,
    # and both sit well above the full-history mean (sales trend upward).
    actual = bases.loc["actual_backtest_days", "working_capital"]
    assert planning["working_capital"].iloc[0] == pytest.approx(actual, rel=0.02)
    assert actual > bases.loc["full_history_mean", "working_capital"] * 1.05
    port = pd.read_csv(INVENTORY_DIR / "portfolio_summary.csv")
    assert planning["working_capital"].iloc[0] == pytest.approx(port["current_wc_total"].sum(), abs=2.0)
