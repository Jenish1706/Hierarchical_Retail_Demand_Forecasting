"""
M5-15: how far the service-level conclusion moves with the assumed gross margin.

The break-even margins come from the cost model's own formula, and every number
the README quotes about them is rendered from the saved CSVs.  The module is
also checked against the pipeline's ``ServiceLevelAnalyzer`` so the closed form
cannot drift from the implementation it describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from src.inventory.margin_sensitivity import (
    DEFAULT_MENU,
    best_menu_level,
    break_even_margin,
    break_even_table,
    category_sl_star_range,
    margin_sensitivity,
    relative_cost,
    sl_star,
    sl_star_equals,
)
from src.inventory.safety_stock import _z_score
from src.inventory.service_level import (
    DEFAULT_COST_MARGINS,
    DEFAULT_HOLDING_COST_RATE,
    ServiceLevelAnalyzer,
    normal_loss,
)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = ROOT / "reports" / "inventory"
H = DEFAULT_HOLDING_COST_RATE
L = 7


def test_relative_cost_matches_the_formula_by_hand() -> None:
    # f = h (1-g) z + (365/L) g G(z) at g = 0.5, SL = 95%
    z = float(ndtri(0.95))
    expected = H * 0.5 * z + (365 / L) * 0.5 * normal_loss(z)
    assert relative_cost(0.5, 0.95, H, L) == pytest.approx(expected)
    # hand-evaluated: z = 1.6449, G = phi(z) - z (1 - Phi(z)) = 0.10314 - 1.6449 * 0.05 = 0.020895
    assert relative_cost(0.5, 0.95, H, L) == pytest.approx(0.25 * 0.5 * 1.6449 + 52.1429 * 0.5 * 0.020895, rel=1e-3)


def test_sl_star_is_the_closed_form_and_ignores_price() -> None:
    # SL* = 1 - h c L / (365 m); with c = p (1-g), m = p g price cancels.
    g = 0.60
    assert sl_star(g, H, L) == pytest.approx(1 - H * (1 - g) * L / (365 * g))
    c, m = 3.0 * (1 - g), 3.0 * g
    assert sl_star(g, H, L) == pytest.approx(1 - H * c * L / (365 * m))


def test_break_even_is_where_two_levels_cost_the_same() -> None:
    for lo, hi in ((0.90, 0.95), (0.95, 0.98)):
        g = break_even_margin(lo, hi, H, L)
        assert 0.0 < g < 1.0
        assert relative_cost(g, lo, H, L) == pytest.approx(relative_cost(g, hi, H, L), rel=1e-9)
        # Above the break-even the higher level is cheaper; below it the lower one is.
        assert relative_cost(g + 0.02, hi, H, L) < relative_cost(g + 0.02, lo, H, L)
        assert relative_cost(g - 0.02, hi, H, L) > relative_cost(g - 0.02, lo, H, L)


def test_best_menu_level_changes_exactly_at_the_break_evens() -> None:
    b95_90 = break_even_margin(0.90, 0.95, H, L)
    b98_95 = break_even_margin(0.95, 0.98, H, L)
    assert b95_90 < b98_95  # convex cost: higher levels win only at higher margins
    assert best_menu_level(b95_90 - 0.01, DEFAULT_MENU, H, L) == 0.90
    assert best_menu_level(b95_90 + 0.01, DEFAULT_MENU, H, L) == 0.95
    assert best_menu_level(b98_95 - 0.01, DEFAULT_MENU, H, L) == 0.95
    assert best_menu_level(b98_95 + 0.01, DEFAULT_MENU, H, L) == 0.98


def test_sl_star_equals_inverts_sl_star() -> None:
    for sl in (0.95, 0.98):
        assert sl_star(sl_star_equals(sl, H, L), H, L) == pytest.approx(sl)


def test_sl_star_rule_of_thumb_is_looser_than_the_menu_break_even() -> None:
    """SL* reaching 98% needs a higher margin than 98% being the cheapest menu level."""
    assert sl_star_equals(0.98, H, L) > break_even_margin(0.95, 0.98, H, L)
    assert sl_star_equals(0.95, H, L) > break_even_margin(0.90, 0.95, H, L)


@pytest.mark.parametrize("g", [0.04, 0.10, 0.30, 0.60])
def test_closed_form_agrees_with_the_pipeline_analyzer(g: float) -> None:
    """The cheapest menu level the analyzer finds equals the one the closed form names."""
    rmse, dem, price = 1.7, 2.0, 3.0
    rows = [
        {
            "item_id": "X",
            "store_id": "CA_1",
            "service_level": sl,
            "lead_time_days": L,
            "avg_daily_demand": dem,
            "safety_stock_units": round(_z_score(sl) * rmse * np.sqrt(L), 2),
            "forecast_error_rmse": rmse,
            "avg_sell_price": price,
            "cat_id": "T",
        }
        for sl in DEFAULT_MENU
    ]
    abc = pd.DataFrame(
        [{"item_id": "X", "store_id": "CA_1", "abc_xyz_class": "AX", "abc_class": "A", "xyz_class": "X"}]
    )
    scen = ServiceLevelAnalyzer(cost_margin_assumptions={"T": g}).analyze(pd.DataFrame(rows), abc)
    picked = float(scen.loc[scen["total_relevant_cost"].idxmin(), "service_level"])
    assert picked == best_menu_level(g, DEFAULT_MENU, H, L)
    # And the analyzer's own unconstrained optimum is the closed form.
    assert float(scen["unconstrained_optimal_sl"].iloc[0]) == pytest.approx(sl_star(g, H, L), abs=1e-4)


def test_sensitivity_table_covers_the_category_margins_and_the_grid() -> None:
    df = margin_sensitivity()
    for cat, g in DEFAULT_COST_MARGINS.items():
        row = df[df["scenario"] == cat]
        assert len(row) == 1
        assert row["gross_margin"].iloc[0] == pytest.approx(g)
    assert (df["scenario"] == "grid").sum() >= 5
    assert df["gross_margin"].is_monotonic_increasing
    # The cheapest level has zero extra cost and every other level costs more.
    for _, r in df.iterrows():
        extras = {sl: r[f"extra_cost_{round(sl * 100)}_pct"] for sl in DEFAULT_MENU}
        assert extras[r["best_menu_level"]] == pytest.approx(0.0, abs=1e-9)
        assert all(v >= -1e-9 for v in extras.values())


def test_category_sl_star_range_is_per_category_not_hand_typed() -> None:
    lo, hi = category_sl_star_range()
    per_cat = [sl_star(g, H, L) for g in DEFAULT_COST_MARGINS.values()]
    assert lo == pytest.approx(min(per_cat))
    assert hi == pytest.approx(max(per_cat))
    # HOBBIES at 45% is the lowest, FOODS at 60% the highest, and the low end is below 99.5%.
    assert lo == pytest.approx(sl_star(DEFAULT_COST_MARGINS["HOBBIES"], H, L))
    assert lo < 0.995


def test_break_even_table_has_all_four_thresholds() -> None:
    t = break_even_table()
    assert set(t["threshold"]) == {
        "cheapest_95_over_90",
        "cheapest_98_over_95",
        "sl_star_reaches_98",
        "sl_star_reaches_95",
    }
    assert t["gross_margin"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Committed artifacts and the README
# ---------------------------------------------------------------------------


def test_committed_artifacts_match_the_module() -> None:
    sens = pd.read_csv(INVENTORY_DIR / "margin_sensitivity.csv")
    be = pd.read_csv(INVENTORY_DIR / "margin_break_even.csv")
    pd.testing.assert_frame_equal(
        margin_sensitivity().reset_index(drop=True), sens, check_dtype=False, rtol=1e-9
    )
    pd.testing.assert_frame_equal(break_even_table().reset_index(drop=True), be, check_dtype=False, rtol=1e-9)


def test_grocery_margins_keep_98_percent_the_cheapest_menu_level() -> None:
    """The point of the sensitivity: realistic 25-30% grocery margins do not change the answer."""
    sens = pd.read_csv(INVENTORY_DIR / "margin_sensitivity.csv")
    grocery = sens[(sens["scenario"] == "grid") & sens["gross_margin"].between(0.25, 0.30)]
    assert len(grocery) == 2
    assert (grocery["best_menu_level"] == 0.98).all()
