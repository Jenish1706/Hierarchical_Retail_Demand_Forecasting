"""
Margin sensitivity of the service-level result (M5-15).

The cost model rests on assumed gross margins (FOODS 60%, HOUSEHOLD 50%,
HOBBIES 45%), and a retail reader will challenge 60% on groceries.  This module
shows how far the conclusion moves when the margin does.

Why one number is enough
------------------------
For a SKU with price p and gross margin g, the unit cost is c = p(1 - g) and the
unit margin lost per unit short is m = p g.  Holding z standard deviations of
lead-time demand costs h·c·z·sigma_L a year; the shortage costs
sigma_L·G(z)·(365/L)·m a year (``service_level.py``).  So the annual cost,
per dollar of price and per unit of sigma_L, is

    f(z; g) = h (1 - g) z + (365 / L) g G(z)

Price and error size scale out, so which of the 90/95/98% service levels is the
cheapest depends on the margin g (and on h and L) and on nothing else.  f is
convex in z, so the cheapest menu level changes only where two *adjacent* levels
cost the same, and each such break-even margin has a closed form.

Two thresholds, easy to confuse
-------------------------------
* ``break_even_margin(lo, hi)`` is where service level ``hi`` costs the same as
  ``lo``.  Above it, ``hi`` is cheaper.  This is the exact answer to "which menu
  level does the cost model pick".
* ``sl_star_equals(sl)`` is the margin at which the *unconstrained* optimum
  SL* = 1 - h c L / (365 m) equals ``sl``.  It is a looser rule of thumb: SL*
  can lie below 98% while 98% is still the cheapest of the three menu levels,
  because SL* is the continuous optimum and the menu is coarse.  Both are
  reported so neither is mistaken for the other.

Nothing here is hand-typed: the README renders these tables from
``margin_sensitivity.csv`` and ``margin_break_even.csv``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtri

from src.inventory.safety_stock import DEFAULT_LEAD_TIME_DAYS, DEFAULT_SERVICE_LEVELS
from src.inventory.service_level import (
    DEFAULT_COST_MARGINS,
    DEFAULT_HOLDING_COST_RATE,
    normal_loss,
)

#: The pipeline's own menu, holding rate, margins and lead time, so the sensitivity cannot
#: drift from the analysis it qualifies.
DEFAULT_MENU: tuple[float, ...] = DEFAULT_SERVICE_LEVELS

#: Margins shown besides the category assumptions: the low end, where the menu
#: choice changes, and the 25-30% a grocery buyer is more likely to quote.
DEFAULT_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def relative_cost(
    gross_margin: float,
    service_level: float,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> float:
    """f(z; g) = h (1 - g) z + (365 / L) g G(z): annual cost per $ of price per unit of sigma_L."""
    z = float(ndtri(service_level))
    return float(
        holding_cost_rate * (1.0 - gross_margin) * z
        + (365.0 / lead_time_days) * gross_margin * normal_loss(z)
    )


def sl_star(
    gross_margin: float,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> float:
    """Unconstrained optimum SL* = 1 - h c L / (365 m), with c = p (1 - g) and m = p g."""
    return float(
        1.0
        - holding_cost_rate * (1.0 - gross_margin) * lead_time_days / (365.0 * gross_margin)
    )


def best_menu_level(
    gross_margin: float,
    menu: Sequence[float] = DEFAULT_MENU,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> float:
    """The cheapest service level on the menu at this margin (ties go to the lower level)."""
    costs = {sl: relative_cost(gross_margin, sl, holding_cost_rate, lead_time_days) for sl in menu}
    return min(sorted(menu), key=lambda sl: costs[sl])


def break_even_margin(
    lower: float,
    higher: float,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> float:
    """
    Gross margin at which service level ``higher`` costs the same as ``lower``.

    Setting f(z_hi; g) = f(z_lo; g) gives (1 - g) / g = r with
    r = (365 / L) (G(z_lo) - G(z_hi)) / (h (z_hi - z_lo)), so g = 1 / (1 + r).
    Above this margin the higher level is cheaper; below it the lower one is.
    """
    z_lo, z_hi = float(ndtri(lower)), float(ndtri(higher))
    ratio = (365.0 / lead_time_days) * (normal_loss(z_lo) - normal_loss(z_hi)) / (
        holding_cost_rate * (z_hi - z_lo)
    )
    return float(1.0 / (1.0 + ratio))


def sl_star_equals(
    service_level: float,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> float:
    """Gross margin at which the unconstrained optimum SL* equals ``service_level``."""
    ratio = (1.0 - service_level) * 365.0 / (holding_cost_rate * lead_time_days)
    return float(1.0 / (1.0 + ratio))


def margin_sensitivity(
    category_margins: Mapping[str, float] | None = None,
    grid: Sequence[float] = DEFAULT_GRID,
    menu: Sequence[float] = DEFAULT_MENU,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> pd.DataFrame:
    """
    The cheapest menu level across gross margins, with what each alternative costs.

    One row per margin: the category assumptions (``scenario`` set to the
    category) and the ``grid`` (``scenario`` = ``"grid"``), ordered by margin.
    ``extra_cost_<sl>_pct`` is how much more the annual cost is at that level
    than at the cheapest one (0 for the cheapest).
    """
    cats = dict(DEFAULT_COST_MARGINS if category_margins is None else category_margins)
    entries = [(float(g), "grid") for g in grid] + [(float(g), cat) for cat, g in cats.items()]
    levels = sorted(menu)

    rows = []
    for g, scenario in sorted(entries, key=lambda e: (e[0], e[1])):
        costs = {sl: relative_cost(g, sl, holding_cost_rate, lead_time_days) for sl in levels}
        best = min(levels, key=lambda sl: costs[sl])
        row: dict[str, object] = {
            "gross_margin": g,
            "scenario": scenario,
            "sl_star": sl_star(g, holding_cost_rate, lead_time_days),
            "best_menu_level": best,
        }
        for sl in levels:
            row[f"extra_cost_{round(sl * 100)}_pct"] = float(100.0 * (costs[sl] / costs[best] - 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def break_even_table(
    menu: Sequence[float] = DEFAULT_MENU,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> pd.DataFrame:
    """Break-even margins between adjacent menu levels, plus where SL* reaches each level."""
    levels = sorted(menu)
    rows = []
    for lo, hi in zip(levels[:-1], levels[1:], strict=True):
        rows.append(
            {
                "threshold": f"cheapest_{round(hi * 100)}_over_{round(lo * 100)}",
                "service_level": hi,
                "gross_margin": break_even_margin(lo, hi, holding_cost_rate, lead_time_days),
                "meaning": (
                    f"{round(hi * 100)}% costs less than {round(lo * 100)}% above this gross margin"
                ),
            }
        )
    for sl in reversed(levels[1:]):
        rows.append(
            {
                "threshold": f"sl_star_reaches_{round(sl * 100)}",
                "service_level": sl,
                "gross_margin": sl_star_equals(sl, holding_cost_rate, lead_time_days),
                "meaning": (
                    f"the unconstrained optimum SL* is at least {round(sl * 100)}% above this "
                    "gross margin (a looser rule of thumb than the menu break-even)"
                ),
            }
        )
    return pd.DataFrame(rows)


def category_sl_star_range(
    category_margins: Mapping[str, float] | None = None,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
) -> tuple[float, float]:
    """(lowest, highest) SL* across the category margin assumptions."""
    cats = DEFAULT_COST_MARGINS if category_margins is None else category_margins
    values = [sl_star(g, holding_cost_rate, lead_time_days) for g in cats.values()]
    return float(np.min(values)), float(np.max(values))
