"""ABC/XYZ dual-axis SKU classification for inventory segmentation.

The problem
-----------
A retailer cannot apply one inventory policy to 30,490 SKUs.  High-revenue,
stable-demand items need different service levels than low-revenue, erratic
ones.  The standard approach is a two-dimensional classification:

    ABC  -- financial importance (revenue contribution)
    XYZ  -- demand predictability (coefficient of variation)

The intersection produces a 3x3 matrix (AX, AY, AZ, ... CZ) that maps
directly to differentiated inventory policies.

ABC method
----------
Sort SKUs by descending revenue.  Cumulate.
    A = top 80% of cumulative revenue  (typically ~20% of SKUs)
    B = next 15%                       (typically ~30% of SKUs)
    C = bottom 5%                      (typically ~50% of SKUs)

The thresholds are configurable but default to the textbook Pareto split.

XYZ method
----------
Compute the coefficient of variation (CV = sigma / mu) of unit sales over a
trailing window.
    X = CV < 0.5   -- stable, forecastable demand
    Y = 0.5 <= CV < 1.0  -- variable demand
    Z = CV >= 1.0  -- erratic, hard to forecast
    N = no sales at all in the window -- "no recent demand"

An item with no sales has std = 0 and mean = 0, so CV = std / mean is undefined,
not 0.  Giving it CV = 0 would label it "perfectly stable" (X), which is the
opposite of what is known about it (M5-09).  Such items get their own class N
and an undefined CV.  Zero sales can also be a stock-out (censored demand)
rather than absent demand, so N means "review", not "discontinue".

SBC cross-reference
--------------------
The existing SBC classification (intermittent.py) segments by *demand pattern*
(inter-demand interval + size variability).  ABC/XYZ segments by *business
value + predictability*.  The two axes are complementary.  This module reports
the SBC distribution within each ABC/XYZ cell so the user can verify that
AZ items (high-revenue, erratic) overlap with SBC LUMPY/INTERMITTENT as
expected.

Data constraint
---------------
ABC requires sell_price × sales = revenue.  In the supplied M5 workbook,
prices cover CA_1 and CA_2 only (15.3% of the full price table).  For stores
without price data, only XYZ classification is computed; ABC defaults to None.

References
----------
Silver, Pyke & Thomas (2017), "Inventory and Production Management in Supply
    Chains", 4th ed., CRC Press -- Chapter 3: ABC analysis.
Syntetos, Boylan & Croston (2005), JORS 56(5):495-503 -- SBC quadrants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds.  These are not magic numbers; they are the standard
# textbook values.  Making them explicit and configurable rather than buried
# in code lets the results report cite them.
# ---------------------------------------------------------------------------
DEFAULT_ABC_THRESHOLDS: tuple[float, float] = (0.80, 0.95)
DEFAULT_XYZ_THRESHOLDS: tuple[float, float] = (0.50, 1.00)
DEFAULT_CV_WINDOW_DAYS: int = 90

# XYZ label for items with no sales in the CV window (CV undefined, not "stable").
NO_DEMAND_CLASS: str = "N"

_EPS = 1e-12


@dataclass(frozen=True)
class ClassificationThresholds:
    """Auditable record of the cut-offs used in a classification run."""

    abc: tuple[float, float]
    xyz: tuple[float, float]
    cv_window_days: int


def _validate_thresholds(
    abc: tuple[float, float], xyz: tuple[float, float]
) -> None:
    """Raise on nonsensical threshold values."""
    if not (0.0 < abc[0] < abc[1] <= 1.0):
        raise ValueError(
            f"ABC thresholds must satisfy 0 < t1 < t2 <= 1, got {abc}"
        )
    if not (0.0 < xyz[0] < xyz[1]):
        raise ValueError(
            f"XYZ thresholds must satisfy 0 < t1 < t2, got {xyz}"
        )


class ABCXYZClassifier:
    """Dual-axis SKU classification: ABC (revenue) × XYZ (demand CV).

    Usage
    -----
    >>> clf = ABCXYZClassifier()
    >>> result = clf.classify(panel_df, prices_df=prices_df)
    >>> result[result["abc_xyz_class"] == "AX"]  # high-value, stable SKUs

    Parameters
    ----------
    abc_thresholds : tuple[float, float]
        Cumulative revenue percentages for A and A+B boundaries.
        Default ``(0.80, 0.95)`` means A = top 80%, B = next 15%, C = bottom 5%.
    xyz_thresholds : tuple[float, float]
        CV cut-offs for X and Y.
        Default ``(0.50, 1.00)`` means X: CV < 0.5, Y: 0.5-1.0, Z: >= 1.0.
    cv_window_days : int
        Trailing window for CV calculation: exactly this many days ending at
        the last date in the panel.  Default 90 days.  The inventory pipeline
        sets it to the backtest evaluation window so that CV, mean demand and
        the forecast-error estimate all describe the same days.
    """

    def __init__(
        self,
        abc_thresholds: tuple[float, float] = DEFAULT_ABC_THRESHOLDS,
        xyz_thresholds: tuple[float, float] = DEFAULT_XYZ_THRESHOLDS,
        cv_window_days: int = DEFAULT_CV_WINDOW_DAYS,
    ) -> None:
        _validate_thresholds(abc_thresholds, xyz_thresholds)
        self.abc_thresholds = abc_thresholds
        self.xyz_thresholds = xyz_thresholds
        self.cv_window_days = int(cv_window_days)
        self.thresholds_ = ClassificationThresholds(
            abc=abc_thresholds, xyz=xyz_thresholds, cv_window_days=self.cv_window_days
        )

    # ------------------------------------------------------------------
    # XYZ: works on unit sales alone -- all 10 stores
    # ------------------------------------------------------------------

    def _compute_xyz(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Compute coefficient of variation per (item_id, store_id).

        Uses the last ``cv_window_days`` days of the panel (exactly that many:
        the window is ``(max_date - cv_window_days, max_date]``).  Items with
        no sales in the window get class ``N`` and an undefined (NaN) CV.
        """
        df = panel.copy()
        max_date = df["date"].max()
        window_start = max_date - pd.Timedelta(days=self.cv_window_days)
        df_window = df[df["date"] > window_start]

        if len(df_window) == 0:
            log.warning(
                "cv_window_empty",
                cv_window_days=self.cv_window_days,
                max_date=str(max_date),
            )
            df_window = df

        stats = (
            df_window.groupby(["item_id", "store_id"], observed=True)["sales"]
            .agg(mean_demand="mean", std_demand="std", n_days="count")
            .reset_index()
        )
        stats["std_demand"] = stats["std_demand"].fillna(0.0)

        # No sales in the window: CV = 0 / 0 is undefined, not "perfectly stable".
        no_demand = stats["mean_demand"] <= 0
        stats["cv"] = np.where(
            no_demand, np.nan, stats["std_demand"] / (stats["mean_demand"] + _EPS)
        )

        # XYZ assignment (plain strings: N is not one of the three CV bins)
        xyz = pd.cut(
            stats["cv"],
            bins=[-np.inf, self.xyz_thresholds[0], self.xyz_thresholds[1], np.inf],
            labels=["X", "Y", "Z"],
        ).astype(object)
        stats["xyz_class"] = np.where(no_demand, NO_DEMAND_CLASS, xyz)

        # Self-describing artifact: which days the XYZ columns were computed on.
        stats["xyz_window_start"] = df_window["date"].min().date().isoformat()
        stats["xyz_window_end"] = df_window["date"].max().date().isoformat()

        log.info(
            "xyz_computed",
            n_series=len(stats),
            x_count=int((stats["xyz_class"] == "X").sum()),
            y_count=int((stats["xyz_class"] == "Y").sum()),
            z_count=int((stats["xyz_class"] == "Z").sum()),
            n_count=int((stats["xyz_class"] == NO_DEMAND_CLASS).sum()),
            cv_median=round(float(stats["cv"].median()), 3),
        )
        return stats

    # ------------------------------------------------------------------
    # ABC: requires sell_price -- CA_1 and CA_2 only
    # ------------------------------------------------------------------

    def _compute_abc(
        self, panel: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute ABC classification from revenue contribution.

        Revenue = sell_price × daily_sales, aggregated over the full training
        window.  The price table is weekly (keyed by wm_yr_wk); we merge via
        the calendar's wm_yr_wk column which must already be in the panel.
        """
        # If panel already contains sell_price, use it directly without re-merging
        if "sell_price" in panel.columns and panel["sell_price"].notna().any():
            df = panel
        else:
            # Merge prices onto the panel
            merge_keys = ["store_id", "item_id", "wm_yr_wk"]
            if "wm_yr_wk" not in panel.columns:
                raise ValueError(
                    "Panel must contain 'wm_yr_wk' for price joining.  "
                    "Load with with_calendar=True."
                )

            df = panel.merge(
                prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
                on=merge_keys,
                how="left",
            )

        # Restrict to rows with actual price data
        has_price = df["sell_price"].notna()
        if has_price.sum() == 0:
            log.warning("no_price_data", msg="No sell_price matches found; ABC skipped.")
            return pd.DataFrame()

        df_priced = df[has_price].copy()
        df_priced["revenue"] = df_priced["sales"] * df_priced["sell_price"]

        # Aggregate revenue per SKU
        sku_revenue = (
            df_priced.groupby(["item_id", "store_id"], observed=True)["revenue"]
            .sum()
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

        total_revenue = sku_revenue["revenue"].sum()
        if total_revenue <= 0:
            log.warning("zero_revenue", msg="Total revenue is zero; ABC skipped.")
            return pd.DataFrame()

        sku_revenue["revenue_share"] = sku_revenue["revenue"] / total_revenue
        sku_revenue["cumulative_revenue_pct"] = sku_revenue["revenue_share"].cumsum()

        # ABC assignment: items that bring cumulative revenue up to and across
        # each boundary are included in that tier (e.g. top item is always A even
        # if it alone accounts for >= 80% of revenue).
        prev_cum = sku_revenue["cumulative_revenue_pct"].shift(1, fill_value=0.0)
        conditions = [
            prev_cum < self.abc_thresholds[0],
            prev_cum < self.abc_thresholds[1],
        ]
        choices = ["A", "B"]
        sku_revenue["abc_class"] = np.select(conditions, choices, default="C")

        log.info(
            "abc_computed",
            n_skus=len(sku_revenue),
            a_count=int((sku_revenue["abc_class"] == "A").sum()),
            b_count=int((sku_revenue["abc_class"] == "B").sum()),
            c_count=int((sku_revenue["abc_class"] == "C").sum()),
            total_revenue=round(float(total_revenue), 2),
        )
        return sku_revenue

    # ------------------------------------------------------------------
    # Combined classification
    # ------------------------------------------------------------------

    def classify(
        self,
        panel: pd.DataFrame,
        prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run dual-axis ABC/XYZ classification.

        Parameters
        ----------
        panel : pd.DataFrame
            Long-format panel with at least ``item_id``, ``store_id``,
            ``date``, ``sales``.  Must contain ``wm_yr_wk`` if prices are
            provided.
        prices : pd.DataFrame | None
            Weekly sell prices with ``store_id``, ``item_id``, ``wm_yr_wk``,
            ``sell_price``.  If None, only XYZ is computed.

        Returns
        -------
        pd.DataFrame
            One row per (item_id, store_id) with columns:
            ``abc_class``, ``xyz_class`` (X, Y, Z, or N for no sales in the
            window), ``abc_xyz_class``, ``revenue_share``,
            ``cumulative_revenue_pct``, ``cv`` (NaN for class N),
            ``mean_demand``, ``std_demand``, ``xyz_window_start`` and
            ``xyz_window_end`` (the days the XYZ columns describe).  ABC revenue
            uses the whole panel passed in; the XYZ columns use only its
            trailing CV window.  Pass a panel that ends *before* the days a
            policy will be scored on, so the class is decided without them.
        """
        required = {"item_id", "store_id", "date", "sales"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"Panel is missing columns: {missing}")

        xyz_df = self._compute_xyz(panel)

        if prices is not None and len(prices) > 0:
            abc_df = self._compute_abc(panel, prices)
            if len(abc_df) > 0:
                result = xyz_df.merge(
                    abc_df[["item_id", "store_id", "abc_class", "revenue_share",
                            "cumulative_revenue_pct", "revenue"]],
                    on=["item_id", "store_id"],
                    how="left",
                )
            else:
                result = xyz_df.copy()
                result["abc_class"] = None
                result["revenue_share"] = np.nan
                result["cumulative_revenue_pct"] = np.nan
                result["revenue"] = np.nan
        else:
            result = xyz_df.copy()
            result["abc_class"] = None
            result["revenue_share"] = np.nan
            result["cumulative_revenue_pct"] = np.nan
            result["revenue"] = np.nan
            log.info("abc_skipped", reason="No price data provided")

        # Combined label
        result["abc_xyz_class"] = (
            result["abc_class"].fillna("?").astype(str)
            + result["xyz_class"].astype(str)
        )

        # Summary
        log.info(
            "classification_complete",
            n_series=len(result),
            classes=result["abc_xyz_class"].value_counts().to_dict(),
            thresholds=str(self.thresholds_),
        )
        return result

    # ------------------------------------------------------------------
    # SBC cross-reference
    # ------------------------------------------------------------------

    @staticmethod
    def cross_reference_sbc(
        abc_xyz_df: pd.DataFrame, sbc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge ABC/XYZ with SBC demand-pattern labels for validation.

        Parameters
        ----------
        abc_xyz_df : pd.DataFrame
            Output of :meth:`classify`.
        sbc_df : pd.DataFrame
            Must contain ``item_id``, ``store_id``, ``demand_pattern``.

        Returns
        -------
        pd.DataFrame
            Contingency table: rows = ABC/XYZ class, columns = SBC pattern,
            values = SKU count.
        """
        merged = abc_xyz_df.merge(
            sbc_df[["item_id", "store_id", "demand_pattern"]],
            on=["item_id", "store_id"],
            how="inner",
        )
        contingency = pd.crosstab(
            merged["abc_xyz_class"], merged["demand_pattern"], margins=True
        )
        log.info(
            "sbc_cross_reference",
            matched=len(merged),
            unmatched=len(abc_xyz_df) - len(merged),
        )
        return contingency
