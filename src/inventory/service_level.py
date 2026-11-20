"""Service-level scenario analysis: cost trade-offs at 90%, 95%, and 98%.

The question
------------
Moving from 95% to 98% service level sounds small, but it can double safety
stock for erratic items.  This module quantifies the *cost* of each service
level so the business can make an informed trade-off rather than setting a
blanket target.

For each (SKU segment, service level) combination it computes:

    Safety Stock (units)      -- from SafetyStockCalculator
    Average Inventory (units) -- avg_demand × L/2 + safety_stock
    Holding Cost              -- avg_inventory × unit_cost × holding_rate
    Expected Stockout Cost    -- (1 - SL) × avg_demand × L × unit_margin
    Total Relevant Cost       -- holding + stockout

The key deliverable is a **scenario comparison table** per ABC/XYZ segment
showing the trade-off curve, plus a marginal analysis showing the incremental
cost of raising the service level by one tier.

Assumptions and limitations
---------------------------
1. M5 has no actual cost-of-goods data.  Unit cost is proxied as
   ``sell_price × (1 - margin)``.  Margin assumptions are configurable per
   category (default: FOODS 60%, HOUSEHOLD 50%, HOBBIES 45%) and documented
   as assumptions in every output.

2. The holding cost rate (default 25% per annum) covers warehousing, capital
   cost, obsolescence, and insurance.  This is a standard industry estimate
   (Chopra & Meindl, Ch. 12).

3. Stockout cost is estimated as lost margin per unit.  The true cost
   (including customer churn and substitution) is higher but unobservable
   in M5.

References
----------
Chopra & Meindl (2016), "Supply Chain Management", 6th ed., Pearson --
    Chapter 12: managing uncertainty in a supply chain.
Silver, Pyke & Thomas (2017), "Inventory and Production Management in Supply
    Chains", 4th ed., CRC Press -- Chapter 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.config.logging import get_logger

log = get_logger(__name__)

DEFAULT_HOLDING_COST_RATE: float = 0.25  # 25% per annum
DEFAULT_COST_MARGINS: dict[str, float] = {
    "FOODS": 0.60,
    "HOUSEHOLD": 0.50,
    "HOBBIES": 0.45,
}

_EPS = 1e-12


class ServiceLevelAnalyzer:
    """Evaluate inventory cost trade-offs across service levels.

    Usage
    -----
    >>> analyzer = ServiceLevelAnalyzer()
    >>> scenarios = analyzer.analyze(safety_stock_df, abc_xyz_df, prices_df, panel_df)
    >>> marginals = analyzer.marginal_analysis(scenarios)

    Parameters
    ----------
    service_levels : tuple[float, ...]
        The service levels to compare.
    holding_cost_rate : float
        Annual holding cost as a fraction of unit cost.
    cost_margin_assumptions : dict[str, float] | None
        Gross margin by category (to back-calculate unit cost from sell price).
        Keys must match ``cat_id`` values in the panel.
    """

    def __init__(
        self,
        service_levels: tuple[float, ...] = (0.90, 0.95, 0.98),
        holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
        cost_margin_assumptions: dict[str, float] | None = None,
    ) -> None:
        self.service_levels = tuple(sorted(service_levels))
        self.holding_cost_rate = float(holding_cost_rate)
        self.cost_margins = cost_margin_assumptions or DEFAULT_COST_MARGINS

    def _resolve_unit_cost(
        self, row: pd.Series, prices_agg: pd.DataFrame | None
    ) -> float:
        """Back-calculate unit cost from sell price and margin assumption.

        unit_cost = sell_price × (1 - margin)
        """
        if prices_agg is None:
            return np.nan

        match = prices_agg[
            (prices_agg["item_id"] == row["item_id"])
            & (prices_agg["store_id"] == row["store_id"])
        ]
        if match.empty:
            return np.nan

        sell_price = float(match.iloc[0]["avg_sell_price"])
        cat = str(row.get("cat_id", ""))
        margin = self.cost_margins.get(cat, 0.50)  # default 50%
        return sell_price * (1.0 - margin)

    def analyze(
        self,
        safety_stock_df: pd.DataFrame,
        abc_xyz_df: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        panel: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run cost/risk scenario analysis across service levels.

        Parameters
        ----------
        safety_stock_df : pd.DataFrame
            Output of SafetyStockCalculator (any method).
        abc_xyz_df : pd.DataFrame
            Output of ABCXYZClassifier.classify().
        prices : pd.DataFrame | None
            Raw prices table for unit-cost estimation.
        panel : pd.DataFrame | None
            Panel for category lookup (``cat_id``).

        Returns
        -------
        pd.DataFrame
            One row per (item_id, store_id, service_level) with cost
            components and the ABC/XYZ class.
        """
        # Merge ABC/XYZ labels
        df = safety_stock_df.merge(
            abc_xyz_df[["item_id", "store_id", "abc_xyz_class", "abc_class", "xyz_class"]],
            on=["item_id", "store_id"],
            how="left",
        )

        # Add category for cost margin lookup
        if panel is not None and "cat_id" in panel.columns:
            cat_map = panel[["item_id", "store_id", "cat_id"]].drop_duplicates()
            df = df.merge(cat_map, on=["item_id", "store_id"], how="left")

        # Compute average sell price per SKU
        prices_agg = None
        if prices is not None and len(prices) > 0:
            prices_agg = (
                prices.groupby(["item_id", "store_id"], observed=True)["sell_price"]
                .mean()
                .reset_index()
                .rename(columns={"sell_price": "avg_sell_price"})
            )
            df = df.merge(prices_agg, on=["item_id", "store_id"], how="left")
        elif "avg_sell_price" not in df.columns:
            df["avg_sell_price"] = np.nan

        # Compute cost components
        results = []
        for _, row in df.iterrows():
            lt = row.get("lead_time_days", 7)
            avg_demand = row.get("avg_daily_demand", 0.0)
            ss = row.get("safety_stock_units", 0.0)
            sl = row.get("service_level", 0.95)

            # Average inventory = cycle stock + safety stock
            # Cycle stock ≈ (avg_demand × L) / 2 for continuous review
            cycle_stock = (avg_demand * lt) / 2.0
            avg_inventory = cycle_stock + ss

            # Unit cost from sell price and margin
            sell_price = row.get("avg_sell_price", np.nan)
            cat = row.get("cat_id", "")
            margin = self.cost_margins.get(str(cat), 0.50)

            if pd.notna(sell_price) and sell_price > 0:
                unit_cost = sell_price * (1.0 - margin)
                unit_margin = sell_price * margin

                inventory_value = avg_inventory * unit_cost
                holding_cost = inventory_value * self.holding_cost_rate

                # Same sigma_e that sized the safety stock (RMSE, bias included). Reading the
                # bias-free std here would pair a bias-inflated SS with a too-small sigma_L.
                sigma_e = row.get("forecast_error_rmse", np.nan)
                if pd.isna(ss):
                    # sigma_e was missing, so the stock could not be sized (M5-09): do not
                    # invent a stockout cost for it; it is excluded and counted downstream.
                    stockout_cost = np.nan
                else:
                    if pd.notna(sigma_e) and sigma_e <= 1e-4:
                        stockout_prob = 0.0
                    elif pd.notna(sigma_e) and sigma_e > 1e-4 and lt > 0:
                        sigma_l = sigma_e * np.sqrt(lt)
                        z_eff = max(0.0, ss / sigma_l)
                        stockout_prob = float(1.0 - ndtr(z_eff))
                    else:
                        stockout_prob = float(1.0 - sl)

                    # Annualized stockout cost: cycle stockout margin * (365 / lt)
                    # = stockout_prob * (avg_demand * lt) * unit_margin * (365 / lt)
                    # = stockout_prob * avg_demand * 365 * unit_margin
                    stockout_cost = stockout_prob * avg_demand * 365.0 * unit_margin
                total_cost = holding_cost + stockout_cost
            else:
                unit_cost = np.nan
                unit_margin = np.nan
                inventory_value = np.nan
                holding_cost = np.nan
                stockout_cost = np.nan
                total_cost = np.nan

            results.append({
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "model": row.get("model", "adida"),
                "abc_xyz_class": row.get("abc_xyz_class", "??"),
                "abc_class": row.get("abc_class"),
                "xyz_class": row.get("xyz_class"),
                "service_level": sl,
                "lead_time_days": lt,
                "avg_daily_demand": round(float(avg_demand), 4),
                "safety_stock_units": round(float(ss), 2),
                "cycle_stock_units": round(float(cycle_stock), 2),
                "avg_inventory_units": round(float(avg_inventory), 2),
                "unit_cost": round(float(unit_cost), 4) if pd.notna(unit_cost) else np.nan,
                "inventory_value": round(float(inventory_value), 2) if pd.notna(inventory_value) else np.nan,
                "holding_cost_annual": round(float(holding_cost), 2) if pd.notna(holding_cost) else np.nan,
                "stockout_cost_annual": round(float(stockout_cost), 2) if pd.notna(stockout_cost) else np.nan,
                "total_relevant_cost": round(float(total_cost), 2) if pd.notna(total_cost) else np.nan,
            })

        result = pd.DataFrame(results)
        self._log_summary(result)
        return result

    def marginal_analysis(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the marginal cost of raising service level by one tier.

        For each SKU, shows the *incremental* cost of going from 90%→95%
        and 95%→98%.  This is the key decision input: "Is the extra
        protection worth it for this segment?"

        Returns
        -------
        pd.DataFrame
            One row per (item_id, store_id, from_sl, to_sl) with
            ``delta_ss``, ``delta_holding_cost``, ``delta_stockout_cost``,
            ``delta_total_cost``, and ``cost_per_pct_point``.
        """
        sorted_sls = sorted(scenarios_df["service_level"].unique())
        if len(sorted_sls) < 2:
            log.warning("marginal_needs_multiple_sls", n_levels=len(sorted_sls))
            return pd.DataFrame()

        rows = []
        for i in range(len(sorted_sls) - 1):
            from_sl, to_sl = sorted_sls[i], sorted_sls[i + 1]
            df_from = scenarios_df[scenarios_df["service_level"] == from_sl].set_index(
                ["item_id", "store_id"]
            )
            df_to = scenarios_df[scenarios_df["service_level"] == to_sl].set_index(
                ["item_id", "store_id"]
            )

            common = df_from.index.intersection(df_to.index)
            for idx in common:
                r_from = df_from.loc[idx]
                r_to = df_to.loc[idx]

                delta_ss = r_to["safety_stock_units"] - r_from["safety_stock_units"]
                delta_holding = (
                    r_to["holding_cost_annual"] - r_from["holding_cost_annual"]
                    if pd.notna(r_to["holding_cost_annual"]) and pd.notna(r_from["holding_cost_annual"])
                    else np.nan
                )
                delta_stockout = (
                    r_to["stockout_cost_annual"] - r_from["stockout_cost_annual"]
                    if pd.notna(r_to["stockout_cost_annual"]) and pd.notna(r_from["stockout_cost_annual"])
                    else np.nan
                )
                delta_total = (
                    r_to["total_relevant_cost"] - r_from["total_relevant_cost"]
                    if pd.notna(r_to["total_relevant_cost"]) and pd.notna(r_from["total_relevant_cost"])
                    else np.nan
                )

                pct_points = (to_sl - from_sl) * 100
                cost_per_pct = delta_total / pct_points if pd.notna(delta_total) and pct_points > 0 else np.nan

                item_id = idx[0] if isinstance(idx, tuple) else idx
                store_id = idx[1] if isinstance(idx, tuple) else ""

                rows.append({
                    "item_id": item_id,
                    "store_id": store_id,
                    "abc_xyz_class": r_from.get("abc_xyz_class", "??"),
                    "from_service_level": from_sl,
                    "to_service_level": to_sl,
                    "delta_safety_stock": round(float(delta_ss), 2),
                    "delta_holding_cost": round(float(delta_holding), 2) if pd.notna(delta_holding) else np.nan,
                    "delta_stockout_cost": round(float(delta_stockout), 2) if pd.notna(delta_stockout) else np.nan,
                    "delta_total_cost": round(float(delta_total), 2) if pd.notna(delta_total) else np.nan,
                    "cost_per_pct_point": round(float(cost_per_pct), 2) if pd.notna(cost_per_pct) else np.nan,
                })

        result = pd.DataFrame(rows)
        log.info("marginal_analysis_complete", n_rows=len(result))
        return result

    def segment_summary(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate scenario results by ABC/XYZ segment.

        Produces the high-level table a business decision-maker needs:
        total inventory value, holding cost, and stockout cost per segment
        at each service level.
        """
        if "abc_xyz_class" not in scenarios_df.columns:
            log.warning("no_abc_xyz_class", msg="Cannot segment; returning empty.")
            return pd.DataFrame()

        numeric_cols = [
            "safety_stock_units", "avg_inventory_units", "inventory_value",
            "holding_cost_annual", "stockout_cost_annual", "total_relevant_cost",
        ]
        # Filter to only existing numeric columns
        agg_cols = [c for c in numeric_cols if c in scenarios_df.columns]

        summary = (
            scenarios_df.groupby(["abc_xyz_class", "service_level"], observed=True)[agg_cols]
            .agg(["sum", "mean", "count"])
        )
        # Flatten multi-level columns
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary = summary.reset_index()

        log.info("segment_summary_complete", n_segments=summary["abc_xyz_class"].nunique())
        return summary

    def _log_summary(self, df: pd.DataFrame) -> None:
        """Log scenario analysis summary."""
        if len(df) == 0:
            log.warning("scenario_empty")
            return

        for sl in self.service_levels:
            sl_df = df[df["service_level"] == sl]
            if len(sl_df) == 0:
                continue
            inv_val = sl_df["inventory_value"].sum()
            hold_cost = sl_df["holding_cost_annual"].sum()
            log.info(
                "scenario_computed",
                service_level=sl,
                n_skus=len(sl_df),
                total_inventory_value=round(float(inv_val), 2) if pd.notna(inv_val) else "N/A",
                total_holding_cost=round(float(hold_cost), 2) if pd.notna(hold_cost) else "N/A",
            )
