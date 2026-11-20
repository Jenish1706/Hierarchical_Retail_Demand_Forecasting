"""Working capital impact analysis and differentiated inventory policy.

The business question
---------------------
"How much working capital can be released without materially increasing
stockout risk?"

The answer is not a single number -- it depends on *which* SKUs you reduce.
Cutting safety stock on AX items (high-value, stable) saves a lot of money
but risks losing your most profitable products.  Cutting CZ items
(low-value, erratic) saves little per SKU but is low-risk and there are
many of them.

This module aggregates safety-stock and service-level results into portfolio-
level KPIs and produces a differentiated policy recommendation:

    AX → 98% service level  (protect revenue, demand is predictable)
    AY → 95% service level  (moderate buffer)
    AZ → 95% service level  (high-value but erratic → watch closely)
    BX → 95% service level  (moderate value, predictable)
    BY → 95% service level  (baseline)
    BZ → 90% service level  (moderate value, unpredictable → reduce buffer)
    CX → 90% service level  (low value, can afford lower fill rate)
    CY → 90% service level  (low value, variable)
    CZ → 90% service level  (low value, erratic → minimal buffer)

The recommendation is transparent: it shows exactly how much capital is
released by each downgrade and how much additional stockout risk is accepted.

KPIs computed
-------------
- Inventory Turnover = Annual demand / Average inventory
- Days of Inventory (DOI) = 365 / Turnover
- Sell-Through Rate = Demand / (Demand + Average inventory)  [proxy]
- Working Capital Tied = Average inventory × unit cost
- Capital Release = WC(current_policy) - WC(recommended_policy)

References
----------
Ramanathan, R. (2006), "ABC inventory classification with multiple-criteria
    using weighted linear optimization", Computers & OR 33(3):695-700.
Scholz-Reiter et al. (2012), "An integrated model for inventory and
    lead-time management", IJPE 134(1):1-10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.logging import get_logger

log = get_logger(__name__)

# Differentiated service-level policy: abc_xyz_class -> recommended SL.
# These are starting recommendations, not commandments.  The marginal
# analysis from ServiceLevelAnalyzer lets the user override any cell.
DEFAULT_POLICY: dict[str, float] = {
    "AX": 0.98,
    "AY": 0.95,
    "AZ": 0.95,
    "BX": 0.95,
    "BY": 0.95,
    "BZ": 0.90,
    "CX": 0.90,
    "CY": 0.90,
    "CZ": 0.90,
    # Fallbacks for unknown / partial classifications
    "?X": 0.95,
    "?Y": 0.90,
    "?Z": 0.90,
}
# Classes ending in "N" (AN, BN, CN, ?N: no sales in the XYZ window) are deliberately
# absent: ``recommend_policy`` falls back to the current service level for them, so they
# are held, not differentiated, and reported as their own row (M5-09).


class WorkingCapitalAnalyzer:
    """Translate inventory scenarios into working-capital and turnover KPIs.

    Usage
    -----
    >>> wca = WorkingCapitalAnalyzer()
    >>> kpis = wca.compute_kpis(scenarios_df)
    >>> recs = wca.recommend_policy(kpis, current_sl=0.95)

    Parameters
    ----------
    cost_margin_assumptions : dict[str, float] | None
        Gross margin by category (same as ServiceLevelAnalyzer).
    policy : dict[str, float] | None
        ABC/XYZ class -> recommended service level.  Defaults to
        ``DEFAULT_POLICY``.
    """

    def __init__(
        self,
        cost_margin_assumptions: dict[str, float] | None = None,
        policy: dict[str, float] | None = None,
    ) -> None:
        self.cost_margins = cost_margin_assumptions or {
            "FOODS": 0.60,
            "HOUSEHOLD": 0.50,
            "HOBBIES": 0.45,
        }
        self.policy = policy or DEFAULT_POLICY

    def compute_kpis(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """Compute inventory performance KPIs per SKU and service level.

        Parameters
        ----------
        scenarios_df : pd.DataFrame
            Output of ServiceLevelAnalyzer.analyze().

        Returns
        -------
        pd.DataFrame
            Extended with ``inventory_turnover``, ``days_of_inventory``,
            ``sell_through_rate``, ``working_capital_tied``.
        """
        df = scenarios_df.copy()

        # Annual demand
        df["annual_demand"] = df["avg_daily_demand"] * 365.0

        # Inventory turnover
        df["inventory_turnover"] = np.where(
            df["avg_inventory_units"] > 0,
            df["annual_demand"] / df["avg_inventory_units"],
            np.nan,
        )

        # Days of inventory
        df["days_of_inventory"] = np.where(
            df["inventory_turnover"] > 0,
            365.0 / df["inventory_turnover"],
            np.nan,
        )

        # Sell-through rate (proxy: demand / (demand + avg inventory))
        denom = df["annual_demand"] + df["avg_inventory_units"]
        df["sell_through_rate"] = np.where(
            denom > 0,
            df["annual_demand"] / denom,
            np.nan,
        )

        # Working capital tied = avg inventory × unit cost
        df["working_capital_tied"] = np.where(
            pd.notna(df["unit_cost"]),
            df["avg_inventory_units"] * df["unit_cost"],
            np.nan,
        )

        # Round for readability
        for col in ["inventory_turnover", "days_of_inventory", "sell_through_rate",
                     "working_capital_tied", "annual_demand"]:
            df[col] = df[col].round(2)

        log.info("kpis_computed", n_rows=len(df))
        return df

    def recommend_policy(
        self,
        kpis_df: pd.DataFrame,
        current_sl: float = 0.95,
    ) -> pd.DataFrame:
        """Generate differentiated policy recommendations per SKU.

        Compares a uniform ``current_sl`` policy against the segment-specific
        recommended policy.  Shows how much capital is released (or consumed)
        per SKU by switching.

        Parameters
        ----------
        kpis_df : pd.DataFrame
            Output of :meth:`compute_kpis`.
        current_sl : float
            The existing blanket service level to compare against.

        Returns
        -------
        pd.DataFrame
            One row per (item_id, store_id) with:
            ``current_sl``, ``recommended_sl``, ``current_wc``,
            ``recommended_wc``, ``capital_released``, ``stockout_risk_delta``.
        """
        # Get the current-policy rows
        current = kpis_df[kpis_df["service_level"] == current_sl].copy()
        if len(current) == 0:
            log.warning(
                "current_sl_not_found",
                current_sl=current_sl,
                available=sorted(kpis_df["service_level"].unique().tolist()),
            )
            return pd.DataFrame()

        rows = []
        for _, row in current.iterrows():
            abc_xyz = str(row.get("abc_xyz_class", "??"))
            rec_sl = self.policy.get(abc_xyz, current_sl)
            if pd.isna(row["safety_stock_units"]):
                # sigma_e missing: the stock was never sized, so no recommendation is made.
                rec_sl = np.nan

            # Find the recommended-SL row for this SKU
            rec_row = kpis_df[
                (kpis_df["item_id"] == row["item_id"])
                & (kpis_df["store_id"] == row["store_id"])
                & (kpis_df["service_level"] == rec_sl)
            ]

            if len(rec_row) == 0:
                # Recommended SL not computed; keep current
                rec_wc = row["working_capital_tied"]
                rec_ss = row["safety_stock_units"]
                rec_stockout = row.get("stockout_cost_annual", np.nan)
                rec_holding = row.get("holding_cost_annual", np.nan)
                rec_total = row.get("total_relevant_cost", np.nan)
            else:
                rec_row = rec_row.iloc[0]
                rec_wc = rec_row["working_capital_tied"]
                rec_ss = rec_row["safety_stock_units"]
                rec_stockout = rec_row.get("stockout_cost_annual", np.nan)
                rec_holding = rec_row.get("holding_cost_annual", np.nan)
                rec_total = rec_row.get("total_relevant_cost", np.nan)

            current_wc = row["working_capital_tied"]
            current_stockout = row.get("stockout_cost_annual", np.nan)
            current_holding = row.get("holding_cost_annual", np.nan)
            current_total = row.get("total_relevant_cost", np.nan)

            holding_cost_delta = (
                (rec_holding - current_holding)
                if pd.notna(rec_holding) and pd.notna(current_holding)
                else np.nan
            )
            total_cost_delta = (
                (rec_total - current_total)
                if pd.notna(rec_total) and pd.notna(current_total)
                else np.nan
            )

            capital_released = (
                (current_wc - rec_wc)
                if pd.notna(current_wc) and pd.notna(rec_wc)
                else np.nan
            )

            stockout_risk_delta = (
                (rec_stockout - current_stockout)
                if pd.notna(rec_stockout) and pd.notna(current_stockout)
                else np.nan
            )

            rows.append({
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "abc_xyz_class": abc_xyz,
                "current_service_level": current_sl,
                "recommended_service_level": rec_sl,
                "policy_change": "excluded" if pd.isna(rec_sl) else (
                    "downgrade" if rec_sl < current_sl else (
                        "upgrade" if rec_sl > current_sl else "maintain"
                    )
                ),
                "current_safety_stock": round(float(row["safety_stock_units"]), 2),
                "recommended_safety_stock": round(float(rec_ss), 2),
                "current_working_capital": round(float(current_wc), 2) if pd.notna(current_wc) else np.nan,
                "recommended_working_capital": round(float(rec_wc), 2) if pd.notna(rec_wc) else np.nan,
                "capital_released": round(float(capital_released), 2) if pd.notna(capital_released) else np.nan,
                "stockout_risk_delta": round(float(stockout_risk_delta), 2) if pd.notna(stockout_risk_delta) else np.nan,
                "holding_cost_delta": round(float(holding_cost_delta), 2) if pd.notna(holding_cost_delta) else np.nan,
                "total_cost_delta": round(float(total_cost_delta), 2) if pd.notna(total_cost_delta) else np.nan,
                "inventory_turnover": round(float(row.get("inventory_turnover", np.nan)), 2) if pd.notna(row.get("inventory_turnover")) else np.nan,
                "days_of_inventory": round(float(row.get("days_of_inventory", np.nan)), 2) if pd.notna(row.get("days_of_inventory")) else np.nan,
            })

        result = pd.DataFrame(rows)
        self._log_summary(result)
        return result

    def portfolio_summary(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate recommendations to portfolio level by ABC/XYZ segment.

        This is the executive summary: total capital released, total extra
        risk accepted, per segment.
        """
        if len(recommendations_df) == 0:
            return pd.DataFrame()

        summary = (
            recommendations_df.groupby("abc_xyz_class", observed=True)
            .agg(
                n_skus=("item_id", "count"),
                current_wc_total=("current_working_capital", "sum"),
                recommended_wc_total=("recommended_working_capital", "sum"),
                capital_released_total=("capital_released", "sum"),
                stockout_risk_delta_total=("stockout_risk_delta", "sum"),
                holding_cost_delta_total=("holding_cost_delta", "sum"),
                total_cost_delta_total=("total_cost_delta", "sum"),
                avg_turnover=("inventory_turnover", "mean"),
                avg_doi=("days_of_inventory", "mean"),
                upgrades=("policy_change", lambda x: (x == "upgrade").sum()),
                downgrades=("policy_change", lambda x: (x == "downgrade").sum()),
                maintained=("policy_change", lambda x: (x == "maintain").sum()),
                excluded=("policy_change", lambda x: (x == "excluded").sum()),
            )
            .reset_index()
        )

        # Round
        for col in ["current_wc_total", "recommended_wc_total",
                     "capital_released_total", "stockout_risk_delta_total",
                     "holding_cost_delta_total", "total_cost_delta_total",
                     "avg_turnover", "avg_doi"]:
            if col in summary.columns:
                summary[col] = summary[col].round(2)

        log.info(
            "portfolio_summary",
            total_capital_released=round(float(summary["capital_released_total"].sum()), 2),
            total_downgrades=int(summary["downgrades"].sum()),
            total_upgrades=int(summary["upgrades"].sum()),
        )
        return summary

    # ------------------------------------------------------------------
    # Policy comparison: capital released must be read next to annual cost (M5-10)
    # ------------------------------------------------------------------

    @staticmethod
    def _complete_priced(kpis_df: pd.DataFrame) -> pd.DataFrame:
        """SKUs with a total cost at every evaluated service level (priced and sized).

        Every policy is compared on exactly these SKUs, so totals are like for like.
        """
        n_levels = kpis_df["service_level"].nunique()
        priced = kpis_df.dropna(subset=["total_relevant_cost"])
        complete = priced.groupby(["item_id", "store_id"])["service_level"].transform("nunique") == n_levels
        return priced[complete]

    @staticmethod
    def _sl_label(sl: float) -> str:
        return f"{sl * 100:g}"

    def segment_cost_optimum(self, kpis_df: pd.DataFrame, current_sl: float = 0.95) -> pd.DataFrame:
        """Per ABC/XYZ segment: total annual cost at each service level and the cheapest one.

        ``cost_minimising_sl`` is the level on the evaluated menu with the lowest summed
        holding + stockout cost (ties within one cent keep the level closest to ``current_sl``).
        Segments ending in ``N`` (no recent demand) are held at ``current_sl``, never optimised.
        ``closed_form_sl_star`` is the unconstrained optimum 1 - h·c·L / (365·m), taken as the
        segment median; it can lie above the whole menu.
        """
        base = self._complete_priced(kpis_df)
        levels = sorted(base["service_level"].unique())
        rows = []
        for cls, g in base.groupby("abc_xyz_class", observed=True):
            totals = g.groupby("service_level")["total_relevant_cost"].sum()
            if str(cls).endswith("N"):
                best = current_sl
            else:
                best_total = float(totals.min())
                tied = [sl for sl in levels if totals[sl] <= best_total + 0.01]
                best = current_sl if current_sl in tied else min(tied, key=lambda s: abs(s - current_sl))
            row = {
                "abc_xyz_class": cls,
                "n_skus": int(g[["item_id", "store_id"]].drop_duplicates().shape[0]),
                "capital_release_sl": self.policy.get(str(cls), current_sl),
                "cost_minimising_sl": best,
            }
            for sl in levels:
                row[f"total_cost_{self._sl_label(sl)}"] = round(float(totals[sl]), 2)
            if "unconstrained_optimal_sl" in g.columns:
                row["closed_form_sl_star"] = round(float(g["unconstrained_optimal_sl"].median()), 4)
            else:
                row["closed_form_sl_star"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def cost_minimising_levels(self, kpis_df: pd.DataFrame, current_sl: float = 0.95) -> dict[str, float]:
        """Map each ABC/XYZ segment to its cost-minimising service level on the evaluated menu."""
        table = self.segment_cost_optimum(kpis_df, current_sl)
        return dict(zip(table["abc_xyz_class"], table["cost_minimising_sl"], strict=True))

    def compare_policies(self, kpis_df: pd.DataFrame, current_sl: float = 0.95) -> pd.DataFrame:
        """Three policies side by side: working capital next to annual cost.

        ``uniform_current``       every SKU at ``current_sl`` (the baseline)
        ``capital_release_rule``  the ABC/XYZ rule in ``policy`` (AX 98%, ..., CZ 90%)
        ``cost_minimising``       the cheapest menu level per segment (``cost_minimising_levels``)

        All three are summed over the same SKUs (priced, sized at every level).  The ``*_change``
        columns are relative to ``uniform_current``; a negative ``wc_change_vs_current`` is capital
        released, and ``total_cost_change`` is the change in holding + stockout cost per year.
        """
        base = self._complete_priced(kpis_df)
        levels = self.cost_minimising_levels(kpis_df, current_sl)
        policies = {
            "uniform_current": ("Uniform current service level", lambda cls: current_sl),
            "capital_release_rule": ("ABC/XYZ capital-release rule", lambda cls: self.policy.get(cls, current_sl)),
            "cost_minimising": ("Cost-minimising level per segment", lambda cls: levels.get(cls, current_sl)),
        }

        def pick(fn) -> pd.DataFrame:
            target = base["abc_xyz_class"].astype(str).map(fn)
            return base[np.isclose(base["service_level"], target)]

        totals = {}
        for key, (_, fn) in policies.items():
            sel = pick(fn)
            totals[key] = {
                "n_skus": int(sel[["item_id", "store_id"]].drop_duplicates().shape[0]),
                "working_capital": float(sel["working_capital_tied"].sum()),
                "holding_cost_annual": float(sel["holding_cost_annual"].sum()),
                "stockout_cost_annual": float(sel["stockout_cost_annual"].sum()),
                "total_cost_annual": float(sel["total_relevant_cost"].sum()),
            }
        ref = totals["uniform_current"]
        rows = []
        for key, (label, _) in policies.items():
            t = totals[key]
            rows.append({
                "policy": key,
                "label": label,
                **{k: (v if k == "n_skus" else round(v, 2)) for k, v in t.items()},
                "wc_change_vs_current": round(t["working_capital"] - ref["working_capital"], 2),
                "holding_cost_change": round(t["holding_cost_annual"] - ref["holding_cost_annual"], 2),
                "stockout_cost_change": round(t["stockout_cost_annual"] - ref["stockout_cost_annual"], 2),
                "total_cost_change": round(t["total_cost_annual"] - ref["total_cost_annual"], 2),
            })
        return pd.DataFrame(rows)

    def demand_basis_comparison(
        self,
        kpis_df: pd.DataFrame,
        bases: dict[str, pd.DataFrame],
        current_sl: float = 0.95,
        planning_basis: str = "forecast",
    ) -> pd.DataFrame:
        """Current working capital under different mean-demand bases.

        Each entry of ``bases`` is a frame with ``item_id``, ``store_id`` and ``demand`` (mean
        units per day).  Safety stock is the same in every row; only cycle stock (demand × L/2)
        moves.  Only ``planning_basis`` is a number a planner could have had at decision time;
        the rest are ``reference only`` (actual sales on the scored days are look-ahead).
        """
        # Same SKUs as the policy comparison: priced and with a sized (non-missing) safety stock.
        cur = kpis_df[
            (kpis_df["service_level"] == current_sl)
            & kpis_df["unit_cost"].notna()
            & kpis_df["safety_stock_units"].notna()
        ]
        rows = []
        for name, d in bases.items():
            m = cur.merge(d[["item_id", "store_id", "demand"]], on=["item_id", "store_id"], how="inner")
            cycle = float((m["demand"] * m["lead_time_days"] / 2.0 * m["unit_cost"]).sum())
            safety = float((m["safety_stock_units"] * m["unit_cost"]).sum())
            rows.append({
                "basis": name,
                "used_for": "planning" if name == planning_basis else "reference only",
                "n_skus": len(m),
                "mean_daily_demand_total": round(float(m["demand"].sum()), 2),
                "cycle_stock_capital": round(cycle, 2),
                "safety_stock_capital": round(safety, 2),
                "working_capital": round(cycle + safety, 2),
            })
        return pd.DataFrame(rows)

    def _log_summary(self, df: pd.DataFrame) -> None:
        """Log policy recommendation summary."""
        if len(df) == 0:
            log.warning("recommendations_empty")
            return

        n_upgrade = (df["policy_change"] == "upgrade").sum()
        n_downgrade = (df["policy_change"] == "downgrade").sum()
        n_maintain = (df["policy_change"] == "maintain").sum()
        total_released = df["capital_released"].sum()

        log.info(
            "policy_recommended",
            n_skus=len(df),
            upgrades=int(n_upgrade),
            downgrades=int(n_downgrade),
            maintained=int(n_maintain),
            total_capital_released=round(float(total_released), 2) if pd.notna(total_released) else "N/A",
        )
