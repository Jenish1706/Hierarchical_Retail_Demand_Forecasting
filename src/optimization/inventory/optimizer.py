"""
Inventory Optimization Engine.

Implements classical and probabilistic inventory models:
  - Economic Order Quantity (EOQ) — minimizes holding + ordering cost
  - Safety Stock — buffers against demand and lead time uncertainty
  - Reorder Point (ROP) — triggers replenishment at right time
  - (s, S) Policy — continuous review with order-up-to level
  - ABC-XYZ Classification — prioritizes optimization effort

Mathematical foundations:
  EOQ = sqrt(2 * D * S / H)
  Safety Stock = z * sqrt(LT * σ_d² + d̄² * σ_LT²)
  ROP = d̄ * LT + Safety Stock

Where:
  D  = annual demand
  S  = ordering cost per order
  H  = holding cost per unit per year
  z  = service level z-score
  LT = lead time in days
  σ_d = daily demand standard deviation
  d̄  = mean daily demand
  σ_LT = lead time standard deviation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.utils.logging import get_logger
from src.utils.settings import settings

log = get_logger(__name__)

# Service level → z-score mapping
SERVICE_LEVEL_Z = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
    0.995: 2.576,
    0.999: 3.090,
}


@dataclass
class InventoryPolicy:
    """Complete inventory policy for one SKU/store combination."""
    item_id: str
    store_id: str

    # Demand parameters
    mean_daily_demand: float
    std_daily_demand: float
    annual_demand: float

    # Lead time parameters
    mean_lead_time_days: float
    std_lead_time_days: float = 0.0

    # Cost parameters
    unit_cost: float = 10.0
    ordering_cost: float = 50.0        # Fixed cost per order
    holding_cost_rate: float = 0.25    # Annual holding cost as fraction of unit cost
    stockout_cost_per_unit: float = 0.0  # Computed from service level if 0

    # Target service level
    service_level: float = 0.95

    # Computed fields (set by optimizer)
    eoq: float = 0.0
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    max_stock_level: float = 0.0
    order_up_to_level: float = 0.0
    expected_annual_orders: float = 0.0
    expected_holding_cost: float = 0.0
    expected_ordering_cost: float = 0.0
    expected_total_cost: float = 0.0
    inventory_turnover: float = 0.0

    @property
    def holding_cost_per_unit_per_year(self) -> float:
        return self.unit_cost * self.holding_cost_rate

    @property
    def z_score(self) -> float:
        return SERVICE_LEVEL_Z.get(self.service_level, norm.ppf(self.service_level))


class InventoryOptimizer:
    """
    Computes optimal inventory policies using classical OR models.

    All formulas based on standard Operations Research textbook models
    (Hillier & Lieberman, "Introduction to Operations Research").
    """

    def __init__(
        self,
        ordering_cost: float = None,
        holding_cost_rate: float = None,
        default_service_level: float = None,
    ):
        cfg = settings.inventory
        self.ordering_cost = ordering_cost or cfg.default_ordering_cost
        self.holding_cost_rate = holding_cost_rate or cfg.holding_cost_rate
        self.default_service_level = default_service_level or cfg.default_service_level

    def optimize(self, policy: InventoryPolicy) -> InventoryPolicy:
        """
        Compute all inventory parameters for a given policy.
        Modifies policy in-place and returns it.
        """
        h = policy.holding_cost_per_unit_per_year

        # ---------------------------------------------------------------
        # 1. Economic Order Quantity (EOQ)
        # ---------------------------------------------------------------
        if h > 0 and policy.ordering_cost > 0:
            policy.eoq = np.sqrt(
                2 * policy.annual_demand * policy.ordering_cost / h
            )
        else:
            policy.eoq = policy.mean_daily_demand * 30  # Fallback: 1 month

        policy.eoq = max(1.0, round(policy.eoq, 2))

        # ---------------------------------------------------------------
        # 2. Safety Stock
        # ---------------------------------------------------------------
        # Combines demand uncertainty AND lead time uncertainty:
        # SS = z * sqrt(LT * σ_d² + d̄² * σ_LT²)
        lt = policy.mean_lead_time_days
        sigma_d = policy.std_daily_demand
        d_bar = policy.mean_daily_demand
        sigma_lt = policy.std_lead_time_days
        z = policy.z_score

        variance_from_demand = lt * (sigma_d ** 2)
        variance_from_leadtime = (d_bar ** 2) * (sigma_lt ** 2)
        combined_std = np.sqrt(variance_from_demand + variance_from_leadtime)
        policy.safety_stock = max(0.0, round(z * combined_std, 2))

        # ---------------------------------------------------------------
        # 3. Reorder Point
        # ---------------------------------------------------------------
        # ROP = mean demand during lead time + safety stock
        policy.reorder_point = round(d_bar * lt + policy.safety_stock, 2)

        # ---------------------------------------------------------------
        # 4. Order-Up-To Level (for periodic review)
        # ---------------------------------------------------------------
        review_period = settings.inventory.review_period_days
        policy.order_up_to_level = round(
            d_bar * (lt + review_period) + policy.safety_stock, 2
        )
        policy.max_stock_level = round(policy.order_up_to_level, 2)

        # ---------------------------------------------------------------
        # 5. Cost Analysis
        # ---------------------------------------------------------------
        policy.expected_annual_orders = policy.annual_demand / policy.eoq

        policy.expected_ordering_cost = (
            policy.expected_annual_orders * policy.ordering_cost
        )
        # Average inventory = EOQ/2 + safety stock
        avg_inventory = policy.eoq / 2 + policy.safety_stock
        policy.expected_holding_cost = avg_inventory * h

        policy.expected_total_cost = (
            policy.expected_ordering_cost + policy.expected_holding_cost
        )

        # ---------------------------------------------------------------
        # 6. Inventory Turnover
        # ---------------------------------------------------------------
        policy.inventory_turnover = (
            policy.annual_demand / avg_inventory
            if avg_inventory > 0 else 0.0
        )

        return policy

    def optimize_batch(
        self, df_demand: pd.DataFrame, unit_costs: dict[str, float] | None = None
    ) -> pd.DataFrame:
        """
        Optimize inventory policies for all item/store combinations.

        Args:
            df_demand: DataFrame with columns:
                item_id, store_id, mean_daily_demand, std_daily_demand,
                mean_lead_time_days, [std_lead_time_days]
            unit_costs: {item_id: unit_cost} lookup

        Returns:
            DataFrame with all computed policy parameters
        """
        results = []
        unit_costs = unit_costs or {}

        for _, row in df_demand.iterrows():
            annual_demand = row["mean_daily_demand"] * 365

            policy = InventoryPolicy(
                item_id=row["item_id"],
                store_id=row["store_id"],
                mean_daily_demand=float(row["mean_daily_demand"]),
                std_daily_demand=float(row.get("std_daily_demand", row["mean_daily_demand"] * 0.3)),
                annual_demand=annual_demand,
                mean_lead_time_days=float(row.get("mean_lead_time_days", settings.inventory.default_lead_time_days)),
                std_lead_time_days=float(row.get("std_lead_time_days", 1.5)),
                unit_cost=float(unit_costs.get(row["item_id"], 10.0)),
                ordering_cost=self.ordering_cost,
                holding_cost_rate=self.holding_cost_rate,
                service_level=self.default_service_level,
            )

            self.optimize(policy)

            results.append({
                "item_id": policy.item_id,
                "store_id": policy.store_id,
                "mean_daily_demand": policy.mean_daily_demand,
                "std_daily_demand": policy.std_daily_demand,
                "eoq": policy.eoq,
                "safety_stock": policy.safety_stock,
                "reorder_point": policy.reorder_point,
                "order_up_to_level": policy.order_up_to_level,
                "max_stock_level": policy.max_stock_level,
                "z_score": policy.z_score,
                "service_level_target": policy.service_level,
                "expected_annual_orders": round(policy.expected_annual_orders, 1),
                "expected_ordering_cost": round(policy.expected_ordering_cost, 2),
                "expected_holding_cost": round(policy.expected_holding_cost, 2),
                "expected_total_cost": round(policy.expected_total_cost, 2),
                "inventory_turnover": round(policy.inventory_turnover, 2),
                "annual_demand": round(annual_demand, 2),
            })

        return pd.DataFrame(results)


class ReorderRecommendationEngine:
    """
    Generates actionable reorder recommendations by comparing
    current inventory levels against computed ROP thresholds.
    """

    URGENCY_THRESHOLDS = {
        "critical": 0.0,    # Below safety stock: reorder immediately
        "high": 1.0,        # Below ROP: reorder now
        "normal": 1.2,      # Within 20% of ROP: reorder soon
        "low": float("inf"),  # Adequate stock
    }

    def generate_recommendations(
        self,
        df_current: pd.DataFrame,   # item_id, store_id, on_hand_qty
        df_policies: pd.DataFrame,  # output of optimize_batch
        df_forecasts: pd.DataFrame, # item_id, store_id, forecast_mean (next 28d)
    ) -> pd.DataFrame:
        """
        Compare on-hand inventory against ROP to flag reorder needs.
        """
        df = df_current.merge(
            df_policies[["item_id", "store_id", "reorder_point", "safety_stock",
                          "eoq", "mean_lead_time_days"]],
            on=["item_id", "store_id"],
            how="left",
        )

        # Join forecasted demand for the lead time period
        df = df.merge(
            df_forecasts[["item_id", "store_id", "forecast_mean"]].rename(
                columns={"forecast_mean": "expected_demand_28d"}
            ),
            on=["item_id", "store_id"],
            how="left",
        )

        recs = []
        for _, row in df.iterrows():
            on_hand = float(row.get("on_hand_qty", 0))
            rop = float(row.get("reorder_point", 0))
            ss = float(row.get("safety_stock", 0))
            eoq = float(row.get("eoq", 0))

            # Determine urgency
            if on_hand <= ss:
                urgency = "critical"
                reason = f"On-hand ({on_hand:.0f}) below safety stock ({ss:.0f})"
            elif on_hand <= rop:
                urgency = "high"
                reason = f"On-hand ({on_hand:.0f}) at or below reorder point ({rop:.0f})"
            elif on_hand <= rop * 1.2:
                urgency = "normal"
                reason = f"On-hand ({on_hand:.0f}) approaching reorder point ({rop:.0f})"
            else:
                urgency = "low"
                reason = "Inventory adequate"

            if urgency in ("critical", "high", "normal"):
                recs.append({
                    "item_id": row["item_id"],
                    "store_id": row["store_id"],
                    "on_hand_qty": on_hand,
                    "reorder_point": rop,
                    "safety_stock": ss,
                    "reorder_quantity": eoq,
                    "urgency": urgency,
                    "reason": reason,
                    "expected_lead_time_days": int(row.get("mean_lead_time_days", 7)),
                })

        df_recs = pd.DataFrame(recs)
        if len(df_recs) > 0:
            # Sort by urgency
            urgency_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
            df_recs["urgency_rank"] = df_recs["urgency"].map(urgency_order)
            df_recs = df_recs.sort_values("urgency_rank").drop("urgency_rank", axis=1)

        log.info(
            "reorder_recommendations_generated",
            total=len(df_recs),
            critical=len(df_recs[df_recs["urgency"] == "critical"]) if len(df_recs) > 0 else 0,
            high=len(df_recs[df_recs["urgency"] == "high"]) if len(df_recs) > 0 else 0,
        )
        return df_recs


class ABCXYZClassifier:
    """
    ABC-XYZ classification for inventory prioritization.

    ABC (by revenue contribution):
      A: top 70% of revenue → tight control, frequent review
      B: next 20% → moderate control
      C: bottom 10% → simple min/max policy

    XYZ (by demand variability):
      X: CV < 0.5 → stable demand
      Y: 0.5 ≤ CV < 1.0 → variable demand
      Z: CV ≥ 1.0 → highly variable / intermittent
    """

    def classify(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Args:
            df: item_id, annual_revenue, mean_daily_demand, std_daily_demand

        Returns:
            df with abc_class, xyz_class, abc_xyz columns
        """
        df = df.copy()

        # ABC Classification
        df["revenue_pct"] = df["annual_revenue"] / df["annual_revenue"].sum()
        df = df.sort_values("annual_revenue", ascending=False)
        df["cum_revenue_pct"] = df["revenue_pct"].cumsum()
        df["abc_class"] = pd.cut(
            df["cum_revenue_pct"],
            bins=[0, 0.70, 0.90, 1.0],
            labels=["A", "B", "C"],
            include_lowest=True,
        )

        # XYZ Classification (Coefficient of Variation)
        df["cv"] = df["std_daily_demand"] / df["mean_daily_demand"].replace(0, np.nan)
        df["xyz_class"] = pd.cut(
            df["cv"],
            bins=[0, 0.5, 1.0, float("inf")],
            labels=["X", "Y", "Z"],
            include_lowest=True,
        )
        df["xyz_class"] = df["xyz_class"].fillna("Z")

        df["abc_xyz"] = df["abc_class"].astype(str) + df["xyz_class"].astype(str)
        return df
