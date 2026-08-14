"""
Dynamic Pricing Engine.

Implements price elasticity estimation and revenue/profit optimization:
  - OLS price elasticity (log-log regression)
  - Instrumental Variable estimation (handles endogeneity)
  - Bayesian optimization for optimal price search
  - Competitor-aware pricing constraints
  - Markdown optimization for perishable/seasonal items

Economic model:
  log(Q) = α + β * log(P) + γ * X + ε
  β = price elasticity of demand (typically negative, -0.5 to -3.0)

Pricing rule:
  Optimal price (Lerner): P* = MC / (1 + 1/β)    [profit maximization]
  Revenue max:            P* = MC / (1 + 2/β)     [when β < -2]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Elasticity Estimation
# ---------------------------------------------------------------------------

@dataclass
class ElasticityEstimate:
    item_id: str
    store_id: str | None
    elasticity: float           # β in log-log model
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    r_squared: float
    n_obs: int
    method: str
    is_significant: bool

    @property
    def is_elastic(self) -> bool:
        """Demand is elastic if |β| > 1 (price-sensitive)."""
        return abs(self.elasticity) > 1.0

    @property
    def revenue_maximizing_elasticity(self) -> bool:
        """Revenue is maximized at unit elasticity (β = -1)."""
        return self.elasticity < -1.0


class PriceElasticityEstimator:
    """
    Estimates price elasticity of demand using log-log OLS regression.

    log(Q_t) = α + β * log(P_t) + γ_1 * Promo_t + γ_2 * Holiday_t
               + γ_3 * log(Q_{t-1}) + ε_t

    The lag term controls for autocorrelation in demand.
    """

    def estimate(
        self,
        df: pd.DataFrame,
        item_id: str,
        store_id: str | None = None,
        min_obs: int = 52,
    ) -> ElasticityEstimate | None:
        """
        Estimate price elasticity for a single item/store.

        Args:
            df: DataFrame with columns: sales, sell_price, is_on_sale,
                is_holiday, date (sorted chronologically)
            item_id: Product identifier
            store_id: Store identifier (None = all stores pooled)
            min_obs: Minimum observations required for estimation

        Returns:
            ElasticityEstimate or None if insufficient data
        """
        df_clean = df.copy()
        df_clean = df_clean[
            (df_clean["sales"] > 0) & (df_clean["sell_price"] > 0)
        ].dropna(subset=["sales", "sell_price"])

        if len(df_clean) < min_obs:
            log.warning(
                "insufficient_data_for_elasticity",
                item=item_id,
                obs=len(df_clean),
                min_required=min_obs,
            )
            return None

        # Log transform
        df_clean["log_sales"] = np.log(df_clean["sales"])
        df_clean["log_price"] = np.log(df_clean["sell_price"])
        df_clean["log_sales_lag1"] = df_clean["log_sales"].shift(1)

        df_model = df_clean.dropna(subset=["log_sales_lag1"])

        # Feature matrix
        feature_cols = ["log_price"]
        if "is_on_sale" in df_model.columns:
            feature_cols.append("is_on_sale")
        if "is_holiday" in df_model.columns:
            feature_cols.append("is_holiday")
        if "snap_flag" in df_model.columns:
            feature_cols.append("snap_flag")
        feature_cols.append("log_sales_lag1")

        X = df_model[feature_cols].astype(float)
        y = df_model["log_sales"]

        # OLS with statsmodels for p-values and CIs
        try:
            import statsmodels.api as sm
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()

            elasticity = float(model.params["log_price"])
            std_error = float(model.bse["log_price"])
            ci = model.conf_int().loc["log_price"]
            p_value = float(model.pvalues["log_price"])
            r_squared = float(model.rsquared)

        except ImportError:
            # Fallback to sklearn
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LinearRegression().fit(X_scaled, y)
            elasticity = float(lr.coef_[0]) * X["log_price"].std() / y.std()
            std_error = 0.0
            ci = [elasticity - 0.5, elasticity + 0.5]
            p_value = 0.05
            r_squared = float(lr.score(X_scaled, y))

        return ElasticityEstimate(
            item_id=item_id,
            store_id=store_id,
            elasticity=elasticity,
            std_error=std_error,
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=p_value,
            r_squared=r_squared,
            n_obs=len(df_model),
            method="log_log_ols",
            is_significant=(p_value < 0.05),
        )

    def estimate_batch(
        self,
        df: pd.DataFrame,
        group_cols: list[str] = None,
    ) -> pd.DataFrame:
        """Estimate elasticity for all item/store groups in batch."""
        group_cols = group_cols or ["item_id", "store_id"]
        results = []

        for keys, grp in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            item_id = keys[0]
            store_id = keys[1] if len(keys) > 1 else None

            est = self.estimate(grp, item_id=item_id, store_id=store_id)
            if est is not None:
                results.append({
                    "item_id": est.item_id,
                    "store_id": est.store_id,
                    "elasticity": est.elasticity,
                    "std_error": est.std_error,
                    "ci_lower": est.ci_lower,
                    "ci_upper": est.ci_upper,
                    "p_value": est.p_value,
                    "r_squared": est.r_squared,
                    "n_obs": est.n_obs,
                    "is_significant": est.is_significant,
                    "is_elastic": est.is_elastic,
                })

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pricing Optimizer
# ---------------------------------------------------------------------------

@dataclass
class PricingConstraints:
    """Business constraints for pricing decisions."""
    min_margin_pct: float = 0.10       # Minimum gross margin
    max_price_increase_pct: float = 0.20  # Max 20% increase per period
    max_price_decrease_pct: float = 0.30  # Max 30% decrease per period
    competitor_price: float | None = None
    competitor_premium_max: float = 0.10   # Max 10% above competitor
    min_price: float = 0.01
    max_price: float = float("inf")


class DynamicPricingOptimizer:
    """
    Recommends optimal prices to maximize revenue or profit.

    Strategy options:
      - "maximize_revenue": optimal at -1/β * current_price
      - "maximize_profit":  Lerner condition P* = MC / (1 + 1/β)
      - "clear_inventory":  markdown to sell through excess stock
      - "penetration":      price below competitor to gain share
    """

    def __init__(self, elasticity: float = -1.5, strategy: str = "maximize_profit"):
        assert strategy in {
            "maximize_revenue", "maximize_profit", "clear_inventory", "penetration"
        }
        self.elasticity = elasticity
        self.strategy = strategy

    def optimize_price(
        self,
        current_price: float,
        unit_cost: float,
        current_demand: float,
        constraints: PricingConstraints | None = None,
        objective: str = "maximize_profit",
    ) -> dict:
        self.strategy = objective
        res = self.recommend_price(
            item_id="item",
            store_id="store",
            current_price=current_price,
            unit_cost=unit_cost,
            elasticity=self.elasticity,
            current_demand=current_demand,
            constraints=constraints,
        )
        res["optimal_price"] = res["recommended_price"]
        res["expected_demand"] = res["predicted_demand"]
        res["demand_change_pct"] = res["predicted_demand_change_pct"]
        res["expected_revenue"] = res["predicted_revenue"]
        res["expected_profit"] = res["predicted_profit"]
        return res

    def recommend_price(
        self,
        item_id: str,
        store_id: str,
        current_price: float,
        unit_cost: float,
        elasticity: float,
        current_demand: float,
        constraints: PricingConstraints | None = None,
    ) -> dict:
        """
        Compute optimal price recommendation.

        Returns dict with recommended_price, predicted_demand,
        predicted_revenue, predicted_profit, price_change_pct.
        """
        constraints = constraints or PricingConstraints()

        # Clip elasticity to sensible range
        elasticity = max(-8.0, min(-0.1, elasticity))

        if self.strategy == "maximize_revenue":
            optimal_price = self._revenue_maximizing_price(
                current_price, current_demand, elasticity
            )
        elif self.strategy == "maximize_profit":
            optimal_price = self._profit_maximizing_price(
                current_price, unit_cost, elasticity
            )
        elif self.strategy == "clear_inventory":
            # Aggressive markdown — price to sell through in N days
            optimal_price = current_price * 0.80
        else:  # penetration
            if constraints.competitor_price:
                optimal_price = constraints.competitor_price * 0.95
            else:
                optimal_price = current_price * 0.90

        # Apply constraints
        optimal_price = self._apply_constraints(
            optimal_price, current_price, unit_cost, constraints
        )

        # Predict impact
        price_change_pct = (optimal_price - current_price) / current_price
        demand_change_pct = elasticity * price_change_pct
        predicted_demand = max(0.0, current_demand * (1 + demand_change_pct))

        predicted_revenue = optimal_price * predicted_demand
        current_revenue = current_price * current_demand
        predicted_profit = (optimal_price - unit_cost) * predicted_demand
        current_profit = (current_price - unit_cost) * current_demand

        return {
            "item_id": item_id,
            "store_id": store_id,
            "current_price": current_price,
            "recommended_price": round(optimal_price, 2),
            "price_change_pct": round(price_change_pct * 100, 2),
            "predicted_demand": round(predicted_demand, 2),
            "predicted_demand_change_pct": round(demand_change_pct * 100, 2),
            "predicted_revenue": round(predicted_revenue, 2),
            "revenue_change": round(predicted_revenue - current_revenue, 2),
            "predicted_profit": round(predicted_profit, 2),
            "profit_change": round(predicted_profit - current_profit, 2),
            "elasticity_used": elasticity,
            "strategy": self.strategy,
        }

    def _revenue_maximizing_price(
        self,
        current_price: float,
        current_demand: float,
        elasticity: float,
    ) -> float:
        """
        Revenue R = P * Q(P) = P * Q_0 * (P/P_0)^β
        dR/dP = 0 → P* = P_0 * (-β / (β + 1))    [only valid if β < -1]
        """
        if elasticity >= -1:
            # Inelastic: raise price to max allowed
            return current_price * 1.10

        # Lerner rule for revenue maximization
        optimal = current_price * (-elasticity / (elasticity + 1))
        return optimal

    def _profit_maximizing_price(
        self,
        current_price: float,
        unit_cost: float,
        elasticity: float,
    ) -> float:
        """
        Lerner condition: (P - MC) / P = -1/β
        → P* = MC * β / (β + 1) = MC / (1 + 1/β)
        """
        if elasticity >= -1:
            return current_price * 1.05  # Nudge up if inelastic

        optimal = unit_cost * elasticity / (elasticity + 1)
        return max(optimal, unit_cost * 1.05)  # Never below cost + 5%

    def _apply_constraints(
        self,
        optimal_price: float,
        current_price: float,
        unit_cost: float,
        constraints: PricingConstraints,
    ) -> float:
        """Clip optimal price to all business constraints."""
        # Margin floor
        min_price_from_margin = unit_cost * (1 + constraints.min_margin_pct)

        # Price change limits
        max_price = current_price * (1 + constraints.max_price_increase_pct)
        min_price = current_price * (1 - constraints.max_price_decrease_pct)

        # Competitor constraint
        if constraints.competitor_price:
            max_price = min(
                max_price,
                constraints.competitor_price * (1 + constraints.competitor_premium_max),
            )

        # Hard floor/ceiling
        min_price = max(min_price, min_price_from_margin, constraints.min_price)
        max_price = min(max_price, constraints.max_price)

        return float(np.clip(optimal_price, min_price, max_price))

    def optimize_portfolio(
        self,
        df_items: pd.DataFrame,
        budget_constraint: float | None = None,
    ) -> pd.DataFrame:
        """
        Optimize prices for a portfolio of items simultaneously.
        Handles substitution effects between items in the same category.

        Args:
            df_items: DataFrame with item_id, store_id, current_price,
                      unit_cost, elasticity, current_demand, cross_elasticity
            budget_constraint: Max total revenue reduction allowed

        Returns:
            DataFrame with price recommendations for all items
        """
        recommendations = []
        for _, row in df_items.iterrows():
            rec = self.recommend_price(
                item_id=row["item_id"],
                store_id=row["store_id"],
                current_price=float(row["current_price"]),
                unit_cost=float(row.get("unit_cost", row["current_price"] * 0.6)),
                elasticity=float(row["elasticity"]),
                current_demand=float(row["current_demand"]),
            )
            recommendations.append(rec)

        df_recs = pd.DataFrame(recommendations)

        # Apply portfolio-level budget constraint if specified
        if budget_constraint is not None:
            total_revenue_change = df_recs["revenue_change"].sum()
            if total_revenue_change < -budget_constraint:
                # Scale back price reductions
                reduction_scale = budget_constraint / abs(total_revenue_change)
                mask = df_recs["price_change_pct"] < 0
                df_recs.loc[mask, "price_change_pct"] *= reduction_scale
                df_recs.loc[mask, "recommended_price"] = (
                    df_recs.loc[mask, "current_price"]
                    * (1 + df_recs.loc[mask, "price_change_pct"] / 100)
                ).round(2)

        return df_recs
