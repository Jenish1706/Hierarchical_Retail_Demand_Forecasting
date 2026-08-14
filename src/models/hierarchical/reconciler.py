"""
Hierarchical Forecasting with Reconciliation.

Forecasts at every level of the M5 hierarchy simultaneously,
then reconciles to ensure coherence (sum of parts = whole).

Hierarchy:
    Total
    └── State (CA, TX, WI)
        └── Store (CA_1, TX_1, ...)
            └── Category (HOBBIES, HOUSEHOLD, FOODS)
                └── Department (HOBBIES_1, ...)
                    └── Item (HOBBIES_1_001_CA_1, ...)

Reconciliation methods:
  - Bottom-Up: aggregate from item level (best for stable items)
  - Top-Down: disaggregate from total (best for new items)
  - MinT (Minimum Trace): optimal linear combination
  - OLS: ordinary least squares projection

Research reference: Hyndman et al. (2011) "Optimal combination forecasts
for hierarchical time series"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.utils.logging import get_logger

log = get_logger(__name__)

# Hierarchy level definitions
HIERARCHY_LEVELS = ["total", "state", "store", "cat", "dept", "item_store"]

ReconciliationMethod = Literal["bottom_up", "top_down", "ols", "mint_shrink"]


@dataclass
class HierarchyNode:
    """One node in the M5 hierarchy."""
    level: str
    entity_id: str
    parent_id: str | None
    children: list[str]
    base_forecast: np.ndarray | None = None
    reconciled_forecast: np.ndarray | None = None


class M5HierarchyBuilder:
    """
    Constructs the M5 hierarchy from item-level sales data.
    """

    def build(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, list[str]]]:
        """
        Returns hierarchy: {level: {entity_id: [child_ids]}}
        """
        hierarchy = {
            "total": {"total": []},
            "state": {},
            "store": {},
            "cat": {},
            "dept": {},
            "item_store": {},
        }

        for _, row in df[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates().iterrows():
            state = row["state_id"]
            store = row["store_id"]
            cat = f"{row['cat_id']}"
            dept = f"{row['dept_id']}"
            item_store = f"{row['item_id']}_{row['store_id']}"

            hierarchy["total"]["total"].append(state)
            hierarchy["state"].setdefault(state, []).append(store)
            hierarchy["store"].setdefault(store, []).append(dept)
            hierarchy["cat"].setdefault(cat, []).append(dept)
            hierarchy["dept"].setdefault(dept, []).append(item_store)
            hierarchy["item_store"].setdefault(item_store, [])

        # Deduplicate children
        for level in hierarchy:
            for key in hierarchy[level]:
                hierarchy[level][key] = sorted(set(hierarchy[level][key]))

        return hierarchy


class HierarchicalForecaster:
    """
    Generates and reconciles forecasts across the full M5 hierarchy.

    Workflow:
      1. Generate base forecasts at item/store level
      2. Aggregate to upper levels (bottom-up pass)
      3. Reconcile using chosen method
      4. Evaluate coherence and accuracy at each level
    """

    def __init__(
        self,
        base_forecaster: Any,
        reconciliation_method: ReconciliationMethod = "mint_shrink",
        horizon: int = 28,
    ):
        self.base_forecaster = base_forecaster
        self.reconciliation_method = reconciliation_method
        self.horizon = horizon
        self._log = get_logger("HierarchicalForecaster")

    def fit_and_reconcile(
        self,
        df_train: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "HierarchicalForecaster":
        """Fit base forecaster. Reconciliation is applied at predict time."""
        self.base_forecaster.fit(X_train, y_train)
        self._df_train = df_train
        return self

    def predict(
        self,
        X_future: pd.DataFrame,
        df_meta: pd.DataFrame,  # item_id, store_id, dept_id, cat_id, state_id per row
    ) -> pd.DataFrame:
        """
        Generate reconciled hierarchical forecasts.

        Returns long-format DataFrame with columns:
          level, entity_id, date, forecast_reconciled
        """
        # 1. Base forecasts at item/store level
        base_preds = self.base_forecaster.predict(X_future).clip(0)
        df_forecasts = df_meta.copy()
        df_forecasts["base_forecast"] = base_preds

        # 2. Aggregate to upper levels (bottom-up pass)
        df_all_levels = self._aggregate_hierarchy(df_forecasts)

        # 3. Reconcile
        if self.reconciliation_method == "bottom_up":
            df_reconciled = self._reconcile_bottom_up(df_all_levels, df_forecasts)
        elif self.reconciliation_method == "top_down":
            df_reconciled = self._reconcile_top_down(df_all_levels, df_forecasts)
        else:
            # OLS / MinT: use summing matrix projection
            df_reconciled = self._reconcile_mint(df_all_levels, df_forecasts)

        return df_reconciled

    def _aggregate_hierarchy(
        self, df_item: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate item-level forecasts to all hierarchy levels."""
        rows = []

        # Item-store level (bottom)
        for _, row in df_item.iterrows():
            rows.append({
                "level": "item_store",
                "entity_id": f"{row['item_id']}_{row['store_id']}",
                "base_forecast": row["base_forecast"],
                "item_id": row.get("item_id"),
                "store_id": row.get("store_id"),
                "dept_id": row.get("dept_id"),
                "cat_id": row.get("cat_id"),
                "state_id": row.get("state_id"),
            })

        # Dept level
        dept_agg = df_item.groupby(["dept_id", "store_id"])["base_forecast"].sum().reset_index()
        for _, row in dept_agg.iterrows():
            rows.append({
                "level": "dept",
                "entity_id": f"{row['dept_id']}_{row['store_id']}",
                "base_forecast": row["base_forecast"],
                "dept_id": row["dept_id"],
                "store_id": row["store_id"],
            })

        # Cat level
        cat_agg = df_item.groupby(["cat_id", "store_id"])["base_forecast"].sum().reset_index()
        for _, row in cat_agg.iterrows():
            rows.append({
                "level": "cat",
                "entity_id": f"{row['cat_id']}_{row['store_id']}",
                "base_forecast": row["base_forecast"],
                "cat_id": row["cat_id"],
                "store_id": row["store_id"],
            })

        # Store level
        store_agg = df_item.groupby("store_id")["base_forecast"].sum().reset_index()
        for _, row in store_agg.iterrows():
            rows.append({
                "level": "store",
                "entity_id": row["store_id"],
                "base_forecast": row["base_forecast"],
                "store_id": row["store_id"],
            })

        # State level
        state_agg = df_item.groupby("state_id")["base_forecast"].sum().reset_index()
        for _, row in state_agg.iterrows():
            rows.append({
                "level": "state",
                "entity_id": row["state_id"],
                "base_forecast": row["base_forecast"],
                "state_id": row["state_id"],
            })

        # Total level
        rows.append({
            "level": "total",
            "entity_id": "total",
            "base_forecast": df_item["base_forecast"].sum(),
        })

        return pd.DataFrame(rows)

    def _reconcile_bottom_up(
        self, df_all: pd.DataFrame, df_item: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Bottom-up reconciliation: upper levels = sum of bottom level.
        Already done in _aggregate_hierarchy; just rename columns.
        """
        df_all["forecast_reconciled"] = df_all["base_forecast"]
        df_all["reconciliation_method"] = "bottom_up"
        return df_all

    def _reconcile_top_down(
        self, df_all: pd.DataFrame, df_item: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Top-down: disaggregate total forecast using historical proportions.
        Useful when top-level forecast is more accurate.
        """
        total_forecast = df_all.loc[df_all["level"] == "total", "base_forecast"].values[0]
        total_historical = df_item["base_forecast"].sum()

        if total_historical == 0:
            proportions = pd.Series(1.0 / len(df_item), index=df_item.index)
        else:
            proportions = df_item["base_forecast"] / total_historical

        df_item = df_item.copy()
        df_item["item_reconciled"] = proportions * total_forecast

        # Re-aggregate from reconciled item forecasts
        df_item_renamed = df_item.copy()
        df_item_renamed["base_forecast"] = df_item_renamed["item_reconciled"]
        df_reag = self._aggregate_hierarchy(df_item_renamed)
        df_reag["forecast_reconciled"] = df_reag["base_forecast"]
        df_reag["reconciliation_method"] = "top_down"
        return df_reag

    def _reconcile_mint(
        self, df_all: pd.DataFrame, df_item: pd.DataFrame
    ) -> pd.DataFrame:
        """
        MinT (Minimum Trace) reconciliation via OLS.
        Optimal reconciliation that minimizes total forecast error variance.
        P = S(S'S)^-1 S' where S is the summing matrix.
        """
        # Build summing matrix S: rows = all levels, cols = bottom level
        item_ids = df_item["entity_id"].tolist() if "entity_id" in df_item.columns else [
            f"{r['item_id']}_{r['store_id']}" for _, r in df_item.iterrows()
        ]
        n_bottom = len(item_ids)
        n_all = len(df_all)

        S = np.zeros((n_all, n_bottom))
        item_idx = {iid: i for i, iid in enumerate(item_ids)}

        for row_i, (_, row) in enumerate(df_all.iterrows()):
            level = row["level"]
            if level == "item_store":
                eid = row["entity_id"]
                if eid in item_idx:
                    S[row_i, item_idx[eid]] = 1
            elif level == "dept":
                dept = row.get("dept_id", "")
                store = row.get("store_id", "")
                for j, iid in enumerate(item_ids):
                    item_row = df_item[df_item.get("entity_id", pd.Series()) == iid]
                    # Simplified: mark all items in this dept/store
                    if dept in iid and store in iid:
                        S[row_i, j] = 1
            elif level == "total":
                S[row_i, :] = 1

        # OLS reconciliation: ŷ_reconciled = S(S'S)^{-1}S' * ŷ_base
        base_bottom = df_all.loc[df_all["level"] == "item_store", "base_forecast"].values
        if len(base_bottom) == n_bottom:
            try:
                STS_inv = np.linalg.pinv(S.T @ S)
                P = S @ STS_inv @ S.T
                base_all = df_all["base_forecast"].values
                reconciled_all = P @ base_all
                df_all["forecast_reconciled"] = reconciled_all.clip(0)
            except np.linalg.LinAlgError:
                # Fall back to bottom-up
                df_all["forecast_reconciled"] = df_all["base_forecast"]
        else:
            df_all["forecast_reconciled"] = df_all["base_forecast"]

        df_all["reconciliation_method"] = "mint_ols"
        return df_all

    def evaluate_by_level(
        self,
        df_reconciled: pd.DataFrame,
        df_actuals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare WAPE at each hierarchy level."""
        results = []
        for level in HIERARCHY_LEVELS:
            level_preds = df_reconciled[df_reconciled["level"] == level]
            if len(level_preds) == 0:
                continue

            # Try to match with actuals
            if "actual" in df_actuals.columns:
                merged = level_preds.merge(
                    df_actuals[df_actuals["level"] == level][["entity_id", "actual"]],
                    on="entity_id",
                    how="inner",
                )
                if len(merged) > 0:
                    wape = (
                        np.abs(merged["actual"] - merged["forecast_reconciled"]).sum()
                        / merged["actual"].sum() * 100
                    )
                    mae = mean_absolute_error(merged["actual"], merged["forecast_reconciled"])
                    results.append({"level": level, "wape": wape, "mae": mae, "n": len(merged)})

        return pd.DataFrame(results)


# Type alias for Any used in the class
from typing import Any
