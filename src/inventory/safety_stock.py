"""Safety stock and reorder point calculation.

Why this exists
---------------
The forecasting platform answers "what will demand be?"  This module answers
the next question: "how much buffer inventory do we need to protect against
forecast error at a given service level?"

Two complementary approaches
-----------------------------
1. **Classical (forecast-error based):**
       SS = z_alpha × sigma_e × sqrt(L),   sigma_e = RMSE of backtest residuals
   This is the textbook formula from Silver, Pyke & Thomas (2017),
   appropriate when the forecast error distribution is approximately normal.

   ``sigma_e`` is the root mean squared error *about zero*, not the standard
   deviation.  A standard deviation subtracts each model's mean error, so a
   forecast that is consistently too low would be costed as if it were
   accurate (M5-12b: that made the historical-mean baseline, which
   under-forecasts by about 0.3 units per SKU per day, look like the cheapest
   model).  Since RMSE^2 = bias^2 + variance, a biased forecast pays for its
   bias.  RMSE is used instead of "std, then add the mean error to the reorder
   point" because the latter assumes the bias measured on the evaluation window
   is known and corrected in advance, which flatters exactly the models that
   drift.  Known approximations: scaling by sqrt(L) treats daily errors as
   independent, so a persistent bias (which accumulates as L, not sqrt(L)) is
   still somewhat understated; and RMSE charges over-forecast bias too, even
   though an over-forecast already sits in the reorder point's forecast term.

2. **Quantile-based:**
       SS = Q_alpha(demand_L) - mu_hat_L
   Uses the upper quantile from the QuantileForecaster at the target service
   level, summed over the lead-time window.  This is distribution-free and
   leverages the existing probabilistic forecasting tier.

Both approaches compute the reorder point:
    ROP = mean_demand × L + SS

The user can compare both and choose the one that better matches the actual
demand distribution.  For zero-inflated M5 series where normality fails, the
quantile approach is likely more accurate.

Data requirements
-----------------
- **Classical:** point forecasts and actuals from backtest (to compute residuals)
- **Quantile:** output of QuantileForecaster.predict() containing per-quantile
  forecasts
- **Both:** the demand panel for mean daily demand computation

The lead time is configurable (default 7 days, matching M5's dominant weekly
seasonality) and can be set per department for heterogeneous supply chains.

References
----------
Silver, Pyke & Thomas (2017), "Inventory and Production Management in Supply
    Chains", 4th ed., CRC Press -- Chapter 7: safety stock determination.
Chopra & Meindl (2016), "Supply Chain Management", 6th ed., Pearson --
    Chapter 12: managing uncertainty in a supply chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config.logging import get_logger

log = get_logger(__name__)

DEFAULT_SERVICE_LEVELS: tuple[float, ...] = (0.90, 0.95, 0.98)
DEFAULT_LEAD_TIME_DAYS: int = 7

_EPS = 1e-12


def _z_score(service_level: float) -> float:
    """Inverse normal CDF for the target service level (cycle service level).

    z(0.90) = 1.282, z(0.95) = 1.645, z(0.98) = 2.054, z(0.99) = 2.326.
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError(f"Service level must be in (0, 1), got {service_level}")
    return float(sp_stats.norm.ppf(service_level))


def summarise_forecast_errors(
    df: pd.DataFrame,
    by: list[str] | tuple[str, ...],
    residual_col: str = "residual",
) -> pd.DataFrame:
    """Summarise forecast residuals (actual - forecast) per group.

    This is the single definition of ``sigma_e`` used for safety stock, so the
    per-model comparison table and the per-SKU safety stock file cannot drift
    apart.

    Returns one row per group with:

    ``forecast_error_rmse``
        sqrt(mean(e^2)): error about zero.  **This is the sigma_e that sizes
        safety stock**, because it includes bias.
    ``forecast_error_std``
        Sample standard deviation (ddof=1).  Bias-free; kept for reference only.
    ``forecast_error_bias``
        Mean error, actual - forecast.  Positive means the forecast is too low
        (under-forecast), negative means too high.
    ``n_obs``
        Number of residuals.
    """
    by = list(by)
    work = df[by].copy()
    err = df[residual_col].astype(float)
    work["_e"] = err
    work["_e2"] = err * err
    out = (
        work.groupby(by, observed=True)
        .agg(
            forecast_error_bias=("_e", "mean"),
            forecast_error_std=("_e", "std"),
            _mean_sq=("_e2", "mean"),
            n_obs=("_e", "count"),
        )
        .reset_index()
    )
    out["forecast_error_rmse"] = np.sqrt(out.pop("_mean_sq"))
    return out[[*by, "forecast_error_rmse", "forecast_error_std", "forecast_error_bias", "n_obs"]]


class SafetyStockCalculator:
    """Compute safety stock and reorder points at multiple service levels.

    Usage
    -----
    >>> calc = SafetyStockCalculator(lead_time_days=7)
    >>> ss_df = calc.from_forecast_errors(forecasts_df, actuals_df, panel_df)
    >>> ss_df[ss_df["service_level"] == 0.95]

    Parameters
    ----------
    lead_time_days : int or dict[str, int]
        Replenishment lead time in days.  If a dict, keys are department IDs
        (e.g., ``{"FOODS_1": 5, "HOBBIES_1": 10}``).  Missing departments
        fall back to ``default_lead_time`` (default 7).
    service_levels : tuple[float, ...]
        Target cycle service levels to evaluate.
    default_lead_time : int
        Fallback lead time when ``lead_time_days`` is a dict and a department
        is not found.
    """

    def __init__(
        self,
        lead_time_days: int | dict[str, int] = DEFAULT_LEAD_TIME_DAYS,
        service_levels: tuple[float, ...] = DEFAULT_SERVICE_LEVELS,
        default_lead_time: int = DEFAULT_LEAD_TIME_DAYS,
    ) -> None:
        self.lead_time_days = lead_time_days
        self.service_levels = tuple(sorted(service_levels))
        self.default_lead_time = int(default_lead_time)

        for sl in self.service_levels:
            if not 0.0 < sl < 1.0:
                raise ValueError(f"Service levels must be in (0, 1), got {sl}")

    def _get_lead_time(self, dept_id: str | None = None) -> int:
        """Resolve lead time for a department."""
        if isinstance(self.lead_time_days, dict):
            return self.lead_time_days.get(dept_id, self.default_lead_time)
        return int(self.lead_time_days)

    # ------------------------------------------------------------------
    # Approach 1: Classical -- from backtest forecast errors
    # ------------------------------------------------------------------

    def from_forecast_errors(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame | None = None,
        panel: pd.DataFrame | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        """Compute safety stock from the RMSE of forecast errors.

        Formula: SS = z(alpha) * sigma_e * sqrt(L), with sigma_e = RMSE (error
        about zero, so a consistently biased forecast needs more stock).

        Parameters
        ----------
        forecasts : pd.DataFrame
            Either:
            1. Standard format containing ``item_id``, ``store_id``, ``date``, ``forecast``
               (and optional ``model``).
            2. Backtest predictions format containing ``series_id``, ``date``, ``y_true``,
               ``y_pred`` (and optional ``model``, ``item_id``, ``store_id``).
        actuals : pd.DataFrame | None
            Must contain ``item_id``, ``store_id``, ``date``, ``sales`` if ``forecasts``
            does not already contain actuals (``y_true``).
        panel : pd.DataFrame | None
            Full panel for mean daily demand computation. If None, daily demand is estimated
            from the provided forecasts/actuals.
        model : str | None
            Specific model to filter for, or name to record in the output table.

        Returns
        -------
        pd.DataFrame
            One row per (item_id, store_id, service_level) with columns:
            ``item_id``, ``store_id``, ``dept_id``, ``model``, ``service_level``,
            ``z_score``, ``lead_time_days``, ``avg_daily_demand``,
            ``forecast_error_rmse`` (the sizing sigma_e), ``forecast_error_std``
            and ``forecast_error_bias`` (reference), ``safety_stock_units``,
            ``reorder_point``.
        """
        # Resolve forecast dataframe and compute residuals
        if actuals is None and "y_true" in forecasts.columns and "y_pred" in forecasts.columns:
            f_df = forecasts.copy()
            if model is not None and "model" in f_df.columns:
                f_df = f_df[f_df["model"] == model]
            if "item_id" not in f_df.columns or "store_id" not in f_df.columns:
                sids = f_df["series_id"].astype(str)
                f_df["item_id"] = sids.str.rsplit("_", n=2).str[0]
                f_df["store_id"] = sids.str.rsplit("_", n=2).str[1] + "_" + sids.str.rsplit("_", n=2).str[2]
            f_df["residual"] = f_df["y_true"].astype(float) - f_df["y_pred"].astype(float)
            merged = f_df
            resolved_model = model or (f_df["model"].iloc[0] if "model" in f_df.columns and len(f_df) > 0 else "forecast_error")
        else:
            if actuals is None:
                raise ValueError("actuals must be provided when forecasts does not contain y_true and y_pred.")
            f_df = forecasts.copy()
            if model is not None and "model" in f_df.columns:
                f_df = f_df[f_df["model"] == model]
            merge_cols = ["item_id", "store_id", "date"]
            merged = f_df.merge(
                actuals[["item_id", "store_id", "date", "sales"]],
                on=merge_cols,
                how="inner",
            )
            merged["residual"] = merged["sales"] - merged["forecast"]
            resolved_model = model or (merged["model"].iloc[0] if "model" in merged.columns and len(merged) > 0 else "forecast_error")

        # Forecast error per series: RMSE sizes the stock; std and bias are kept for reference.
        error_stats = summarise_forecast_errors(merged, by=["item_id", "store_id"])
        error_stats["forecast_error_std"] = error_stats["forecast_error_std"].fillna(0.0)

        # Mean daily demand
        if panel is not None:
            demand_stats = (
                panel.groupby(["item_id", "store_id"], observed=True)["sales"]
                .mean()
                .reset_index()
                .rename(columns={"sales": "avg_daily_demand"})
            )
            if "dept_id" in panel.columns:
                dept_map = (
                    panel[["item_id", "store_id", "dept_id"]]
                    .drop_duplicates()
                )
                demand_stats = demand_stats.merge(dept_map, on=["item_id", "store_id"], how="left")
        else:
            sales_col = "sales" if "sales" in merged.columns else "y_true"
            demand_stats = (
                merged.groupby(["item_id", "store_id"], observed=True)[sales_col]
                .mean()
                .reset_index()
                .rename(columns={sales_col: "avg_daily_demand"})
            )
            if "dept_id" in merged.columns:
                dept_map = merged[["item_id", "store_id", "dept_id"]].drop_duplicates()
                demand_stats = demand_stats.merge(dept_map, on=["item_id", "store_id"], how="left")

        base = error_stats.merge(demand_stats, on=["item_id", "store_id"], how="inner")

        # Expand across service levels
        rows = []
        for _, row in base.iterrows():
            dept = row.get("dept_id")
            lt = self._get_lead_time(dept)
            for sl in self.service_levels:
                z = _z_score(sl)
                sigma_e = row["forecast_error_rmse"]
                ss = z * sigma_e * np.sqrt(lt)
                rop = row["avg_daily_demand"] * lt + ss
                rows.append({
                    "item_id": row["item_id"],
                    "store_id": row["store_id"],
                    "dept_id": dept,
                    "model": resolved_model,
                    "service_level": sl,
                    "z_score": round(z, 4),
                    "lead_time_days": lt,
                    "avg_daily_demand": round(float(row["avg_daily_demand"]), 4),
                    "forecast_error_rmse": round(float(sigma_e), 4),
                    "forecast_error_std": round(float(row["forecast_error_std"]), 4),
                    "forecast_error_bias": round(float(row["forecast_error_bias"]), 4),
                    "safety_stock_units": round(float(max(ss, 0.0)), 2),
                    "reorder_point": round(float(max(rop, 0.0)), 2),
                })

        result = pd.DataFrame(rows)
        self._log_summary(result, method="forecast_error")
        return result

    # ------------------------------------------------------------------
    # Approach 2: Quantile-based -- from probabilistic forecasts
    # ------------------------------------------------------------------

    def from_quantiles(
        self,
        quantile_forecasts: pd.DataFrame,
        panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute safety stock from quantile forecast spread.

        Formula: SS = Q(alpha, lead_time) - mean_demand × L

        The quantile forecast at the target service level gives the demand
        level that will not be exceeded with probability ``alpha``.  Safety
        stock is the difference between that level and the expected demand
        over the lead time.

        Parameters
        ----------
        quantile_forecasts : pd.DataFrame
            Must contain ``item_id``, ``store_id``, ``forecast_point``, and
            quantile columns named ``q0.9``, ``q0.95``, ``q0.98``, etc.
        panel : pd.DataFrame
            Full panel for mean daily demand computation.

        Returns
        -------
        pd.DataFrame
            Same schema as :meth:`from_forecast_errors`.
        """
        # Mean daily demand
        demand_stats = (
            panel.groupby(["item_id", "store_id"], observed=True)["sales"]
            .mean()
            .reset_index()
            .rename(columns={"sales": "avg_daily_demand"})
        )

        if "dept_id" in panel.columns:
            dept_map = panel[["item_id", "store_id", "dept_id"]].drop_duplicates()
            demand_stats = demand_stats.merge(dept_map, on=["item_id", "store_id"], how="left")

        # Aggregate quantile forecasts per series (mean across forecast dates)
        qf_agg = quantile_forecasts.groupby(["item_id", "store_id"], observed=True).mean(numeric_only=True).reset_index()

        base = qf_agg.merge(demand_stats, on=["item_id", "store_id"], how="inner")

        rows = []
        for _, row in base.iterrows():
            dept = row.get("dept_id")
            lt = self._get_lead_time(dept)
            mean_demand_lt = row["avg_daily_demand"] * lt

            for sl in self.service_levels:
                # Find the quantile column closest to the service level
                q_col = f"q{sl}"
                if q_col not in row.index:
                    # Try alternative naming: q0.9, q0.95, etc.
                    q_col_alt = f"q{sl:g}"
                    if q_col_alt in row.index:
                        q_col = q_col_alt
                    else:
                        log.warning(
                            "quantile_missing",
                            service_level=sl,
                            available=[c for c in row.index if c.startswith("q")],
                        )
                        continue

                q_demand_lt = float(row[q_col]) * lt
                ss = max(q_demand_lt - mean_demand_lt, 0.0)
                rop = mean_demand_lt + ss

                rows.append({
                    "item_id": row["item_id"],
                    "store_id": row["store_id"],
                    "dept_id": dept,
                    "service_level": sl,
                    "z_score": np.nan,  # not applicable for quantile method
                    "lead_time_days": lt,
                    "avg_daily_demand": round(float(row["avg_daily_demand"]), 4),
                    "forecast_error_rmse": np.nan,  # not applicable
                    "forecast_error_std": np.nan,  # not applicable
                    "forecast_error_bias": np.nan,  # not applicable
                    "safety_stock_units": round(float(ss), 2),
                    "reorder_point": round(float(rop), 2),
                })

        result = pd.DataFrame(rows)
        self._log_summary(result, method="quantile")
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, df: pd.DataFrame, method: str) -> None:
        """Log a summary of safety stock calculations."""
        if len(df) == 0:
            log.warning("safety_stock_empty", method=method)
            return

        for sl in self.service_levels:
            sl_df = df[df["service_level"] == sl]
            if len(sl_df) == 0:
                continue
            log.info(
                "safety_stock_computed",
                method=method,
                service_level=sl,
                n_series=len(sl_df),
                ss_median=round(float(sl_df["safety_stock_units"].median()), 2),
                ss_mean=round(float(sl_df["safety_stock_units"].mean()), 2),
                ss_max=round(float(sl_df["safety_stock_units"].max()), 2),
                rop_median=round(float(sl_df["reorder_point"].median()), 2),
            )
