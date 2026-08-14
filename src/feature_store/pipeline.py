"""
Feature Engineering Pipeline.

Generates all feature groups required for demand forecasting:
  - Time features (cyclical encoding, holidays)
  - Lag features (t-1, t-7, t-14, t-28, t-90)
  - Rolling statistics (mean, std, min, max)
  - Price features (elasticity-relevant signals)
  - Demand features (velocity, volatility)
  - External features (weather, events, economic)

Design decisions:
  - Polars for vectorized operations over large datasets
  - Feature versioning to support reproducible experiments
  - Separate computation from storage (pipeline pattern)
  - Handles cold-start (insufficient history) gracefully
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import holidays

from src.utils.logging import get_logger, PipelineLogger
from src.utils.settings import settings

log = get_logger(__name__)


class FeatureEngineeringPipeline:
    """
    End-to-end feature engineering for demand forecasting.

    Usage:
        pipeline = FeatureEngineeringPipeline()
        df_features = pipeline.transform(df_sales, df_prices, df_calendar)
    """

    LAG_WINDOWS = [1, 7, 14, 28, 90]
    ROLLING_WINDOWS = [7, 28, 90]

    def __init__(self, feature_version: int = 1):
        self.feature_version = feature_version
        self._us_holidays = holidays.US()
        self._log = get_logger("FeatureEngineeringPipeline")

    def transform(
        self,
        df_sales: pd.DataFrame,
        df_prices: pd.DataFrame,
        df_calendar: pd.DataFrame,
        df_weather: pd.DataFrame | None = None,
        df_economic: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Full feature engineering pass.

        Args:
            df_sales: Long-format sales (item_id, store_id, date, sales)
            df_prices: (item_id, store_id, wm_yr_wk, sell_price)
            df_calendar: (date, event_name_1, event_type_1, snap_ca, ...)
            df_weather: Optional (store_id, date, temp_max, temp_min, ...)
            df_economic: Optional (date, cpi, unemployment, ...)

        Returns:
            Feature DataFrame with all engineered columns
        """
        with PipelineLogger("feature_engineering"):
            df = df_sales.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

            self._log.info("feature_engineering_start", rows=len(df))

            # Feature groups
            df = self._add_time_features(df)
            df = self._add_price_features(df, df_prices, df_calendar)
            df = self._add_lag_features(df)
            df = self._add_rolling_features(df)
            df = self._add_demand_features(df)
            df = self._add_calendar_features(df, df_calendar)

            if df_weather is not None:
                df = self._add_weather_features(df, df_weather)

            if df_economic is not None:
                df = self._add_economic_features(df, df_economic)

            df["feature_version"] = self.feature_version

            self._log.info(
                "feature_engineering_complete",
                rows=len(df),
                feature_count=len(df.columns),
            )
            return df

    # -----------------------------------------------------------------------
    # Time Features
    # -----------------------------------------------------------------------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode temporal features."""
        dt = df["date"].dt

        df["day_of_week"] = dt.dayofweek       # 0=Monday, 6=Sunday
        df["day_of_month"] = dt.day
        df["week_of_year"] = dt.isocalendar().week.astype(int)
        df["month"] = dt.month
        df["quarter"] = dt.quarter
        df["year"] = dt.year
        df["is_weekend"] = (dt.dayofweek >= 5).astype(bool)
        df["is_month_start"] = dt.is_month_start
        df["is_month_end"] = dt.is_month_end
        df["days_in_month"] = dt.days_in_month

        # Cyclical encoding for periodic features (prevents ordinal assumption)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
        df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

        # Holiday features
        df["is_holiday"] = df["date"].apply(
            lambda d: d in self._us_holidays
        ).astype(bool)
        df["holiday_name"] = df["date"].apply(
            lambda d: self._us_holidays.get(d, None)
        )

        # Days to/from nearest holiday
        holiday_dates = pd.DatetimeIndex(sorted(self._us_holidays.keys()))
        df["days_to_holiday"] = df["date"].apply(
            lambda d: self._days_to_nearest(d, holiday_dates, direction="forward")
        )
        df["days_after_holiday"] = df["date"].apply(
            lambda d: self._days_to_nearest(d, holiday_dates, direction="backward")
        )

        return df

    @staticmethod
    def _days_to_nearest(
        d: pd.Timestamp,
        holiday_dates: pd.DatetimeIndex,
        direction: str,
    ) -> int:
        if direction == "forward":
            future = holiday_dates[holiday_dates >= d]
            return (future[0] - d).days if len(future) > 0 else 365
        else:
            past = holiday_dates[holiday_dates <= d]
            return (d - past[-1]).days if len(past) > 0 else 365

    # -----------------------------------------------------------------------
    # Price Features
    # -----------------------------------------------------------------------

    def _add_price_features(
        self,
        df: pd.DataFrame,
        df_prices: pd.DataFrame,
        df_calendar: pd.DataFrame,
    ) -> pd.DataFrame:
        """Engineer price-related features for elasticity modeling."""
        # Join wm_yr_wk from calendar
        cal = df_calendar[["date", "wm_yr_wk"]].copy()
        cal["date"] = pd.to_datetime(cal["date"])
        df = df.merge(cal, on="date", how="left")

        # Join sell prices
        df = df.merge(
            df_prices[["item_id", "store_id", "wm_yr_wk", "sell_price"]],
            on=["item_id", "store_id", "wm_yr_wk"],
            how="left",
        )

        # Week-over-week price change
        df = df.sort_values(["item_id", "store_id", "date"])
        df["prev_week_price"] = df.groupby(["item_id", "store_id"])["sell_price"].shift(7)
        df["price_change"] = df["sell_price"] - df["prev_week_price"]
        df["price_change_pct"] = (
            df["price_change"] / df["prev_week_price"].replace(0, np.nan)
        )
        df["is_on_sale"] = (df["price_change_pct"] < -0.05).fillna(False)
        df["discount_pct"] = df["price_change_pct"].clip(upper=0).abs()

        # Relative price: this item's price vs category mean price
        cat_avg_price = (
            df.groupby(["date", "dept_id"])["sell_price"]
            .transform("mean")
            .rename("cat_avg_price")
        )
        df["relative_price"] = df["sell_price"] / cat_avg_price.replace(0, np.nan)

        # Price volatility (rolling 4-week std)
        df["price_vol_28"] = (
            df.groupby(["item_id", "store_id"])["sell_price"]
            .transform(lambda x: x.rolling(4, min_periods=1).std())
        )

        return df

    # -----------------------------------------------------------------------
    # Lag Features
    # -----------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for autoregressive demand signals."""
        for lag in self.LAG_WINDOWS:
            df[f"lag_{lag}"] = (
                df.groupby(["item_id", "store_id"])["sales"]
                .shift(lag)
            )

        # Lag of price (for price elasticity modeling)
        if "sell_price" in df.columns:
            for lag in [1, 7]:
                df[f"price_lag_{lag}"] = (
                    df.groupby(["item_id", "store_id"])["sell_price"]
                    .shift(lag)
                )

        return df

    # -----------------------------------------------------------------------
    # Rolling Features
    # -----------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling window statistics."""
        for window in self.ROLLING_WINDOWS:
            grp = df.groupby(["item_id", "store_id"])["sales"]
            df[f"rolling_mean_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window // 2)).mean()
            )
            df[f"rolling_std_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window // 2)).std()
            )
            df[f"rolling_max_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )
            df[f"rolling_min_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).min()
            )
            df[f"rolling_median_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window // 2)).median()
            )

        # Expanding mean (all-history average) — useful for new items
        df["expanding_mean"] = (
            df.groupby(["item_id", "store_id"])["sales"]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

        return df

    # -----------------------------------------------------------------------
    # Demand Features
    # -----------------------------------------------------------------------

    def _add_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Demand velocity and volatility — key for safety stock calculations.
        """
        # Demand velocity: rate of change
        df["demand_velocity_7"] = (
            df["rolling_mean_7"] - df.groupby(["item_id", "store_id"])["rolling_mean_7"].shift(7)
        )

        # Demand volatility: coefficient of variation
        df["demand_cv_28"] = (
            df["rolling_std_28"] / df["rolling_mean_28"].replace(0, np.nan)
        )

        # Trend indicator: is demand trending up or down?
        df["trend_7_28"] = df["rolling_mean_7"] / df["rolling_mean_28"].replace(0, np.nan)
        df["trend_up"] = (df["trend_7_28"] > 1.05).astype(int)
        df["trend_down"] = (df["trend_7_28"] < 0.95).astype(int)

        # Intermittent demand indicator (many zeros = intermittent)
        df["zero_7d"] = (
            df.groupby(["item_id", "store_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).apply(lambda v: (v == 0).sum()))
        )

        return df

    # -----------------------------------------------------------------------
    # Calendar / Event Features
    # -----------------------------------------------------------------------

    def _add_calendar_features(
        self, df: pd.DataFrame, df_calendar: pd.DataFrame
    ) -> pd.DataFrame:
        """Add M5-specific calendar events (SNAP, sporting events, etc.)."""
        cal = df_calendar.copy()
        cal.columns = [c.lower() for c in cal.columns]
        cal["date"] = pd.to_datetime(cal["date"])

        # Map SNAP flags by state
        state_snap_map = {"CA": "snap_ca", "TX": "snap_tx", "WI": "snap_wi"}
        cols_to_merge = [c for c in ["date", "event_name_1", "event_type_1", "event_name_2",
                                     "event_type_2", "snap_ca", "snap_tx", "snap_wi"] if c in cal.columns]
        df = df.merge(
            cal[cols_to_merge],
            on="date",
            how="left",
        )

        # Apply state-specific SNAP flag
        if "state_id" in df.columns:
            df["snap_flag"] = df.apply(
                lambda row: bool(row.get(state_snap_map.get(row.get("state_id", ""), ""), 0)),
                axis=1,
            )
        else:
            df["snap_flag"] = False

        # One-hot encode event types
        df["is_sporting_event"] = (df["event_type_1"] == "Sporting").fillna(False)
        df["is_cultural_event"] = (df["event_type_1"] == "Cultural").fillna(False)
        df["is_national_event"] = (df["event_type_1"] == "National").fillna(False)
        df["is_religious_event"] = (df["event_type_1"] == "Religious").fillna(False)

        return df

    # -----------------------------------------------------------------------
    # Weather Features
    # -----------------------------------------------------------------------

    def _add_weather_features(
        self, df: pd.DataFrame, df_weather: pd.DataFrame
    ) -> pd.DataFrame:
        """Join weather data to sales DataFrame."""
        weather = df_weather.copy()
        weather["date"] = pd.to_datetime(weather["date"])

        df = df.merge(
            weather[["store_id", "date", "temp_max", "temp_min", "precipitation", "wind_speed"]],
            on=["store_id", "date"],
            how="left",
        )

        # Derived weather features
        df["temp_range"] = df["temp_max"] - df["temp_min"]
        df["is_rainy"] = (df["precipitation"] > 5.0).fillna(False)
        df["is_cold"] = (df["temp_max"] < 5.0).fillna(False)  # Celsius
        df["is_hot"] = (df["temp_max"] > 35.0).fillna(False)

        return df

    # -----------------------------------------------------------------------
    # Economic Features
    # -----------------------------------------------------------------------

    def _add_economic_features(
        self, df: pd.DataFrame, df_economic: pd.DataFrame
    ) -> pd.DataFrame:
        """Join macroeconomic indicators."""
        econ = df_economic.copy()
        econ["date"] = pd.to_datetime(econ["date"])

        # Economic data is usually monthly — forward-fill to daily
        df = df.merge(econ[["date", "cpi", "unemployment", "consumer_confidence"]], on="date", how="left")
        for col in ["cpi", "unemployment", "consumer_confidence"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        return df

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names generated (excluding target and IDs)."""
        return [
            # Time
            "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
            "is_weekend", "is_month_start", "is_month_end",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
            "is_holiday", "days_to_holiday", "days_after_holiday",
            # Price
            "sell_price", "price_change", "price_change_pct", "is_on_sale",
            "discount_pct", "relative_price", "price_vol_28",
            # Lags
            *[f"lag_{w}" for w in self.LAG_WINDOWS],
            # Rolling
            *[f"rolling_mean_{w}" for w in self.ROLLING_WINDOWS],
            *[f"rolling_std_{w}" for w in self.ROLLING_WINDOWS],
            *[f"rolling_max_{w}" for w in [7, 28]],
            "rolling_median_28", "expanding_mean",
            # Demand
            "demand_velocity_7", "demand_cv_28", "trend_7_28", "zero_7d",
            # Calendar
            "snap_flag", "is_sporting_event", "is_cultural_event",
            "is_national_event", "is_religious_event",
            # Weather
            "temp_max", "temp_min", "temp_range", "precipitation",
            "is_rainy", "is_cold", "is_hot",
            # Economic
            "cpi", "unemployment", "consumer_confidence",
        ]
