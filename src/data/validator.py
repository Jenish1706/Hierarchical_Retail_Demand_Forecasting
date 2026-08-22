"""
Data validation framework using Great Expectations and Pandera.

Implements multi-layer validation:
  Layer 1: Schema validation (column types, nullability, cross-column logic)
  Layer 2: Completeness validation (continuity, missing values)
  Layer 3: Business rule validation (domain-specific constraints, outliers)
  Layer 4: Statistical validation (distribution tests, zero-inflation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera as pa
from pandera.typing import Series

from src.config.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pandera Schemas (Layer 1 & 2: Type + Statistical)
# ---------------------------------------------------------------------------

class RawSalesSchema(pa.DataFrameModel):
    """Schema validation for raw M5 sales data."""

    item_id: Series[str] = pa.Field(nullable=False, str_contains="HOBBIES|HOUSEHOLD|FOODS")
    dept_id: Series[str] = pa.Field(nullable=False)
    cat_id: Series[str] = pa.Field(nullable=False, isin=["HOBBIES", "HOUSEHOLD", "FOODS"])
    store_id: Series[str] = pa.Field(nullable=False)
    state_id: Series[str] = pa.Field(nullable=False, isin=["CA", "TX", "WI"])
    date: Series[pa.DateTime] = pa.Field(nullable=False)
    sales: Series[float] = pa.Field(
        nullable=False,
        ge=0,          # Sales cannot be negative
        le=10_000,     # Outlier cap — flag anything above
    )

    @pa.dataframe_check
    def check_store_state_match(cls, df: pd.DataFrame) -> Series[bool]:
        """Store ID prefix must match State ID."""
        return df["store_id"].str[:2] == df["state_id"]

    class Config:
        name = "RawSalesSchema"
        strict = False
        coerce = True


class FeatureSchema(pa.DataFrameModel):
    """Schema for engineered features before model training."""

    item_id: Series[str] = pa.Field(nullable=False)
    store_id: Series[str] = pa.Field(nullable=False)
    date: Series[pa.DateTime] = pa.Field(nullable=False)
    sales: Series[float] = pa.Field(nullable=False, ge=0, le=10_000)
    sell_price: Series[float] = pa.Field(nullable=True, ge=0)
    lag_1: Series[float] = pa.Field(nullable=True, ge=0, le=10_000)
    lag_7: Series[float] = pa.Field(nullable=True, ge=0, le=10_000)
    rolling_mean_7: Series[float] = pa.Field(nullable=True, ge=0, le=10_000)
    rolling_mean_28: Series[float] = pa.Field(nullable=True, ge=0, le=10_000)
    day_of_week: Series[int] = pa.Field(nullable=False, ge=0, le=6)
    month: Series[int] = pa.Field(nullable=False, ge=1, le=12)
    is_weekend: Series[bool] = pa.Field(nullable=False)
    is_holiday: Series[bool] = pa.Field(nullable=False)

    class Config:
        name = "FeatureSchema"
        strict = False
        coerce = True


# ---------------------------------------------------------------------------
# Validation Result Types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Structured result from a validation run."""
    dataset_name: str
    total_rows: int
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "total_rows": self.total_rows,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class SalesDataValidator:
    """
    Multi-layer validator for sales data.
    Run after ingestion, before feature engineering.
    """

    # Thresholds
    MAX_ZERO_SALE_PCT = 0.40      # Flag if >40% zero sales for a SKU/store
    HIGH_ZERO_SALE_PCT = 0.80     # Flag if >80% zero sales (highly intermittent)
    MIN_COMPLETENESS = 0.95       # Require 95% non-null for critical fields
    Z_SCORE_THRESHOLD = 4.0       # Dynamic outlier detection threshold

    def validate(self, df: pd.DataFrame, dataset_name: str = "sales") -> ValidationResult:
        """Run all validation layers. Returns consolidated result."""
        result = ValidationResult(
            dataset_name=dataset_name,
            total_rows=len(df),
            passed=True,
        )

        self._validate_schema(df, result)
        self._validate_completeness(df, result)
        self._validate_business_rules(df, result)
        self._validate_statistical(df, result)

        if result.errors:
            result.passed = False

        log.info(
            "validation_complete",
            dataset=dataset_name,
            rows=result.total_rows,
            passed=result.passed,
            errors=result.error_count,
            warnings=result.warning_count,
        )
        return result

    def _validate_schema(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Layer 1: Check required columns exist and have correct types."""
        required_cols = {
            "item_id": "object",
            "store_id": "object",
            "date": None,  # Will check as datetime
            "sales": "float64",
        }

        for col, expected_dtype in required_cols.items():
            if col not in df.columns:
                result.errors.append(f"Missing required column: '{col}'")
            elif expected_dtype and not pd.api.types.is_numeric_dtype(df[col]) and expected_dtype == "float64":
                result.errors.append(f"Column '{col}' has wrong dtype: {df[col].dtype}")

        # Check date parsing
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
            except Exception:
                result.errors.append("Column 'date' cannot be parsed as datetime")

        # Try Pandera Schema (will catch state/store mismatch and negative values)
        if all(c in df.columns for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sales"]):
            try:
                RawSalesSchema.validate(df)
            except pa.errors.SchemaError as e:
                result.errors.append(f"Schema error: {e}")

    def _validate_completeness(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Layer 2: Check completeness for critical fields and time series continuity."""
        critical_cols = ["item_id", "store_id", "date", "sales"]
        for col in critical_cols:
            if col not in df.columns:
                continue
            null_pct = df[col].isna().mean()
            if null_pct > (1 - self.MIN_COMPLETENESS):
                result.errors.append(
                    f"Column '{col}' completeness {1-null_pct:.1%} < threshold {self.MIN_COMPLETENESS:.1%}"
                )

        # Time Series Continuity Check
        if "date" in df.columns and "item_id" in df.columns and "store_id" in df.columns:
            # Check if there are gaps in the dates for any item_id, store_id
            df_dates = df.groupby(["item_id", "store_id"])["date"].agg(["min", "max", "count"])
            df_dates["expected_count"] = (pd.to_datetime(df_dates["max"]) - pd.to_datetime(df_dates["min"])).dt.days + 1
            missing_dates = df_dates[df_dates["count"] < df_dates["expected_count"]]
            if not missing_dates.empty:
                result.warnings.append(
                    f"Found {len(missing_dates)} item/store combinations with missing dates in their time series"
                )

    def _validate_business_rules(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Layer 3: Domain-specific business rule checks."""
        if "sales" in df.columns:
            # Negative sales
            neg_count = (df["sales"] < 0).sum()
            if neg_count > 0:
                result.errors.append(f"Found {neg_count} rows with negative sales")

            # Dynamic Z-Score Outlier Detection
            sales = df["sales"]
            mean_sales = sales.mean()
            std_sales = sales.std()
            if std_sales > 0:
                z_scores = (sales - mean_sales) / std_sales
                outliers = (z_scores > self.Z_SCORE_THRESHOLD).sum()
                if outliers > 0:
                    result.warnings.append(
                        f"Found {outliers} sales values exceeding Z-score threshold of {self.Z_SCORE_THRESHOLD}"
                    )

            # Extreme outliers (> 99.99th percentile is suspicious)
            q9999 = df["sales"].quantile(0.9999)
            extreme_count = (df["sales"] > q9999).sum()
            if extreme_count > 0:
                result.warnings.append(
                    f"Found {extreme_count} sales values above 99.99th percentile ({q9999:.2f})"
                )


        # Date range check
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            if dates.min().year < 2010 or dates.max().year > 2030:
                result.warnings.append(
                    f"Date range seems unusual: {dates.min().date()} to {dates.max().date()}"
                )

    def _validate_statistical(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Layer 4: Statistical distribution checks."""
        if "sales" not in df.columns:
            return

        sales = df["sales"].dropna()
        result.stats.update({
            "sales_mean": float(sales.mean()),
            "sales_std": float(sales.std()),
            "sales_median": float(sales.median()),
            "sales_p95": float(sales.quantile(0.95)),
            "sales_p99": float(sales.quantile(0.99)),
            "zero_sale_pct": float((sales == 0).mean()),
        })

        # Zero-Inflation Classification
        if "item_id" in df.columns and "store_id" in df.columns:
            zero_pct_by_sku = (
                df.groupby(["item_id", "store_id"])["sales"]
                .apply(lambda x: (x == 0).mean())
            )
            intermittent_skus = (zero_pct_by_sku > self.MAX_ZERO_SALE_PCT).sum()
            highly_intermittent_skus = (zero_pct_by_sku > self.HIGH_ZERO_SALE_PCT).sum()

            if highly_intermittent_skus > 0:
                result.warnings.append(
                    f"{highly_intermittent_skus} SKU/store combinations have >{self.HIGH_ZERO_SALE_PCT:.0%} zero sales (highly intermittent)"
                )
            elif intermittent_skus > 0:
                result.warnings.append(
                    f"{intermittent_skus} SKU/store combinations have >{self.MAX_ZERO_SALE_PCT:.0%} zero sales (intermittent demand)"
                )

        # Coefficient of variation check
        cv = sales.std() / sales.mean() if sales.mean() > 0 else 0
        if cv > 10:
            result.warnings.append(
                f"Very high coefficient of variation ({cv:.2f}) — data may contain extreme outliers"
            )


class PriceDataValidator:
    """Validator for pricing data."""

    MAX_PRICE_CHANGE_WOW = 0.50  # 50% week-over-week change is suspicious

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult(
            dataset_name="sell_prices",
            total_rows=len(df),
            passed=True,
        )

        if "sell_price" in df.columns:
            # No zero or negative prices
            invalid = (df["sell_price"] <= 0).sum()
            if invalid > 0:
                result.errors.append(f"Found {invalid} rows with zero or negative sell_price")

            # Price range check
            p999 = df["sell_price"].quantile(0.999)
            if p999 > 10_000:
                result.warnings.append(
                    f"99.9th percentile price is ${p999:.2f} — check for data entry errors"
                )

            # Week-over-week Price Volatility Check
            if "item_id" in df.columns and "store_id" in df.columns and "wm_yr_wk" in df.columns:
                # Sort to ensure chronological order per item/store
                sorted_df = df.sort_values(["store_id", "item_id", "wm_yr_wk"])

                # Calculate percentage change week over week
                sorted_df["price_pct_change"] = sorted_df.groupby(["store_id", "item_id"])["sell_price"].pct_change().abs()

                volatile_prices = (sorted_df["price_pct_change"] > self.MAX_PRICE_CHANGE_WOW).sum()
                if volatile_prices > 0:
                    result.warnings.append(
                        f"Found {volatile_prices} instances where price changed >{self.MAX_PRICE_CHANGE_WOW:.0%} week-over-week"
                    )

            result.stats["price_min"] = float(df["sell_price"].min())
            result.stats["price_max"] = float(df["sell_price"].max())
            result.stats["price_mean"] = float(df["sell_price"].mean())

        if result.errors:
            result.passed = False

        return result
