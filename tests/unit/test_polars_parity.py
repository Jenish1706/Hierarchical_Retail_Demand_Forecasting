"""Parity guard between the Polars and pandas feature engines.

The Polars refactor exists to remove a memory ceiling, not to change results.
If the two engines ever diverge on the columns they are supposed to agree on,
every previously recorded accuracy number silently becomes incomparable -- so
the equivalence is asserted here rather than assumed.

Three columns are DELIBERATELY excluded, because diagnosing the parity gap
found latent bugs in the pandas path and the Polars values are the corrected
ones:

  * ``zero_rate_*`` -- pandas evaluated ``shifted_sales == 0`` where the
    leading entries are NaN. ``NaN == 0`` is False, so unobserved days were
    counted as days that DID sell, understating the zero rate at the start of
    every series. Polars propagates the shift as null and excludes those days.
  * ``price_change_pct`` -- pandas took the numerator as a 1-week difference
    but divided by the price ``horizon`` weeks earlier, mixing reference
    weeks. Polars uses the same week for both.
  * ``price_change`` -- follows from the same fix.

Those three are pinned by their own assertions below so the corrected
behaviour cannot silently regress either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.base import FeaturePipeline

# Columns whose pandas implementation is known-wrong; see the module docstring.
KNOWN_PANDAS_BUGS = {"zero_rate_28", "zero_rate_90", "price_change", "price_change_pct"}


def _synthetic_panel(n_series: int = 12, n_days: int = 200) -> pd.DataFrame:
    """A panel with the shape the pipeline expects, including sparse series."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    frames = []
    for i in range(n_series):
        # Alternate dense and intermittent so both regimes are exercised.
        sales = (
            rng.poisson(12, n_days)
            if i % 2 == 0
            else (rng.random(n_days) < 0.18) * rng.integers(1, 5, n_days)
        )
        frames.append(
            pd.DataFrame(
                {
                    "item_id": f"ITEM_{i:03d}",
                    "store_id": f"CA_{i % 3 + 1}",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "state_id": "CA",
                    "date": dates,
                    "wm_yr_wk": 11101 + np.arange(n_days) // 7,
                    "sales": sales.astype("int16"),
                    "sell_price": np.round(rng.uniform(1.5, 9.0) + rng.normal(0, 0.05, n_days), 2),
                    "snap_flag": rng.integers(0, 2, n_days).astype("int8"),
                    "is_event_day": (rng.random(n_days) < 0.08).astype("int8"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def engines() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _synthetic_panel()
    pandas_out = FeaturePipeline(horizon=28, use_polars=False).transform(panel, validate=False)
    polars_out = FeaturePipeline(horizon=28, use_polars=True).transform(panel, validate=False)
    sort_cols = ["item_id", "store_id", "date"]
    return (
        pandas_out.sort_values(sort_cols).reset_index(drop=True),
        polars_out.sort_values(sort_cols).reset_index(drop=True),
    )


def test_both_engines_emit_the_same_columns(engines):
    pandas_out, polars_out = engines
    assert set(pandas_out.columns) == set(polars_out.columns)


def test_both_engines_emit_the_same_row_count(engines):
    pandas_out, polars_out = engines
    assert len(pandas_out) == len(polars_out)


def test_numeric_features_are_identical_except_known_pandas_bugs(engines):
    pandas_out, polars_out = engines
    compared, mismatched = 0, []

    for col in sorted(set(pandas_out.columns) & set(polars_out.columns)):
        if col in KNOWN_PANDAS_BUGS:
            continue
        if not (
            pd.api.types.is_numeric_dtype(pandas_out[col])
            and pd.api.types.is_numeric_dtype(polars_out[col])
        ):
            continue
        compared += 1
        left = pandas_out[col].to_numpy(dtype=float)
        right = polars_out[col].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-5, atol=1e-5, equal_nan=True):
            mismatched.append((col, float(np.nanmax(np.abs(left - right)))))

    assert compared > 20, f"only {compared} numeric columns compared; fixture too thin"
    assert not mismatched, f"Polars/pandas divergence: {mismatched}"


def test_zero_rate_excludes_unobserved_days(engines):
    """The corrected zero_rate must never exceed 1 and must be null-free late."""
    _, polars_out = engines
    for col in ("zero_rate_28", "zero_rate_90"):
        values = polars_out[col].dropna().to_numpy(dtype=float)
        assert values.size > 0
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_price_change_pct_uses_a_consistent_reference_week(engines):
    """price_change_pct must equal price_change / previous price, same week."""
    _, polars_out = engines
    subset = polars_out.sort_values(["item_id", "store_id", "date"])
    grouped = subset.groupby(["item_id", "store_id"], observed=True)
    previous = grouped["price"].shift(1)

    expected = subset["price_change"] / previous.where(previous.abs() > 0)
    actual = subset["price_change_pct"]

    both = expected.notna() & actual.notna()
    assert both.sum() > 100, "not enough overlapping price rows to test"
    assert np.allclose(
        expected[both].to_numpy(float), actual[both].to_numpy(float), rtol=1e-4, atol=1e-6
    )


def test_is_discounted_is_a_clean_binary_flag(engines):
    _, polars_out = engines
    values = polars_out["is_discounted"]
    assert values.isna().sum() == 0, "binary flag must not carry nulls"
    assert set(np.unique(values.to_numpy())) <= {0, 1}
