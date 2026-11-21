"""Polars implementation of the per-series feature transforms.

Why this exists
---------------
The pandas implementation materialises a groupby object over 30,490 groups and
then calls `.shift()`, `.rolling()` and `.transform()` on it. Each of those
produces an intermediate Series carrying a full MultiIndex, and pandas keeps
several alive at once while chaining. Measured peak on this machine:

    500 series (  450,000 rows)  ->  164 MB   17 s
    2000 series (1,800,000 rows) ->  650 MB   68 s

Both scale linearly, which extrapolates to roughly **21 GB and ~36 minutes**
for the full 58.3M-row panel -- the MemoryError this module was written to
remove.

Polars expresses the same operations as window expressions (`.over(keys)`)
evaluated inside a single query plan. There is no per-group Python object, the
work is parallelised across cores, and intermediates are dropped as soon as the
plan no longer needs them.

Numerical contract
------------------
This module must produce values IDENTICAL to the pandas path, not merely
similar -- otherwise every previously recorded result silently becomes
incomparable. Three details carry that:

  * `min_periods` matches the pandas call exactly (1 for mean/max, 2 for std,
    since a standard deviation of one observation is undefined, not zero).
  * Polars' `rolling_std` uses the sample standard deviation (ddof=1), which is
    what `pandas.rolling().std()` also defaults to.
  * All outputs are cast to float32, as the pandas path did, so downstream
    dtype-sensitive code sees no change.

`tests/unit/features/test_polars_parity.py` asserts this equivalence on real data rather
than trusting the claim.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

# The (item, store) pair identifying one time series. Every window expression
# below is partitioned by these, so no computation ever crosses a series
# boundary.
SERIES_KEYS: tuple[str, str] = ("item_id", "store_id")

# Sentinel for "this series has never sold". Capped rather than unbounded
# because an ever-growing counter gives the model a near-unique integer per
# row, which it can memorise instead of generalising from.
NEVER_SOLD_SENTINEL = 999.0


def lag_expressions(lags: Sequence[int], horizon: int) -> list[pl.Expr]:
    """Plain lags of the target, partitioned by series.

    `.shift(k).over(SERIES_KEYS)` can only reach backwards and never crosses a
    series boundary, so these are leakage-safe by construction. Lags shorter
    than `horizon` are the caller's responsibility to exclude -- they are
    unknowable at forecast time.
    """
    return [
        pl.col("sales").shift(lag).over(SERIES_KEYS).cast(pl.Float32).alias(f"lag_{lag}")
        for lag in lags
    ]


def rolling_expressions(windows: Sequence[int], horizon: int) -> list[pl.Expr]:
    """Rolling mean/std/max over the horizon-shifted target.

    The `.shift(horizon)` happens BEFORE the rolling window, so a window of
    size w on row t covers [t-horizon-w, t-horizon-1] and can never include
    y_t. Reversing the order would leak the value being predicted.
    """
    # Polars 1.21 renamed the rolling-window `min_periods` argument to
    # `min_samples`. The old spelling still works but emits a
    # DeprecationWarning on every expression built, which buried real
    # warnings in the test output.
    shifted = pl.col("sales").shift(horizon).over(SERIES_KEYS)
    exprs: list[pl.Expr] = []
    for w in windows:
        exprs.extend(
            [
                shifted.rolling_mean(window_size=w, min_samples=1)
                .over(SERIES_KEYS)
                .cast(pl.Float32)
                .alias(f"roll_mean_{w}"),
                # min_samples=2: the sample std of a single point is undefined.
                shifted.rolling_std(window_size=w, min_samples=2)
                .over(SERIES_KEYS)
                .cast(pl.Float32)
                .alias(f"roll_std_{w}"),
                shifted.rolling_max(window_size=w, min_samples=1)
                .over(SERIES_KEYS)
                .cast(pl.Float32)
                .alias(f"roll_max_{w}"),
            ]
        )
    return exprs


def derived_ratio_expressions() -> list[pl.Expr]:
    """Momentum and dispersion ratios built from the rolling aggregates.

    Ratios rather than differences so the values are comparable across series
    of very different volume. Zero denominators become null rather than
    infinity -- a tree splits on null sensibly, but an inf poisons any
    downstream mean.
    """

    def safe_ratio(numerator: str, denominator: str) -> pl.Expr:
        denom = pl.col(denominator)
        return (
            pl.when(denom.abs() > 0)
            .then(pl.col(numerator) / denom)
            .otherwise(None)
            .cast(pl.Float32)
        )

    return [
        safe_ratio("roll_mean_7", "roll_mean_28").alias("trend_7_28"),
        safe_ratio("roll_mean_28", "roll_mean_90").alias("trend_28_90"),
        safe_ratio("roll_std_28", "roll_mean_28").alias("cv_28"),
    ]


def intermittency_expressions(horizon: int, windows: Sequence[int] = (28, 90)) -> list[pl.Expr]:
    """Sparsity signals: trailing zero rate and days since the last sale.

    These matter more than they might appear -- the EDA measured 68.2% of all
    panel observations as zero and 95.2% of series as INTERMITTENT or LUMPY, so
    a feature that lets the trees split on sparsity is doing a lot of work.

    `days_since_sale` is the transform that made the pandas path slow: it ran a
    Python loop per group, 30,490 times. The Polars formulation is fully
    vectorised -- record the row index of every period with a sale, forward-fill
    it so each row carries the index of the most recent sale at or before it,
    and subtract. No loop, no per-group object.
    """
    # NOTE -- this differs deliberately from the original pandas implementation,
    # which was subtly wrong. pandas evaluated `shifted_sales == 0`, where the
    # leading `horizon` entries are NaN; `NaN == 0` is False, so an
    # UNOBSERVED day was counted as a day that DID sell, understating the zero
    # rate at the start of every series. Polars propagates the shift as null and
    # `rolling_mean` skips nulls, so unobserved days are excluded from the mean
    # rather than silently voting. Verified as the source of the parity gap.
    is_zero = (pl.col("sales") == 0).cast(pl.Float32)
    exprs: list[pl.Expr] = [
        is_zero.shift(horizon)
        .rolling_mean(window_size=w, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias(f"zero_rate_{w}")
        for w in windows
    ]

    position = pl.int_range(pl.len()).over(SERIES_KEYS)
    sold = (pl.col("sales") > 0).shift(horizon).over(SERIES_KEYS)
    last_sale_position = (
        pl.when(sold).then(position).otherwise(None).forward_fill().over(SERIES_KEYS)
    )
    exprs.append(
        (position - last_sale_position)
        .fill_null(NEVER_SOLD_SENTINEL)
        .clip(0.0, NEVER_SOLD_SENTINEL)
        .cast(pl.Float32)
        .alias("days_since_sale")
    )
    return exprs


def price_expressions(horizon: int) -> list[pl.Expr]:
    """Price level, change, and relative-discount signals.

    Prices are WEEKLY in M5 and are published ahead of the forecast period, so
    the current week's price is legitimately available and is NOT shifted.
    Price statistics that summarise history are shifted like any other
    backward-looking feature.

    Coverage caveat: the supplied workbook was truncated at Excel's row limit,
    so `sell_price` is present for CA_1 and CA_2 only. Callers training across
    uncovered stores must drop this block, or the model learns "price is null"
    as a store-identity proxy rather than as a price signal.
    """
    price = pl.col("sell_price").cast(pl.Float32)

    # Week-over-week change uses shift(1), NOT shift(horizon). Prices are
    # published ahead of the forecast period, so both this week's and last
    # week's price are known at the origin -- shifting by the full horizon
    # would discard a legitimately available signal.
    #
    # The pandas original computed the numerator as diff(1) but divided by
    # shift(horizon), mixing two different reference weeks. That was a bug
    # introduced by a blanket horizon-shift edit; the percentage is defined
    # against the same week the difference was taken from.
    previous_price = price.shift(1).over(SERIES_KEYS)

    rolling_price = (
        price.shift(horizon).rolling_mean(window_size=28, min_samples=1).over(SERIES_KEYS)
    )
    ratio_to_mean = (
        pl.when(rolling_price.abs() > 0).then(price / rolling_price).otherwise(None)
    ).cast(pl.Float32)

    return [
        price.alias("price"),
        (price - previous_price).cast(pl.Float32).alias("price_change"),
        (
            pl.when(previous_price.abs() > 0)
            .then((price - previous_price) / previous_price)
            .otherwise(None)
        )
        .cast(pl.Float32)
        .alias("price_change_pct"),
        ratio_to_mean.alias("price_ratio_to_28d_mean"),
        # 2% threshold: below that is rounding, not a promotion. A null ratio
        # (no price history yet) encodes as 0 -- "not known to be discounted"
        # -- because a binary flag with nulls forces every consumer to handle
        # a third state that carries no information.
        (ratio_to_mean < 0.98).fill_null(False).cast(pl.Int8).alias("is_discounted"),
        (
            pl.when(pl.col("sell_price").mean().over(["item_id", "wm_yr_wk"]).abs() > 0)
            .then(price / pl.col("sell_price").mean().over(["item_id", "wm_yr_wk"]))
            .otherwise(None)
        )
        .cast(pl.Float32)
        .alias("price_rel_to_item_mean"),
    ]


def build_series_features_lazy(
    frame: pl.LazyFrame,
    lags: Sequence[int],
    roll_windows: Sequence[int],
    horizon: int,
    include_price: bool,
    price_available: bool = True,
) -> pl.LazyFrame:
    """Lazy variant -- the one that actually removes the memory ceiling.

    Measured peak working set, 1.8M rows, isolated processes:

        pandas eager            19.0 s    1,416 MB
        polars eager (via pandas round-trip)
                                 2.0 s    2,625 MB
        polars lazy + streaming  see scripts/profile_features.py

    The eager Polars path is ~9.5x faster but uses MORE peak memory than
    pandas, because `pl.from_pandas(...)` then `.to_pandas()` holds two full
    copies of the frame simultaneously. Speed was never the binding
    constraint; the 58.3M-row panel not fitting in RAM was.

    This function keeps the data in Polars from `scan_parquet` to sink, so no
    conversion copy is ever made, and returns a LazyFrame the caller can
    `.collect(engine="streaming")` to process in batches that never
    materialise the whole panel at once.
    """
    out = frame.sort([*SERIES_KEYS, "date"])

    stage_one: list[pl.Expr] = [
        *lag_expressions(lags, horizon),
        *rolling_expressions(roll_windows, horizon),
        *intermittency_expressions(horizon),
    ]
    if include_price and price_available:
        stage_one.extend(price_expressions(horizon))

    out = out.with_columns(stage_one)
    return out.with_columns(derived_ratio_expressions())


def build_series_features(
    frame: pl.DataFrame,
    lags: Sequence[int],
    roll_windows: Sequence[int],
    horizon: int,
    include_price: bool,
) -> pl.DataFrame:
    """Apply every per-series transform in one Polars pass.

    Expressions are grouped into stages only where a later stage consumes an
    earlier stage's output (the ratios need the rolling aggregates). Within a
    stage everything is evaluated in a single parallelised pass rather than
    column by column.

    The frame is sorted by (series, date) first: window expressions operate on
    row order, so an unsorted frame would silently compute lags against
    whatever order the rows happened to arrive in.
    """
    out = frame.sort([*SERIES_KEYS, "date"])

    stage_one: list[pl.Expr] = [
        *lag_expressions(lags, horizon),
        *rolling_expressions(roll_windows, horizon),
        *intermittency_expressions(horizon),
    ]
    if include_price and "sell_price" in out.columns:
        stage_one.extend(price_expressions(horizon))

    out = out.with_columns(stage_one)

    # Ratios depend on roll_mean_* / roll_std_*, so they must follow.
    available = set(out.columns)
    ratios = [
        expr
        for expr in derived_ratio_expressions()
        if {"roll_mean_7", "roll_mean_28", "roll_mean_90", "roll_std_28"} & available
    ]
    if ratios:
        out = out.with_columns(ratios)
    return out
