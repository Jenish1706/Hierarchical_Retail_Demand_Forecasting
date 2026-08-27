"""
M5 data loader: Parquet -> long-format panel, with optional PostgreSQL sink.

Design
------
Parquet-first, database-optional. The previous loader could only write to
PostgreSQL, which meant no one could compute a feature or run a backtest
without standing up docker-compose first. Since the modelling work needs the
data far more often than the serving layer does, the dependency is inverted
here: `load_panel()` returns a DataFrame from local Parquet, and pushing to
PostgreSQL is a separate, opt-in step.

The wide -> long melt
---------------------
`sales_train_validation` is stored wide: one row per series, one column per
day (30,490 x 1,913). Melting produces 58.3 million rows, which is fine for
Polars but wasteful when an experiment only needs three departments or a
90-day window. So every subsetting option is applied BEFORE the melt, on the
narrow wide frame, rather than after it on 58 million rows.

Memory notes:
  * `d_*` columns are int16 on disk (max daily unit sale in M5 is in the
    hundreds; int16 holds 32,767).
  * `item_id` / `store_id` / `dept_id` / `cat_id` / `state_id` become
    Categorical in the returned frame -- 30,490 series across 3,049 item
    strings would otherwise cost more in repeated Python strings than the
    sales data itself.

Price coverage
--------------
The supplied workbook's price sheet was truncated at Excel's row limit, so
`sell_price` is present only for the stores in `M5.STORES_WITH_PRICE_DATA`.
`load_panel()` therefore returns an explicit `has_price_data` flag rather
than leaving callers to infer coverage from nulls -- a null price means "this
store was cut from the file", not "this item was not for sale".
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from src.config.logging import PipelineLogger, get_logger
from src.config.settings import M5, settings

log = get_logger(__name__)

# Output stems written by scripts/extract_workbook.py.
PARQUET_FILES = {
    "sales": "sales_train_validation.parquet",
    "calendar": "calendar.parquet",
    "prices": "sell_prices.parquet",
    "submission": "sales_submission.parquet",
}

ID_COLS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")


class M5Loader:
    """
    Reads the extracted Parquet files and assembles the modelling panel.

    Usage:
        loader = M5Loader()
        df = loader.load_panel(stores=["CA_1"], categories=["FOODS"])
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else settings.raw_data_dir
        self.run_id = str(uuid.uuid4())[:8]
        self._log = get_logger("M5Loader")
        self._calendar: pd.DataFrame | None = None

    # -- path helpers -------------------------------------------------------

    def _path(self, key: str) -> Path:
        path = self.data_dir / PARQUET_FILES[key]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/extract_workbook.py` "
                f"to generate the Parquet files from the source workbook."
            )
        return path

    def available(self) -> dict[str, bool]:
        """Which extracted tables are present on disk."""
        return {k: (self.data_dir / v).exists() for k, v in PARQUET_FILES.items()}

    # -- calendar -----------------------------------------------------------

    def load_calendar(self) -> pd.DataFrame:
        """
        Calendar with derived event and SNAP columns.

        Cached: it is only 1,969 rows but is joined on every panel load.
        """
        if self._calendar is not None:
            return self._calendar

        cal = pd.read_parquet(self._path("calendar"))
        cal["date"] = pd.to_datetime(cal["date"])

        # `d` is the join key back to the wide sales columns.
        cal["d"] = cal["d"].astype(str)

        # Single flag for "any named event on this day". Both event slots are
        # checked because a handful of dates carry two (e.g. Easter falling on
        # the same day as Orthodox Easter).
        cal["is_event_day"] = (cal["event_name_1"].notna() | cal["event_name_2"].notna()).astype(
            "int8"
        )

        for col in ("snap_CA", "snap_TX", "snap_WI"):
            cal[col] = pd.to_numeric(cal[col], errors="coerce").fillna(0).astype("int8")

        self._calendar = cal
        self._log.info(
            "calendar_loaded",
            rows=len(cal),
            event_days=int(cal["is_event_day"].sum()),
            date_min=str(cal["date"].min().date()),
            date_max=str(cal["date"].max().date()),
        )
        return cal

    # -- prices -------------------------------------------------------------

    def load_prices(self) -> pd.DataFrame:
        """Weekly sell prices. Coverage is partial -- see the module docstring."""
        prices = pd.read_parquet(self._path("prices"))
        prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
        prices["sell_price"] = prices["sell_price"].astype("float32")

        covered = sorted(prices["store_id"].unique().tolist())
        self._log.info(
            "prices_loaded",
            rows=len(prices),
            stores_covered=covered,
            coverage_note=f"{M5.PRICE_ROW_COVERAGE_PCT}% of the full M5 price table",
        )
        return prices

    # -- the main entry point ------------------------------------------------

    def load_panel(
        self,
        stores: Sequence[str] | None = None,
        states: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        departments: Sequence[str] | None = None,
        items: Sequence[str] | None = None,
        sample_series: int | None = None,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        last_n_days: int | None = None,
        with_prices: bool = True,
        with_calendar: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Build a long-format panel: one row per (item, store, date).

        Every filter is applied to the WIDE frame before melting, so memory
        scales with the subset requested rather than with the full 58.3M-row
        panel.

        Args:
            stores/states/categories/departments/items: categorical filters.
            sample_series: take a random sample of this many series AFTER the
                categorical filters. Sampling series (not rows) preserves each
                selected series' full history, which is what lag features and
                the RMSSE scale require -- sampling rows would shred it.
            start_date/end_date: inclusive date bounds.
            last_n_days: keep only the final N days. Applied after the date
                bounds; convenient for backtests that only need recent history.
            with_prices: join weekly sell_price on (store_id, item_id, wm_yr_wk).
            with_calendar: join calendar attributes (weekday, SNAP, events).

        Returns:
            DataFrame with at least: id, item_id, dept_id, cat_id, store_id,
            state_id, date, d, sales -- plus calendar and price columns when
            requested, and a `has_price_data` coverage flag.
        """
        with PipelineLogger("m5_load_panel", self.run_id):
            wide = pl.read_parquet(self._path("sales"))

            # --- categorical filters, applied while still wide ---
            filters: list[pl.Expr] = []
            if stores:
                filters.append(pl.col("store_id").is_in(list(stores)))
            if states:
                filters.append(pl.col("state_id").is_in(list(states)))
            if categories:
                filters.append(pl.col("cat_id").is_in(list(categories)))
            if departments:
                filters.append(pl.col("dept_id").is_in(list(departments)))
            if items:
                filters.append(pl.col("item_id").is_in(list(items)))
            for expr in filters:
                wide = wide.filter(expr)

            if wide.height == 0:
                raise ValueError("no series matched the requested filters")

            if sample_series is not None and sample_series < wide.height:
                wide = wide.sample(n=sample_series, seed=random_state)

            n_series = wide.height

            # --- restrict day columns before melting ---
            day_cols = [c for c in wide.columns if c.startswith("d_")]
            keep_days = self._select_day_columns(day_cols, start_date, end_date, last_n_days)
            if not keep_days:
                raise ValueError("no days matched the requested date range")

            id_cols = [c for c in ID_COLS if c in wide.columns]
            wide = wide.select(id_cols + keep_days)

            # --- wide -> long ---
            long = wide.unpivot(
                index=id_cols,
                on=keep_days,
                variable_name="d",
                value_name="sales",
            )
            df = long.to_pandas()
            del wide, long

            df["sales"] = df["sales"].astype("int16")

            # --- attach dates ---
            cal = self.load_calendar()
            df = df.merge(cal[["d", "date", "wm_yr_wk"]], on="d", how="left")

            if with_calendar:
                cal_cols = [
                    "d",
                    "weekday",
                    "wday",
                    "month",
                    "year",
                    "event_name_1",
                    "event_type_1",
                    "event_name_2",
                    "event_type_2",
                    "is_event_day",
                    "snap_CA",
                    "snap_TX",
                    "snap_WI",
                ]
                cal_cols = [c for c in cal_cols if c in cal.columns]
                df = df.merge(cal[cal_cols], on="d", how="left")

                # Collapse the three state SNAP columns to the flag that
                # applies to each row's own state. Keeping all three would
                # hand the model two columns of pure noise per row, and a tree
                # will happily split on another state's benefit calendar.
                df["snap_flag"] = self._resolve_snap(df)

            if with_prices:
                df = self._attach_prices(df)

            df = self._downcast_categoricals(df)
            df = df.sort_values(["item_id", "store_id", "date"], kind="mergesort")
            df = df.reset_index(drop=True)

            self._log.info(
                "panel_loaded",
                rows=len(df),
                series=n_series,
                days=len(keep_days),
                date_min=str(df["date"].min().date()),
                date_max=str(df["date"].max().date()),
                price_coverage_pct=(
                    round(100.0 * df["sell_price"].notna().mean(), 1)
                    if "sell_price" in df.columns
                    else None
                ),
                memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 1),
            )
            return df

    # -- helpers ------------------------------------------------------------

    def _select_day_columns(
        self,
        day_cols: list[str],
        start_date: date | str | None,
        end_date: date | str | None,
        last_n_days: int | None,
    ) -> list[str]:
        """Map a date range onto the `d_N` column names via the calendar."""
        cal = self.load_calendar()
        d_to_date = dict(zip(cal["d"], cal["date"]))

        # Sort by the numeric suffix, not lexically: "d_10" < "d_2" as strings.
        ordered = sorted(day_cols, key=lambda c: int(c.split("_")[1]))

        if start_date is not None:
            lo = pd.Timestamp(start_date)
            ordered = [c for c in ordered if d_to_date.get(c) is not None and d_to_date[c] >= lo]
        if end_date is not None:
            hi = pd.Timestamp(end_date)
            ordered = [c for c in ordered if d_to_date.get(c) is not None and d_to_date[c] <= hi]
        if last_n_days is not None:
            ordered = ordered[-last_n_days:]
        return ordered

    @staticmethod
    def _resolve_snap(df: pd.DataFrame) -> pd.Series:
        """Pick each row's own-state SNAP flag from snap_CA / snap_TX / snap_WI."""
        snap = pd.Series(0, index=df.index, dtype="int8")
        for state in M5.STATES:
            col = f"snap_{state}"
            if col in df.columns:
                mask = df["state_id"].astype(str) == state
                snap.loc[mask] = df.loc[mask, col].astype("int8")
        return snap

    def _attach_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join weekly prices and flag coverage explicitly.

        `has_price_data` distinguishes the two very different reasons a price
        can be null:
          * the store was cut from the truncated workbook (flag = 0), versus
          * the item genuinely had no price that week (flag = 1, price NaN),
            which in M5 means it was not stocked yet.
        Collapsing both into "NaN" would let a model treat store coverage as
        a demand signal.
        """
        prices = self.load_prices()
        df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        covered = set(prices["store_id"].unique())
        df["has_price_data"] = df["store_id"].astype(str).isin(covered).astype("int8")

        n_missing_store = int((df["has_price_data"] == 0).sum())
        if n_missing_store:
            self._log.warning(
                "price_coverage_incomplete",
                rows_without_price_coverage=n_missing_store,
                pct=round(100.0 * n_missing_store / len(df), 1),
                stores_with_prices=sorted(covered),
                reason="source workbook truncated at Excel row limit",
            )
        return df

    @staticmethod
    def _downcast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        for col in (
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "weekday",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ):
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    # -- series-level access (for baselines, Croston, the router) ------------

    def load_series_dict(
        self,
        **panel_kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        """
        Return {series_id: sales array} plus the series metadata table.

        This is the shape the per-series models want (baselines, Croston,
        ADIDA, the router), as opposed to the flat feature matrix the global
        tree models consume.
        """
        df = self.load_panel(with_prices=False, with_calendar=False, **panel_kwargs)
        df["series_id"] = df["item_id"].astype(str) + "_" + df["store_id"].astype(str)

        series: dict[str, np.ndarray] = {
            sid: g.sort_values("date")["sales"].to_numpy(dtype=float)
            for sid, g in df.groupby("series_id", observed=True)
        }
        meta = (
            df.groupby("series_id", observed=True)
            .agg(
                item_id=("item_id", "first"),
                store_id=("store_id", "first"),
                dept_id=("dept_id", "first"),
                cat_id=("cat_id", "first"),
                state_id=("state_id", "first"),
                n_days=("sales", "size"),
                total_units=("sales", "sum"),
            )
            .reset_index()
        )
        self._log.info("series_dict_loaded", n_series=len(series))
        return series, meta
