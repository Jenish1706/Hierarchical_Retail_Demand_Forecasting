"""
M5 Forecasting Competition Data Ingestion Pipeline.

Handles ingestion of all M5 data sources:
  - Sales data (sales_train_evaluation.csv)
  - Calendar data (calendar.csv)
  - Sell prices (sell_prices.csv)

Design decisions:
  - Polars for high-performance reading (10M+ rows)
  - Chunked ingestion to control memory usage
  - Idempotent upserts (safe to re-run)
  - Full audit trail per ingestion run
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Iterator

import polars as pl
import pandas as pd
from sqlalchemy import text

from src.utils.database import get_sync_db
from src.utils.logging import PipelineLogger, get_logger
from src.utils.settings import settings

log = get_logger(__name__)

# M5 data files
M5_FILES = {
    "sales": "sales_train_evaluation.csv",
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
}

# Chunk size for batch inserts
CHUNK_SIZE = 50_000


class M5DataIngester:
    """
    Ingests M5 Forecasting Competition data into PostgreSQL.

    The M5 dataset has wide-format sales (one column per day).
    This class melts it to long format and stores it efficiently.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.run_id = str(uuid.uuid4())[:8]
        self._log = get_logger("M5DataIngester")

    def run(self) -> dict[str, int]:
        """Execute full ingestion pipeline. Returns row counts per table."""
        results = {}
        with PipelineLogger("m5_ingestion", self.run_id):
            results["calendar"] = self._ingest_calendar()
            results["prices"] = self._ingest_prices()
            results["sales"] = self._ingest_sales()
        return results

    def _ingest_calendar(self) -> int:
        """Ingest calendar.csv → raw.m5_calendar."""
        path = self.data_dir / M5_FILES["calendar"]
        self._log.info("ingesting_calendar", path=str(path))

        df = pl.read_csv(path)

        # Normalize column names and types
        calendar_records = []
        for row in df.iter_rows(named=True):
            calendar_records.append({
                "date": row["date"],
                "wm_yr_wk": row["wm_yr_wk"],
                "weekday": row["weekday"],
                "wday": row["wday"],
                "month": row["month"],
                "year": row["year"],
                "event_name_1": row.get("event_name_1"),
                "event_type_1": row.get("event_type_1"),
                "event_name_2": row.get("event_name_2"),
                "event_type_2": row.get("event_type_2"),
                "snap_ca": row.get("snap_CA", 0),
                "snap_tx": row.get("snap_TX", 0),
                "snap_wi": row.get("snap_WI", 0),
            })

        with get_sync_db() as db:
            db.execute(
                text("""
                    INSERT INTO raw.m5_calendar
                        (date, wm_yr_wk, weekday, wday, month, year,
                         event_name_1, event_type_1, event_name_2, event_type_2,
                         snap_ca, snap_tx, snap_wi)
                    VALUES
                        (:date, :wm_yr_wk, :weekday, :wday, :month, :year,
                         :event_name_1, :event_type_1, :event_name_2, :event_type_2,
                         :snap_ca, :snap_tx, :snap_wi)
                    ON CONFLICT (date) DO UPDATE SET
                        event_name_1 = EXCLUDED.event_name_1,
                        event_type_1 = EXCLUDED.event_type_1,
                        event_name_2 = EXCLUDED.event_name_2,
                        event_type_2 = EXCLUDED.event_type_2
                """),
                calendar_records,
            )

        self._log.info("calendar_ingested", rows=len(calendar_records))
        return len(calendar_records)

    def _ingest_prices(self) -> int:
        """Ingest sell_prices.csv → raw.m5_sell_prices."""
        path = self.data_dir / M5_FILES["prices"]
        self._log.info("ingesting_prices", path=str(path))

        # Use Polars for fast reading
        df = pl.read_csv(path)
        total_rows = 0

        for chunk in self._chunked_dataframe(df, CHUNK_SIZE):
            records = chunk.to_dicts()
            with get_sync_db() as db:
                db.execute(
                    text("""
                        INSERT INTO raw.m5_sell_prices
                            (store_id, item_id, wm_yr_wk, sell_price)
                        VALUES
                            (:store_id, :item_id, :wm_yr_wk, :sell_price)
                        ON CONFLICT DO NOTHING
                    """),
                    records,
                )
            total_rows += len(records)

        self._log.info("prices_ingested", rows=total_rows)
        return total_rows

    def _ingest_sales(self) -> int:
        """
        Ingest sales_train_evaluation.csv → raw.m5_sales.

        Input is wide format: item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ...
        Output is long format: one row per (item, store, date).
        """
        path = self.data_dir / M5_FILES["sales"]
        self._log.info("ingesting_sales", path=str(path))

        # Read with Polars (much faster than pandas for this size)
        df = pl.read_csv(path)

        # Identify day columns
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        day_cols = [c for c in df.columns if c.startswith("d_")]

        self._log.info("sales_shape", rows=df.shape[0], day_cols=len(day_cols))

        # Melt from wide to long format
        df_long = df.unpivot(
            index=id_cols,
            on=day_cols,
            variable_name="d",
            value_name="sales",
        )

        # Join with calendar to get actual dates
        calendar_df = pl.read_csv(self.data_dir / M5_FILES["calendar"])
        calendar_df = calendar_df.select([
            pl.col("date"),
            pl.col("d").alias("d_col"),
        ]).rename({"d_col": "d"})

        df_long = df_long.join(calendar_df, on="d", how="left")
        df_long = df_long.drop("d")

        # Select final columns
        df_final = df_long.select([
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "date", "sales",
        ])

        # Filter out null sales (pre-launch periods)
        df_final = df_final.filter(pl.col("sales").is_not_null())

        total_rows = 0
        for chunk in self._chunked_dataframe(df_final, CHUNK_SIZE * 5):
            records = []
            for row in chunk.iter_rows(named=True):
                records.append({
                    "item_id": row["item_id"],
                    "dept_id": row["dept_id"],
                    "cat_id": row["cat_id"],
                    "store_id": row["store_id"],
                    "state_id": row["state_id"],
                    "date": row["date"],
                    "sales": float(row["sales"]) if row["sales"] is not None else 0.0,
                    "source_file": M5_FILES["sales"],
                })

            with get_sync_db() as db:
                db.execute(
                    text("""
                        INSERT INTO raw.m5_sales
                            (item_id, dept_id, cat_id, store_id, state_id, date, sales, source_file)
                        VALUES
                            (:item_id, :dept_id, :cat_id, :store_id, :state_id,
                             :date::date, :sales, :source_file)
                        ON CONFLICT DO NOTHING
                    """),
                    records,
                )
            total_rows += len(records)
            self._log.info("sales_chunk_loaded", rows_so_far=total_rows)

        self._log.info("sales_ingested", total_rows=total_rows)
        return total_rows

    @staticmethod
    def _chunked_dataframe(
        df: pl.DataFrame, chunk_size: int
    ) -> Iterator[pl.DataFrame]:
        """Yield DataFrame in chunks for memory-efficient processing."""
        n = len(df)
        for start in range(0, n, chunk_size):
            yield df.slice(start, min(chunk_size, n - start))


class WeatherDataIngester:
    """Ingests weather data from NOAA or Open-Meteo API."""

    def __init__(self, store_locations: dict[str, tuple[float, float]]):
        """
        Args:
            store_locations: {store_id: (latitude, longitude)}
        """
        self.store_locations = store_locations
        self._log = get_logger("WeatherIngester")

    def ingest_weather_for_date_range(
        self, start_date: date, end_date: date
    ) -> int:
        """
        Fetch and store weather data for all stores for the date range.
        Uses Open-Meteo API (free, no auth required for historical data).
        """
        import httpx

        total_rows = 0
        base_url = "https://archive-api.open-meteo.com/v1/archive"

        for store_id, (lat, lon) in self.store_locations.items():
            try:
                response = httpx.get(
                    base_url,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                        "timezone": "America/Chicago",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                records = self._parse_weather_response(store_id, data)
                self._store_weather_records(records)
                total_rows += len(records)

            except Exception as e:
                self._log.error("weather_fetch_failed", store_id=store_id, error=str(e))

        return total_rows

    def _parse_weather_response(
        self, store_id: str, data: dict
    ) -> list[dict]:
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        return [
            {
                "store_id": store_id,
                "date": d,
                "temp_max": daily.get("temperature_2m_max", [None])[i],
                "temp_min": daily.get("temperature_2m_min", [None])[i],
                "precipitation": daily.get("precipitation_sum", [None])[i],
                "wind_speed": daily.get("windspeed_10m_max", [None])[i],
                "source": "open_meteo",
            }
            for i, d in enumerate(dates)
        ]

    def _store_weather_records(self, records: list[dict]) -> None:
        with get_sync_db() as db:
            db.execute(
                text("""
                    INSERT INTO raw.weather_data
                        (store_id, date, temp_max, temp_min, precipitation, wind_speed, source)
                    VALUES
                        (:store_id, :date::date, :temp_max, :temp_min, :precipitation, :wind_speed, :source)
                    ON CONFLICT DO NOTHING
                """),
                records,
            )
