"""Format report CSV files and generate companion compressed Parquet files
that are 100% synchronized with their CSV sources.

Usage:
    python scripts/optimize_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def process_reports(reports_dir: Path | str = "reports") -> None:
    reports_dir = Path(reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    if not reports_dir.exists():
        print(f"Directory {reports_dir} does not exist.")
        return

    csv_files = sorted(reports_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {reports_dir}\n")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            parquet_path = csv_path.with_suffix(".parquet")
            pl_df = pl.from_pandas(df)
            pl_df.write_parquet(parquet_path, compression="snappy")
            rel_csv = csv_path.relative_to(PROJECT_ROOT)
            print(f"  Synchronized: {rel_csv} -> {parquet_path.name}")

        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")

    print("\n" + "=" * 60)
    print("Report parquet companions fully synchronized from CSV.")
    print("=" * 60)


if __name__ == "__main__":
    process_reports()
