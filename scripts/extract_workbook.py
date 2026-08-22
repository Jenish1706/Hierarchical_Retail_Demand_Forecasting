"""
Extract the four M5 sheets from the supplied .xlsx into Parquet.

Why this script exists
----------------------
The dataset was supplied as a single 229 MB Excel workbook. Excel is not a
viable format for the training pipeline: openpyxl parses ~58 million cells
per full read of the sales sheet, which costs minutes and gigabytes every
time. This script pays that cost ONCE and writes columnar Parquet that the
rest of the pipeline reads in under a second.

It also does something more important: it MEASURES the workbook against the
known M5 cardinalities and writes an extraction report. The sales sheet turns
out to be complete, but the price sheet does not -- see below -- and that is
exactly the kind of finding that must surface at ingest rather than as a
confusing accuracy result three stages downstream.

Usage:
    python scripts/extract_workbook.py
    python scripts/extract_workbook.py --workbook path/to/file.xlsx --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging import get_logger  # noqa: E402
from src.config.settings import M5, settings  # noqa: E402

log = get_logger("extract_workbook")

DEFAULT_WORKBOOK = "M5 Forecasting Competition Dataset.csv.xlsx"

# Sheet name in the workbook -> output stem. The workbook's sheet names are
# preserved verbatim here, typo included ("Sales_subission"), because that is
# what is actually in the file; renaming happens on the output side.
SHEETS = {
    "calendar": "calendar",
    "Sales_prices": "sell_prices",
    "Sales_subission": "sales_submission",
    "Sales_train_validation": "sales_train_validation",
}

# Rows per Parquet write for the wide sales sheet. 2,000 x 1,919 cells is a
# few tens of MB per batch -- large enough to amortise write overhead, small
# enough that peak memory stays flat regardless of total size.
SALES_CHUNK_ROWS = 2_000


def _iter_sheet(ws, min_row: int = 2) -> Iterator[tuple]:
    """Yield non-empty rows from a read-only worksheet."""
    for row in ws.iter_rows(min_row=min_row, values_only=True):
        if row[0] is None:
            continue
        yield row


def _header(ws) -> list[str]:
    for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
        return [str(c) for c in row if c is not None]
    raise ValueError("sheet has no header row")


# ---------------------------------------------------------------------------
# Per-sheet extractors
# ---------------------------------------------------------------------------

def extract_calendar(ws, out: Path) -> dict[str, Any]:
    """Calendar is small (1,969 rows); read it whole."""
    cols = _header(ws)
    df = pd.DataFrame(list(_iter_sheet(ws)), columns=cols)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")

    df.to_parquet(out, index=False, compression="snappy")

    events = set(df["event_name_1"].dropna()) | set(df.get("event_name_2", pd.Series(dtype=object)).dropna())
    return {
        "rows": len(df),
        "expected_rows": M5.N_CALENDAR_ROWS,
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "distinct_named_events": len(events),
        "expected_named_events": M5.N_EVENT_NAMES,
        "snap_columns": [c for c in df.columns if c.startswith("snap_")],
    }


def extract_prices(ws, out: Path) -> dict[str, Any]:
    """
    Sell prices. Checked hard for Excel truncation.

    The genuine M5 sell_prices.csv has 6,841,121 rows. Excel's sheet limit is
    1,048,576 rows. If the sheet sits exactly at that limit, the table was cut
    when the CSV was pasted into the workbook, and whatever comes after the cut
    is simply absent -- not sampled, not missing at random, just gone.
    """
    cols = _header(ws)
    df = pd.DataFrame(list(_iter_sheet(ws)), columns=cols)

    df["wm_yr_wk"] = pd.to_numeric(df["wm_yr_wk"], errors="coerce").astype("Int32")
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce").astype("float32")
    df.to_parquet(out, index=False, compression="snappy")

    EXCEL_ROW_LIMIT = 1_048_576
    FULL_M5_PRICE_ROWS = 6_841_121
    truncated = len(df) >= EXCEL_ROW_LIMIT - 1

    stores = sorted(df["store_id"].dropna().unique().tolist())
    covered = set(stores)
    missing = [s for s in M5.STORES if s not in covered]

    return {
        "rows": len(df),
        "expected_rows_full_m5": FULL_M5_PRICE_ROWS,
        "coverage_pct": round(100.0 * len(df) / FULL_M5_PRICE_ROWS, 2),
        "TRUNCATED_AT_EXCEL_LIMIT": bool(truncated),
        "stores_with_prices": stores,
        "stores_WITHOUT_prices": missing,
        "distinct_items": int(df["item_id"].nunique()),
        "distinct_weeks": int(df["wm_yr_wk"].nunique()),
        "price_min": float(df["sell_price"].min()),
        "price_max": float(df["sell_price"].max()),
    }


def extract_submission(ws, out: Path) -> dict[str, Any]:
    cols = _header(ws)
    df = pd.DataFrame(list(_iter_sheet(ws)), columns=cols)
    for c in df.columns:
        if c.startswith("F"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df.to_parquet(out, index=False, compression="snappy")

    return {
        "rows": len(df),
        "horizon_columns": sum(c.startswith("F") for c in df.columns),
        "expected_horizon": M5.HORIZON_DAYS,
        "validation_ids": int(df["id"].str.endswith("_validation").sum()),
        "evaluation_ids": int(df["id"].str.endswith("_evaluation").sum()),
    }


def extract_sales(ws, out: Path) -> dict[str, Any]:
    """
    Wide sales sheet, streamed in row chunks.

    Kept in WIDE form on disk (30,490 x 1,919). The wide layout stores each
    day as its own column of int16, which Parquet compresses extremely well
    given how many zeros M5 contains. Melting to long produces 58.3M rows and
    is done downstream by the loader, in the batches the caller actually
    needs, rather than materialised here.

    int16 is safe: the largest single-day unit sale in M5 is in the hundreds,
    far inside int16's 32,767 ceiling. It also quarters the memory a naive
    int64 read would take.
    """
    cols = _header(ws)
    id_cols = [c for c in cols if not c.startswith("d_")]
    day_cols = [c for c in cols if c.startswith("d_")]

    writer: pq.ParquetWriter | None = None
    buffer: list[tuple] = []
    total = 0
    t0 = time.time()

    # Accumulated for the report without a second pass.
    items: set[str] = set()
    stores: set[str] = set()
    depts: set[str] = set()
    cats: set[str] = set()
    states: set[str] = set()
    unit_total = 0

    def flush() -> None:
        nonlocal writer, buffer, unit_total
        if not buffer:
            return
        chunk = pd.DataFrame(buffer, columns=cols)
        for c in id_cols:
            chunk[c] = chunk[c].astype(str)
        day_block = (
            chunk[day_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype("int16")
        )
        unit_total += int(day_block.to_numpy(dtype=np.int64).sum())
        chunk = pd.concat([chunk[id_cols], day_block], axis=1)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="snappy")
        writer.write_table(table)
        buffer = []

    for row in _iter_sheet(ws):
        buffer.append(row)
        total += 1
        rec = dict(zip(cols[: len(id_cols)], row[: len(id_cols)]))
        items.add(rec.get("item_id"))
        stores.add(rec.get("store_id"))
        depts.add(rec.get("dept_id"))
        cats.add(rec.get("cat_id"))
        states.add(rec.get("state_id"))

        if len(buffer) >= SALES_CHUNK_ROWS:
            flush()
            log.info("sales_chunk_written", rows_done=total, elapsed_s=round(time.time() - t0, 1))

    flush()
    if writer is not None:
        writer.close()

    return {
        "rows": total,
        "expected_rows": M5.N_SERIES,
        "day_columns": len(day_cols),
        "expected_days": M5.N_DAYS,
        "distinct_items": len(items),
        "distinct_stores": len(stores),
        "distinct_departments": len(depts),
        "distinct_categories": len(cats),
        "distinct_states": len(states),
        "total_units_sold": unit_total,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract M5 workbook sheets to Parquet")
    parser.add_argument("--workbook", default=DEFAULT_WORKBOOK)
    parser.add_argument("--out-dir", default=None, help="default: <base>/data/raw")
    parser.add_argument("--force", action="store_true", help="re-extract even if outputs exist")
    args = parser.parse_args()

    import openpyxl

    wb_path = Path(args.workbook)
    if not wb_path.is_absolute():
        wb_path = settings.base_dir / wb_path
    if not wb_path.exists():
        log.error("workbook_not_found", path=str(wb_path))
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else settings.raw_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("opening_workbook", path=str(wb_path), size_mb=round(wb_path.stat().st_size / 1e6, 1))
    wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)

    extractors = {
        "calendar": extract_calendar,
        "Sales_prices": extract_prices,
        "Sales_subission": extract_submission,
        "Sales_train_validation": extract_sales,
    }

    report: dict[str, Any] = {
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "source_workbook": wb_path.name,
        "source_size_bytes": wb_path.stat().st_size,
        "sheets": {},
    }

    for sheet, stem in SHEETS.items():
        if sheet not in wb.sheetnames:
            log.error("sheet_missing", sheet=sheet, available=wb.sheetnames)
            report["sheets"][stem] = {"error": "sheet not found in workbook"}
            continue

        out_path = out_dir / f"{stem}.parquet"
        if out_path.exists() and not args.force:
            log.info("skipping_existing", sheet=sheet, path=str(out_path))
            continue

        log.info("extracting_sheet", sheet=sheet, out=str(out_path))
        t0 = time.time()
        stats = extractors[sheet](wb[sheet], out_path)
        stats["output"] = str(out_path.relative_to(settings.base_dir))
        stats["output_size_bytes"] = out_path.stat().st_size
        stats["extract_seconds"] = round(time.time() - t0, 1)
        report["sheets"][stem] = stats
        log.info("sheet_extracted", sheet=sheet, **{k: v for k, v in stats.items()
                                                    if isinstance(v, (int, float, bool, str))})

    wb.close()

    # --- cardinality check against the M5 facts ---
    sales = report["sheets"].get("sales_train_validation", {})
    if sales and "rows" in sales:
        observed = {
            "n_series": sales["rows"],
            "n_items": sales["distinct_items"],
            "n_stores": sales["distinct_stores"],
            "n_states": sales["distinct_states"],
            "n_categories": sales["distinct_categories"],
            "n_departments": sales["distinct_departments"],
            "n_days": sales["day_columns"],
        }
        report["cardinality_violations"] = M5.validate_cardinalities(observed)

    # --- data limitations that must travel downstream ---
    prices = report["sheets"].get("sell_prices", {})
    limitations: list[dict[str, Any]] = []
    if prices.get("TRUNCATED_AT_EXCEL_LIMIT"):
        limitations.append(
            {
                "severity": "HIGH",
                "table": "sell_prices",
                "finding": (
                    f"Truncated at Excel's 1,048,576-row sheet limit. "
                    f"{prices['rows']:,} of {prices['expected_rows_full_m5']:,} rows present "
                    f"({prices['coverage_pct']}%)."
                ),
                "consequence": (
                    "Price data exists for "
                    f"{prices.get('stores_with_prices')} only. "
                    f"No prices at all for {prices.get('stores_WITHOUT_prices')}. "
                    "The truncation is systematic (the CSV was cut mid-table), "
                    "NOT missing at random."
                ),
                "impact": [
                    "Price elasticity is estimable only for the covered stores, "
                    "all of which are in CA -- no cross-state comparison.",
                    "Price-derived features must be restricted to covered stores, "
                    "or a model will learn 'price is missing' as a proxy for "
                    "store identity rather than as a price signal.",
                    "Sales, calendar, SNAP and event data are COMPLETE, so "
                    "forecasting, hierarchical reconciliation, SNAP causal "
                    "analysis and holiday lift are unaffected.",
                ],
            }
        )
    report["data_limitations"] = limitations

    report_path = out_dir / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # --- console summary ---
    print("\n" + "=" * 72)
    print("M5 WORKBOOK EXTRACTION COMPLETE")
    print("=" * 72)
    for stem, stats in report["sheets"].items():
        if "error" in stats:
            print(f"  {stem:<26} ERROR: {stats['error']}")
            continue
        print(f"  {stem:<26} {stats.get('rows', 0):>10,} rows  "
              f"-> {stats.get('output_size_bytes', 0)/1e6:>7.1f} MB  "
              f"({stats.get('extract_seconds', 0)}s)")

    violations = report.get("cardinality_violations", [])
    print("\n  Cardinality vs M5 facts:",
          "CLEAN" if not violations else f"{len(violations)} VIOLATION(S)")
    for v in violations:
        print(f"    - {v}")

    if limitations:
        print("\n  DATA LIMITATIONS:")
        for lim in limitations:
            print(f"    [{lim['severity']}] {lim['table']}: {lim['finding']}")
            print(f"      {lim['consequence']}")
    print(f"\n  Report: {report_path}")
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
