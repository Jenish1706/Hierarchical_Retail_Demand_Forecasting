# 03: Data Architecture, Schemas, & Quality Safeguards

This document covers the empirical dataset, schema definitions, storage formats, data quality validation rules, and known source limitations.

---

## 1. Data Source & Licence

The platform utilizes the **M5 Forecasting Competition dataset**, made publicly available by Walmart and the Makridakis Open Forecasting Center (MOFC) under the competition rules hosted on Kaggle:
- **Original Source**: Walmart retail point-of-sale data (10 stores across 3 US states: California, Texas, Wisconsin).
- **Time Range**: 1,913 days from 2011-01-29 to 2016-04-24.
- **Licence**: Creative Commons / MOFC Competition Terms for academic and educational research.
- **Repository Storage**: The raw data is stored as Snappy-compressed Parquet tables under `data/raw/`. A 15-file lightweight slice of summary reports and backtest predictions is tracked in version control under `reports/`.

---

## 2. Table Schemas & Row Counts

The platform consumes three primary relational tables:

### 1. `sales_train_validation.parquet`
Stores daily sales units for every product across every store in wide format.
- **Row Count**: Exactly **30,490 series** (SKUs across 10 stores).
- **Column Count**: 1,919 columns (6 metadata columns + 1,913 daily sales columns $d_1 \dots d_{1913}$).
- **Total Cell Count**: $30,490 \times 1,913 = \mathbf{58,327,370}$ daily observation records.

| Column Name | Data Type | Description | Domain / Example |
| :--- | :--- | :--- | :--- |
| `id` | `Categorical` / `String` | Unique series identifier | `HOBBIES_1_001_CA_1_validation` |
| `item_id` | `Categorical` / `String` | Product identifier (3,049 unique items) | `HOBBIES_1_001` |
| `dept_id` | `Categorical` / `String` | Department identifier (7 departments) | `HOBBIES_1`, `FOODS_3` |
| `cat_id` | `Categorical` / `String` | Product category (3 categories) | `HOBBIES`, `HOUSEHOLD`, `FOODS` |
| `store_id` | `Categorical` / `String` | Retail store identifier (10 stores) | `CA_1`, `CA_2`, `TX_1`, `WI_3` |
| `state_id` | `Categorical` / `String` | State identifier (3 states) | `CA`, `TX`, `WI` |
| `d_1` $\dots$ `d_1913` | `int16` | Units sold on calendar day $t$ | Non-negative integers $\ge 0$ |

### 2. `calendar.parquet`
Provides calendar dates, event schedules, and federal SNAP food assistance benefit schedules.
- **Row Count**: **1,969 calendar days** (covering $d_1$ through $d_{1969}$, including evaluation horizons).
- **Column Count**: 14 columns.

| Column Name | Data Type | Description | Examples |
| :--- | :--- | :--- | :--- |
| `date` | `Date` / `String` | Calendar date | `2011-01-29` to `2016-06-19` |
| `wm_yr_wk` | `int32` | Walmart retail accounting week | `11101` |
| `weekday` | `String` | Day of week string | `Saturday`, `Sunday` |
| `wday` | `int8` | Walmart day of week (Saturday=1) | `1` to `7` |
| `month` | `int8` | Calendar month | `1` to `12` |
| `year` | `int16` | Calendar year | `2011` to `2016` |
| `d` | `String` | Day indicator string | `d_1` to `d_1969` |
| `event_name_1` | `String` (nullable) | Primary holiday / cultural event | `SuperBowl`, `Easter`, `Ramadan` |
| `event_type_1` | `String` (nullable) | Category of primary event | `Sporting`, `Cultural`, `National` |
| `event_name_2` | `String` (nullable) | Secondary concurrent event | `Cinco De Mayo` |
| `event_type_2` | `String` (nullable) | Category of secondary event | `Cultural` |
| `snap_CA` | `int8` | SNAP assistance active in California | `0` or `1` |
| `snap_TX` | `int8` | SNAP assistance active in Texas | `0` or `1` |
| `snap_WI` | `int8` | SNAP assistance active in Wisconsin | `0` or `1` |

### 3. `sell_prices.parquet`
Provides weekly unit retail selling prices per store and item.
- **Row Count**: **1,048,575 records** (truncated source slice; see Section 4).
- **Column Count**: 4 columns.

| Column Name | Data Type | Description | Examples |
| :--- | :--- | :--- | :--- |
| `store_id` | `Categorical` / `String` | Store identifier | `CA_1`, `CA_2` |
| `item_id` | `Categorical` / `String` | Product identifier | `FOODS_1_001` |
| `wm_yr_wk` | `int32` | Walmart retail week | `11101` |
| `sell_price` | `float32` | Unit retail price in USD | `$0.05` to `$107.32` |

---

## 3. Data Ingestion Architecture: Narrow-Before-Melt

Melting the full $30,490 \times 1,913$ matrix into a long panel produces **58,327,370 rows**. In unoptimized pandas, storing this table in float/object types requires over **6 GB of RAM**, causing Out-Of-Memory (OOM) crashes on standard developer workstations.

The platform solves this with a two-phase lazy ingestion design in [`src/data/loader.py`](../../src/data/loader.py):

1. **Downcasting at Rest**: Daily unit sales are stored as `int16` on disk (sufficient since the maximum recorded single-day sales quantity for any item in M5 is 763 units, well within the 32,767 `int16` limit). Categorical identifiers (`item_id`, `store_id`, `dept_id`) are mapped to dictionary-encoded categories.
2. **Narrow-Before-Melt Filtering**: Subsetting filters (e.g., selecting stores `CA_1` and `CA_2`, or selecting a 900-day evaluation window) are executed on the **wide** dataframe before melting. Filtering 30,490 wide rows takes less than 20 milliseconds, avoiding the generation of tens of millions of unused rows.
3. **Out-of-Core Streaming**: For feature transformations, Polars streaming expressions (`pl.scan_parquet`) process data in vector chunks, holding peak memory below **600 MB**.

---

## 4. Known Data Quality Problems & Handling

### 1. Structural Lifecycle Zeros (Late Product Introductions)
- **Problem**: 23,511 of 30,490 series (77%) were introduced into the Walmart catalog after Day 1, with a median first sale date of Day 160. Treating pre-launch inactivity as zero-demand artificially inflates $ADI$ ($n / n_{\text{nonzero}}$) and dilutes naive error scale denominators ($0 - 0 = 0$), distorting both demand classification and RMSSE.
- **Resolution**: `M5Loader.load_first_sale_dates()` scans the catalog from $d_1$ onward to identify each series' true launch date. Both `classify_demand()` and `naive_scale()` slice arrays from the first historical non-zero sale. Crucially, mature products evaluated in later windows keep their in-window leading zeros, preventing mid-life zero truncation.

### 2. Truncated Sell Prices (Excel Limit)
- **Problem**: The raw price file extracted from the competition workbook contains 1,048,575 rows—the exact historical row limit of Microsoft Excel ($2^{20} - 1$). Consequently, price coverage is restricted to stores `CA_1` and `CA_2` (covering 15.3% of the catalog; 4,591 of 6,098 California series).
- **Resolution**: Rather than dropping unpriced series or improvising synthetic prices, the platform explicitly isolates inventory capital optimization to stores `CA_1` and `CA_2`. The remaining 1,507 unpriced California SKUs are tracked transparently as `Unpriced (?X/?Y/?Z)` with $0.00 valuation, ensuring capital claims are strictly verifiable against committed price records.

### 3. Christmas Zero Closure
- **Problem**: Walmart retail stores close completely on Christmas Day (December 25). Recorded sales on Christmas days ($d_{331}, d_{696}, d_{1062}, d_{1427}, d_{1792}$) are identically zero for all 30,490 series.
- **Resolution**: Calendar feature encodings explicitly mark `event_name_1 == 'Christmas'` as a zero-demand calendar override, preventing models from learning artificial structural dips.

---

## 5. Schema Contracts (Pandera Enforcement)

Incoming panel data is validated at runtime against strict Pandera schema contracts (`src/data/schema.py`):
- `series_id`: String / Categorical; non-null; matches regex `^[A-Z]+_[0-9]+_[A-Z]+_[0-9]+$`.
- `date`: DateTime / Date; non-null; monotonically increasing per series.
- `y_true`: Float / Integer; non-null; non-negative ($\ge 0$).
- `sell_price`: Float; nullable; when present, must be strictly positive ($> 0$).
- `snap`: Integer; must belong to binary domain $\{0, 1\}$.
