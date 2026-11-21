import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    print("Computing M5 WRMSSE weights...")
    last_28_d = [f"d_{i}" for i in range(1886, 1914)]

    # We can use pl.scan_parquet directly on the raw data, or the interim parquet if it exists
    # M5 weights are based on the last 28 days of the training period.
    # The training data is sales_train_validation.csv

    try:
        sales_path = "data/raw/sales_train_validation.parquet"
        prices_path = "data/raw/sell_prices.parquet"
        calendar_path = "data/raw/calendar.parquet"

        # This approach might still take memory, let's just do a manual loop over the series if we have to.
        # Let's try loading it efficiently with polars

        # Load prices
        prices = pl.scan_parquet(prices_path)
        cal = pl.scan_parquet(calendar_path)

        # Join cal and prices to get week ID for the last 28 dates
        cal_28 = cal.filter(pl.col("d").is_in(last_28_d))

        # Read sales lazy
        sales = pl.scan_parquet(sales_path)

        # We only need 'id', 'item_id', 'store_id' and the last 28 days
        cols_to_keep = ["id", "item_id", "store_id"] + last_28_d
        sales_28 = sales.select(cols_to_keep)

        # Unpivot (melt) the last 28 days
        sales_melt = sales_28.unpivot(
            index=["id", "item_id", "store_id"],
            on=last_28_d,
            variable_name="d",
            value_name="sales"
        )

        # Join with calendar to get wm_yr_wk
        sales_melt = sales_melt.join(cal_28.select(["d", "wm_yr_wk"]), on="d", how="left")

        # Join with prices
        sales_melt = sales_melt.join(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left"
        )

        # Compute dollar sales
        res = (
            sales_melt
            .with_columns(
                (pl.col("sales") * pl.col("sell_price")).alias("dollar_sales")
            )
            .group_by("id")
            .agg(pl.col("dollar_sales").sum())
            .collect(streaming=True)
        )

        weights = res.to_pandas().set_index("id")["dollar_sales"]
        weights = weights / weights.sum()

        # Remove the validation suffix to match series_id
        weights.index = weights.index.str.replace("_validation", "")
        weights.index.name = "series_id"

        out_dir = Path("reports/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        weights.to_csv(out_dir / "wrmsse_weights.csv")
        print(f"Weights saved to {out_dir / 'wrmsse_weights.csv'}")

    except Exception as e:
        print(f"Failed to compute weights: {e}")

if __name__ == "__main__":
    main()
