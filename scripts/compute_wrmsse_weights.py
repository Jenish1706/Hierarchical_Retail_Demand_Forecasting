import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import M5Loader

def main():
    print("Computing M5 WRMSSE weights...")
    loader = M5Loader()
    
    calendar = loader.load_calendar()
    
    dates = calendar.set_index("d")["date"]
    last_28_d = [f"d_{i}" for i in range(1886, 1914)]
    last_28_dates = dates.loc[last_28_d].values
    
    panel = loader.load_panel(sample_series=30490, random_state=42, with_prices=True)
    
    panel_28 = panel[panel["date"].isin(last_28_dates)].copy()
    
    panel_28["dollar_sales"] = panel_28["sales"] * panel_28["sell_price"]
    
    weights = panel_28.groupby("series_id")["dollar_sales"].sum()
    weights = weights / weights.sum()
    
    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out_dir / "wrmsse_weights.csv")
    print(f"Weights saved to {out_dir / 'wrmsse_weights.csv'}")

if __name__ == "__main__":
    main()
