"""
Router validation (empirical re-derivation per §4.3)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statsforecast import StatsForecast
from statsforecast.models import ADIDA, SeasonalNaive

from src.data.loader import M5Loader
from src.evaluation.splits import rolling_origins
from src.models.router import DEFAULT_ROUTING, SegmentRouter


def main():
    loader = M5Loader()
    panel = loader.load_panel(
        sample_series=3000, random_state=42, with_prices=False, with_calendar=False
    )
    panel["unique_id"] = panel["item_id"].astype(str) + "_" + panel["store_id"].astype(str)
    panel = panel.rename(columns={"date": "ds", "sales": "y"})
    panel["ds"] = pd.to_datetime(panel["ds"])
    panel["y"] = panel["y"].astype(float)

    horizon = 28
    origin = list(
        rolling_origins(
            n_origins=1, horizon=horizon, last_date=panel["ds"].max(), first_date=panel["ds"].min()
        )
    )[0]

    train = panel[panel["ds"] <= pd.to_datetime(origin.train_end)]
    test = panel[
        (panel["ds"] >= pd.to_datetime(origin.test_start))
        & (panel["ds"] <= pd.to_datetime(origin.test_end))
    ]

    print("Fitting statistical models for routing comparison...")
    sf = StatsForecast(models=[SeasonalNaive(season_length=7), ADIDA()], freq="D")
    sf.fit(train[["unique_id", "ds", "y"]])
    preds = sf.predict(h=horizon).reset_index()

    global_forecasts = {}
    for uid, g in preds.groupby("unique_id"):
        global_forecasts[uid] = g.sort_values("ds")["SeasonalNaive"].values

    actuals = {}
    for uid, g in test.groupby("unique_id"):
        actuals[uid] = g.sort_values("ds")["y"].values

    history = {}
    for uid, g in train.groupby("unique_id"):
        history[uid] = g.sort_values("ds")["y"].values

    print("Fitting router...")
    router = SegmentRouter(routing=DEFAULT_ROUTING)
    router.fit(history)

    print("Evaluating routing benefit...")
    benefit = router.evaluate_routing_benefit(actuals=actuals, global_forecasts=global_forecasts)

    print("\nRouting Benefit by Segment:")
    print(benefit.by_segment.to_string())

    print("\nOverall:")
    print(f"Improved: {benefit.improved}")
    print(f"Routed WAPE: {benefit.routed_wape:.4f}")
    print(f"Global WAPE: {benefit.global_wape:.4f}")


if __name__ == "__main__":
    main()
