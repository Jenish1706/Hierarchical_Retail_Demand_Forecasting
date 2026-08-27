"""
Head-to-head comparison of DLinear vs SNaive/Stats on M5.
Testing regimes:
1. High-volume series (top decile)
2. Higher aggregation levels (store level)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from src.data.loader import M5Loader
from src.evaluation.point_metrics import metrics_by_group, naive_scale
from src.evaluation.splits import rolling_origins
from src.models.deep_learning.library_models.neuralforecast_adapter import NeuralForecastAdapter


def run_comparison():
    loader = M5Loader()
    print("Loading data...")
    # Load 3000 series, find top 300
    panel = loader.load_panel(
        sample_series=3000, random_state=42, with_prices=False, with_calendar=False
    )
    panel["unique_id"] = panel["item_id"].astype(str) + "_" + panel["store_id"].astype(str)

    # store level aggregation
    store_level = panel.groupby(["store_id", "date"])["sales"].sum().reset_index()
    store_level = store_level.rename(columns={"store_id": "unique_id", "date": "ds", "sales": "y"})
    store_level["ds"] = pd.to_datetime(store_level["ds"])
    store_level["y"] = store_level["y"].astype(float)

    # item-store high volume
    vols = panel.groupby("unique_id")["sales"].sum()
    top_300 = vols.nlargest(300).index
    item_level = panel[panel["unique_id"].isin(top_300)].copy()
    item_level = item_level.rename(columns={"date": "ds", "sales": "y"})
    item_level["ds"] = pd.to_datetime(item_level["ds"])
    item_level["y"] = item_level["y"].astype(float)

    horizon = 28

    for name, df in [("Store Level", store_level), ("Top 300 Item-Store", item_level)]:
        print(f"\n--- {name} ---")

        origin = list(
            rolling_origins(
                n_origins=1, horizon=horizon, last_date=df["ds"].max(), first_date=df["ds"].min()
            )
        )[0]

        train = df[df["ds"] <= pd.to_datetime(origin.train_end)]
        test = df[
            (df["ds"] >= pd.to_datetime(origin.test_start))
            & (df["ds"] <= pd.to_datetime(origin.test_end))
        ]

        scales = {
            uid: naive_scale(g.sort_values("ds")["y"].to_numpy(float), seasonality=1)
            for uid, g in train.groupby("unique_id")
        }

        # DLinear
        dlinear = NeuralForecastAdapter(
            name="dlinear", horizon=horizon, input_size=104, max_steps=500
        )
        dlinear.fit(train)
        dl_preds = dlinear.predict()
        dl_preds["model"] = "DLinear"

        # StatsForecast (SNaive, ETS)
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=7), AutoETS(season_length=7)], freq="D"
        )
        sf.fit(train[["unique_id", "ds", "y"]])
        sf_preds = sf.predict(h=horizon)
        sf_preds = sf_preds.reset_index()

        # SNaive
        snaive_preds = sf_preds[["unique_id", "ds", "SeasonalNaive"]].rename(
            columns={"SeasonalNaive": "y_pred"}
        )
        snaive_preds["model"] = "SNaive"

        # ETS
        ets_preds = sf_preds[["unique_id", "ds", "AutoETS"]].rename(columns={"AutoETS": "y_pred"})
        ets_preds["model"] = "AutoETS"

        all_preds = pd.concat([dl_preds, snaive_preds, ets_preds], ignore_index=True)
        merged = all_preds.merge(test[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
        merged = merged.rename(columns={"y": "y_true"})
        merged["scale"] = merged["unique_id"].map(scales)

        res = metrics_by_group(merged, ["model"], scale_col="scale")
        print(res[["model", "wape", "rmsse", "mae"]].to_string(index=False))


if __name__ == "__main__":
    run_comparison()
