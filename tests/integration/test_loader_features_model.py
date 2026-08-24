"""Real-data smoke test for the shared forecasting path on 50 M5 series.

This is intentionally small enough for CI when raw files are supplied, yet it
crosses the actual boundaries that unit tests cannot: Parquet loader ->
horizon-safe features -> one global model -> common metric bundle.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config.settings import settings
from src.data.loader import M5Loader
from src.evaluation.point_metrics import compute_metrics
from src.features.base import FeaturePipeline


pytestmark = pytest.mark.integration


def _raw_data_available() -> bool:
    raw = settings.raw_data_dir
    return (raw / "sales_train_validation.parquet").exists() and (raw / "calendar.parquet").exists()


@pytest.mark.skipif(not _raw_data_available(), reason="requires extracted M5 Parquet files")
def test_loader_features_lightgbm_metric_path_on_fifty_series():
    from lightgbm import LGBMRegressor

    loader = M5Loader()
    panel = loader.load_panel(sample_series=50, last_n_days=430, with_prices=False, random_state=42)
    # Keep the pipeline instance: `model_feature_columns` is a FeaturePipeline
    # method (it depends on the configured horizon and price settings), not a
    # loader method.
    pipeline = FeaturePipeline(horizon=28, include_price=False)
    features = pipeline.transform(panel)
    dates = np.sort(features["date"].unique())
    cutoff = dates[-29]
    train = features[features["date"] <= cutoff]
    test = features[features["date"] > cutoff]
    feature_cols = pipeline.model_feature_columns(features, drop_price=True)

    model = LGBMRegressor(n_estimators=8, learning_rate=0.1, n_jobs=1, random_state=42, verbosity=-1)
    model.fit(train[feature_cols].fillna(0.0), train["sales"].astype(float))
    prediction = np.clip(model.predict(test[feature_cols].fillna(0.0)), 0.0, None)
    metrics = compute_metrics(
        test["sales"].to_numpy(float),
        prediction,
        y_insample=train["sales"].to_numpy(float),
    )

    assert len(test) == 50 * 28
    assert np.isfinite(metrics["wape"])
    assert np.isfinite(metrics["rmsse"])
