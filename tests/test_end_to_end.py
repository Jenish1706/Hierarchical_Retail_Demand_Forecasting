import numpy as np
import pandas as pd

from src.data.loader import M5Loader
from src.evaluation.point_metrics import metrics_by_group
from src.features.base import FeaturePipeline
from src.mlops.registry import PromotionGate
from src.models.classical_ml.trainer import DemandTrainer, TrainingConfig


def test_end_to_end():
    # 1. Loader
    loader = M5Loader()
    # Need price features to compute WRMSSE weights and features
    panel = loader.load_panel(sample_series=50, random_state=42, with_prices=True)

    assert len(panel) > 0, "Loader returned empty panel"

    # 2. Feature Pipeline
    pipeline = FeaturePipeline(horizon=28, use_polars=True)
    features = pipeline.transform(panel)

    # Ensure no leakage on shift
    assert features["lag_28"].isna().sum() >= 28 * 50

    # Drop NAs
    features = features.dropna(subset=["sales", "lag_28"]).copy()
    assert len(features) > 0, "Feature pipeline dropped all rows"

    # 3. Model Training
    # Train very fast models (Ridge) on 1 fold to test harness
    cfg = TrainingConfig(
        models=("ridge",),
        tune_trials=0,
        outer_splits=2,
        build_stack=False
    )
    trainer = DemandTrainer(cfg)
    features["wrmsse_weight"] = np.random.rand(len(features))
    trainer.fit(features)

    # 4. Metric Evaluation
    # Evaluate predictions using the trainer's predict method
    oof = pd.Series(
        trainer.predict(features, model_name="ridge"),
        name="ridge",
    )
    # Join with features to get series_id, sales, and scale
    oof_eval = pd.concat(
        [
            features[["date", "id", "sales", "wrmsse_weight"]].rename(columns={"id": "series_id"}).reset_index(drop=True),
            oof.reset_index(drop=True),
        ],
        axis=1,
    )

    # Add dummy scale column and model label for RMSSE grouping
    oof_eval["scale"] = 1.0
    oof_eval["model"] = "ridge"

    metrics = metrics_by_group(
        df=oof_eval.rename(columns={"sales": "y_true", "ridge": "y_pred"}),
        group_cols=["model"],
        y_true_col="y_true",
        y_pred_col="y_pred",
        scale_col="scale",
        weight_col="wrmsse_weight",
        series_id_col="series_id",
    )

    assert not metrics.empty
    assert "rmsse" in metrics.columns

    # 5. Promotion Gate
    gate = PromotionGate()
    # Check if a model passes the gate
    passed, improvement, reason = gate.assess(
        challenger_wape=0.8,
        incumbent_wape=1.0,
        min_improvement_pct=2.0,
    )
    assert isinstance(passed, bool)
    assert passed is True
