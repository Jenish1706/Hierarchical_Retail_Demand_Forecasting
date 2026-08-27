import pytest
import pandas as pd
import numpy as np

from src.data.loader import M5Loader
from src.features.base import FeaturePipeline
from src.models.classical_ml.trainer import TrainingConfig, DemandTrainer
from src.evaluation.point_metrics import metrics_by_group
from src.mlops.registry import PromotionGate

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
    # Evaluate predictions
    oof = pd.Series(trainer.fitted_models_["ridge"].predict(features.drop(columns=["sales", "date", "item_id", "store_id", "wrmsse_weight"], errors="ignore")), name="ridge")
    # Join with features to get series_id and scale
    oof_eval = pd.concat([features[["date", "series_id", "wrmsse_weight"]].reset_index(drop=True), 
                          oof.reset_index(drop=True)], axis=1)
    
    # Add dummy scale column for RMSSE
    oof_eval["scale"] = 1.0
    
    metrics = metrics_by_group(
        df=oof_eval.rename(columns={"sales": "y_true", "ridge": "y_pred"}),
        group_cols=["model"],
        y_true_col="y_true",
        y_pred_col="y_pred",
        scale_col="scale",
        weight_col="wrmsse_weight",
        series_id_col="series_id"
    )
    
    assert not metrics.empty
    assert "rmsse" in metrics.columns
    
    # 5. Promotion Gate
    gate = PromotionGate()
    # Check if a model passes the gate
    # Just asserting it runs without error
    passed, reason = gate.evaluate_candidate(
        candidate_name="ridge",
        candidate_wrmsse=0.8,
        experiment_name="test_experiment"
    )
    
    assert isinstance(passed, bool)
