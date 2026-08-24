"""
Recency-weighting ablation.
Evaluate the LightGBM decay hyperparameter against no-decay on the full subsample.
"""
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import M5Loader
from src.features.base import FeaturePipeline
from src.models.classical_ml.trainer import DemandTrainer, TrainingConfig

def main():
    print("Loading data...")
    loader = M5Loader()
    # Loading a 3000 series subsample to run the ablation
    panel = loader.load_panel(sample_series=3000, random_state=42)
    
    print("Building features...")
    pipeline = FeaturePipeline()
    df_features = pipeline.transform(panel)
    
    print("Running without recency weighting (half_life = 0.0)...")
    cfg_no_decay = TrainingConfig(
        models=("lightgbm",),
        recency_half_life_days=0.0,
        build_stack=False,
        build_blend=False,
        tune_trials=0,
        outer_splits=1,
    )
    trainer_no_decay = DemandTrainer(cfg_no_decay)
    trainer_no_decay.fit(df_features)
    res_no = trainer_no_decay.leaderboard_
    print("No decay results:")
    print(res_no[["model", "cv_wape", "cv_rmsse"]])
    
    print("\nRunning with recency weighting (half_life = 365.0)...")
    cfg_decay = TrainingConfig(
        models=("lightgbm",),
        recency_half_life_days=365.0,
        build_stack=False,
        build_blend=False,
        tune_trials=0,
        outer_splits=1,
    )
    trainer_decay = DemandTrainer(cfg_decay)
    trainer_decay.fit(df_features)
    res_decay = trainer_decay.leaderboard_
    print("Decay results:")
    print(res_decay[["model", "cv_wape", "cv_rmsse"]])
    
if __name__ == "__main__":
    main()
