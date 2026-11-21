"""
Hyperparameter optimization for Classical ML (Tier 2) & Probabilistic (Tier 3) on M5.
Target Models: LightGBM Regressor and HurdleForecaster.
Framework: LightGBM via DemandTrainer.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import M5Loader
from src.features.base import FeaturePipeline
from src.models.classical_ml.trainer import DemandTrainer, TrainingConfig


def main():
    print("Loading data...")
    loader = M5Loader()
    # Using a 3000-series sample for HPO speed
    panel = loader.load_panel(sample_series=3000, random_state=42)

    print("Building features...")
    pipeline = FeaturePipeline()
    # Ensure to use transform as fit_transform was shown to be missing
    df_features = pipeline.transform(panel)

    # 1. Tune Standard LightGBM
    print("\n--- Tuning LightGBM Regressor ---")
    cfg_lgb = TrainingConfig(
        models=("lightgbm",),
        tune_trials=50,  # Phase A requirement: 50 trials
        outer_splits=2,  # Phase B requirement: Fix TimeSeriesSplit >= 2
        build_stack=False,
        build_blend=False,
    )

    trainer_lgb = DemandTrainer(cfg_lgb)
    trainer_lgb.fit(df_features)
    lgb_leaderboard = trainer_lgb.leaderboard_
    print("LightGBM HPO Leaderboard:")
    print(lgb_leaderboard[["model", "cv_wape", "cv_rmsse"]])

    from src.models.probabilistic.hurdle import HurdleForecaster

    # 2. Tune HurdleForecaster
    print("\n--- Tuning HurdleForecaster ---")
    cfg_hurdle = TrainingConfig(
        models=(HurdleForecaster(),),
        tune_trials=50,  # Phase A requirement: 50 trials
        outer_splits=2,  # Phase B requirement: Fix TimeSeriesSplit >= 2
        build_stack=False,
        build_blend=False,
    )

    trainer_hurdle = DemandTrainer(cfg_hurdle)
    trainer_hurdle.fit(df_features)
    hurdle_leaderboard = trainer_hurdle.leaderboard_
    print("HurdleForecaster HPO Leaderboard:")
    print(hurdle_leaderboard[["model", "cv_wape", "cv_rmsse"]])

    # Save the best parameters
    best_params = {
        "lightgbm": trainer_lgb.results_["lightgbm"].best_params,
        "HurdleForecaster": trainer_hurdle.results_["HurdleForecaster"].best_params,
    }

    out_path = Path("reports/tuning/best_ml_configs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\nBest configurations saved to {out_path}")


if __name__ == "__main__":
    main()
