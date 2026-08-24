"""
Hyperparameter optimization for Deep Learning models on M5.
Target Models: DLinear, PatchTST, N-HiTS.
Framework: neuralforecast with ray/optuna.
"""
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import M5Loader
from src.evaluation.splits import rolling_origins

# neuralforecast imports
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoDLinear, AutoPatchTST, AutoNHITS
from ray import tune

def main():
    print("Loading data...")
    loader = M5Loader()
    # Load sample to find top 300 volume item-store series
    panel = loader.load_panel(sample_series=3000, random_state=42, with_prices=False, with_calendar=False)
    panel["unique_id"] = panel["item_id"].astype(str) + "_" + panel["store_id"].astype(str)
    
    # Calculate volumes and filter
    vols = panel.groupby("unique_id")["sales"].sum()
    top_300 = vols.nlargest(300).index
    df = panel[panel["unique_id"].isin(top_300)].copy()
    
    df = df.rename(columns={"date": "ds", "sales": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    
    horizon = 28
    # Keep only required columns
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    
    print("Setting up search spaces...")
    # Define search spaces for tuning
    dlinear_config = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "input_size": tune.choice([horizon * 2, horizon * 3, horizon * 4]),
        "max_steps": 500,
    }
    
    patchtst_config = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "input_size": tune.choice([horizon * 2, horizon * 3]),
        "hidden_size": tune.choice([16, 64, 128]),
        "dropout": tune.uniform(0.1, 0.3),
        "max_steps": 500,
    }
    
    nhits_config = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "input_size": tune.choice([horizon * 2, horizon * 4]),
        "dropout_prob_theta": tune.uniform(0.1, 0.3),
        "max_steps": 500,
    }
    
    models = [
        AutoDLinear(h=horizon, config=dlinear_config, num_samples=10),
        AutoPatchTST(h=horizon, config=patchtst_config, num_samples=10),
        AutoNHITS(h=horizon, config=nhits_config, num_samples=10)
    ]
    
    nf = NeuralForecast(models=models, freq='D')
    
    print(f"Starting HPO on {len(top_300)} series...")
    # fit performs the HPO when using Auto-models
    nf.fit(df=df)
    
    print("Tuning complete. Saving best configurations...")
    best_configs = {}
    for m in nf.models:
        # Extract best parameters from the fitted Auto model
        name = m.__class__.__name__.replace("Auto", "")
        # m.results is the ray tune ResultGrid, we can extract the best config
        try:
            best_configs[name] = m.results.get_best_result().config
        except Exception:
            best_configs[name] = "Failed to extract config"
            
    out_path = Path("reports/best_dl_configs.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(best_configs, f)
        
    print(f"Configurations saved to {out_path}")

if __name__ == "__main__":
    main()
