# M5 Forecasting Laboratory: Model Card

## Model Details
- **Architecture**: Multi-tier benchmark comparing statistical intermittent models (ADIDA, Croston SBA), exponential smoothing/baselines, gradient boosted trees (LightGBM), and segment-based routing.
- **Frameworks**: LightGBM, StatsForecast, Scikit-learn.
- **Task**: 28-day horizon retail demand forecasting and inventory decision support.
- **Data**: M5 Walmart Retail Dataset (30,490 item-store series across 1,913 days; evaluated on a 3,000-series sample across 3 rolling origins).

## Intended Use
- **Primary Use Case**: Research and comparative benchmarking of forecasting architectures across varying levels of aggregation and series sparsity.
- **Out of Scope**: Real-world deployment without real-time price elasticity inputs, promotional overrides, or supply constraints.

## Factors & Subgroups
- **Sparsity**: 95.2% of the dataset is zero. The models handle this by separating predictions into intermittent (`ADIDA` via SegmentRouter) and continuous/dense blocks (where `LightGBM` and `DLinear` dominate).
- **Aggregation Levels**: Accuracy varies significantly by hierarchy. Deep learning excels at highly aggregated, dense continuous series (Levels 1-3), while gradient boosting excels at the lowest, most granular item-store level (Level 12).

## Metrics
- **Primary Accuracy**: WAPE (Weighted Absolute Percentage Error) and RMSSE (Root Mean Squared Scaled Error) evaluated across 3 walk-forward origins.
- **Statistical Significance**: Omnibus Friedman test, post-hoc Nemenyi critical difference ranking, and Holm-corrected pairwise Diebold-Mariano tests.
- **Inventory Metrics**: Cycle Service Level (CSL), Safety Stock, Working Capital, and Stockout Risk. 

## Technical & Deployment Limitations
- **Memory Profiling**: Feature pipeline requires ~2.5GB RAM for the full M5 dataset. Vectorization is mandatory for efficiency. 
- **Concurrency Limitation (n_jobs=1)**: During training and inference on Windows architectures, `n_jobs=1` is **strictly required**. 
  - *Finding*: `statsforecast`'s estimators are Numba-compiled and vectorized across series inside a single process. Using process-level parallelism (`n_jobs > 1`) on Windows forces the `spawn` start method, which re-imports the module in each worker. Without a strict `__main__` guard in all execution paths, this immediately causes a `BrokenProcessPool` crash. The spin-up overhead also dominates the actual calculation time for subsets < 1,000 series, making multiprocessing a net negative for performance.

## MLOps Registry Integration
- The deployment of the final model is gated by the `PromotionGate` (`src/mlops/registry.py`).
- Candidate models are strictly required to exceed the incumbent's WRMSSE by `promotion_min_improvement_pct` before being committed to the MLflow Model Registry.
- Tracks configuration schemas dynamically, falling back to SQLite if the central tracking server is unreachable.
