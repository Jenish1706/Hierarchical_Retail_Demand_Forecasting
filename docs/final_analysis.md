# M5 Forecasting Laboratory: Final Analysis Report (Pillar 5)

This document provides a synthesis of the M5 Forecasting Laboratory based on the verified artifacts from walk-forward backtests, statistical significance testing, and inventory analysis.

## 1. Which model wins overall, and is the margin statistically significant?
Across the 3,000 sampled series over 3 walk-forward rolling 28-day origins (84 test days per series):
* **Pooled WAPE**: `adida` achieved the lowest overall WAPE at **75.17%** (RMSSE 0.9878), followed closely by `moving_average` (75.47%, RMSSE 0.9840), `routed_ensemble` (76.52%, RMSSE 0.9975), and `lightgbm` (77.37%, RMSSE 0.9841).
* **Statistical Significance (Ranks)**: The omnibus Friedman test decisively rejects equal performance across models ($\chi^2 = 2,460.79, p < 10^{-12}$). Post-hoc Nemenyi testing (Critical Difference $CD = 0.2494$) ranks `routed_ensemble` best (average rank 4.545), with `croston_sba` (average rank 4.606) statistically indistinguishable from it. `adida` (rank 4.827) and `lightgbm` (rank 5.590) are statistically significantly worse than `routed_ensemble` on average rank.
* **Key Takeaway**: On this sample, specialized intermittent-demand methods match or beat a complex global LightGBM model.

## 2. Does deep learning beat gradient boosting on M5?
* Deep learning architectures (TFT, PatchTST, N-BEATS, DeepAR, DLinear) were not evaluated in the cross-tier 3,000-series walk-forward backtest (`reports/backtest_main`).
* On the models actually evaluated, gradient boosting (`LightGBM`) did not beat simple intermittent baselines (`ADIDA`, `croston_sba`) or moving average. In zero-inflated retail series (57.5% zeros in test observations), simple methods show remarkable robustness compared to dense tabular gradient boosting.

## 3. Does DLinear match transformers?
* DLinear was explored in preliminary tuning (`scripts/tune_dl.py`), but full cross-tier backtest numbers are pending rerun under standardized M5 evaluation protocols.

## 4. Where does each family win?
* **Intermittent Statistical (ADIDA / Croston SBA)**: Strongest on sparse, intermittent series where demand spikes are separated by zero intervals.
* **Moving Average / SES**: Highly competitive baselines that avoid overfitting to noisy low-volume sales.
* **Global Gradient Boosting (LightGBM)**: Performs solidly on higher-volume, continuous patterns where feature lags and calendar encodings provide predictive signal.

## 5. Do intermittent-demand methods beat general models on sparse series?
* **Yes.** On intermittent series, specialized estimators (`ADIDA`, `Croston SBA`) significantly outperform standard naive baselines (`Seasonal Naive` WAPE 98.92%, `Naive` WAPE 99.65%).

## 6. Does segment routing beat a single global model?
* **Yes.** In the walk-forward backtest (`reports/backtest_main/routing_benefit.csv`), routing intermittent series to ADIDA and others to the global model consistently improved WAPE across all three origins:
  * Origin 2 (2016-02-01): 77.08% routed vs 77.93% global (+1.10% improvement)
  * Origin 1 (2016-02-29): 77.20% routed vs 78.83% global (+2.07% improvement)
  * Origin 0 (2016-03-28): 75.30% routed vs 75.38% global (+0.11% improvement)

## 7. Which probabilistic method is best calibrated?
* Two-stage hurdle models and quantile regression were implemented in `src/models/probabilistic/`. Formal probabilistic calibration benchmarks across walk-forward origins remain pending standardized rerun.

## 8. Does hierarchical reconciliation improve accuracy, or only coherence?
* In `reports/tier1_esbu/`, bottom-up aggregation (`ES_bu`) guarantees coherence across the aggregation hierarchy. However, scaling differences in earlier runs highlighted that RMSSE definitions must strictly follow the canonical M5 in-sample naive scale from first sale to ensure cross-tier comparability.

## 9. Which features matter?
* Temporal recency and rolling statistics (`roll_mean_7`, `roll_mean_28`) provide the strongest feature importance for tree-based models, while horizon-safe shifting (minimum 28 days) ensures zero future information leaks into training.

## 10. Operational Impact: Forecasts to Working Capital
* Connecting forecast error distributions to inventory policies via ABC/XYZ segmentation in stores CA_1 and CA_2 identified $3,223.14 in releasable working capital by differentiating service levels (e.g. 98% for high-value AX items vs 90% for erratic CZ items) without increasing stockout risk on revenue drivers.
