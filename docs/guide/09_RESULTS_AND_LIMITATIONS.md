# Empirical Results, Statistical Verification & Limitations

This document provides a comprehensive synthesis of the platform's empirical findings, null results, threats to validity, and prioritized future directions.

All quantitative rankings and statistical intervals are backed by committed artifacts in `reports/` and synchronized with the [Main Leaderboard in README.md](../../README.md#walk-forward-backtest-leaderboard-3000-series--3-origins--10-models).

---

## 1. Synthesis of Core Research Findings

### Q1: Which model wins overall, and is the margin statistically significant?
- **Finding**: There is no single, uncontested "best" model across all criteria.
  - **By Volume-Weighted Aggregate Error (WAPE)**: `adida` achieves the lowest aggregate WAPE (**75.16%**), closely followed by `moving_average` (75.47%) and `routed_ensemble` (75.66%).
  - **By Equal-Weighted Series Rank (Nemenyi)**: `croston_sba` achieves the best average rank (**4.55**), followed by `adida` (4.77) and `routed_ensemble` (4.79).
- **Omnibus Hypothesis Test**: The Friedman test decisively rejects equal performance across the 10 models ($\chi^2 = 2429.30, p < 10^{-12}$).
- **Top Clique**: Post-hoc Nemenyi testing with Critical Difference $CD = 0.2494$ identifies a top statistical clique comprising **`croston_sba`**, **`adida`**, and **`routed_ensemble`**. Within this clique, average rank differences are statistically indistinguishable.
- **Bootstrap Confidence Intervals**: Paired series bootstrap (2,000 resamples) demonstrates that:
  - `routed_ensemble` is indistinguishable from `adida` on WAPE: $\Delta\text{WAPE} = +0.50\%$ (95% CI $[-0.01\%, +1.04\%]$).
  - `moving_average` is indistinguishable from `adida` on RMSSE: $\Delta\text{RMSSE} = +0.0011$ (95% CI $[-0.0001, +0.0023]$).
- **Artifact Backing**: `reports/backtest_main/accuracy_overall.csv`, `reports/backtest_main/significance/significance_report.json`, `reports/backtest_main/significance/bootstrap_intervals.csv`.

---

### Q2: Does deep learning beat gradient boosting on M5?
- **Finding**: Deep learning architectures (TFT, PatchTST, N-BEATS, DeepAR, DLinear) were not evaluated in the cross-tier 3,000-series walk-forward backtest.
- **Evaluated Baseline Reality**: On the models actually evaluated under identical rolling-origin conditions, gradient boosted trees (`lightgbm` at 77.39% WAPE / 0.7858 RMSSE; `lightgbm_tweedie` at 77.26% WAPE / 0.7829 RMSSE) did not outperform specialized intermittent methods (`adida` at 75.16% WAPE) or a 28-day `moving_average` (75.47% WAPE / 0.7687 RMSSE).
- **Artifact Backing**: `reports/backtest_main/accuracy_overall.csv`.

---

### Q3: Does DLinear match transformers on retail forecasting?
- **Finding**: Preliminary tuning was performed (`scripts/tune_dl.py`), but full 3,000-series cross-tier evaluations under standardized M5 scaling protocols remain pending. DLinear serves as the formal null hypothesis for future transformer benchmarks.

---

### Q4: Where does each model family excel?
- **Intermittent Statistical (`adida`, `croston_sba`)**: Dominates sparse, zero-inflated series where demand arrives in discrete lumps separated by multi-day zero intervals. Croston SBA wins head-to-head rank tests on sparse series by decoupling inter-arrival intervals from non-zero sales size.
- **Rolling Baselines (`moving_average`, `ses`)**: Highly robust baselines that prevent overfitting to erratic sales spikes, particularly effective on low-volume retail categories.
- **Global Gradient Boosting (`lightgbm`, `lightgbm_tweedie`)**: Performs reliably on high-velocity, continuous series where price momentum, calendar holidays (SNAP events), and cross-series feature distributions provide clear predictive signal.

---

### Q5: Do intermittent-demand methods beat general models on sparse series?
- **Finding**: **Yes.** On intermittent and lumpy series (which represent 72.6% of the catalog post-trimming), specialized estimators (`adida`, `croston_sba`) substantially outperform standard naive baselines (`seasonal_naive_28d` WAPE 98.92%, `naive` WAPE 99.65%).

---

### Q6: Does segment routing beat a single global model?
- **Finding**: **Yes.** Dynamically routing series based on Syntetos-Boylan demand classification (routing intermittent series to `adida` and continuous series to `lightgbm`) consistently improved forecast accuracy over a single global model across all three walk-forward origins:
  - **Origin 2 (2016-02-01)**: 75.82% routed vs 77.87% global (+2.63% improvement)
  - **Origin 1 (2016-02-29)**: 76.72% routed vs 78.81% global (+2.64% improvement)
  - **Origin 0 (2016-03-28)**: 74.44% routed vs 75.51% global (+1.42% improvement)
- **Artifact Backing**: `reports/backtest_main/routing_benefit.csv`.

---

### Q7: Which probabilistic method is best calibrated?
- **Finding**: Two-stage hurdle models and conformalized quantile regression were implemented in `src/models/probabilistic/`. Conformal prediction guarantees finite-sample coverage at $1 - \alpha$, though cross-origin empirical evaluation across the full panel remains pending standardized rerun.

---

### Q8: Does hierarchical reconciliation improve accuracy or only coherence?
- **Finding**: Bottom-up aggregation (`ES_bu`) guarantees mathematical coherence across all 12 levels of the M5 hierarchy. However, scaling differences observed across earlier runs highlighted that RMSSE definitions must strictly adhere to in-sample naive scale from first sale to ensure cross-tier comparability.
- **Artifact Backing**: `reports/tier1/tier1_by_level.csv`, `reports/tier1_esbu/tier1_by_level.csv`.

---

### Q9: Which features drive gradient boosting performance?
- **Finding**: Tree-based feature importance indicates that temporal recency and rolling statistics (`roll_mean_7`, `roll_mean_28`) provide the strongest predictive power, while horizon-safe shifting (minimum 28 days) ensures zero leakage of future observations into training frames.

---

### Q10: Operational Impact: Working Capital Optimization
- **Finding**: Translating forecast uncertainty distributions into safety stock requirements via ABC/XYZ segmentation across California stores (`CA_1`, `CA_2`) identified **$3,223.14** in releasable working capital.
- **Operational Trade-Off**: Releasing this capital requires explicitly trading off planned stockout risk on low-velocity, erratic CZ items (reducing Cycle Service Level from 95% to 90%) in order to upgrade critical revenue-driving AX items (increasing CSL from 95% to 98%).
- **Artifact Backing**: `reports/inventory/portfolio_summary.csv`.

---

## 2. Null and Negative Results

Documenting what *did not work* is critical for engineering integrity:

1. **Tweedie Loss vs. Standard MSE in LightGBM**:
   - Textbook theory indicates Tweedie compound Poisson-Gamma loss ($p=1.5$) is ideal for zero-inflated retail counts.
   - Empirical Result: In post-hoc pairwise testing with Holm-Bonferroni correction, `lightgbm_tweedie` vs `lightgbm` yielded an adjusted p-value of **$1.0000$**. The Tweedie objective yielded no statistically detectable rank improvement over standard least-squares regression.
2. **The Model Complexity Paradox**:
   - A zero-parameter 28-day `moving_average` achieved **75.47% WAPE** and **0.7687 RMSSE**, beating both LightGBM models on RMSSE and trailing the champion ADIDA by only 0.31 percentage points. Complex global feature engineering provided negligible aggregate advantage over simple rolling averages on this sample.
3. **High Head-to-Head Indistinguishability**:
   - In pairwise per-series Diebold-Mariano tests with Holm-Bonferroni correction across 45 model pairs, `adida` and `croston_sba` were statistically indistinguishable on **93.80% of individual series** (2,814 out of 3,000 series).

---

## 3. Threats to Validity

1. **Truncated Pricing History**:
   - In the M5 dataset, 1,507 series in California stores lack recorded prices during initial backtest origins because Walmart only records prices for active weeks. Inventory revenue calculations partition these into an explicit `UNPRICED` segment.
2. **California-Specific Sample Scope**:
   - Walk-forward backtests evaluated a representative 3,000-series sample across California stores (`CA_1`, `CA_2`, `CA_3`, `CA_4`). While stratified across categories, store-level dynamics in Texas and Wisconsin may exhibit differing seasonal patterns.
3. **Gaussian Safety Stock Approximation on Intermittent Demand**:
   - Standard analytical safety stock formulas ($SS = z \cdot \sigma_e \cdot \sqrt{L}$) assume Gaussian error distributions. On zero-inflated retail series with 57.5% zeros, error distributions are discrete, right-skewed, and non-negative. Working capital figures represent an analytical baseline subject to discrete simulation in Stage B.
4. **Direct Horizon Feature Freezing**:
   - Shifting features by $h=28$ days guarantees zero leakage, but assumes the supply chain planner makes decisions strictly at day $t_0$ for the entire 28-day window without mid-horizon daily reforecasting.

---

## 4. Prioritized Future Work (Roadmap to Stage B)

1. **Discrete Lead-Time Inventory Simulation**:
   - Simulate actual reorder point inventory dynamics using Poisson / Negative Binomial stochastic lead-time demand rather than normal approximations.
2. **Cross-Tier Deep Learning Benchmarking**:
   - Evaluate modern neural architectures (PatchTST, TiDE, DLinear) across all 12 hierarchical aggregation levels to determine the exact crossover point where neural representations surpass statistical baselines.
3. **Hierarchical Reconciliation with Shrinkage Covariance**:
   - Implement MinT (Minimum Trace) optimal reconciliation using Ledoit-Wolf shrinkage covariance estimation across bottom-up series.
