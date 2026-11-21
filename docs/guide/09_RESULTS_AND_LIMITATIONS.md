# Empirical Results, Statistical Verification & Limitations

This document provides a comprehensive synthesis of the platform's empirical findings, null results, threats to validity, and prioritized future directions.

All quantitative rankings and statistical intervals are backed by committed artifacts in `reports/` and synchronized with the [Main Leaderboard in README.md](../../README.md#5-are-the-differences-real).

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
- **Finding**: Preliminary tuning was performed (`scripts/experiments/tune_dl.py`), but full 3,000-series cross-tier evaluations under standardized M5 scaling protocols remain pending. DLinear serves as the formal null hypothesis for future transformer benchmarks.

---

### Q4: Where does each model family excel?
- **Intermittent Statistical (`adida`, `croston_sba`)**: Dominates sparse, zero-inflated series where demand arrives in discrete lumps separated by multi-day zero intervals. Croston SBA has the lower absolute error on more individual series than ADIDA; this benchmark does not establish why.
- **Rolling Baselines (`moving_average`, `ses`)**: Highly robust baselines that prevent overfitting to erratic sales spikes, particularly effective on low-volume retail categories.
- **Global Gradient Boosting (`lightgbm`, `lightgbm_tweedie`)**: Evaluated on direct horizon-shifted demand features, calendar indicators, and SNAP events without price features (`price_features_included: false`). Performs reliably on higher-velocity continuous series, but lacks the specialized variance shrinkage of intermittent methods when prices are unobserved.

---

### Q5: Do intermittent-demand methods beat general models on sparse series?
- **Finding**: **Yes.** On intermittent and lumpy series (which represent 91.0% of the catalog after pre-launch trimming), specialized estimators (`adida`, `croston_sba`) substantially outperform standard naive baselines (`seasonal_naive_28d` WAPE 98.92%, `naive` WAPE 99.65%).

---

### Q6: Does segment routing beat the best single model?
- **Finding**: **No.** Routing by SBC (Syntetos-Boylan-Croston) quadrant (INTERMITTENT to `adida`, no-demand series to a zero forecast, SMOOTH, ERRATIC and LUMPY to `lightgbm`) beats standalone LightGBM at every origin, but it does not beat applying `adida` to every series: the routed ensemble's WAPE is higher than ADIDA's at every origin and pooled, so the best single model is ADIDA at every origin.
- **Where it loses**: in the segments handed to LightGBM. LightGBM, run without price features, is worse than ADIDA on ERRATIC and LUMPY and better on SMOOTH, so it was better on one of the three segments it was given, not on all. INTERMITTENT is ADIDA either way.
- **Numbers**: every figure, per origin and per segment, is in the README table under "6. Does demand classification help?", rendered from `routing_vs_best_single.csv`. This guide does not repeat them.
- **Caveats**: the INTERMITTENT to ADIDA assignment was chosen on these same evaluation origins, so a clean test needs an earlier selection origin; and the best single model is picked per row after the scores are known, which favours the single model.
- **Artifact Backing**: `reports/backtest_main/routing_vs_best_single.csv`, `reports/backtest_main/routing_benefit.csv` (per-origin comparison columns), `reports/backtest_main/routing_assignments.csv` (where the router sent each series at each origin).

---

### Q7: Which probabilistic method is best calibrated?
- **Finding**: Two-stage hurdle models and conformalized quantile regression were implemented in `src/models/probabilistic/`. Conformal prediction guarantees finite-sample coverage at $1 - \alpha$, though cross-origin empirical evaluation across the full panel remains pending standardized rerun.

---

### Q8: Does hierarchical reconciliation improve accuracy or only coherence?
- **Finding**: Bottom-up aggregation (`ES_bu`) guarantees mathematical coherence across all 12 levels of the M5 hierarchy. However, scaling differences observed across earlier runs highlighted that RMSSE definitions must strictly adhere to in-sample naive scale from first sale to ensure cross-tier comparability.
- **Artifact Backing**: `reports/tier1/tier1_by_level.csv`, `reports/tier1_esbu/tier1_by_level.csv`.

---

### Q9: Which features drive gradient boosting performance?
- **Finding**: Not established. No committed artifact records the global model's feature importances, so no claim is made about which features drive it. What holds by construction is that every lag and rolling feature is horizon-shifted (at least 28 days), which the perturbation tests in `tests/unit/evaluation/test_leakage_and_recompute.py` check.

---

### Q10: Operational Impact: Working Capital Optimization
- **Finding**: Translating forecast uncertainty into safety stock via ABC/XYZ segmentation across California stores (`CA_1`, `CA_2`), the capital-release rule frees **$3,170** of working capital but raises total annual cost by **+$12,444/yr** under the project's cost model (holding -$792/yr, expected stockout cost +$13,237/yr). It is a capital-for-risk trade, not a saving.
- **Operational Trade-Off**: The rule cuts BZ/CY/CZ items from 95% to 90% CSL and upgrades AX from 95% to 98%. The cheapest level on the 90/95/98% menu is 98% in every optimised segment (-$18,684/yr for +$11,455 of working capital); the unconstrained optimum is 99.4% to 99.7% by category (HOBBIES at a 45% margin to FOODS at 60%), because a lost unit of margin costs far more than a week of holding it.
- **Margin sensitivity**: the cheapest menu level depends on the gross margin alone. 98% is the cheapest above a 12.6% margin, 95% between 6.2% and 12.6%, and 90% below 6.2%; the looser rule of thumb "SL* reaches 98%" needs 19.3%. At 25% to 30% grocery margins 98% is still the cheapest, so differentiating service levels by ABC/XYZ does not pay under lost-margin costing. Rendered in the README margin table from `reports/inventory/margin_sensitivity.csv` and `margin_break_even.csv`.
- **Decisions versus scoring**: classes are decided on the 84 days before the first test day (ending 2016-01-31), planning demand is ADIDA's mean forecast, and $\sigma_e$ is a forecast-error size measured on the backtest days, not a realised inventory outcome.
- **Artifact Backing**: `reports/inventory/policy_comparison.csv`, `reports/inventory/segment_cost_optimum.csv`, `reports/inventory/portfolio_summary.csv`.

---

## 2. Null and Negative Results

Documenting what *did not work* is critical for engineering integrity:

1. **Tweedie Loss vs. Standard MSE in LightGBM**:
   - Textbook theory indicates Tweedie compound Poisson-Gamma loss is well suited to zero-inflated retail counts; the project uses a power of 1.1.
   - Empirical Result: In post-hoc pairwise testing with Holm-Bonferroni correction, `lightgbm_tweedie` vs `lightgbm` yielded an adjusted p-value of **$1.0000$**. The Tweedie objective yielded no statistically detectable rank improvement over standard least-squares regression.
2. **The Model Complexity Paradox**:
   - A zero-parameter 28-day `moving_average` achieved **75.47% WAPE** and **0.7687 RMSSE**, beating both LightGBM models on RMSSE and trailing the champion ADIDA by only 0.31 percentage points. Complex global feature engineering provided negligible aggregate advantage over simple rolling averages on this sample.
   - *Global Model Handicaps*: Global GBDTs ran under three significant handicaps compared to winning M5 competition setups: (1) **no price or promotion features** (`price_features_included: false` due to workbook truncation at Excel's 1,048,575-row limit), (2) a **3,000-series sample** rather than the full 30,490 catalog, and (3) a **900-day window** rather than the full 1,913-day history. In the original competition, price momentum and catalog-wide cross-learning were decisive factors.
3. **High Head-to-Head Indistinguishability**:
   - In pairwise per-series Diebold-Mariano tests with Holm-Bonferroni correction across series within the pair, `adida` and `croston_sba` were statistically indistinguishable on **93.80% of individual series** (2,814 out of 3,000 series).

---

## 3. Threats to Validity

1. **Truncated Pricing History & Global Model Handicaps**:
   - In the M5 dataset, the source `sell_prices.csv` workbook was truncated at Excel's 1,048,575-row boundary, capturing only 15.3% of the M5 price table (covering CA_1 and CA_2, with 1,507 California series unpriced).
   - Because prices were missing across 85% of store-item combinations, the backtest was executed with `price_features_included: false`. LightGBM was therefore evaluated with three deliberate handicaps: no price features, a 3,000-series sample, and a 900-day history window. The empirical finding that intermittent statistical methods beat global GBDTs carries this essential caveat.
2. **California-Specific Sample Scope**:
   - Walk-forward backtests evaluated a random 3,000-series sample drawn from all ten stores (California, Texas and Wisconsin), not a stratified design; the inventory layer covers only the two California stores that have prices.
3. **Gaussian Safety Stock Approximation on Intermittent Demand**:
   - Standard analytical safety stock formulas ($SS = z \cdot \sigma_e \cdot \sqrt{L}$) assume Gaussian error distributions. On zero-inflated retail series with 57.5% zeros, error distributions are discrete, right-skewed, and non-negative. Working capital figures represent an analytical baseline subject to discrete simulation in Stage B. Here $\sigma_e$ is the forecast-error RMSE (about zero, so bias is paid for), not a standard deviation; the $\sqrt{L}$ scaling treats daily errors as independent and so still understates a persistent bias, and RMSE also charges over-forecast bias. The cross-model comparison (`reports/inventory/safety_stock_by_model.csv`) therefore credits smoothing over Naive with about 30% less safety stock capital, of which part (about 22% on the bias-free standard deviation alone) is variability and the rest is Naive's over-forecast bias.
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
