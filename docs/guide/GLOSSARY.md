# Comprehensive Project Glossary

This glossary defines all domain-specific acronyms, mathematical terms, and software concepts used across the repository.

---

### A
- **ABC Classification**: Pareto inventory segmentation dividing products by cumulative revenue contribution into three tiers:
  - **A**: Top 80% of revenue (~20% of items).
  - **B**: Next 15% of revenue (~30% of items).
  - **C**: Bottom 5% of revenue (~50% of items).
- **ADI (Average Inter-Demand Interval)**: The average number of time periods elapsed between successive non-zero demand events:
  $$\text{ADI} = \frac{T}{k}$$
  where $T$ is total active periods and $k$ is the number of periods with non-zero sales.
- **ADIDA (Aggregate-Disaggregate Intermittent Demand Approach)**: A temporal aggregation forecasting technique developed by Nikolopoulos et al. High-frequency intermittent series are aggregated into non-overlapping temporal buckets of size $W$, forecasted using a base model, and disaggregated down to the original daily level.
- **ADR (Architecture Decision Record)**: A structured software engineering document recording a significant architectural decision, along with its context, considered alternatives, and consequences.
- **Autocorrelation**: The degree of linear correlation between values of the same time series at different points in time, measured across varying lag distances.

---

### C
- **CatBoost**: Categorical Boosting; an open-source gradient boosting decision tree library optimizing symmetric (oblivious) decision trees.
- **CD (Critical Difference)**: In post-hoc Nemenyi hypothesis testing, the minimum distance between the average ranks of two competing models required to conclude that the performance difference is statistically significant at significance level $\alpha$.
- **CSL (Cycle Service Level / Type I Service Level)**: The probability that customer demand during a replenishment lead-time cycle will not exceed available stock ($P(\text{Demand} \le \text{Reorder Point})$).
- **Croston's Method**: A classical intermittent demand forecasting algorithm (Croston, 1972) that separately updates demand magnitude ($z$) and inter-arrival intervals ($p$) via single exponential smoothing upon the arrival of non-zero demand:
  $$\hat{y} = \frac{\hat{z}}{\hat{p}}$$
- **Croston SBA (Syntetos-Boylan Approximation)**: A bias-corrected formulation of Croston's estimator (Syntetos & Boylan, 2005). Because $E[1/p] > 1/E[p]$ by Jensen's inequality, classical Croston is positively biased. SBA applies the multiplicative factor $(1 - \alpha / 2)$:
  $$\hat{y}_{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{\hat{z}}{\hat{p}}$$
- **CV (Coefficient of Variation)**: A normalized measure of dispersion defined as the ratio of standard deviation to mean ($\sigma / \mu$).
- **$CV^2$ (Squared Coefficient of Variation)**: The square of the coefficient of variation ($s^2 / \bar{y}^2$), used in Syntetos-Boylan demand categorization to quantify demand size volatility.

---

### D
- **Diebold-Mariano (DM) Test**: A parametric econometric hypothesis test (Diebold & Mariano, 1995) comparing the predictive accuracy of two competing forecasts, adjusting for serial autocorrelation in multi-step forecast errors.
- **Direct Forecasting**: A multi-step forecasting strategy where all feature lags and rolling windows are shifted by the full forecast horizon $h$. The model directly predicts $h$ steps ahead without recursively feeding predictions back as autoregressive inputs.
- **DLinear**: An efficient neural time-series model (Zeng et al., 2023) decomposing input sequences into trend and seasonal components using moving averages, then mapping each component to predictions via a single linear layer.
- **Drift**: A change in dataset distributions over time:
  - *Data Drift (Covariate Shift)*: Change in the marginal feature distribution $P(X)$ over time while $P(y|X)$ remains constant.
  - *Concept Drift*: Change in the conditional relationship between features and target $P(y|X)$.
- **DVC (Data Version Control)**: A git-native version control system managing data pipelines, large model weights, and intermediate evaluation datasets.

---

### E
- **EDA (Exploratory Data Analysis)**: Statistical inspection, summarization, and visual profiling of dataset distributions, missingness, and structural properties.
- **ES_bu (Exponential Smoothing Bottom-Up)**: A hierarchical forecasting baseline generating base forecasts using univariate exponential smoothing at Level 12 (item-store) and aggregating predictions upward through the summing matrix.

---

### F
- **Fill Rate (Type II Service Level)**: The proportion of total customer demand units satisfied directly from shelf inventory without backordering:
  $$\text{Fill Rate} = 1 - \frac{E[\text{Expected Units Short per Cycle}]}{Q}$$
- **Friedman Test**: A non-parametric omnibus hypothesis test (Friedman, 1937) evaluating whether three or more models exhibit statistically significant differences in performance ranks across repeated-measures series.
- **FWER (Family-Wise Error Rate)**: The probability of committing at least one Type I error (false positive) among a family of multiple statistical hypothesis tests.

---

### G
- **GBDT (Gradient Boosted Decision Trees)**: An ensemble machine learning technique constructing decision trees iteratively to minimize a differentiable objective function using gradient descent.

---

### H
- **Holm-Bonferroni Method**: A sequentially rejective step-down statistical procedure (Holm, 1979) controlling the Family-Wise Error Rate (FWER) under multiple pairwise hypothesis testing.
- **Hurdle Model**: A two-stage probabilistic model designed for zero-inflated data. A binary classification model predicts the probability of positive demand ($P(Y > 0)$), and a truncated regression model predicts the positive demand magnitude given that demand is non-zero.

---

### I
- **Intermittent Demand**: A demand pattern characterized by frequent zero-demand periods interspersed with sporadic non-zero sales ($ADI > 1.32$ and $CV^2 < 0.49$).

---

### K
- **Kolmogorov-Smirnov (KS) Test**: A non-parametric test comparing the cumulative distribution functions of two continuous data samples to identify significant distribution drift.

---

### L
- **Lead Time ($L$)**: The physical time latency elapsed between placing a purchase or replenishment order and receiving available stock at the distribution center or retail store shelf.
- **LightGBM**: An optimized, distributed gradient boosting framework (Ke et al., 2017) using histogram-based feature binning, Gradient-based One-Side Sampling (GOSS), and Exclusive Feature Bundling (EFB).
- **Lumpy Demand**: An erratic demand pattern characterized by both infrequent occurrences and volatile demand sizes ($ADI > 1.32$ and $CV^2 \ge 0.49$).

---

### M
- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual values:
  $$\text{MAE} = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|$$
- **MASE (Mean Absolute Scaled Error)**: A scale-free error metric (Hyndman & Koehler, 2006) dividing forecast MAE by the in-sample mean absolute error of a naive one-step random walk.
- **MCB (Multiple Comparisons with the Best)**: A statistical procedure identifying which models are statistically indistinguishable from the top-performing candidate.
- **MinT (Minimum Trace)**: An optimal hierarchical reconciliation algorithm (Wickramasuriya et al., 2019) projecting base forecasts using generalized least squares with error covariance weighting.
- **MLflow**: An open-source MLOps platform managing the machine learning lifecycle, tracking experiment runs, code parameters, and registering candidate models.
- **Moving Average Forecaster**: A simple baseline projecting the unweighted arithmetic mean of the last $W$ historical observations.

---

### N
- **Naive Forecaster**: A standard baseline forecaster projecting the most recent historical observation into all future forecast horizons ($\hat{y}_{T+h} = y_T$).
- **Nemenyi Test**: A post-hoc non-parametric rank test (Nemenyi, 1963) comparing all pairwise differences between average ranks following an omnibus Friedman test rejection.
- **Nested Cross-Validation**: A model validation framework where an inner cross-validation loop optimizes hyperparameters and an independent outer loop evaluates unbiased generalization error.

---

### O
- **OOF (Out-Of-Fold)**: Predictions generated on holdout validation folds during cross-validation, preventing information leakage when training second-stage ensemble models.
- **Optuna**: An automatic hyperparameter optimization framework utilizing Tree-structured Parzen Estimators (TPE) and median stopping rules.

---

### P
- **Pandera**: A data validation and testing library providing statistical schema enforcement and domain constraint verification for Pandas and Polars dataframes.
- **PatchTST**: A transformer-based time series architecture (Nie et al., 2023) extracting local semantic features by slicing 1D series into sub-series patches and applying channel-independent self-attention.
- **Pinball Loss (Quantile Loss)**: An asymmetric linear loss function evaluating quantile forecast accuracy:
  $$L_q(y, \hat{y}) = \max(q(y - \hat{y}), (1 - q)(\hat{y} - y))$$
- **Polars**: A blazingly fast columnar DataFrame library written in Rust and based on Apache Arrow, providing streaming query execution and lazy computation.
- **PSI (Population Stability Index)**: A statistical metric measuring the magnitude of shift between two categorical or binned continuous probability distributions.

---

### R
- **Reorder Point ($ROP$)**: The inventory level threshold that triggers an automated replenishment purchase order:
  $$ROP = \mu_D \cdot L + SS$$
- **RMSSE (Root Mean Squared Scaled Error)**: The official M5 competition scale-free metric dividing forecast RMSE by the in-sample naive one-step random walk RMSE anchored to first sale:
  $$\text{RMSSE} = \sqrt{\frac{\frac{1}{h} \sum_{t=1}^h (y_t - \hat{y}_t)^2}{\frac{1}{T - d_{\text{first}}} \sum_{t=d_{\text{first}}+1}^T (y_t - y_{t-1})^2}}$$
- **Rolling-Origin Backtesting**: An evaluation framework sequentially advancing forecast origin cutoffs through time to test multi-step forecast consistency across multiple temporal horizons.

---

### S
- **Safety Stock ($SS$)**: Buffer inventory held to protect customer service levels against demand variability and lead-time volatility:
  $$SS = z \cdot \sigma_L = z \cdot \sigma_e \cdot \sqrt{L}$$
  Here $\sigma_e$ is the RMSE of the forecast errors (about zero, bias included), not their standard deviation.
- **Seasonal Naive**: A baseline forecaster projecting historical actuals from the exact corresponding seasonal cycle period ($h=7$ for day-of-week seasonality, $h=28$ for 4-week retail cycle).
- **Segment Router**: A modular dispatch engine that classifies time series into SBC quadrants and directs each series to its assigned algorithm. It did not beat applying ADIDA to every series (README router table).
- **SES (Simple Exponential Smoothing)**: A classical univariate time-series model applying geometrically declining weights to past observations.
- **SNAP (Supplemental Nutrition Assistance Program)**: The U.S. federal nutritional assistance program; retail demand in grocery categories displays predictable spikes on designated state SNAP distribution dates.
- **SBC Classification (Syntetos-Boylan-Croston, 2005)**: A 2D demand categorization taxonomy dividing time series into Smooth, Intermittent, Erratic, or Lumpy quadrants based on thresholds $ADI = 1.32$ and $CV^2 = 0.49$. Not to be confused with TSB, which is a forecasting method, not a classification.

---

### T
- **TFT (Temporal Fusion Transformer)**: A deep neural network architecture (Lim et al., 2021) combining recurrent layers, gating mechanisms, and temporal self-attention for interpretable multi-horizon forecasting.
- **TSB (Teunter-Syntetos-Babai)**: An intermittent demand forecaster updating demand arrival probability at every time period ($p_t$), preventing obsolescence over long zero periods.
- **Tweedie Distribution**: An exponential dispersion model whose variance scales as $\text{Var}(Y) = \phi \mu^p$. The compound Poisson-Gamma case ($1 < p < 2$) possesses a discrete mass at zero and continuous positive values, matching retail sales.

---

### W
- **WAPE (Weighted Absolute Percentage Error)**: A volume-weighted forecast accuracy metric:
  $$\text{WAPE} = \frac{\sum |y_t - \hat{y}_t|}{\sum y_t}$$
- **Working Capital**: The total monetary value tied up in physical inventory holding across cycle stock and safety stock buffers:
  $$\text{Working Capital} = \left(\frac{Q}{2} + SS\right) \times \text{Unit Cost}$$
- **WRMSSE (Weighted Root Mean Squared Scaled Error)**: The M5 competition aggregate metric weighting RMSSE by cumulative historical dollar sales contribution across the 12 levels of the retail hierarchy.

---

### X
- **XYZ Classification**: Assortment volatility segmentation partitioning items by coefficient of variation ($CV$):
  - **X**: Stable demand ($CV \le 0.5$).
  - **Y**: Moderate variability ($0.5 < CV \le 1.0$).
  - **Z**: High variability / erratic ($CV > 1.0$).
  - **N**: No sales in the window; $CV$ is undefined, so the item is not "stable". Held at the current service level and flagged for review.
