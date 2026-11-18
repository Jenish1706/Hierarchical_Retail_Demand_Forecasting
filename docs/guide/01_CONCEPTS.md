# 01: Core Concepts & Mathematical Formulations

This document explains every fundamental theoretical and operational concept used across the platform. For every concept, we provide:
1. **Intuition**: What problem it solves in retail forecasting and inventory management.
2. **Mathematical Formula**: Formal mathematical definition with exact notation.
3. **Worked Numerical Example**: A step-by-step arithmetic walk-through.
4. **Code Link**: The exact module and function implementing the formulation.

---

## 1. Intermittent Demand Classification (ADI and $CV^2$)

### Intuition
In retail, most products do not sell every day. Fast-moving goods sell constantly, whereas slow-moving spare parts or specialty groceries experience many days of zero demand punctuated by occasional non-zero sales spikes. The **Syntetos-Boylan-Teunter (SBT)** framework segments series across two orthogonal axes:
- **Average Inter-Demand Interval ($ADI$)**: How often demand occurs (time sparsity).
- **Squared Coefficient of Variation of Demand Sizes ($CV^2$)**: How volatile the non-zero transaction quantities are (magnitude volatility).

Products are partitioned into four distinct quadrants with sharp operational boundaries:
- **Smooth** ($ADI < 1.32, CV^2 < 0.49$): Regular, predictable sales.
- **Intermittent** ($ADI \ge 1.32, CV^2 < 0.49$): Sporadic occurrences, but predictable quantities when sales happen.
- **Erratic** ($ADI < 1.32, CV^2 \ge 0.49$): Frequent occurrences, but wildly varying transaction sizes.
- **Lumpy** ($ADI \ge 1.32, CV^2 \ge 0.49$): Both sporadic timing and volatile quantities (the hardest forecasting regime).

### Mathematical Formula
Given a demand series $y = (y_1, y_2, \dots, y_n)$ with non-zero sales occurring at times $\{t_1, t_2, \dots, t_k\}$:
$$\text{Inter-demand intervals: } p_i = t_i - t_{i-1} \quad \text{for } i = 2, \dots, k$$
$$\text{Average Demand Interval: } ADI = \frac{1}{k-1} \sum_{i=2}^k p_i = \frac{t_k - t_1}{k - 1}$$
$$\text{Demand sizes: } z_i = y_{t_i} \quad \text{for } i = 1, \dots, k$$
$$\text{Mean size: } \mu_z = \frac{1}{k}\sum_{i=1}^k z_i, \quad \text{Variance: } \sigma_z^2 = \frac{1}{k-1}\sum_{i=1}^k (z_i - \mu_z)^2$$
$$CV^2 = \left(\frac{\sigma_z}{\mu_z}\right)^2 = \frac{\sigma_z^2}{\mu_z^2}$$

### Worked Example with Numbers
Suppose an item sells across a 10-day active window:
$$y = [0, 2, 0, 0, 4, 0, 0, 0, 2, 0]$$
- Non-zero sales occur on days: $t_1 = 2$ ($z_1 = 2$), $t_2 = 5$ ($z_2 = 4$), $t_3 = 9$ ($z_3 = 2$). Total transactions $k = 3$.
- Inter-arrival times: $p_1 = 5 - 2 = 3$, $p_2 = 9 - 5 = 4$.
- $ADI = \frac{3 + 4}{2} = 3.50$. Since $3.50 \ge 1.32$, demand is **sparse**.
- Non-zero quantities: $\{2, 4, 2\}$.
  - $\mu_z = \frac{2 + 4 + 2}{3} = \frac{8}{3} \approx 2.667$.
  - $\sigma_z^2 = \frac{(2 - 2.667)^2 + (4 - 2.667)^2 + (2 - 2.667)^2}{2} = \frac{0.444 + 1.778 + 0.444}{2} = 1.333$.
  - $CV^2 = \frac{1.333}{2.667^2} = \frac{1.333}{7.111} \approx 0.187$.
- Since $ADI = 3.50 \ge 1.32$ and $CV^2 = 0.187 < 0.49$, the series is classified as **INTERMITTENT**.

### Code Link
Implemented in [`src/models/statistical/intermittent.py:classify_demand()`](../../src/models/statistical/intermittent.py#L40-L105).

---

## 2. Intermittent Forecasting Methods: Croston, SBA, TSB, ADIDA

Standard Exponential Smoothing fails on intermittent series: when demand is zero, it updates the forecast downward; when a spike occurs, it over-reacts upward, inducing artificial demand oscillation.

### Croston's Method (1972)
#### Intuition
Separates the process into two independent components: updating the non-zero demand size $z_t$ and the inter-arrival interval $p_t$ only when a non-zero transaction occurs.

#### Formula
When demand occurs ($y_t > 0$):
$$\hat{z}_{t} = \alpha y_t + (1 - \alpha) \hat{z}_{t-1}$$
$$\hat{p}_{t} = \alpha q_t + (1 - \alpha) \hat{p}_{t-1}$$
where $q_t$ is the elapsed days since the last sale. When $y_t = 0$, both states remain unchanged ($\hat{z}_t = \hat{z}_{t-1}, \hat{p}_t = \hat{p}_{t-1}$).
The rate forecast is:
$$\hat{y}_{t+h|t} = \frac{\hat{z}_t}{\hat{p}_t}$$

### Syntetos-Boylan Approximation (SBA, 2005)
#### Intuition
Syntetos & Boylan proved mathematically that Croston's estimator is positively biased: by Jensen's inequality, $E[z / p] > E[z] / E[p]$. They introduced an exact shrinkage factor that removes this upward bias.

#### Formula
$$\hat{y}_{t+h|t}^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{\hat{z}_t}{\hat{p}_t}$$

#### Worked Numerical Example (Croston vs. SBA)
Let smoothing parameter $\alpha = 0.10$. Previous state: $\hat{z} = 4.0$ units, $\hat{p} = 2.0$ days.
- Croston point forecast:
  $$\hat{y}^{\text{Croston}} = \frac{4.0}{2.0} = 2.00 \text{ units/day}$$
- SBA debiased point forecast:
  $$\hat{y}^{\text{SBA}} = \left(1 - \frac{0.10}{2}\right) \cdot 2.00 = 0.95 \cdot 2.00 = 1.90 \text{ units/day}$$
The SBA correction eliminates the 5% upward bias caused by the harmonic mean property.

### Teunter-Syntetos-Babai (TSB, 2011)
#### Intuition
Croston and SBA cannot detect product obsolescence: if an item never sells again, $y_t=0$ forever, but $\hat{z}$ and $\hat{p}$ freeze at their last values, continuously forecasting positive sales. TSB replaces the interval $\hat{p}$ with a continuously updating probability of demand occurrence $P(y_t > 0)$.

#### Formula
Every period $t$ (whether demand occurs or not):
$$\hat{P}_t = \beta \cdot \mathbb{I}(y_t > 0) + (1 - \beta) \hat{P}_{t-1}$$
When $y_t > 0$:
$$\hat{z}_t = \alpha y_t + (1 - \alpha) \hat{z}_{t-1}$$
Forecast:
$$\hat{y}_{t+h|t}^{\text{TSB}} = \hat{P}_t \cdot \hat{z}_t$$

### ADIDA (Aggregate-Disaggregate Intermittent Demand Approach, Nikolopoulos et al. 2011)
#### Intuition
Instead of modeling sparse daily zeros, ADIDA aggregates daily sales into non-overlapping temporal buckets of size $K$ (e.g., $K = \text{round}(ADI)$). Aggregating removes zero sparsity and converts intermittent demand into a continuous time series. A standard forecaster (SES) is applied at the aggregate level, and the resulting bucket forecast is disaggregated equally across the $K$ days.

#### Worked Numerical Example (ADIDA)
Suppose active $ADI = 3.8 \approx 4$. We aggregate daily data into $K=4$-day buckets.
- Daily actuals over 12 days: $[0, 2, 0, 0 \mid 1, 0, 0, 3 \mid 0, 0, 2, 0]$
- Bucket totals: $B_1 = 2$, $B_2 = 4$, $B_3 = 2$.
- Applying SES ($\alpha=0.2$) across buckets yields aggregate forecast $\hat{B}_4 = 2.8$ units for the next 4-day block.
- Disaggregating across the 4 days yields a daily rate:
  $$\hat{y}_{t} = \frac{2.8}{4} = 0.70 \text{ units/day}$$

### Code Links
Implemented in [`src/models/statistical/intermittent.py`](../../src/models/statistical/intermittent.py):
- `croston()`: [Line 115](../../src/models/statistical/intermittent.py#L115)
- `croston_sba()`: [Line 160](../../src/models/statistical/intermittent.py#L160)
- `tsb()`: [Line 200](../../src/models/statistical/intermittent.py#L200)
- `adida()`: [Line 250](../../src/models/statistical/intermittent.py#L250)

---

## 3. Evaluation Metrics: RMSSE, WAPE, and WRMSSE

### Root Mean Squared Scaled Error (RMSSE)
#### Intuition
Standard percentage errors (MAPE) explode when actual sales are zero ($\text{division by zero}$). Absolute errors (MAE) favor high-volume items and penalize sparse items arbitrarily. The M5 competition metric, **RMSSE**, normalizes squared forecast errors against the in-sample one-step-ahead naive random walk scale. An RMSSE $< 1.0$ indicates the model outperforms a naive persistence baseline.

#### Formula
For a single series $s$ evaluated over horizon $h$ with in-sample historical active training length $n$:
$$\text{In-sample naive scale: } s_i = \frac{1}{n - d_{\text{first}}} \sum_{t = d_{\text{first}} + 1}^n (y_{i,t} - y_{i,t-1})^2$$
$$\text{RMSSE}_{i} = \sqrt{\frac{\frac{1}{h} \sum_{t = n+1}^{n+h} (y_{i,t} - \hat{y}_{i,t})^2}{s_i}}$$
Across $M$ series in a panel, overall RMSSE is the unweighted mean:
$$\text{RMSSE} = \frac{1}{M} \sum_{i=1}^M \text{RMSSE}_i$$

#### Worked Numerical Example (RMSSE)
Suppose active training history after first sale has 4 days: $[1, 3, 2, 4]$.
- In-sample differences: $(3-1)=2, (2-3)=-1, (4-2)=2$.
- Squared differences: $4, 1, 4$.
- Naive scale: $s_i = \frac{4 + 1 + 4}{3} = \frac{9}{3} = 3.00$.
- Evaluation period ($h=2$ days): actuals $y = [2, 5]$, forecast $\hat{y} = [3, 4]$.
- Forecast errors: $(2 - 3) = -1$, $(5 - 4) = 1$.
- Evaluation MSE: $\frac{(-1)^2 + 1^2}{2} = \frac{2}{2} = 1.00$.
- $\text{RMSSE}_i = \sqrt{\frac{1.00}{3.00}} = \sqrt{0.3333} \approx 0.577$.

### Weighted Absolute Percentage Error (WAPE)
#### Intuition
WAPE (also known as MAD/Mean ratio) evaluates volume-weighted accuracy across an entire portfolio. It computes the total absolute unit error across all series and divides by total units demanded. Unlike MAPE, WAPE never divides by zero at the series level.

#### Formula
$$\text{WAPE} = 100\% \times \frac{\sum_{i=1}^M \sum_{t=1}^h |y_{i,t} - \hat{y}_{i,t}|}{\sum_{i=1}^M \sum_{t=1}^h y_{i,t}}$$

#### Worked Numerical Example (WAPE)
Two series over 1 day:
- Series 1 (high volume): actual $y_1 = 100$, forecast $\hat{y}_1 = 80$. $|e_1| = 20$.
- Series 2 (low volume): actual $y_2 = 2$, forecast $\hat{y}_2 = 0$. $|e_2| = 2$.
$$\text{WAPE} = 100\% \times \frac{20 + 2}{100 + 2} = 100\% \times \frac{22}{102} \approx 21.57\%$$

### Weighted Root Mean Squared Scaled Error (WRMSSE)
#### Intuition
The official M5 ranking metric weights each series' RMSSE by its dollar sales volume share ($w_i = \frac{\sum \text{DollarSales}_i}{\sum_{\text{all}} \text{DollarSales}}$) across the final 28 days of the training window:
$$\text{WRMSSE} = \sum_{i=1}^{30,490} w_i \cdot \text{RMSSE}_i$$

### Code Links
Implemented in [`src/evaluation/point_metrics.py`](../../src/evaluation/point_metrics.py):
- `naive_scale()`: [Line 35](../../src/evaluation/point_metrics.py#L35)
- `rmsse()`: [Line 80](../../src/evaluation/point_metrics.py#L80)
- `wape()`: [Line 110](../../src/evaluation/point_metrics.py#L110)
- `metrics_by_group()`: [Line 160](../../src/evaluation/point_metrics.py#L160)

---

## 4. Hypothesis Testing: Friedman, Nemenyi, and Diebold-Mariano

### Non-Parametric Friedman Omnibus Test
#### Intuition
Before claiming model A is better than model B, we must test the omnibus null hypothesis: *Are all models performing identically across the panel?*
Averaging metric values directly is invalid because a few erratic series dominate any mean. Friedman replaces metric values with **within-series ranks** (1 = lowest error, $k$ = highest error), making the test robust to scale and outliers.

#### Formula
Given $N$ series and $k$ models, let $r_i^j$ be the rank of model $j$ on series $i$, and $R_j = \frac{1}{N}\sum_{i=1}^N r_i^j$ be the average rank of model $j$:
$$\chi_F^2 = \frac{12N}{k(k+1)} \left[ \sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4} \right]$$
Under $H_0$, $\chi_F^2$ follows a chi-squared distribution with $k-1$ degrees of freedom.

### Nemenyi Post-Hoc Test & Critical Difference ($CD$)
#### Intuition
When the Friedman test rejects $H_0$, the Nemenyi post-hoc test determines exactly which model pairs differ significantly. Two models whose average ranks differ by more than the Critical Difference ($CD$) are statistically separable.

#### Formula
$$CD = q_\alpha \sqrt{\frac{k(k+1)}{6N}}$$
where $q_\alpha$ is the critical value of the Studentized range statistic divided by $\sqrt{2}$. At $\alpha=0.05$ for $k=10$ models, $q_{0.05} = 3.164$.
With $N = 2,950$ complete series:
$$CD = 3.164 \times \sqrt{\frac{10 \times 11}{6 \times 2,950}} = 3.164 \times \sqrt{\frac{110}{17,700}} = 3.164 \times 0.07883 \approx 0.2494$$
Any two models with $|\bar{R}_a - \bar{R}_b| \le 0.2494$ form a statistically indistinguishable clique.

### Diebold-Mariano Test (Per-Series vs. Panel-Pooled)
#### Intuition
Tests whether the loss differential $d_t = L(e_{a,t}) - L(e_{b,t})$ between two forecasts has zero mean, while accounting for the fact that multi-step forecast errors are serially correlated (autocorrelated up to lag $h-1$).

#### Formula
$$\text{Mean differential: } \bar{d} = \frac{1}{n}\sum_{t=1}^n d_t$$
$$\text{Long-run variance: } \hat{V}(\bar{d}) = \gamma_0 + 2\sum_{k=1}^{h-1} \gamma_k, \quad \text{where } \gamma_k = \frac{1}{n}\sum_{t=k+1}^n (d_t - \bar{d})(d_{t-k} - \bar{d})$$
$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})/n}}$$
Harvey, Leybourne & Newbold (1997) small-sample correction modifies this by:
$$DM^* = DM \times \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$
and compares against Student's $t_{n-1}$.

#### The Panel Pooling Pitfall
Concatenating 3,000 series into a 252,000-observation vector creates an artificially huge $N$, driving $\sqrt{\hat{V}/N} \to 0$ and yielding $p < 10^{-12}$ on practically irrelevant differences while ignoring cross-series spatial correlation. The platform therefore replaces pooled DM with **per-series testing** (computing $DM$ per series, adjusting $p$-values with Holm's step-down method) and **paired series bootstrap confidence intervals**.

### Code Links
Implemented in [`src/evaluation/significance.py`](../../src/evaluation/significance.py):
- `friedman_test()`: [Line 346](../../src/evaluation/significance.py#L346)
- `nemenyi_test()`: [Line 471](../../src/evaluation/significance.py#L471)
- `paired_series_bootstrap()`: [Line 660](../../src/evaluation/significance.py#L660)
- `pairwise_per_series_tests()`: [Line 815](../../src/evaluation/significance.py#L815)

---

## 5. Direct Multi-Horizon Features and Information Leakage

### The Information Leakage Hazard
In a 28-day forecasting deployment, the model predicts day $t+1$ through $t+28$ simultaneously. If a feature uses sales from day $t-1$ to predict day $t+15$, that feature will be unavailable in production (since day $t+14$ sales are in the future).

### Horizon-Safe Direct Lags
To prevent leakage, all rolling features and lags must be shifted by **at least the maximum forecast horizon $h_{\text{max}} = 28$ days**:
$$\text{Safe lag: } x_{t, k} = y_{t - 28 - k} \quad \text{for } k \ge 0$$
$$\text{Safe rolling mean: } \bar{y}_{t, w} = \frac{1}{w}\sum_{j=0}^{w-1} y_{t - 28 - j}$$
At inference time for any day $t+1 \le \tau \le t+28$, all conditioning inputs are strictly historical observations ($t \le 0$).

### Code Link
Implemented in [`src/features/polars_engine.py`](../../src/features/polars_engine.py#L75-L160) with runtime assertion `assert_no_temporal_leakage()`.

---

## 6. Safety Stock, Cycle Service Level (CSL) vs. Fill Rate

### Cycle Service Level (CSL)
CSL is the probability that no stockout occurs during replenishment lead time $L$:
$$\text{CSL} = P(\text{Demand in } L \le \text{Reorder Point})$$
Targeting 95% CSL means 5% of replenishment cycles will experience at least one stockout.

### Fill Rate ($FR$)
Fill rate is the fraction of customer unit demand met directly from shelf inventory:
$$FR = 1 - \frac{E[\text{Units Short in Lead Time}]}{E[\text{Lead Time Demand}]}$$
Fill rate is almost always higher than CSL: in a cycle with a stockout, 90% of requested units may still be fulfilled.

### Parametric Normal Safety Stock Formulation
Under the standard normal approximation with lead time $L = 7$ days and forecast error standard deviation $\sigma_e$:
$$\text{Safety Stock} = z_{\text{CSL}} \cdot \sigma_e \cdot \sqrt{L}$$
$$\text{Reorder Point (ROP)} = \hat{\mu}_L + \text{Safety Stock} = \hat{\mu} \cdot L + z_{\text{CSL}} \cdot \sigma_e \cdot \sqrt{L}$$
where:
- 90% CSL: $z = 1.282$
- 95% CSL: $z = 1.645$
- 98% CSL: $z = 2.054$

#### Worked Numerical Example (Safety Stock)
Item has mean daily forecast $\hat{\mu} = 10$ units, forecast root-mean-squared error $\sigma_e = 3.0$ units, lead time $L = 7$ days.
- Lead-time demand: $\hat{\mu}_L = 10 \times 7 = 70$ units.
- At 95% CSL ($z = 1.645$):
  $$\text{Safety Stock} = 1.645 \times 3.0 \times \sqrt{7} = 1.645 \times 3.0 \times 2.6458 \approx 13.06 \approx 14 \text{ units}$$
  $$\text{ROP} = 70 + 14 = 84 \text{ units}$$
- At 98% CSL ($z = 2.054$):
  $$\text{Safety Stock} = 2.054 \times 3.0 \times 2.6458 \approx 16.31 \approx 17 \text{ units}$$
  Upgrading service level from 95% to 98% requires 3 additional units of safety stock (+21.4% working capital buffer).

### Intermittent Demand Reality
Because retail sales have a discrete point mass at zero and heavy positive tails, the Gaussian normal approximation underestimates tail risk on erratic items and overestimates buffer on zero-heavy items. It serves as an analytical baseline in Stage A, pending empirical trace-driven stockout simulation in Stage B.

### Code Link
Implemented in [`src/inventory/safety_stock.py:calculate_safety_stock()`](../../src/inventory/safety_stock.py#L45-L95).

---

## 7. ABC/XYZ Inventory Classification Matrix

### ABC Analysis (Revenue Pareto Principle)
Sorts catalog SKUs by historical revenue contribution:
- **Class A**: Top 80% of cumulative revenue (~10-20% of SKUs). Critical revenue drivers.
- **Class B**: Next 15% of cumulative revenue (~20-30% of SKUs). Secondary items.
- **Class C**: Bottom 5% of cumulative revenue (~50-60% of SKUs). Long-tail items.

### XYZ Analysis (Demand Volatility)
Classifies SKUs by demand predictability using the coefficient of variation ($CV = \sigma / \mu$):
- **Class X** ($CV \le 0.50$): Low volatility, stable demand.
- **Class Y** ($0.50 < CV \le 1.0$): Moderate variability.
- **Class Z** ($CV > 1.0$): Extreme volatility, highly erratic.

### The 3x3 Policy Matrix
Combining ABC and XYZ yields 9 distinct replenishment strategies:
- **AX**: High revenue, stable demand -> Low relative safety buffer, tight automated reordering, high CSL (98%).
- **AZ**: High revenue, volatile demand -> Strategic safety buffer, executive exception handling, guarded CSL (95%).
- **CZ**: Low revenue, erratic demand -> Minimized buffer, periodic batch replenishment, relaxed CSL (90%) to release locked working capital.

### Code Link
Implemented in [`src/inventory/abc_xyz.py:classify_abc_xyz()`](../../src/inventory/abc_xyz.py#L30-L110).
