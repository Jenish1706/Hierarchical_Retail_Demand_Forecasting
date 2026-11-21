# M5 Retail Demand Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DVC Pipeline](https://img.shields.io/badge/pipeline-DVC%20v3-orange.svg)](https://dvc.org)
[![Data Engine: Polars](https://img.shields.io/badge/engine-Polars%20lazy%20streaming-blueviolet.svg)](https://pola.rs)
[![Documentation](https://img.shields.io/badge/docs-comprehensive%20guide-blue.svg)](docs/guide/00_START_HERE.md)

An empirical demand-forecasting and inventory benchmark on the Walmart M5 retail data (store-item daily unit sales). It is organised as one chain of eight questions, each raised by the answer to the one before, from "what does the demand look like?" to "which service level should each segment get?". Every number in this README is rendered from a committed artifact (`python scripts/render_readme_tables.py --check` verifies that) or tied to one by a test, and where a result is negative or limited the limit is stated next to it.

> **Deep Documentation Available**: For first-principles math, worked numerical examples, architecture diagrams, runbooks, and design decisions, see the [Comprehensive Guide](docs/guide/00_START_HERE.md).

### The chain at a glance

1. [What does the demand look like?](#1-what-does-the-demand-look-like) Mostly zeros, and many series launch late.
2. [Which metric is fair for it?](#2-which-metric-is-fair-for-it) WAPE for volume, RMSSE per (series, origin) for scale-free comparison, with pre-launch zeros trimmed only before a series' first sale.
3. [How do we evaluate without leakage?](#3-how-do-we-evaluate-without-leakage) Rolling origins, horizon-shifted features and training-window-only statistics.
4. [Which models, and what does each rung test?](#4-which-models-and-what-does-each-rung-test) A ladder from naive, through intermittent-demand methods, to a global gradient-boosted model with stated handicaps.
5. [Are the differences real?](#5-are-the-differences-real) Friedman, Nemenyi and a paired series bootstrap separate real gaps from noise.
6. [Does demand classification help?](#6-does-demand-classification-help) Not against ADIDA applied to every series.
7. [What is a better forecast worth?](#7-what-is-a-better-forecast-worth) Safety-stock capital, by forecast model.
8. [Which service level should each segment get?](#8-which-service-level-should-each-segment-get) Under lost-margin costing the highest service level on the menu is the cheapest for every segment, so differentiating by ABC/XYZ trades capital for risk instead of saving money.

---

## Forecast benchmark (steps 1 to 6)

### 1. What does the demand look like?

Store-item sales are mostly zeros, and many series start late. Before choosing a metric or a model, look at the shape (rendered from `reports/eda/`):

<!-- BEGIN_DEMAND_PROFILE -->
- **Scale**: 30,490 store-item series over 1,913 days, 58.3 million daily observations.
- **Sparsity**: 68.2% of daily records are zero (Foods 62.0%, Household 71.8%, Hobbies 77.3% by category).
- **Late launches**: 77% of series (23,511) had no sale on day 1, and the median first sale is on day 160. Zeros before a series' first sale mean "not on sale yet", not "no demand".
- **Demand patterns** (SBC quadrants, full history, pre-launch zeros trimmed): 91.0% of series are intermittent or lumpy.

| SBC quadrant | Rule | Series | % of series | % of volume |
|---|---|---|---|---|
| **SMOOTH** | ADI < 1.32, CV² < 0.49 | 1,895 | 6.2% | 28.5% |
| **ERRATIC** | ADI < 1.32, CV² ≥ 0.49 | 859 | 2.8% | 13.8% |
| **LUMPY** | ADI ≥ 1.32, CV² ≥ 0.49 | 5,598 | 18.4% | 23.7% |
| **INTERMITTENT** | ADI ≥ 1.32, CV² < 0.49 | 22,138 | 72.6% | 34.0% |
<!-- END_DEMAND_PROFILE -->

*Raises the next question*: zero-heavy, late-starting series break the usual accuracy measures. Which metric is fair for them?

### 2. Which metric is fair for it?

- **WAPE** (weighted absolute percentage error) is pooled absolute error divided by pooled actual sales. It is defined on zeros, so it works for sparse series, and it is volume-weighted, so it follows units sold.
- **RMSSE** (the M5 competition's scale-free error) divides the root mean squared error by the in-sample one-step naive error of the same series, so series of different size are comparable. It is computed per (series, origin) with that origin's own scale and then averaged, never with one scale shared across origins.
- **Trimming rule** (the M5 rule): the scale and the demand classification drop zeros only before a series' first sale in its *full* history, never zeros in the middle of its life. Trimming a mid-life window would turn genuine zero-demand days into "not on sale yet".
- The two measures can disagree on sparse items (equal-weighted ranks versus volume), which is why both are reported and why step 5 tests the gaps.

*Raises the next question*: a fair metric still needs a fair test. How do we evaluate without leakage?

### 3. How do we evaluate without leakage?

A rolling-origin walk-forward backtest: at each origin the models see only data up to that origin and forecast the next block of days, and the next origin starts after the previous test window (rendered from `reports/backtest_main/manifest.json`).

<!-- BEGIN_EVAL_PROTOCOL -->
| Origin index | Training ends | Test window |
|---|---|---|
| 2 | 2016-01-31 | 2016-02-01 to 2016-02-28 |
| 1 | 2016-02-28 | 2016-02-29 to 2016-03-27 |
| 0 | 2016-03-27 | 2016-03-28 to 2016-04-24 |

- **Sample and size**: 3,000 series drawn at random with a fixed seed, 900 days of history each. 3 origins x 28 days = 84 test days per series, 252,000 predictions per model. Origin indices count back from the end of the data, so index 0 is the latest window.
- **Horizon-shifted features**: every lag and rolling feature the global model sees is shifted by the full 28-day horizon (38 features), so a test row never references an actual inside its own window.
- **Training-window-only statistics**: the RMSSE scale and the router's SBC quadrants for each origin use data up to that origin's training end, never the test window.
<!-- END_EVAL_PROTOCOL -->

*Raises the next question*: with a leakage-free test in place, which models do we run, and what is each one there to test?

### 4. Which models, and what does each rung test?

Each rung adds one idea to the one below it, so a gap between two rungs can be attributed to that idea.

<!-- BEGIN_MODEL_LADDER -->
| Rung | Model | Tier | What this rung tests |
|---|---|---|---|
| 1 | Naive | Baseline | The floor: repeat the last observed value. |
| 2 | Seasonal Naive (Weekly) | Baseline | Does a weekly cycle alone help? |
| 3 | Seasonal Naive (28d) | Baseline | Does a four-week cycle alone help? |
| 4 | Moving Average (28d) | Simple Baseline | Does averaging out the noise help? |
| 5 | SES | Exponential Smoothing | Does recency-weighted smoothing help? |
| 6 | Croston SBA | Intermittent Statistical | Does modelling demand size and interval separately (debiased) help on intermittent series? |
| 7 | ADIDA | Intermittent Statistical | Does aggregating to shorter, denser series help on intermittent series? |
| 8 | LightGBM | Global GBDT | Does a global model that learns across series help? (squared-error loss) |
| 9 | LightGBM (Tweedie) | Global GBDT | The same global model with a loss suited to zero-inflated counts. |
| 10 | Routed SBC Ensemble | Hybrid (ADIDA + Global) | Does sending each demand pattern to a different method help? (step 6) |

- **Handicaps on the global model**: LightGBM ran under three handicaps relative to the M5 competition: (1) no price features (`price_features_included: false`; the price table came from a workbook truncated at Excel's row limit and covers about 15% of M5 prices), (2) a 3,000-series sample of the 30,490-series catalogue, and (3) a 900-day window of the 1,913-day history. In the M5 competition price and promotion features and catalogue-wide learning were a large part of why GBDTs won, so "intermittent methods match or beat global GBDTs here" carries that qualification.
<!-- END_MODEL_LADDER -->

*Raises the next question*: the rungs score differently, but are those differences larger than noise?

### 5. Are the differences real?

Rankings are gated by an omnibus Friedman test, Nemenyi critical-difference ranking and paired series bootstrap confidence intervals (resampling whole series, so within-series autocorrelation is kept). The leaderboard is ordered by Nemenyi average rank, with WAPE and RMSSE and their intervals alongside:

<!-- BEGIN_LEADERBOARD_TABLE -->
| Rank | Model Name | Architecture Tier | Nemenyi Avg Rank | Top Clique? | WAPE (%) [95% CI] | RMSSE [95% CI] | $\Delta$WAPE vs Best [95% CI] | Statistical Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **Croston SBA** | Intermittent Statistical | **4.55** | **Yes** | 77.80% [74.63%, 81.09%] | 0.789 [0.777, 0.810] | +2.64% [+2.06%, +3.30%] | **Rank Leader** |
| **2** | **ADIDA** | Intermittent Statistical | 4.77 | **Yes** | **75.16% [72.16%, 78.23%]** | **0.768 [0.756, 0.788]** | **+0.00% [+0.00%, +0.00%]** | **WAPE Leader** (within CD of rank leader) |
| **3** | **Routed SBC Ensemble** | Hybrid (ADIDA + Global) | 4.79 | **Yes** | 75.66% [72.40%, 78.87%] | 0.769 [0.757, 0.789] | +0.50% [-0.01%, +1.04%] | **Indistinguishable from Best** (rank & WAPE) |
| **4** | **SES** | Exponential Smoothing | 4.91 | No | 76.99% [74.20%, 79.72%] | 0.772 [0.760, 0.792] | +1.83% [+1.13%, +2.62%] | Significantly worse than best rank |
| **5** | **Moving Average (28d)** | Simple Baseline | 4.94 | No | 75.47% [72.39%, 78.54%] | 0.769 [0.757, 0.789] | +0.31% [+0.05%, +0.59%] | Significantly worse than best rank |
| **6** | **LightGBM (Tweedie)** | Global GBDT | 5.38 | No | 77.26% [74.10%, 80.34%] | 0.783 [0.771, 0.803] | +2.10% [+1.40%, +2.84%] | Significantly worse than best rank |
| **7** | **LightGBM** | Global GBDT | 5.56 | No | 77.39% [74.11%, 80.59%] | 0.786 [0.774, 0.805] | +2.23% [+1.58%, +2.90%] | Significantly worse than best rank |
| **8** | **Naive** | Baseline | 6.16 | No | 99.65% [96.06%, 103.28%] | 1.002 [0.984, 1.027] | +24.49% [+22.62%, +26.27%] | Baseline |
| **9** | **Seasonal Naive (Weekly)** | Baseline | 6.74 | No | 88.08% [84.37%, 91.79%] | 1.000 [0.985, 1.025] | +12.92% [+11.99%, +13.77%] | Baseline |
| **10** | **Seasonal Naive (28d)** | Baseline | 7.20 | No | 98.92% [95.11%, 102.55%] | 1.084 [1.070, 1.106] | +23.76% [+22.14%, +25.41%] | Baseline |

*Statistical significance: Omnibus Friedman test $\chi^2 = 2429.30$ ($p < 10^{-12}$, 10 models $\times$ 2,950 complete series). Post-hoc Nemenyi critical difference $CD = 0.2494$ ($\alpha = 0.05$). Top rank clique $\{croston\_sba, adida, routed\_ensemble\}$ is statistically indistinguishable in average rank. Paired series bootstrap (2,000 resamples) demonstrates that Routed SBC Ensemble is indistinguishable from ADIDA in pooled WAPE ($\Delta\text{WAPE} = +0.50\%$, 95% CI [-0.01%, +1.04%]; assignment chosen on the evaluation origins; a clean test needs an earlier selection origin). Global model caveat: LightGBM was evaluated under three handicaps compared to M5 competition conditions: (1) no price features (`price_features_included: false` due to Excel row-limit truncation of the raw price workbook to ~15% coverage), (2) a 3,000-series sample rather than the full 30,490 catalog, and (3) a 900-day window rather than the full 1,913-day history.*
<!-- END_LEADERBOARD_TABLE -->

#### Statistical significance and error analysis

<!-- BEGIN_STATISTICS_SECTION -->
- **Non-Parametric Omnibus Test (Friedman)**: Evaluated across 10 models over 2,950 complete series (84 evaluation days per series). The omnibus Friedman test decisively rejects the null hypothesis of equal model performance ($\chi^2 = 2429.30$, $p < 10^{-12}$), justifying post-hoc ranking comparisons.
- **Post-Hoc Critical Difference (Nemenyi)**: Critical difference $CD = 0.2494$ at $\alpha = 0.05$. **Croston SBA** achieved the best average rank (4.55), forming a top statistically indistinguishable clique with **ADIDA**, **Routed SBC Ensemble**. Standard global models (**LightGBM (Tweedie)** at 5.38, **LightGBM** at 5.56) scored significantly worse in average rank across the panel.
- **Paired Series Bootstrap (95% CI)**: Resampling whole series over 2,000 replicates (preserving within-series autocorrelation) reveals that **ADIDA** leads pooled volume-weighted accuracy at 75.16% WAPE [72.16%, 78.23%]. Relative to ADIDA, **Routed SBC Ensemble** is statistically indistinguishable in WAPE ($\Delta\text{WAPE} = +0.50\%$, 95% CI [-0.01%, +1.04%]; routing assignment chosen on the evaluation origins; a clean test needs an earlier selection origin), while **Moving Average (28d)** is indistinguishable in RMSSE ($\Delta\text{RMSSE} = +0.0011$, 95% CI [-0.0001, +0.0023]).
- **Per-Series Pairwise Win Rates**: Comparing models series-by-series resolves the apparent divergence between rank tests and volume metrics: **Croston SBA** beats **ADIDA** on 57.0% of individual series (favoring rank metrics on sparse items), whereas **ADIDA** achieves lower aggregate error on higher-volume items. On 93.8% of series, the two models are statistically indistinguishable under Holm-corrected per-series Diebold-Mariano testing.
<!-- END_STATISTICS_SECTION -->

*Raises the next question*: the routed ensemble sits in the top clique, so does classifying demand help at all?

### 6. Does demand classification help?

The router classifies each series into an SBC (Syntetos-Boylan-Croston, 2005) quadrant from its training window and sends it to a per-segment method. The right yardstick is the best single method applied everywhere, which here is ADIDA, not LightGBM. WAPE is the same pooled definition as the leaderboard.

<!-- BEGIN_ROUTER_COMPARISON -->
| Scope | Router sends to | Series-origins | Router WAPE (%) | ADIDA everywhere (%) | LightGBM everywhere (%) | Best single model (WAPE) | Router - ADIDA (pp) |
|---|---|---|---|---|---|---|---|
| **SMOOTH** | LightGBM | 610 | 45.94 | 47.21 | 45.94 | LightGBM (45.94%) | -1.26 |
| **ERRATIC** | LightGBM | 215 | 71.73 | 70.23 | 71.73 | Moving Average (28d) (69.83%) | +1.50 |
| **LUMPY** | LightGBM | 1,428 | 91.29 | 87.74 | 91.29 | ADIDA (87.74%) | +3.55 |
| **INTERMITTENT** | ADIDA | 6,736 | 88.10 | 88.10 | 92.06 | ADIDA (88.10%) | +0.00 |
| **NO_DEMAND** | zero forecast | 11 | 100.00 | 100.00 | 137.09 | ADIDA (100.00%) | +0.00 |
| **All series, pooled over origins** | mixed | 9,000 | 75.66 | 75.16 | 77.39 | ADIDA (75.16%) | +0.50 |
| All series, origin 2 (test 2016-02-01 to 2016-02-28) | mixed | 3,000 | 75.82 | 75.37 | 77.87 | ADIDA (75.37%) | +0.45 |
| All series, origin 1 (test 2016-02-29 to 2016-03-27) | mixed | 3,000 | 76.72 | 75.95 | 78.81 | ADIDA (75.95%) | +0.77 |
| All series, origin 0 (test 2016-03-28 to 2016-04-24) | mixed | 3,000 | 74.44 | 74.17 | 75.51 | ADIDA (74.17%) | +0.27 |

- **Result**: classification-based routing did not beat applying ADIDA everywhere. Pooled over the origins the router's WAPE is 75.66% against 75.16% for ADIDA (+0.50 pp, positive meaning the router is worse), and it is worse at every origin (+0.45, +0.77, +0.27 pp). The best single model at each origin is ADIDA.
- **Where the difference comes from**: INTERMITTENT series go to ADIDA (the same method as ADIDA everywhere; the router fits it after trimming pre-launch zeros, so a few forecasts differ slightly from the standalone model's), NO_DEMAND series get a zero forecast, and SMOOTH, ERRATIC and LUMPY series go to LightGBM, which here has no price features. On those segments LightGBM beat ADIDA on SMOOTH and lost to it on ERRATIC and LUMPY (LightGBM vs ADIDA WAPE: SMOOTH 45.94% vs 47.21%; ERRATIC 71.73% vs 70.23%; LUMPY 91.29% vs 87.74%). The global model was not better on every segment it was given, which is why routing on demand pattern did not pay.
- **Reading it**: the best single model is the lowest-WAPE single model within each row, chosen after the scores are known, which favours the single model. The INTERMITTENT to ADIDA assignment was chosen on these same evaluation origins, so the router is scored on the data that selected its rule; a clean test needs an earlier selection origin. LightGBM ran with the handicaps listed under Limitation 1.
<!-- END_ROUTER_COMPARISON -->

*Raises the next question*: if the best forecasts are all close, what is a better forecast worth in operational terms?

---

## Inventory Decision Layer (steps 7 and 8)

### 7. What is a better forecast worth?

Evaluating required safety stock capital at a fixed 95% Cycle Service Level (CSL) across California stores `CA_1` and `CA_2` (4,591 priced SKUs) connects each model's rolling-origin forecast error to the stock needed to cover it ($SS = z \cdot \sigma_e \cdot \sqrt{L}$). Here $\sigma_e$ is the **RMSE** of the forecast errors, taken about zero, not their standard deviation: a standard deviation subtracts a model's mean error, so a forecast that is consistently too low would be costed as if it were accurate. The *Mean bias* column is that mean error (actual - forecast; positive means the forecast runs low):

<!-- BEGIN_SAFETY_STOCK_MODEL_TABLE -->
| Model Name | Architecture Tier | Mean RMSE ($\sigma_e$) | Mean bias (actual - forecast) | Safety Stock Units (95% CSL) | Safety Stock Capital ($) | Capital Reduction vs Naive ($) | Reduction (%) |
|---|---|---|---|---|---|---|---|
| **Moving Average (28d)** | Simple Baseline | 1.4891 | +0.0385 | 39,517.03 | $47,896.83 | +$20,639.58 | +30.11% |
| **ADIDA** | Intermittent Statistical | 1.4898 | -0.0022 | 39,534.77 | $47,909.05 | +$20,627.36 | +30.10% |
| **SES** | Exponential Smoothing | 1.5048 | -0.0389 | 39,934.20 | $48,135.60 | +$20,400.81 | +29.77% |
| **Croston SBA** | Intermittent Statistical | 1.5269 | +0.0072 | 40,520.74 | $49,268.17 | +$19,268.24 | +28.11% |
| **Historical Mean (Baseline)** | Simple Baseline | 1.6426 | +0.2949 | 43,589.60 | $52,085.50 | +$16,450.91 | +24.00% |
| **Seasonal Naive (Weekly)** | Baseline | 1.9040 | +0.0326 | 50,526.82 | $61,980.25 | +$6,556.16 | +9.57% |
| **Seasonal Naive (28d)** | Baseline | 2.0800 | +0.3290 | 55,197.38 | $67,325.36 | +$1,211.05 | +1.77% |
| **Naive** | Baseline | 2.1345 | -0.4870 | 56,645.13 | $68,536.41 | $0.00 | 0.00% |
<!-- END_SAFETY_STOCK_MODEL_TABLE -->

- **Reading the table**: Sized on RMSE, **Naive** needs about 2.13 units per day of error cover against about 1.49 for **Moving Average (28d)** and **ADIDA**, so required safety stock capital falls from **$68,536.41** (Naive) to **$47,909.05** (ADIDA), **$20,627.36 (30.10%)** less at the same 95% CSL. The four smoothing methods (Moving Average, ADIDA, SES, Croston SBA) sit within 3% of each other ($47,896.83 to $49,268.17), so the saving comes from **smoothing versus naive, not from ADIDA specifically**. The **Historical Mean** baseline under-forecasts by 0.29 units per SKU per day: a standard deviation hid that and ranked it cheapest, but on RMSE it ranks fifth.
- **Caveat on the 30%**: RMSE charges bias in either direction. Naive over-forecasts (mean error -0.49 per day), and an over-forecast already sits in the reorder point's forecast term, so part of Naive's gap is bias rather than variability. RMSE is the sizing rule used here and credits smoothing with the full 30%; on the bias-free standard deviation alone the gap is about 22% (`mean_std` in `safety_stock_by_model.csv`).

### 8. Which service level should each segment get?

Feeding champion **ADIDA** forecast errors ($\sigma_e$ = RMSE) into 3x3 ABC/XYZ Pareto segmentation across California stores `CA_1` and `CA_2` (6,098 SKUs, 4,591 with verified price data). ADIDA is the inventory champion because it leads the main 3,000-series benchmark; on these 6,098 CA series alone **Moving Average (28d)** has the lower WAPE (76.73% vs 77.23%), and the two differ by under 0.1% in safety stock capital.

**Decisions and scoring use different days.** A planner decides before the test days, so the policy is built only from what is known then:

- **Decision window (ends 2016-01-31)**: ABC revenue comes from the history up to that day, and the XYZ coefficient of variation, the mean demand behind the classes and the no-sales check (class **N**) from its last 84 days. The class decides who gets 98% or 90%, so it is never chosen with knowledge of the test days.
- **Planning demand**: ADIDA's mean *forecast* on the 84 scored days (2016-02-01 to 2016-04-24) sets cycle stock and the reorder point. Actual sales on those days are used only for scoring.
- **$\sigma_e$ is a forecast-error size measured on the backtest days** (the same 84 days). It is the standard way a backtest compares models, not a realised inventory outcome.

#### Two policies, side by side

Capital released only means something next to what it costs. Holding cost uses a 25% annual rate and a unit short costs its lost margin (FOODS 60%, HOUSEHOLD 50%, HOBBIES 45%). Expected shortage per replenishment cycle is $\sigma_L \cdot G(z)$ (the normal loss function), there are $365/L$ cycles a year (each order covers one lead time, as the cycle-stock term assumes), and shortage cost is units short $\times$ cycles $\times$ unit margin. All three policies are summed over the same priced SKUs:

<!-- BEGIN_POLICY_COMPARISON_TABLE -->
| Policy | SKUs | Working Capital ($) | Change in Working Capital ($) | Change in Holding Cost ($/yr) | Change in Stockout Cost ($/yr) | Change in Total Annual Cost ($/yr) |
|---|---|---|---|---|---|---|
| **Uniform current service level** | 4,591 | $81,265 | $0 | $0 | $0 | **$0** |
| **ABC/XYZ capital-release rule** | 4,591 | $78,095 | -$3,170 | -$792 | +$13,237 | **+$12,444** |
| **Cost-minimising level per segment** | 4,591 | $92,720 | +$11,455 | +$2,864 | -$21,547 | **-$18,684** |
<!-- END_POLICY_COMPARISON_TABLE -->

- **Capital-release rule** (AX 98%, BZ/CY/CZ 90%, the rest held at 95%): frees $3,170 of working capital but raises total annual cost by **+$12,444/yr**. Holding cost falls by $792/yr while expected stockout cost rises by $13,237/yr. Under the project's own cost model it is not a saving; it is a capital-for-risk trade, worth taking only if working capital is the binding constraint.
- **Cost-minimising level**: on the 90/95/98% menu it is **98% in every optimised segment**, for **-$18,684/yr** at **+$11,455** of working capital. The unconstrained optimum $SL^* = 1 - hcL/(365m)$ sits above the whole menu in every category (range in the margin table below), because a lost unit of margin costs far more than a week of holding it. Class N segments are held at the current level in both policies.
- **Limits of this result**: the cost model charges only lost margin per unit short (no churn or substitution) and assumes normal errors on zero-heavy demand, so it is an analytical baseline; empirical lead-time stockout simulation is scheduled for Stage B.

#### Does the answer survive a different gross margin?

The cost model rests on assumed gross margins, and a buyer will challenge 60% on groceries. For a SKU with price $p$ and gross margin $g$, unit cost is $c = p(1-g)$ and the margin lost per unit short is $m = pg$, so $SL^* = 1 - hcL/(365m)$ and the cheapest menu level depend on $g$ alone (price and error size cancel). Break-even margins are computed in `src/inventory/margin_sensitivity.py`:

<!-- BEGIN_MARGIN_SENSITIVITY -->
| Gross margin | Scenario | Unconstrained optimum SL* | Cheapest menu level | Extra holding + shortage cost at 90% | Extra holding + shortage cost at 95% | Extra holding + shortage cost at 98% |
|---|---|---|---|---|---|---|
| 5% | grid | 90.9% | 90% | 0% | +4% | +18% |
| 10% | grid | 95.7% | 95% | +12% | 0% | +4% |
| 15% | grid | 97.3% | 98% | +30% | +4% | 0% |
| 20% | grid | 98.1% | 98% | +54% | +12% | 0% |
| 25% | grid | 98.6% | 98% | +78% | +21% | 0% |
| 30% | grid | 98.9% | 98% | +103% | +30% | 0% |
| 45% | HOBBIES assumption | 99.4% | 98% | +183% | +58% | 0% |
| 50% | HOUSEHOLD assumption | 99.5% | 98% | +211% | +67% | 0% |
| 60% | FOODS assumption | 99.7% | 98% | +270% | +88% | 0% |

- **Break-evens** (exact, from the same cost formula the analysis uses): 98% costs less than 95% above a 12.6% gross margin, and 95% costs less than 90% above 6.2%. So the cheapest menu level is 98% above 12.6%, 95% between 6.2% and 12.6%, and 90% below 6.2%. The cost per unit of error size scales with price, so the answer depends on the margin alone.
- **Rule of thumb versus exact**: the unconstrained optimum SL* only reaches 98% above a 19.3% margin and 95% above 8.7%. Those are looser than the break-evens above: SL* can sit below 98% while 98% is still the cheapest of the three menu levels, because SL* is a continuous optimum and the menu is coarse.
- **Realistic grocery margins**: at 25% and 30% gross margin the cheapest level is 98%, and holding 95% instead would cost 21% and 30% more in safety-stock holding plus shortage cost.
- **Range of SL***: by category it is 99.4% (HOBBIES, 45% margin) to 99.7% (FOODS, 60%); the per-segment medians in `segment_cost_optimum.csv` run from 99.5% to 99.7%. All of them sit above the whole 90/95/98% menu.
- **What it means**: because the cheapest level depends on the margin and not on volume or variability, every segment within a category shares one cheapest level. Differentiating service levels by ABC/XYZ cannot lower cost under lost-margin costing while the gross margin is above 12.6%; it can pay only when working capital is the binding constraint, or for items whose margin is below that.
<!-- END_MARGIN_SENSITIVITY -->

Per-segment detail of the capital-release rule, with capital released always shown next to its annual cost change:

<!-- BEGIN_INVENTORY_TABLE -->
| ABC/XYZ Cell | SKU Count | Current Working Capital ($) | Recommended Working Capital ($) | Capital Released ($) | Annual Cost Change ($/yr) | Target Service Level (CSL) | Policy Impact |
|---|---|---|---|---|---|---|---|
| **AX** | 36 | $3,195.44 | $3,475.42 | -$279.98 | -$526/yr | **98%** | High-turnover revenue drivers; upgraded buffer |
| **AY** | 636 | $25,186.16 | $25,186.16 | $0.00 | $0/yr | **95%** | Moderate volume; maintained standard buffer |
| **AZ** | 1,085 | $24,916.55 | $24,916.55 | $0.00 | $0/yr | **95%** | High volatility; guarded safety buffer |
| **BY** | 157 | $2,817.09 | $2,817.09 | $0.00 | $0/yr | **95%** | Moderate secondary volume; maintained |
| **BZ** | 1,113 | $13,749.84 | $11,704.96 | +$2,044.88 | +$7,778/yr | **90%** | Volatile secondary; working capital released |
| **CY** | 32 | $398.01 | $351.12 | +$46.89 | +$197/yr | **90%** | Slow-moving; conservative buffer |
| **CZ** | 1,307 | $8,666.22 | $7,308.40 | +$1,357.82 | +$4,996/yr | **90%** | Erratic low volume; working capital released |
| **No recent demand (AN/BN/CN)** | 222 | $2,291.13 | $2,291.13 | $0.00 | $0/yr | 95% (held) | Zero sales in the 84-day window before the first test day; not stable, no differentiated policy, review (launches, intermittent items, stock-outs) |
| **No ABC class (?X/?Y/?Z/?N)** | 1,510 | $44.09 | $44.09 | $0.00 | $0/yr | — | CA_2 items lacking price coverage, plus SKUs with no priced sales before the cutoff |
| **Total Portfolio** | **6,098** | **$81,264.53** | **$78,094.92** | **+$3,169.61** | **+$12,444/yr** | — | **$3,169.61 Net Working Capital Released, at +$12,444/yr total annual cost** |
<!-- END_INVENTORY_TABLE -->

- **Items with no recent demand (class N)**: a SKU with no sales in the decision window has an undefined CV, so it is not "stable": it gets its own class and row and is held at the current service level. Decided on pre-test days, 322 SKUs are class N, and 291 of them sold during the scored days, so N is a review flag (launches, intermittent items, stock-outs), not a delisting verdict.

#### Why working capital moved from $74,396 to $81,263

<!-- BEGIN_DEMAND_BASIS_TABLE -->
| Mean-demand basis | Used for | Mean demand (units/day) | Cycle-Stock Capital ($) | Safety-Stock Capital ($) | Current Working Capital ($) |
|---|---|---|---|---|---|
| ADIDA mean forecast on the scored days | planning | 5,888.5 | $33,354 | $47,909 | $81,263 |
| Mean actual sales on the scored days (look-ahead) | reference only | 5,894.7 | $33,463 | $47,909 | $81,372 |
| Full-history mean (the earlier figure) | reference only | 5,141.0 | $26,487 | $47,909 | $74,396 |
<!-- END_DEMAND_BASIS_TABLE -->

- **Current working capital is $81,263** on the planning basis (ADIDA's mean forecast). The earlier **$74,396** used the full-history mean, which still includes the 84 test days at about 4% weight; mean actual sales on the scored days (look-ahead, reference only) would give $81,372. The rise is the recent demand level (5,141 to 5,889 units a day across the priced SKUs), which the forecast also sees: ADIDA's portfolio mean forecast is within about 0.1% of the actuals, so planning on the forecast instead of the actuals moves capital by only $109. Safety-stock capital is identical ($47,909) in all three rows.

---

---

## Quickstart & Reproduction

```powershell
# 1. Setup virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Run the test suite
pytest

# 3. Regenerate backtest, significance, and inventory analysis
python scripts/run_backtest.py --sample-series 3000 --n-origins 3
python scripts/run_significance.py --backtest-dir reports/backtest_main
python scripts/run_inventory_analysis.py

# 4. Verify documentation synchronisation
python scripts/render_readme_tables.py --check
```

---

## Technical Documentation Roadmap

The full engineering guide is available in `docs/guide/`:
- [00. Start Here & Orientation](docs/guide/00_START_HERE.md): Executive summary and reading paths.
- [01. Mathematical Concepts](docs/guide/01_CONCEPTS.md): First-principles formulas and worked numerical examples.
- [02. Subsystem Architecture](docs/guide/02_ARCHITECTURE.md): Component diagrams and record lifecycles.
- [03. Data Engineering & Schemas](docs/guide/03_DATA.md): Polars lazy streaming and data anomalies.
- [04. Evaluation Methodology](docs/guide/04_METHODOLOGY.md): Rolling-origin protocol and scaling semantics.
- [05. Code Walkthrough](docs/guide/05_CODE_WALKTHROUGH.md): Function signatures and non-obvious lines.
- [06. Operations Runbook](docs/guide/06_RUNBOOK.md): Windows setup, execution schedule, and troubleshooting.
- [07. Testing Architecture](docs/guide/07_TESTING.md): Suite coverage, fresh-clone skips, and test creation.
- [08. Architecture Decision Records](docs/guide/08_DESIGN_DECISIONS.md): 11 ADR entries.
- [09. Results & Limitations](docs/guide/09_RESULTS_AND_LIMITATIONS.md): Empirical findings, null results, and roadmap.
- [Glossary](docs/guide/GLOSSARY.md): Complete index of acronyms and domain terms.

---

## Limitations

1. **Truncated Pricing History & Global Model Handicaps**: The raw price table originated from a workbook truncated at Excel's 1,048,575-row limit (covering only 15.3% of M5 price records, CA_1 and CA_2 only), leaving 1,507 California series without price data. Consequently, global models (LightGBM) ran under three major handicaps: (a) **no price or promotion features** (`price_features_included: false` in the backtest manifest), (b) trained on a **3,000-series sample** rather than the full 30,490-series catalog, and (c) trained on a **900-day window** rather than the full 1,913-day history. In the M5 competition, price momentum and catalog-wide cross-series learning were decisive drivers of GBDT performance; our finding that specialized intermittent methods match or outperform global GBDTs carries this essential qualification.
2. **Bottom-Level Evaluation**: The benchmark evaluates bottom-level series (Level 12) across California stores rather than full 12-level WRMSSE competition aggregation.
3. **Parametric Normal Safety Stock**: Analytical safety stock ($z \cdot \sigma_e \cdot \sqrt{L}$, with $\sigma_e$ the RMSE so that bias is paid for) and the loss-function shortage cost ($\sigma_L \cdot G(z)$ per cycle) serve as an operational baseline; $\sigma_e$ is a forecast-error size measured on the backtest days, not a realised inventory outcome. The $\sqrt{L}$ scaling treats daily errors as independent, so a persistent bias, which accumulates linearly over the lead time, is still understated; RMSE also charges over-forecast bias. Empirical lead-time stockout simulations are scheduled for Stage B.
4. **Unevaluated Architectures**: Deep learning architectures and two-stage hurdle models were implemented but not scored in the main cross-tier walk-forward backtest.

---

## License

This project is licensed under the MIT License.
