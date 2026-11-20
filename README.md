# M5 Retail Demand Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DVC Pipeline](https://img.shields.io/badge/pipeline-DVC%20v3-orange.svg)](https://dvc.org)
[![Data Engine: Polars](https://img.shields.io/badge/engine-Polars%20lazy%20streaming-blueviolet.svg)](https://pola.rs)
[![Documentation](https://img.shields.io/badge/docs-comprehensive%20guide-blue.svg)](docs/guide/00_START_HERE.md)

An empirical demand forecasting and inventory intelligence benchmark built on **58.3 million daily retail observations** from the Walmart M5 dataset (30,490 SKUs across 10 stores over 1,913 days).

The platform bridges the gap between raw statistical forecasts and inventory operations, evaluating where specialized intermittent demand methods match or beat complex gradient boosted machines on zero-inflated counts.

> **Deep Documentation Available**: For first-principles math, worked numerical examples, architecture diagrams, runbooks, and design decisions, see the [Comprehensive Guide](docs/guide/00_START_HERE.md).

---

## The Problem & Methodological Approach

Retail demand at store-item level is characterized by extreme sparsity: **68.2% of daily records across the catalog are zero**, and 77% of SKUs launched after Day 1 (median first sale on day 160). When evaluated without strict leakage controls or lifecycle zero trimming, standard benchmarks distort both model accuracy and inventory buffers.

### Core Architecture & Safeguards
- **Data Engineering**: Memory-efficient Polars streaming engine applying `narrow-before-melt` to eliminate memory spikes across 58 million rows.
- **Leakage Prevention**: All rolling and lag features are shifted by the full 28-day horizon (`FeaturePipeline(horizon=28)`), eliminating future lookahead bias.
- **Canonical Scaling**: RMSSE scaling is anchored strictly to the first non-zero sale ($d_{\text{first}}$) and evaluated per (series, origin) to prevent scale dilution.
- **Statistical Gating**: Rankings are gated by omnibus Friedman testing, Nemenyi critical difference ranking, and paired series bootstrap confidence intervals (2,000 resamples).

---

## Walk-Forward Backtest Leaderboard (3,000 Series $\times$ 3 Origins $\times$ 10 Models)

Evaluated across 3,000 sampled series over 3 rolling 28-day origins (84 test days per series = 252,000 predictions per model) under canonical M5 per-series evaluation rules.

<!-- BEGIN_LEADERBOARD_TABLE -->
| Rank | Model Name | Architecture Tier | Nemenyi Avg Rank | Top Clique? | WAPE (%) [95% CI] | RMSSE [95% CI] | $\Delta$WAPE vs Best [95% CI] | Statistical Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **Croston SBA** | Intermittent Statistical | **4.55** | **Yes** | 77.80% [74.63%, 81.09%] | 0.789 [0.777, 0.810] | +2.64% [+2.06%, +3.30%] | **Rank Leader** |
| **2** | **ADIDA** | Intermittent Statistical | 4.77 | **Yes** | **75.16% [72.16%, 78.23%]** | **0.768 [0.756, 0.788]** | **+0.00% [+0.00%, +0.00%]** | **WAPE Leader** (within CD of rank leader) |
| **3** | **Routed SBT Ensemble** | Hybrid (ADIDA + Global) | 4.79 | **Yes** | 75.66% [72.40%, 78.87%] | 0.769 [0.757, 0.789] | +0.50% [-0.01%, +1.04%] | **Indistinguishable from Best** (rank & WAPE) |
| **4** | **SES** | Exponential Smoothing | 4.91 | No | 76.99% [74.20%, 79.72%] | 0.772 [0.760, 0.792] | +1.83% [+1.13%, +2.62%] | Significantly worse than best rank |
| **5** | **Moving Average (28d)** | Simple Baseline | 4.94 | No | 75.47% [72.39%, 78.54%] | 0.769 [0.757, 0.789] | +0.31% [+0.05%, +0.59%] | Significantly worse than best rank |
| **6** | **LightGBM (Tweedie)** | Global GBDT | 5.38 | No | 77.26% [74.10%, 80.34%] | 0.783 [0.771, 0.803] | +2.10% [+1.40%, +2.84%] | Significantly worse than best rank |
| **7** | **LightGBM** | Global GBDT | 5.56 | No | 77.39% [74.11%, 80.59%] | 0.786 [0.774, 0.805] | +2.23% [+1.58%, +2.90%] | Significantly worse than best rank |
| **8** | **Naive** | Baseline | 6.16 | No | 99.65% [96.06%, 103.28%] | 1.002 [0.984, 1.027] | +24.49% [+22.62%, +26.27%] | Baseline |
| **9** | **Seasonal Naive (Weekly)** | Baseline | 6.74 | No | 88.08% [84.37%, 91.79%] | 1.000 [0.985, 1.025] | +12.92% [+11.99%, +13.77%] | Baseline |
| **10** | **Seasonal Naive (28d)** | Baseline | 7.20 | No | 98.92% [95.11%, 102.55%] | 1.084 [1.070, 1.106] | +23.76% [+22.14%, +25.41%] | Baseline |

*Statistical significance: Omnibus Friedman test $\chi^2 = 2429.30$ ($p < 10^{-12}$, 10 models $\times$ 2,950 complete series). Post-hoc Nemenyi critical difference $CD = 0.2494$ ($\alpha = 0.05$). Top rank clique $\{croston\_sba, adida, routed\_ensemble\}$ is statistically indistinguishable in average rank. Paired series bootstrap (2,000 resamples) demonstrates that Routed SBT Ensemble is indistinguishable from ADIDA in pooled WAPE ($\Delta\text{WAPE} = +0.50\%$, 95% CI [-0.01%, +1.04%]; assignment chosen on the evaluation origins; a clean test needs an earlier selection origin). Global model caveat: LightGBM was evaluated under three handicaps compared to M5 competition conditions: (1) no price features (`price_features_included: false` due to Excel row-limit truncation of the raw price workbook to ~15% coverage), (2) a 3,000-series sample rather than the full 30,490 catalog, and (3) a 900-day window rather than the full 1,913-day history.*
<!-- END_LEADERBOARD_TABLE -->

### Statistical Significance & Error Analysis
<!-- BEGIN_STATISTICS_SECTION -->
- **Non-Parametric Omnibus Test (Friedman)**: Evaluated across 10 models over 2,950 complete series (84 evaluation days per series). The omnibus Friedman test decisively rejects the null hypothesis of equal model performance ($\chi^2 = 2429.30$, $p < 10^{-12}$), justifying post-hoc ranking comparisons.
- **Post-Hoc Critical Difference (Nemenyi)**: Critical difference $CD = 0.2494$ at $\alpha = 0.05$. **Croston SBA** achieved the best average rank (4.55), forming a top statistically indistinguishable clique with **ADIDA**, **Routed SBT Ensemble**. Standard global models (**LightGBM (Tweedie)** at 5.38, **LightGBM** at 5.56) scored significantly worse in average rank across the panel.
- **Paired Series Bootstrap (95% CI)**: Resampling whole series over 2,000 replicates (preserving within-series autocorrelation) reveals that **ADIDA** leads pooled volume-weighted accuracy at 75.16% WAPE [72.16%, 78.23%]. Relative to ADIDA, **Routed SBT Ensemble** is statistically indistinguishable in WAPE ($\Delta\text{WAPE} = +0.50\%$, 95% CI [-0.01%, +1.04%]; routing assignment chosen on the evaluation origins; a clean test needs an earlier selection origin), while **Moving Average (28d)** is indistinguishable in RMSSE ($\Delta\text{RMSSE} = +0.0011$, 95% CI [-0.0001, +0.0023]).
- **Per-Series Pairwise Win Rates**: Comparing models series-by-series resolves the apparent divergence between rank tests and volume metrics: **Croston SBA** beats **ADIDA** on 57.0% of individual series (favoring rank metrics on sparse items), whereas **ADIDA** achieves lower aggregate error on higher-volume items. On 93.8% of series, the two models are statistically indistinguishable under Holm-corrected per-series Diebold-Mariano testing.
<!-- END_STATISTICS_SECTION -->

---

## Inventory Decision Layer (Operations Analytics)

### The Dollar Value of a Better Forecast (Model-Driven Working Capital)

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

### Differentiated Service Levels & ABC/XYZ Optimization

Feeding champion **ADIDA** forecast errors ($\sigma_e$ = RMSE) into 3x3 ABC/XYZ Pareto segmentation across California stores `CA_1` and `CA_2` (6,098 SKUs, 4,591 with verified price data). ADIDA is the inventory champion because it leads the main 3,000-series benchmark; on these 6,098 CA series alone **Moving Average (28d)** has the lower WAPE (76.73% vs 77.23%), and the two differ by under 0.1% in safety stock capital:

<!-- BEGIN_INVENTORY_TABLE -->
| ABC/XYZ Cell | SKU Count | Current Working Capital ($) | Recommended Working Capital ($) | Capital Released ($) | Target Service Level (CSL) | Policy Impact |
|---|---|---|---|---|---|---|
| **AX** | 51 | $3,298.47 | $3,614.45 | -$315.98 | **98%** | High-turnover revenue drivers; upgraded buffer |
| **AY** | 714 | $23,037.90 | $23,037.90 | $0.00 | **95%** | Moderate volume; maintained standard buffer |
| **AZ** | 1,086 | $25,116.21 | $25,116.21 | $0.00 | **95%** | High volatility; guarded safety buffer |
| **BX** | 9 | $39.40 | $39.40 | $0.00 | **95%** | Stable secondary SKUs; maintained |
| **BY** | 129 | $1,477.01 | $1,477.01 | $0.00 | **95%** | Moderate secondary volume; maintained |
| **BZ** | 1,212 | $13,766.33 | $11,530.12 | +$2,236.21 | **90%** | Volatile secondary; working capital released |
| **CX** | 45 | $61.56 | $58.21 | +$3.35 | **90%** | Stable low volume; bulk periodic order |
| **CY** | 30 | $198.50 | $162.98 | +$35.52 | **90%** | Slow-moving; conservative buffer |
| **CZ** | 1,315 | $7,402.19 | $6,056.09 | +$1,346.10 | **90%** | Erratic low volume; working capital released |
| **Unpriced (?X/?Y/?Z)** | 1,507 | $0.00 | $0.00 | $0.00 | — | CA_2 items lacking price coverage |
| **Total Portfolio** | **6,098** | **$74,397.57** | **$71,092.37** | **+$3,305.20** | — | **$3,305.20 Net Working Capital Released** |
<!-- END_INVENTORY_TABLE -->

- **Operational Trade-Off**: Releasing **+$3,305.20 in net working capital** requires an explicit trade-off: accepting an increase in theoretical stockout probability (CSL 95% $\to$ 90%) on slow-moving tail items (BZ, CX, CY, CZ) to fund higher safety stock (CSL 95% $\to$ 98%) on revenue-driving AX items. Note that under the project's current total relevant cost model, stockout penalty trade-offs are evaluated in detail (see Limitations).

---

## Quickstart & Reproduction

```powershell
# 1. Setup virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Run the test suite (110 tests, ~25s)
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
3. **Parametric Normal Safety Stock**: Analytical safety stock ($z \cdot \sigma_e \cdot \sqrt{L}$, with $\sigma_e$ the RMSE so that bias is paid for) serves as an operational baseline. The $\sqrt{L}$ scaling treats daily errors as independent, so a persistent bias, which accumulates linearly over the lead time, is still understated; RMSE also charges over-forecast bias. Empirical lead-time stockout simulations are scheduled for Stage B.
4. **Unevaluated Architectures**: Deep learning architectures and two-stage hurdle models were implemented but not scored in the main cross-tier walk-forward backtest.

---

## License

This project is licensed under the MIT License.
