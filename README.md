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

Evaluating required safety stock capital at a fixed 95% Cycle Service Level (CSL) across California stores `CA_1` and `CA_2` (4,591 priced SKUs) directly connects empirical forecast accuracy ($\sigma_e$) across rolling origins to balance sheet capital requirements ($SS = z \cdot \sigma_e \cdot \sqrt{L}$):

<!-- BEGIN_SAFETY_STOCK_MODEL_TABLE -->
| Model Name | Architecture Tier | Mean $\sigma_e$ | Safety Stock Units (95% CSL) | Safety Stock Capital ($) | Capital Reduction vs Naive ($) | Reduction (%) |
|---|---|---|---|---|---|---|
| **Historical Mean (Baseline)** | Simple Baseline | 1.4336 | 38,044.01 | $46,390.24 | +$14,504.95 | +23.82% |
| **Croston SBA** | Intermittent Statistical | 1.4690 | 38,982.90 | $47,332.95 | +$13,562.24 | +22.27% |
| **ADIDA** | Intermittent Statistical | 1.4708 | 39,030.58 | $47,441.79 | +$13,453.40 | +22.09% |
| **Moving Average (28d)** | Simple Baseline | 1.4724 | 39,074.33 | $47,535.17 | +$13,360.02 | +21.94% |
| **SES** | Exponential Smoothing | 1.4798 | 39,270.54 | $47,592.80 | +$13,302.39 | +21.84% |
| **Naive** | Baseline | 1.8872 | 50,081.65 | $60,895.19 | $0.00 | 0.00% |
| **Seasonal Naive (Weekly)** | Baseline | 1.8853 | 50,030.91 | $61,460.50 | -$565.31 | -0.93% |
| **Seasonal Naive (28d)** | Baseline | 1.8930 | 50,235.18 | $63,274.63 | -$2,379.44 | -3.91% |
<!-- END_SAFETY_STOCK_MODEL_TABLE -->

- **Working Capital Impact**: Specialized intermittent models (**ADIDA**, **Croston SBA**) reduce required safety stock capital from **$60,895.19** under the Naive baseline to **$47,441.79** (ADIDA) and **$47,332.95** (Croston SBA) — releasing **$13,453.40 (22.09%)** in working capital purely through lower forecast error variance, holding service level constant at 95% CSL.

### Differentiated Service Levels & ABC/XYZ Optimization

Feeding champion **ADIDA** forecast errors ($\sigma_e$) into 3x3 ABC/XYZ Pareto segmentation across California stores `CA_1` and `CA_2` (6,098 SKUs, 4,591 with verified price data):

<!-- BEGIN_INVENTORY_TABLE -->
| ABC/XYZ Cell | SKU Count | Current Working Capital ($) | Recommended Working Capital ($) | Capital Released ($) | Target Service Level (CSL) | Policy Impact |
|---|---|---|---|---|---|---|
| **AX** | 51 | $3,295.91 | $3,611.34 | -$315.43 | **98%** | High-turnover revenue drivers; upgraded buffer |
| **AY** | 714 | $22,933.61 | $22,933.61 | $0.00 | **95%** | Moderate volume; maintained standard buffer |
| **AZ** | 1,086 | $24,917.08 | $24,917.08 | $0.00 | **95%** | High volatility; guarded safety buffer |
| **BX** | 9 | $36.15 | $36.15 | $0.00 | **95%** | Stable secondary SKUs; maintained |
| **BY** | 129 | $1,470.96 | $1,470.96 | $0.00 | **95%** | Moderate secondary volume; maintained |
| **BZ** | 1,212 | $13,685.73 | $11,466.21 | +$2,219.52 | **90%** | Volatile secondary; working capital released |
| **CX** | 45 | $53.66 | $52.09 | +$1.57 | **90%** | Stable low volume; bulk periodic order |
| **CY** | 30 | $197.10 | $161.89 | +$35.21 | **90%** | Slow-moving; conservative buffer |
| **CZ** | 1,315 | $7,340.09 | $6,007.13 | +$1,332.96 | **90%** | Erratic low volume; working capital released |
| **Unpriced (?X/?Y/?Z)** | 1,507 | $0.00 | $0.00 | $0.00 | — | CA_2 items lacking price coverage |
| **Total Portfolio** | **6,098** | **$73,930.29** | **$70,656.46** | **+$3,273.83** | — | **$3,273.83 Net Working Capital Released** |
<!-- END_INVENTORY_TABLE -->

- **Operational Trade-Off**: Releasing **+$3,273.83 in net working capital** requires an explicit trade-off: accepting an increase in theoretical stockout probability (CSL 95% $\to$ 90%) on slow-moving tail items (BZ, CX, CY, CZ) to fund higher safety stock (CSL 95% $\to$ 98%) on revenue-driving AX items. Note that under the project's current total relevant cost model, stockout penalty trade-offs are evaluated in detail (see Limitations).

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
3. **Parametric Normal Safety Stock**: Analytical safety stock ($z \cdot \sigma_e \cdot \sqrt{L}$) serves as an operational baseline; empirical lead-time stockout simulations are scheduled for Stage B.
4. **Unevaluated Architectures**: Deep learning architectures and two-stage hurdle models were implemented but not scored in the main cross-tier walk-forward backtest.

---

## License

This project is licensed under the MIT License.
