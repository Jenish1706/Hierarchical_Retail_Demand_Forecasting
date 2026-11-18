# 00: Start Here

Welcome to the **M5 Retail Demand & Inventory Intelligence Platform** technical guide.

---

## 1. What This Project Is

This project is an empirical forecasting and supply chain operations platform built on the **Walmart M5 Competition dataset** (30,490 distinct retail items across 10 stores in California, Texas, and Wisconsin, spanning 1,913 days).

Unlike Kaggle notebooks that optimize arbitrary leaderboard loss functions, this platform connects data engineering, statistical forecasting, and operations research into a single reproducible pipeline. It evaluates classical baselines, specialized intermittent-demand estimators, and gradient-boosted trees under a rolling-origin walk-forward backtest, subjects the predictions to non-parametric hypothesis tests, and translates the resulting forecast distributions into differentiated inventory replenishment policies.

---

## 2. The Core Questions It Answers

1. **The Intermittent Demand Reality Check**: In high-sparsity retail panels (where 68.2% of daily catalog records are zero), does a complex global gradient boosting model (LightGBM) outperform specialized statistical estimators designed for intermittent demand (ADIDA, Croston SBA) or even simple rolling averages?
2. **Rank vs. Volume Divergence**: Why do equal-weighted non-parametric rank tests (Friedman / Nemenyi) identify a different leader than volume-weighted aggregate percentage loss (WAPE), and how does series-by-series head-to-head analysis reconcile them?
3. **Operational Translation**: How does forecast error uncertainty directly dictate safety stock, working capital requirements, and planned stockout risks across an empirical ABC/XYZ retail catalog?

---

## 3. Honest Headline Results & Backing Artifacts

Every metric reported in this guide is generated directly from committed artifacts in the repository:

| Metric / Question | Verified Empirical Result | Backing Committed Artifact |
| :--- | :--- | :--- |
| **Lowest Pooled WAPE** | **ADIDA** leads at **75.16%** (95% bootstrap CI: [72.16%, 78.23%]) | [`reports/backtest_main/accuracy_overall.csv`](../../reports/backtest_main/accuracy_overall.csv) |
| **Lowest Mean RMSSE** | **ADIDA** leads at **0.768** (95% bootstrap CI: [0.756, 0.788]) | [`reports/backtest_main/accuracy_overall.csv`](../../reports/backtest_main/accuracy_overall.csv) |
| **Best Nemenyi Average Rank** | **Croston SBA** leads with average rank **4.55** | [`reports/backtest_main/significance/significance_report.json`](../../reports/backtest_main/significance/significance_report.json) |
| **Top Rank Clique** | `croston_sba`, `adida`, and `routed_ensemble` form a single indistinguishable clique ($CD = 0.2494, \alpha=0.05$) | [`reports/backtest_main/significance/critical_difference.png`](../../reports/backtest_main/significance/critical_difference.png) |
| **WAPE Equivalence** | **Routed SBT Ensemble** is statistically indistinguishable from ADIDA ($\Delta\text{WAPE} = +0.50\%$, 95% CI [-0.01%, +1.04%]) | [`reports/backtest_main/significance/bootstrap_intervals.csv`](../../reports/backtest_main/significance/bootstrap_intervals.csv) |
| **RMSSE Equivalence** | **Moving Average (28d)** is statistically indistinguishable from ADIDA on RMSSE ($\Delta\text{RMSSE} = +0.0011$, 95% CI [-0.0001, +0.0023]) | [`reports/backtest_main/significance/bootstrap_intervals.csv`](../../reports/backtest_main/significance/bootstrap_intervals.csv) |
| **Global GBDT Comparison** | **LightGBM (Tweedie)** scores 77.26% WAPE (rank 5.38), significantly worse in average rank across the panel | [`reports/backtest_main/accuracy_overall.csv`](../../reports/backtest_main/accuracy_overall.csv) |
| **Inventory Capital Impact** | Differentiating CSL (98% for AX, 90% for CZ) across stores CA_1 & CA_2 releases **+$3,223.14 in working capital** ($72,426.36 -> $69,203.22) | [`reports/inventory/portfolio_summary.csv`](../../reports/inventory/portfolio_summary.csv) |

---

## 4. Reading Order for This Guide

This documentation is written for an engineer, technical reviewer, or interviewer who wants to understand and falsify every component of the system:

1. [**00_START_HERE.md**](00_START_HERE.md) *(this file)*: Orientation, headline results, and 10-minute quickstart.
2. [**01_CONCEPTS.md**](01_CONCEPTS.md): Core mathematical formulations explained from scratch with intuition and small numerical examples.
3. [**02_ARCHITECTURE.md**](02_ARCHITECTURE.md): Component diagrams, responsibility matrices, and the complete life of a data record.
4. [**03_DATA.md**](03_DATA.md): M5 dataset schema, memory-efficient Polars transformations, data quality safeguards, and price truncation limitations.
5. [**04_METHODOLOGY.md**](04_METHODOLOGY.md): Rolling-origin walk-forward backtesting, metric scaling semantics, paired bootstrap protocol, and hypothesis testing gates.
6. [**05_CODE_WALKTHROUGH.md**](05_CODE_WALKTHROUGH.md): Tour of critical pipeline scripts, model implementations, and routing logic.
7. [**06_RUNBOOK.md**](06_RUNBOOK.md): Exact step-by-step reproduction commands on Windows PowerShell.
8. [**07_TESTING.md**](07_TESTING.md): Test suite architecture, regression guarantees, and fresh clone validation.
9. [**08_DESIGN_DECISIONS.md**](08_DESIGN_DECISIONS.md): Architectural Decision Records (ADRs) explaining key design trade-offs and alternatives rejected.
10. [**09_RESULTS_AND_LIMITATIONS.md**](09_RESULTS_AND_LIMITATIONS.md): Deep-dive into empirical results, threats to validity, and scope boundaries.
11. [**GLOSSARY.md**](GLOSSARY.md): Alphabetical reference of terms, acronyms, and operational metrics.

---

## 5. 10-Minute Quickstart

You can verify the entire project from a clean clone without downloading raw M5 data:

```powershell
# 1. Clone repository
git clone -b audit/stage-a https://github.com/your-username/m5-retail-intelligence.git
cd "m5-retail-intelligence"

# 2. Create virtual environment & install pinned dependencies
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .

# 3. Run full test suite (108 unit tests pass in ~20s; integration tests skip gracefully)
pytest -q

# 4. Verify that public README tables match committed artifacts exactly
python scripts/render_readme_tables.py --check

# 5. Launch interactive Streamlit dashboard
streamlit run streamlit_app.py
```
