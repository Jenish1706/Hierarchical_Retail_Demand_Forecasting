# 00: Start Here

Welcome to the **M5 Retail Demand & Inventory Intelligence Platform** technical guide.

---

## 1. What This Project Is

This project is an empirical forecasting and supply chain operations platform built on the **Walmart M5 Competition dataset** (30,490 store-item series: 3,049 items in each of 10 stores in California, Texas, and Wisconsin, spanning 1,913 days).

Unlike Kaggle notebooks that optimize arbitrary leaderboard loss functions, this platform connects data engineering, statistical forecasting, and operations research into a single reproducible pipeline. It evaluates classical baselines, specialized intermittent-demand estimators, and gradient-boosted trees under a rolling-origin walk-forward backtest, subjects the predictions to non-parametric hypothesis tests, and translates the resulting forecast distributions into differentiated inventory replenishment policies.

---

## 2. The Eight-Step Chain

The project answers one chain of eight questions, each raised by the answer to the one before. The [README](../../README.md) has the same chain with every number rendered from the committed artifacts; this table says where each step lives in this guide.

| Step | Question | Answer in one line | Read |
| :--- | :--- | :--- | :--- |
| 1 | What does the demand look like? | Mostly zeros, and many series launch late | [03_DATA.md](03_DATA.md), [01_CONCEPTS.md](01_CONCEPTS.md) |
| 2 | Which metric is fair for it? | WAPE for volume, RMSSE per (series, origin) with pre-launch zeros trimmed only before the first sale | [04_METHODOLOGY.md](04_METHODOLOGY.md) |
| 3 | How do we evaluate without leakage? | Rolling origins, horizon-shifted features, training-window-only statistics | [04_METHODOLOGY.md](04_METHODOLOGY.md) |
| 4 | Which models, and what does each rung test? | A ladder from naive, through intermittent-demand methods, to a global GBDT with stated handicaps | [04_METHODOLOGY.md](04_METHODOLOGY.md), [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) |
| 5 | Are the differences real? | Friedman, Nemenyi and a paired series bootstrap separate real gaps from noise | [04_METHODOLOGY.md](04_METHODOLOGY.md) |
| 6 | Does demand classification help? | Not against ADIDA applied to every series | [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) (Q6) |
| 7 | What is a better forecast worth? | Safety-stock capital by forecast model | [01_CONCEPTS.md](01_CONCEPTS.md), [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) |
| 8 | Which service level should each segment get? | The highest level on the menu is the cheapest for every segment; differentiating by ABC/XYZ trades capital for risk | [02_ARCHITECTURE.md](02_ARCHITECTURE.md), [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) (Q10) |

---

## 3. Honest Headline Results & Backing Artifacts

Every metric reported in this guide is generated directly from committed artifacts in the repository:

| Metric / Question | Verified Empirical Result | Backing Committed Artifact |
| :--- | :--- | :--- |
| **Lowest Pooled WAPE** | **ADIDA** leads at **75.16%** (95% bootstrap CI: [72.16%, 78.23%]) | [`reports/backtest_main/accuracy_overall.csv`](../../reports/backtest_main/accuracy_overall.csv) |
| **Lowest Mean RMSSE** | **ADIDA** leads at **0.768** (95% bootstrap CI: [0.756, 0.788]) | [`reports/backtest_main/accuracy_overall.csv`](../../reports/backtest_main/accuracy_overall.csv) |
| **Best Nemenyi Average Rank** | **Croston SBA** leads with average rank **4.55** | [`reports/backtest_main/significance/significance_report.json`](../../reports/backtest_main/significance/significance_report.json) |
| **Top Rank Clique** | `croston_sba`, `adida`, and `routed_ensemble` form a single indistinguishable clique ($CD = 0.2494, \alpha=0.05$) | [`reports/backtest_main/significance/critical_difference.png`](../../reports/backtest_main/significance/critical_difference.png) |
| **WAPE Equivalence** | **Routed SBC Ensemble** is statistically indistinguishable from ADIDA ($\Delta\text{WAPE} = +0.50\%$, 95% CI [-0.01%, +1.04%]) | [`reports/backtest_main/significance/bootstrap_intervals.csv`](../../reports/backtest_main/significance/bootstrap_intervals.csv) |
| **RMSSE Equivalence** | **Moving Average (28d)** is statistically indistinguishable from ADIDA on RMSSE ($\Delta\text{RMSSE} = +0.0011$, 95% CI [-0.0001, +0.0023]) | [`reports/backtest_main/significance/bootstrap_intervals.csv`](../../reports/backtest_main/significance/bootstrap_intervals.csv) |
| **Router vs Best Single Model** | The router beats standalone LightGBM at every origin but does not beat ADIDA applied to every series (worse at every origin); per-segment and per-origin numbers are in the README router table | [`reports/backtest_main/routing_vs_best_single.csv`](../../reports/backtest_main/routing_vs_best_single.csv) |
| **Global GBDT Comparison** | **LightGBM (Tweedie)** scores 77.26% WAPE (rank 5.38), significantly worse in average rank across the panel | [`reports/backtest_main/accuracy_overall.csv`](../../reports/backtest_main/accuracy_overall.csv) |
| **Inventory Capital Impact** | The ABC/XYZ capital-release rule (98% for AX, 90% for BZ/CY/CZ) across CA_1 & CA_2 frees **$3,170 of working capital** but raises total annual cost by **+$12,444/yr** (holding -$792, stockout +$13,237); the cheapest level on the 90/95/98% menu (98% everywhere) lowers cost by $18,684/yr for +$11,455 of capital | [`reports/inventory/policy_comparison.csv`](../../reports/inventory/policy_comparison.csv) |

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
