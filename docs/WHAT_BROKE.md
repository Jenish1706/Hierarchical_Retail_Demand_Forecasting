# Engineering Defect & Rectification Log

This document records technical defects identified during platform audits and the mathematical/architectural fixes implemented.

---

### [2026-11-16] Observation-Pooled vs. Canonical Per-Series RMSSE Discrepancy (P0-3)
- **Symptom**: Evaluated models produced inconsistent RMSSE scores across pipelines (e.g. ADIDA and AutoETS scoring ~0.842 in Tier 1 vs ~0.988 in cross-tier backtest) despite identical volume-weighted WAPE (75.317%).
- **Root Cause**: Divergence in aggregation order. An earlier backtest script pooled all squared errors and scales across all observations before taking the square root ($\sqrt{\text{mean}(\text{error}^2 / \text{scale})}$), which disproportionately penalized sparse series with volatile spikes. The official M5 competition protocol evaluates $\text{RMSSE}_i = \sqrt{\text{MSE}_i / s_i}$ per series first, then averages across the portfolio.
- **Fix**: Standardized `metrics_by_group()` in `src/evaluation/point_metrics.py` and `scripts/run_tier1.py` to enforce canonical per-series averaging. Rescored backtest artifacts (`reports/backtest_main/`). Added known-answer unit test `test_rmsse_per_series_averaging_matches_m5_rule`.

---

### [2026-11-16] Pre-Launch Zero Padding in Demand Classification and Scaling (P0-4)
- **Symptom**: Unadjusted catalog profiling reported 95.2% of series as sparse (75.6% intermittent, 19.5% lumpy) and only 3.2% smooth, while in-sample naive scale baselines were artificially diluted.
- **Root Cause**: Slicing the full 1,913-day panel without accounting for catalog introduction dates. In the M5 dataset, 77% of products launched after Day 1 (median first-sale day 160). Treating pre-introduction periods as zero demand artificially inflated the Average Inter-Demand Interval ($ADI = n / n_{\text{nonzero}}$) and padded naive squared differences with $0 - 0 = 0$ steps.
- **Fix**: Added `trim_pre_launch=True` to `classify_demand()` (`src/models/statistical/intermittent.py`) and `naive_scale()` (`src/evaluation/point_metrics.py`), slicing series from their first non-zero sale. Re-running the full catalog EDA revealed that the smooth series share is actually 6.2% (1,895 series), with both trimmed and unadjusted baselines documented. Added regression unit tests `test_classify_demand_trims_pre_launch_zeros` and `test_naive_scale_trims_pre_launch_zeros`.

---

### [2026-11-16] Stockout Cost Disconnection from Safety Stock Held (P1-6)
- **Symptom**: In marginal service-level analysis, zero-variance item `FOODS_1_004` showed a stockout cost reduction of $-\$3.21$ between 90% and 95% service levels despite held safety stock remaining unchanged at $0.0$.
- **Root Cause**: Stockout cost in `src/inventory/service_level.py` was calculated as $(1.0 - sl) \times \mu_D \times L \times \text{margin}$, which evaluated the nominal policy label rather than actual safety stock held.
- **Fix**: Reformulated effective stockout probability using the normal CDF of the held buffer: $\Phi(SS / \sigma_L)$. When forecast error variance is zero ($\sigma_e \le 10^{-4}$), stockout probability is strictly 0.0. Added regression test `test_marginal_stockout_cost_delta_zero_when_safety_stock_unchanged`.

---

### [2026-11-16] Dimensional Incoherence in Stockout vs. Holding Cost (P1-6)
- **Symptom**: Replenishment cycle stockout cost was labeled `stockout_cost_annual` and directly added to annual holding cost ($25\% \times \text{inventory value}$).
- **Root Cause**: A 7-day replenishment cycle occurs $365 / L \approx 52.14$ times per year. Comparing a single 7-day stockout margin loss to an annual holding cost created an apples-to-oranges comparison that understated annual stockout risk by $\approx 52\times$.
- **Fix**: Annualized stockout margin loss by multiplying cycle cost by $365.0 / lt$ ($\text{stockout\_prob} \times \text{annual\_demand} \times \text{unit\_margin}$).

---

### [2026-11-16] Documentation Drift in Public README Tables (P0-1, P0-2)
- **Symptom**: The public README leaderboard and inventory tables contained manually typed numbers that did not match artifact outputs.
- **Root Cause**: Lack of automated continuous rendering between committed artifacts and Markdown documentation.
- **Fix**: Implemented `scripts/render_readme_tables.py` to parse `accuracy_overall.csv`, `significance_report.json`, and `portfolio_summary.csv` and inject Markdown tables between HTML delimiter comments (`<!-- BEGIN_LEADERBOARD_TABLE -->`, `<!-- BEGIN_INVENTORY_TABLE -->`). Added continuous integration test `tests/test_readme_tables.py` which fails on any drift.
