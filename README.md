# M5 Forecasting Model Laboratory

A comparative study of statistical, probabilistic, machine learning, and deep
learning forecasting models on the M5 retail dataset, with reproducible data
engineering, MLOps discipline, and rigorous results analysis.

**This is a research project, not a product.** There is no web application, no
API, no dashboard. The deliverables are notebooks, reproducible pipelines,
tracked experiments, figures, and written analysis. That is the correct shape
for a project whose purpose is demonstrating modelling breadth and analytical
rigour.

---

## The five pillars

| # | Pillar | Status |
|---|---|---|
| 1 | Exploratory Data Analysis | Complete — 6 figures, 8 tables, all 30,490 series |
| 2 | Data Engineering | Complete — loader, validator, 48-feature pipeline |
| 3 | Model Zoo | Partial — Tiers 1, 2, 5 built; 3 and 4 outstanding |
| 4 | MLOps | Partial — CI green, tracking/registry not yet extracted |
| 5 | Results Analysis | Not started — 0 of 20 figures |

---

## Headline findings so far

### The dataset is far sparser than the literature's rule of thumb

Measured over all 30,490 series × 1,913 days (`reports/eda/`):

| Demand pattern (Syntetos-Boylan-Teunter) | % of series | % of volume |
|---|---|---|
| INTERMITTENT | 75.6% | 42.6% |
| LUMPY | 19.5% | 27.7% |
| SMOOTH | 3.2% | 19.9% |
| ERRATIC | 1.6% | 9.9% |

**95.2% of series are sparse**, and 68.2% of all observations are zero. This is
the property that drives every other result here: intermittent-demand methods
are not a niche branch of this study, they are the main event.

### Classical methods from 1972 are competitive with gradient boosting

On 3,000 series across 3 rolling origins, with Holm-corrected Diebold-Mariano:

- ADIDA significantly beats LightGBM (DM = −7.02, p < 0.0001)
- ADIDA significantly beats LightGBM-Tweedie (DM = −5.81, p < 0.0001)
- Every model beats the naive family (RMSSE > 1.2 for naive)

### The Tweedie null result

`lightgbm` vs `lightgbm_tweedie`: Holm-adjusted p = 1.0000. Tweedie is the
standard recommendation for zero-inflated retail counts, and at 95.2% sparsity
it showed **no detectable difference** from a plain L2 objective.

Stated precisely: this is *no evidence of a difference*, not *evidence of no
difference*. Holm correction can push an adjusted p to exactly 1.0; the raw DM
statistic is reported alongside it in `reports/*/significance/`.

### Pooled and rank-based metrics disagree on the winner

| Test | Weighting | Leader |
|---|---|---|
| Diebold-Mariano (pooled per-observation loss) | volume-weighted | ADIDA |
| MCB / Nemenyi (per-series ranks) | equal-weighted | routed_ensemble |

Both are correct — they answer different questions. Pooled loss is dominated by
the ~5% of series carrying most volume; per-series ranks are dominated by the
95.2% that are sparse.

This is not an artefact of our setup. Makridakis, Spiliotis & Assimakopoulos
(2022), *IJF* 38:1346–1364 found the same divergence in the competition itself:
the winner ranked 1st on WRMSSE but 13th on MCB, while another entry ranked 8th
on WRMSSE and 1st on MCB. Their reading is that winning teams optimised for
expensive fast-moving products, at the cost of everything else.

Two caveats travel with every rank-based number here:

1. **Nemenyi CD = 0.2494** at N = 2,950 — nearly any rank gap registers as
   significant at that sample size.
2. **The independence assumption is violated.** MCB assumes the compared
   forecasts are independent; these series sit in a nested hierarchy. The M5
   paper flags the same limitation for its own conclusions.

---

## Dataset

Supplied as a single 4-sheet `.xlsx`. Sheet names differ from the Kaggle CSV
filenames, and one contains a typo that is preserved in the loader because it
is what the file actually says:

| Sheet (exact) | Contents |
|---|---|
| `Sales_train_validation` | Wide format, `d_1` … `d_1913` |
| `Sales_prices` | `store_id`, `item_id`, `wm_yr_wk`, `sell_price` |
| `calendar` | dates, events, SNAP flags (`snap_CA`, `snap_TX`, `snap_WI`) |
| `Sales_subission` | `id`, `F1`…`F28` — *typo is in the source file* |

3,049 items × 10 stores = **30,490 series** over **1,913 days**. Sales are
integer unit counts, not revenue. Prices are weekly.

### Verified corrections to the commonly-quoted facts

These were checked against the workbook and differ from the project brief. Where
they differ, the data won:

| Claim | Reality |
|---|---|
| 1,913 days ending 2016-06-19 | Ends **2016-04-24**. 2016-06-19 is calendar row 1,969 — the calendar runs 56 days past the sales data so the forecast window has known events. |
| Test split 2016-04-25 → 2016-05-22 | That is `d_1914`–`d_1941`. **No actuals exist** for it in this release. Splits shifted back accordingly. |
| 161 named events | **30 distinct event names** across **162 event days**. |

### A limitation of this copy of the data

`Sales_prices` was truncated at Excel's 1,048,576-row ceiling — 15.3% of the
true table. Because the cut is positional rather than a sample, price data
exists for **CA_1 and CA_2 only**; the other eight stores have none.

Sales, calendar, SNAP and event data are complete and verified clean against
every M5 cardinality, so forecasting, hierarchical reconciliation and event
analysis are unaffected. Price-derived features are dropped whenever a run
includes an uncovered store, because otherwise a model learns "price is null"
as a store-identity proxy rather than as a price signal.

### Not in M5 — never fabricated here

Unit cost / COGS · inventory levels · lead times · promotion flags · supply
chain network · transport data · warehouse metadata · customer data · store
coordinates · competitor prices.

---

## Quick start

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

```bash
python scripts/extract_workbook.py
```

```bash
python scripts/run_eda.py
```

```bash
python scripts/run_tier1.py --sample-series 3000 --n-origins 3
```

```bash
python scripts/run_significance.py --backtest-dir reports/tier1
```

MLflow is optional — every script falls back to a local SQLite tracking store
when no server is running:

```bash
docker compose up -d mlflow
```

---

## Repository layout

```
src/
├── config/          M5 facts as a frozen dataclass; logging
├── data/            loader (58.3M-row melt), Pandera validator
├── features/        base pipeline (48 features), advanced (target enc, Fourier)
├── models/
│   ├── statistical/     naive, intermittent (Croston/SBA/TSB/ADIDA), statsforecast
│   ├── classical_ml/    nested-CV trainer, ensembles, promotion gate
│   ├── probabilistic/   quantile forecasting
│   ├── deep_learning/   (Tier 4, outstanding)
│   ├── hierarchical/    MinT / bottom-up / top-down reconciliation
│   └── router.py        SBT segment routing
├── evaluation/      point + probabilistic metrics, splits, backtest, significance
├── visualization/   (Pillar 5, outstanding)
└── mlops/           drift detection

scripts/             extract_workbook, run_eda, run_tier1, run_backtest, run_significance
tests/               54 unit tests (known-answer, not smoke)
reports/eda/         Pillar 1 figures and tables
reports/tier1/       Tier 1 accuracy tables
docs/                model card
```

---

## Methodological commitments

These are enforced in code, not just stated:

- **Chronological splits only.** `assert_no_temporal_leakage` fails the run if
  any training row is dated at or after the first test row.
- **Horizon-safe features.** Every backward-looking feature is shifted by the
  full forecast horizon, so a test row never sees an actual from inside the test
  window. `FeaturePipeline` carries a runtime leakage assertion that perturbs
  targets and confirms feature values do not move.
- **Out-of-fold only** for stacking and blending.
- **No ranking is reported as a finding without a significance test** behind it.
- **Coverage and sharpness are always reported together** for probabilistic
  models — coverage alone is trivially gamed by widening the interval.
- **Every model goes through the same harness.** No bespoke scoring paths.

---

## References

Makridakis, Spiliotis & Assimakopoulos (2022). "The M5 accuracy competition:
Results, findings and conclusions." *International Journal of Forecasting*
38(4):1346–1364.

Syntetos, Boylan & Croston (2005). "On the categorization of demand patterns."
*Journal of the Operational Research Society* 56(5):495–503.

Cawley & Talbot (2010). "On over-fitting in model selection and subsequent
selection bias in performance evaluation." *JMLR* 11:2079–2107.

Demšar (2006). "Statistical comparisons of classifiers over multiple data sets."
*JMLR* 7:1–30.

Chernozhukov, Fernández-Val & Galichon (2010). "Quantile and probability curves
without crossing." *Econometrica* 78(3):1093–1125.

Hyndman, Ahmed, Athanasopoulos & Shang (2011). "Optimal combination forecasts
for hierarchical time series." *Computational Statistics & Data Analysis*
55(9):2579–2589.
