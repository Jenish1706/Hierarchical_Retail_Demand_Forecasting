# Testing Architecture & Verification Strategy

This document details the automated testing suite: what each test module covers, what is deliberately excluded from fast tests, how fresh-clone skips are architected, and how to write new tests.

---

## 1. Test Suite Coverage by Module

The repository maintains an automated test suite (`pytest`) that runs on CPU in about a minute. The suite is organized into unit, integration, and system-level checks. Test counts are not quoted here because they change with every item; the count and the fresh-clone result of the latest gate are recorded in the final report, and `pytest` prints the live figure.

### Unit Tests (`tests/unit/`)

| Test File | Modules Tested | Key Assertions & Properties Verified |
| :--- | :--- | :--- |
| `test_m5_01_scale_semantics.py` | `src.evaluation.point_metrics` | Multi-origin scale pooling; anchor scaling from first historical sale $d_1$; pre-launch zero trimming; zero-division edge cases ($s_i = 1.0$). |
| `test_m5_03_tier1_parity.py` | `src.evaluation.point_metrics`, `reports/tier1/` | Cross-pipeline metric consistency; AutoETS Level-12 RMSSE parity harmonized at $0.7675$ between `tier1` and `backtest_main`. |
| `test_m5_04_source_of_truth.py` | `src.dashboard.data_loader`, `reports/` | Lossless parity across all companion CSV-Parquet pairs; verifies `read_table()` treats CSV as canonical source of truth. |
| `test_m5_06_statistical_defense.py` | `src.evaluation.significance` | Paired series bootstrap (2,000 resamples); preservation of within-series temporal autocorrelation; per-series win rates; Holm-Bonferroni correction across series within each model pair. |
| `test_m5_09_zero_demand_classes.py` | `src.inventory.abc_xyz`, `src.inventory.safety_stock` | No-sales items are class N, never X; classes are decided on the window before the first test day; degenerate error estimates are missing, not zero risk; zero-demand and constant-demand series. |
| `test_m5_10_cost_model.py` | `src.inventory.service_level`, `src.inventory.working_capital` | Normal loss function against hand values; shortage cost independent of the demand level; planning demand is the forecast; capital released never shown without its annual cost; README and guide numbers tied to the CSVs. |
| `test_m5_11_global_model_handicaps.py` | README, guides | The three global-model handicaps and the price-feature caveat are stated. |
| `test_m5_12_inventory_backtest_link.py` | `scripts/run_inventory_analysis.py` | The inventory layer reads the benchmarked backtest predictions; safety stock by model. |
| `test_m5_12b_safety_stock_bias.py` | `src.inventory.safety_stock` | Safety stock sized on RMSE: a constant under-forecast raises the required stock. |
| `test_m5_13_router_vs_best_single.py` | `src.evaluation.routing_comparison` | Router scored against every single model per origin and segment; committed comparison recomputed from `predictions.parquet`; assignments agree with the routed forecasts; no tracked text file uses the retired classification name. |
| `test_m5_15_margin_sensitivity.py` | `src.inventory.margin_sensitivity` | Closed-form break-even margins against the pipeline's own analyzer; committed margin CSVs. |
| `test_m5_15_readme_chain.py` | `README.md`, `docs/guide/00_START_HERE.md` | The eight-step order, working anchors, hand-over between steps, no typed numbers in the chain prose. |
| `test_m5_15_guide_numbers.py` | `docs/guide/` | Headline results in guides 00, 04 and 09 derived from the significance report, bootstrap and accuracy CSVs. |
| `test_g4b_dashboard_truth.py` | `src.dashboard.views`, `streamlit_app.py` | Views rendered against a stub Streamlit show artifact numbers; no typed result numbers in dashboard strings. |
| `test_g4b_manifest_provenance.py` | `src.evaluation.backtest` | The manifest records the predictions' SHA-256 and regeneration command, and a rerun keeps writing them. |
| `test_metrics.py` | `src.evaluation.point_metrics` | Analytical correctness of MAE, RMSE, WAPE, RMSSE, MASE, and quantile pinball losses across synthetic vectors with known ground truths. |
| `test_models_and_facts.py` | `src.models.statistical.intermittent`, `src.models.statistical.naive` | Invariants of Croston SBA, TSB, ADIDA, Moving Average, and SES; demand quadrant classifications ($ADI$ and $CV^2$ thresholds). |
| `test_polars_parity.py` | `src.features.polars_engine` | Bitwise/numerical parity between Polars expressions and equivalent Pandas implementations for lags, rolling means, and price ratios. |
| `test_probabilistic_models.py` | `src.models.probabilistic` | Two-stage hurdle model fitting; empirical coverage calibration of conformalized quantile regression intervals ($1 - \alpha$). |
| `test_deep_learning.py` | `src.models.deep_learning` | PyTorch Lightning and NeuralForecast adapter interfaces; tensor shape assertions; forward pass validation on minimal synthetic batches. |
| `test_ensembles_and_mlops.py` | `src.mlops.registry`, `src.mlops.drift` | `PromotionGate` threshold validation (requires $\ge 1\%$ improvement); Kolmogorov-Smirnov drift alarms; Population Stability Index (PSI) thresholds. |

### Integration & System Tests

| Test File | Focus | Assertions Verified |
| :--- | :--- | :--- |
| `tests/integration/test_loader_features_model.py` | Data $\to$ Features $\to$ Model | End-to-end flow: loading sample sales, generating Polars features, fitting LightGBM, and asserting forecast output shapes. |
| `tests/test_dashboard.py` | Dashboard Integrity | Verifies all Streamlit KPI cards load dynamically from artifact files; fails if hardcoded metric strings or numbers are detected. |
| `tests/test_inventory.py` | Inventory Engine | ABC/XYZ classification boundary conditions; safety stock calculations; annualized stockout margin loss calculations. |
| `tests/test_readme_tables.py` | Documentation Synchronization | Parses Markdown tables in `README.md` and validates exact cell-by-cell match against backing CSV/JSON report files. |
| `tests/test_end_to_end.py` | Full Pipeline | Execution of complete analytical pipeline from raw input to output reports. |

---

## 2. Deliberate Omissions (What the Suite Does Not Cover)

To keep the test suite fast in continuous integration, the following workloads are deliberately omitted from automated unit tests:

1. **Full 30,490-Series Multi-Hour Backtests**:
   - Running 10 models across 30,490 series over 3 rolling origins requires ~12 hours of compute. The test suite evaluates algorithms on small synthetic fixtures (10–50 series) or pre-sampled panels.
2. **Heavy GPU Training of Deep Learning Models**:
   - Deep learning architectures (PatchTST, TFT, DLinear) require significant GPU resources for multi-epoch convergence. Tests verify model initialization, input/output tensor shapes, and loss calculations on tiny synthetic batches on CPU.
3. **Live External Database or Tracking Servers**:
   - The test suite does not require a running PostgreSQL or remote MLflow server. All MLOps and registry tests use in-process mocks or local temporary SQLite databases.
4. **Interactive Browser GUI Rendering**:
   - Streamlit UI rendering is not tested via headless browser automation (e.g. Selenium/Playwright). Instead, dashboard logic is tested via data loader validation and static AST code inspection for hardcoded literals.

---

## 3. Fresh-Clone Isolation & Skip Semantics

### The Raw Data Challenge
The raw M5 dataset (`sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`) totals ~229 MB uncompressed and is excluded from git tracking to keep repository clone sizes small and fast.

### Skip Architecture
Tests that strictly require raw data files are decorated with conditional skips:
```python
import pytest
from pathlib import Path

DATA_AVAILABLE = Path("data/raw/sales_train_validation.csv").exists()

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Raw M5 data not found in data/raw/")
def test_full_pipeline_ingestion():
    ...
```

### Fresh-Clone Guarantees
In a fresh clone without raw data:
- Every unit and system test passes immediately.
- Tests that need the raw M5 data or a downloaded model are cleanly skipped with a stated reason (`DATA_AVAILABLE`, `importorskip`).
- No test fails.
- All report artifacts (`reports/`), benchmark results, significance outputs, and dashboard views are fully tested because all derived tables are tracked in the repository.

---

## 4. How to Add a New Test

When introducing new models, features, or metrics, follow these guidelines:

### Standard Unit Test Pattern
Create a new test file in `tests/unit/test_<feature>.py`:

```python
import numpy as np
import pytest
from src.evaluation.point_metrics import wape

def test_wape_calculation_known_values():
    # Arrange: synthetic actuals and predictions
    y_true = np.array([10.0, 20.0, 0.0, 5.0])
    y_pred = np.array([12.0, 18.0, 2.0, 5.0])
    
    # Act: compute metric
    score = wape(y_true, y_pred)
    
    # Assert: hand-calculated expected WAPE (a percentage) = 100 * (2 + 2 + 2 + 0) / (10 + 20 + 0 + 5) = 17.1428...
    expected = 100.0 * 6.0 / 35.0
    assert pytest.approx(score, rel=1e-5) == expected

def test_wape_zero_actuals_boundary():
    # Assert zero actuals returns nan or handled gracefully
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    score = wape(y_true, y_pred)
    assert np.isnan(score) or score == 1.0
```

### Best Practices Checklist
1. **Determinism**: Always set random seeds (`np.random.seed(42)` or `random_state=42`) for tests involving stochastic methods or bootstrapping.
2. **Speed**: Unit tests must complete in $< 1.0$ second per test file.
3. **Floating Point Assertions**: Never assert exact equality `a == b` for floating point calculations; use `pytest.approx(expected, rel=1e-4)` or `np.testing.assert_allclose()`.
4. **Hermeticism**: Tests must not leave residual files on disk. Use pytest's built-in `tmp_path` fixture for any file write tests.
