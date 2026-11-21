# Code Walkthrough and Repository Tour

This document provides a comprehensive, file-by-file walkthrough of the codebase. It details what each component does, its inputs and outputs, key function signatures, execution patterns, and subtle lines that readers frequently trip on.

---

## 1. Data Ingestion & Validation

### `src/data/loader.py`
- **Purpose**: Loads the raw M5 tabular datasets (`calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv`) using Polars with automatic fallback to Pandas.
- **Inputs**: File system paths to CSV/Parquet files in `data/raw/`.
- **Outputs**: Polars `LazyFrame` / `DataFrame` or Pandas `DataFrame` containing parsed sales, calendar encodings, and price histories.
- **Key Classes & Methods**:
  ```python
  class M5Loader:
      def __init__(self, data_dir: Path | str = "data"): ...
      def load_panel(self, n_series: int | None = None) -> pl.DataFrame: ...
      def load_calendar(self) -> pl.DataFrame: ...
      def load_prices(self) -> pl.DataFrame: ...
  ```
- **How It Is Called**:
  ```python
  from src.data.loader import M5Loader
  loader = M5Loader(data_dir="data")
  sales_df = loader.load_panel(n_series=3000)
  ```
- **Non-Obvious Lines / Gotchas**:
  - `load_panel()` loads wide format and melts to long format. Reshaping to long format must be filtered *before* melting (`narrow-before-melt`), otherwise melting all 30,490 series across 1,913 days creates a 58.3-million row table that consumes >5 GB of RAM.

### `src/data/validator.py`
- **Purpose**: Enforces strict structural, schema, and statistical domain constraints on incoming data using Pandera schemas and custom validation classes.
- **Inputs**: Raw data frames from `loader.py`.
- **Outputs**: `ValidationResult` dataclass with boolean flag `is_valid`, error logs, and checked row counts.
- **Key Classes & Functions**:
  ```python
  class RawSalesSchema(pa.DataFrameModel): ...
  class FeatureSchema(pa.DataFrameModel): ...
  class SalesDataValidator:
      def validate(self, df: pd.DataFrame) -> ValidationResult: ...
  class PriceDataValidator:
      def validate(self, df: pd.DataFrame) -> ValidationResult: ...
  ```
- **Non-Obvious Lines / Gotchas**:
  - `PriceDataValidator` checks that `sell_price > 0.0`. However, in the M5 dataset, 1,507 series in California lack pricing rows during initial backtest origins because Walmart only recorded prices for active weeks. Pandera validation must handle nullable prices without dropping active sales records.

### `scripts/extract_workbook.py`
- **Purpose**: Command-line utility to unpack compressed Kaggle zip archives (`m5-forecasting-accuracy.zip`) and standardize raw file layouts.
- **Execution**: `python scripts/extract_workbook.py --zip-path <path_to_zip> --target data/raw`

---

## 2. Feature Engineering & Expressions

### `src/features/polars_engine.py`
- **Purpose**: High-performance, streaming feature engineering engine built entirely on Polars expression trees (`pl.Expr`).
- **Inputs**: Long-format sales panel (`id`, `date`, `sales`, `sell_price`, calendar event columns).
- **Outputs**: Enriched feature matrix with horizon-safe lags, rolling statistics, calendar indicators, and intermittency metrics.
- **Key Functions**:
  ```python
  def lag_expressions(lags: list[int], target_col: str = "sales") -> list[pl.Expr]: ...
  def rolling_expressions(windows: list[int], target_col: str = "sales") -> list[pl.Expr]: ...
  def price_expressions() -> list[pl.Expr]: ...
  def intermittency_expressions() -> list[pl.Expr]: ...
  def build_series_features(df: pl.DataFrame, horizon: int = 28) -> pl.DataFrame: ...
  def build_series_features_lazy(lf: pl.LazyFrame, horizon: int = 28) -> pl.LazyFrame: ...
  ```
- **How It Is Called**:
  ```python
  import polars as pl
  from src.features.polars_engine import build_series_features_lazy
  features_lf = build_series_features_lazy(sales_long_lf, horizon=28)
  df = features_lf.collect()
  ```
- **Non-Obvious Lines / Gotchas**:
  - `horizon=28` enforcement: To prevent temporal data leakage, all backward-looking rolling statistics are shifted by `horizon`. For example, rolling 7-day mean uses `.shift(horizon).rolling_mean(7)`. Using `.shift(1)` would leak actual test-set sales into the multi-step test window.

### `src/features/base.py`
- **Purpose**: Abstract base pipeline and Scikit-learn-compatible wrapper for feature extraction and scaling.
- **Key Classes**:
  ```python
  class FeaturePipeline:
      def __init__(self, horizon: int = 28, drop_na: bool = True): ...
      def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline": ...
      def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
  ```

---

## 3. Forecasting Models Zoo

### `src/models/statistical/intermittent.py`
- **Purpose**: State-of-the-art specialized intermittent demand algorithms designed for sparse retail series.
- **Inputs**: 1D historical sales series `y` ($y_1, y_2, \dots, y_T$) and forecast horizon $h$.
- **Outputs**: Point forecasts of length $h$.
- **Key Classes & Functions**:
  ```python
  def classify_demand(series: np.ndarray, trim_pre_launch: bool = True) -> DemandClassification: ...
  class CrostonForecaster:
      def __init__(self, variant: str = "sba", alpha: float = 0.1): ...
      def fit(self, y: np.ndarray) -> "CrostonForecaster": ...
      def predict(self, h: int = 28) -> np.ndarray: ...
  class ADIDAForecaster:
      def __init__(self, aggregation_level: int | None = None, loss: str = "mse"): ...
      def fit(self, y: np.ndarray) -> "ADIDAForecaster": ...
      def predict(self, h: int = 28) -> np.ndarray: ...
  class ZeroForecaster: ...
  ```
- **Non-Obvious Lines / Gotchas**:
  - In `classify_demand(series, trim_pre_launch=True)`, pre-launch leading zeros are trimmed before computing $ADI = n / k$ and $CV^2$. Without trimming, a product launched on day 1,000 has 1,000 leading zeros that falsely inflate its ADI, misclassifying smooth or erratic products as intermittent.
  - In `CrostonForecaster(variant="sba")`, Syntetos-Boylan applies a multiplicative deflator $(1 - \alpha / 2)$ to the classical Croston ratio $\hat{z} / \hat{p}$. Classical Croston is strictly positively biased ($E[\hat{y}] > E[y]$) because $E[1/p] > 1/E[p]$ by Jensen's inequality.

### `src/models/statistical/naive.py`
- **Purpose**: Fast baseline models serving as null hypotheses for backtesting benchmarks.
- **Key Classes**:
  ```python
  class NaiveForecaster: ...                  # Repeats y_T
  class SeasonalNaiveForecaster: ...          # Repeats y_{T - 28 + (i % 28)}
  class SeasonalNaiveWeeklyForecaster: ...    # Repeats y_{T - 7 + (i % 7)}
  class MovingAverageForecaster: ...          # Mean of last W observations (W=28)
  class SimpleExponentialSmoothing: ...       # Holt-Winters SES with optimal alpha
  def build_baseline(name: str) -> BaselineForecaster: ...
  ```

### `src/models/classical_ml/trainer.py`
- **Purpose**: Global machine learning forecasters (LightGBM standard MSE and LightGBM Tweedie) with nested cross-validation and Optuna hyperparameter optimization.
- **Inputs**: Tabular matrix of direct features and target labels.
- **Outputs**: Fitted booster models, out-of-fold predictions, and feature importance rankings.
- **Key Classes & Structs**:
  ```python
  class TrainingConfig(BaseModel): ...
  class DemandTrainer:
      def __init__(self, config: TrainingConfig): ...
      def train(self, df_train: pd.DataFrame, features: list[str]) -> ModelResult: ...
      def predict(self, df_test: pd.DataFrame) -> np.ndarray: ...
  class StackEnsemble: ...
  class BlendEnsemble: ...
  ```
- **Non-Obvious Lines / Gotchas**:
  - `_lgbm_tweedie()` sets `tweedie_variance_power=1.5`. The compound Poisson-Gamma distribution assumes variance scales as $\mu^p$ with $1 < p < 2$. Setting $p=1.5$ models zero inflation at the origin while handling continuous right-skewed positive sales.

### `src/models/router.py`
- **Purpose**: Demand-pattern segment router. Classifies series into SBC (Syntetos-Boylan-Croston) quadrants (intermittent, lumpy, smooth, erratic, plus no-demand) from the training window and dispatches each series to its assigned estimator: INTERMITTENT to ADIDA, no-demand to zeros, every other quadrant to the supplied global forecast.
- **Key Classes**:
  ```python
  class SegmentRouter:
      def __init__(self, routing: Mapping[DemandPattern, str] | None = None, croston_alpha: float = 0.1, adida_alpha: float = 0.2): ...
      def fit(self, history: Mapping[str, np.ndarray], first_sale_indices: Mapping[str, int] | None = None) -> "SegmentRouter": ...
      def predict(self, h: int, global_forecasts: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]: ...
      def routing_table(self) -> pd.DataFrame: ...
      def evaluate_routing_benefit(self, actuals: Mapping[str, np.ndarray], global_forecasts: Mapping[str, np.ndarray]) -> RoutingBenefit: ...
  ```
- **What `evaluate_routing_benefit` answers**: routed versus the global model only. The question that matters, routed versus the best *single* model, is answered by `compare_router(preds, assignments)` in `src/evaluation/routing_comparison.py`, which scores the routed ensemble against every single model per origin and per segment from `predictions.parquet` and `routing_assignments.csv` (`routing_vs_best_single.csv`, and the per-origin columns appended to `routing_benefit.csv`).

### `src/models/hierarchical/reconciler.py`
- **Purpose**: Hierarchical reconciliation across aggregation levels (Level 1 total down to Level 12 item-store).
- **Key Classes**:
  ```python
  class M5HierarchyBuilder:
      def build(self, df: pd.DataFrame) -> HierarchyNode: ...
  class HierarchicalForecaster:
      def fit_and_reconcile(self, base_forecasts: np.ndarray, method: str = "bottom_up") -> np.ndarray: ...
      def predict(self, h: int = 28) -> np.ndarray: ...
  ```

### `src/models/probabilistic/hurdle.py` & `src/models/probabilistic/quantile_forecaster.py`
- **Purpose**: Two-stage zero-inflated hurdle models and conformalized quantile forecasters.
- **Key Classes**:
  ```python
  class HurdleForecaster:
      def fit(self, X: np.ndarray, y: np.ndarray) -> "HurdleForecaster": ...
      def predict(self, X: np.ndarray) -> np.ndarray: ...
      def predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
  class QuantileForecaster:
      def predict_quantiles(self, X: np.ndarray, quantiles: list[float]) -> dict[float, np.ndarray]: ...
  class ConformalPredictor: ...
  ```

---

## 4. Evaluation & Statistical Testing

### `src/evaluation/point_metrics.py`
- **Purpose**: Canonical M5 evaluation metrics implemented exactly according to competition specifications.
- **Key Functions**:
  ```python
  def naive_scale(y_train: np.ndarray, trim_pre_launch: bool = True) -> float: ...
  def rmsse(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float: ...
  def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
  def metrics_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame: ...
  ```
- **Non-Obvious Lines / Gotchas**:
  - In `naive_scale(y_train, trim_pre_launch=True)`:
    $$\text{scale} = \frac{1}{T - d_{\text{start}}} \sum_{t=d_{\text{start}}+1}^{T} (y_t - y_{t-1})^2$$
    If $y_{\text{train}}$ contains only zeros or never changes value in-sample, the naive one-step MSE is 0. Rather than substituting an arbitrary $1.0$ (which would silently invent an artificial scale), the function returns `NaN` and honestly excludes the series from RMSSE aggregation.
  - In `rmsse()`, evaluation is strictly per-(series, origin) before aggregation. Pooling squared errors across multiple series into an overall root-mean-square artificially distorts high-variance items and breaks canonical M5 competition parity.

### `src/evaluation/splits.py`
- **Purpose**: Leakage-free walk-forward rolling-origin split generator.
- **Key Functions**:
  ```python
  def rolling_origins(df: pd.DataFrame, origins: list[str], horizon: int = 28) -> list[Split]: ...
  def assert_no_temporal_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None: ...
  ```

### `src/evaluation/backtest.py`
- **Purpose**: High-throughput walk-forward backtest harness. Orchestrates rolling-origin evaluations across 3,000 series and 10 models.
- **Outputs**: `reports/backtest_main/predictions.parquet`, `reports/backtest_main/accuracy_overall.csv`, `reports/backtest_main/accuracy_by_origin.csv`, `reports/backtest_main/routing_benefit.csv`, `reports/backtest_main/routing_assignments.csv`, `reports/backtest_main/routing_vs_best_single.csv`, and `reports/backtest_main/manifest.json`.
- **Key Functions**:
  ```python
  def build_origins(cutoff_dates: list[str], horizon: int = 28) -> list[Origin]: ...
  def run_series_models(series_data: dict, models: list[str]) -> dict: ...
  def main(): ...
  ```

### `src/evaluation/routing_comparison.py`
- **Purpose**: Scores the routed ensemble against every single model, per origin and per demand segment, on the same predictions as the leaderboard (pooled WAPE).
- **Key Functions**:
  ```python
  def compare_router(preds: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame: ...
  def per_origin_columns(comparison: pd.DataFrame) -> pd.DataFrame: ...
  ```
- **Non-obvious**: the best single model in each row is picked after the scores are known (an oracle choice that favours the single model), and the router's INTERMITTENT to ADIDA rule was chosen on these same origins.

### `src/evaluation/significance.py`
- **Purpose**: Comprehensive econometric hypothesis testing engine.
- **Key Functions & Signatures**:
  ```python
  def friedman_test(matrix: np.ndarray) -> FriedmanResult: ...
  def nemenyi_test(matrix: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float]: ...
  def paired_series_bootstrap(
      df_preds: pd.DataFrame,
      models: list[str],
      n_resamples: int = 2000,
      random_state: int = 42
  ) -> pd.DataFrame: ...
  def pairwise_per_series_tests(
      df_preds: pd.DataFrame,
      models: list[str]
  ) -> pd.DataFrame: ...
  def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 28) -> DMResult: ...
  ```
- **Non-Obvious Lines / Gotchas**:
  - `paired_series_bootstrap()` resamples whole series (with replacement), preserving within-series temporal autocorrelation across all 84 test steps.
  - `pairwise_per_series_tests()` computes Holm-Bonferroni adjusted p-values across all 45 model pairs to maintain Family-Wise Error Rate (FWER) $\le 0.05$.

---

## 5. Inventory Optimization Subsystem

### `src/inventory/abc_xyz.py`
- **Purpose**: 2D ABC/XYZ demand classification for retail assortments.
- **Key Classes**:
  ```python
  class ClassificationThresholds(BaseModel):
      cum_revenue_a: float = 0.80
      cum_revenue_b: float = 0.95
      cov_x: float = 0.50
      cov_y: float = 1.00
  class ABCXYZClassifier:
      def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
  ```

### `src/inventory/safety_stock.py` & `src/inventory/service_level.py`
- **Purpose**: Calculates safety stock requirements, reorder points, and optimized cycle service levels (CSL) trading off holding cost against stockout margin loss.
- **Key Classes**:
  ```python
  class SafetyStockCalculator:
      def calculate(self, lead_time_days: int = 7, service_level: float = 0.95) -> pd.DataFrame: ...
  class ServiceLevelAnalyzer:
      def analyze(self, df: pd.DataFrame) -> pd.DataFrame: ...
      def marginal_analysis(self, df: pd.DataFrame) -> pd.DataFrame: ...
  ```
- **Non-Obvious Lines / Gotchas**:
  - In `ServiceLevelAnalyzer`, replenishment occurs every $L=7$ days ($365/7 \approx 52.14$ cycles/year). Stockout margin loss must be multiplied by $365 / L$ to achieve dimensional coherence when summed with annual inventory holding cost ($25\% \times \text{inventory value}$).

### `src/inventory/margin_sensitivity.py`
- **Purpose**: How far the service-level conclusion moves with the assumed gross margin. Per dollar of price and per unit of sigma_L the annual cost is $h(1-g)z + (365/L)\,g\,G(z)$, so the cheapest menu level depends on the gross margin $g$ alone.
- **Key Functions**:
  ```python
  def break_even_margin(lower: float, higher: float, holding_cost_rate: float = ..., lead_time_days: float = ...) -> float: ...
  def sl_star_equals(service_level: float, holding_cost_rate: float = ..., lead_time_days: float = ...) -> float: ...
  def margin_sensitivity(category_margins=None, grid=..., menu=..., holding_cost_rate=..., lead_time_days=...) -> pd.DataFrame: ...
  def break_even_table(menu=..., holding_cost_rate=..., lead_time_days=...) -> pd.DataFrame: ...
  ```
- **Non-obvious**: `break_even_margin` is where two *menu* levels cost the same; `sl_star_equals` is where the *unconstrained* optimum reaches a level. The second is looser (it needs a higher margin before 98% counts as optimal); both are reported so neither is mistaken for the other.

### `src/inventory/working_capital.py`
- **Purpose**: Simulates portfolio working capital releases under tailored inventory policies.
- **Key Classes**:
  ```python
  class WorkingCapitalAnalyzer:
      def portfolio_summary(self, df: pd.DataFrame) -> pd.DataFrame: ...
      def recommend_policy(self, df: pd.DataFrame) -> pd.DataFrame: ...
  ```

---

## 6. MLOps, Registry & Dashboard

### `src/mlops/registry.py` & `src/mlops/tracking.py`
- **Purpose**: Gates automated model promotions into the MLflow model registry and tracks pipeline artifacts.
- **Key Classes**:
  ```python
  class PromotionGate:
      def evaluate(self, candidate: PromotionCandidate, incumbent: PromotionCandidate) -> PromotionDecision: ...
  class ExperimentTracker:
      def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
  ```

### `src/mlops/drift.py`
- **Purpose**: Continuous drift monitoring detecting covariate shifts (Kolmogorov-Smirnov test) and demand distribution changes (Population Stability Index).
- **Key Classes**:
  ```python
  class StatisticalDriftDetector:
      def ks_test(self, reference: np.ndarray, current: np.ndarray) -> DriftReport: ...
      def compute_psi(self, reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float: ...
  class DriftMonitoringPipeline: ...
  ```

### `src/dashboard/data_loader.py` & Views
- **Purpose**: Streamlit dashboard loader and presentation views.
- **Key Methods**:
  ```python
  def read_table(csv_path: Path, parquet_path: Path | None = None) -> pd.DataFrame: ...
  ```
- **Views**:
  - `src/dashboard/views/overview.py`: Executive KPI summary cards.
  - `src/dashboard/views/models.py`: 10-model walk-forward leaderboard and error breakdown.
  - `src/dashboard/views/inventory.py`: ABC/XYZ matrix and working capital simulator.
  - `src/dashboard/views/significance.py`: Critical difference diagram and bootstrap CIs.
  - `src/dashboard/views/eda.py`: Catalog sparsity and launch date distributions.

---

## 7. Pipeline Drivers & Execution Scripts

| Script | Purpose | Primary Output |
| :--- | :--- | :--- |
| `scripts/run_eda.py` | Full catalog exploration and zero profiling | `reports/eda/` |
| `scripts/run_tier1.py` | Evaluates baseline models on 3,000 series | `reports/tier1/` |
| `scripts/run_backtest.py` | Full walk-forward backtest (3k series $\times$ 3 origins $\times$ 10 models) | `reports/backtest_main/` |
| `scripts/run_significance.py` | Friedman, Nemenyi, DM, and Paired Bootstrap | `reports/backtest_main/significance/significance_report.json` |
| `scripts/run_inventory_analysis.py` | ABC/XYZ classification and safety stock analysis | `reports/inventory/` |
| `scripts/render_readme_tables.py` | Dynamic Markdown table injector for README | `README.md` updates |
| `scripts/tune_ml.py` | Optuna hyperparameter optimization for LightGBM | MLflow runs |
| `scripts/tune_dl.py` | Preliminary neural network tuning | MLflow runs |
| `scripts/rescore_tier1.py` | Recalculates tier1 RMSSE on canonical scales | `reports/tier1/tier1_overall.csv` |

---

## 8. Configuration & Build Specifications

- `configs/model_config.yml`: Central YAML configuration declaring hyperparameters for LightGBM, CatBoost, StatsForecast baselines, and cross-validation fold schedules.
- `pyproject.toml`: Poetry build configuration, Python version lock (`>=3.11,<3.13`), dependencies, and Ruff linter rules.
- `dvc.yaml`: Data Version Control DAG orchestrating data extraction, feature generation, backtesting, and reporting.
- `.github/workflows/ci_cd.yml`: GitHub Actions workflow enforcing Ruff linting, dynamic package import checks, unit tests, and integration pipelines on pull requests.

---

## 9. Test Suite Architecture

- `tests/unit/test_m5_01_scale_semantics.py`: Verifies per-(series, origin) RMSSE aggregation and pre-launch zero trimming.
- `tests/unit/test_m5_03_tier1_parity.py`: Enforces metric parity between `tier1` and `backtest_main`.
- `tests/unit/test_m5_04_source_of_truth.py`: Ensures CSV-Parquet parity and canonical CSV-first loading.
- `tests/unit/test_m5_06_statistical_defense.py`: Validates paired bootstrap CIs and per-series DM tests.
- `tests/unit/test_metrics.py`: Unit tests for point and quantile loss functions.
- `tests/unit/test_models_and_facts.py`: Asserts model parameter constraints and baseline invariants.
- `tests/unit/test_polars_parity.py`: Verifies exact numerical equivalence between Polars and Pandas feature engines.
- `tests/integration/test_loader_features_model.py`: End-to-end integration test verifying data loading, feature generation, and LightGBM model fitting.
- `tests/test_dashboard.py`: Guards against hardcoded metric literals in Streamlit dashboard views.
- `tests/test_readme_tables.py`: CI test asserting README tables match backing artifact files with zero drift.
