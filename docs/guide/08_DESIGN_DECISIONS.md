# Architecture Decision Records (ADRs)

This document provides formal Architecture Decision Records for the primary technical, mathematical, and infrastructural decisions of the platform. Each entry details the context, the decision taken, considered alternatives, and resulting consequences.

---

## ADR-001: Scope Discipline & Removal of Non-ML/Fictitious Serving Layers

- **Status**: Accepted
- **Context**: The project initially started as a broad "supply chain intelligence platform" containing a FastAPI microservice, a digital twin simulation, supply-chain network graph analytics, carbon emissions accounting, and NSGA-II multi-objective inventory optimization. However, the M5 competition dataset contains only four tables: unit sales, calendar events, weekly sell prices, and a submission template. It contains no supplier network, transport legs, warehouse coordinates, inventory holding balances, actual lead times, or unit purchase costs. All downstream modules were producing outputs derived from arbitrary synthetic constants.
- **Decision**: Wholesale removal of the simulated supply chain modules, digital twin, and REST API. Focus the repository strictly on what the empirical data supports: hierarchical retail demand forecasting, rigorous econometric evaluation, and analytical inventory policy trade-offs.
- **Alternatives Considered**:
  - *Keep simulated modules with synthetic parameters*: Rejected because presenting synthetic supply chain simulations alongside real Kaggle data undermines portfolio credibility.
- **Consequences**:
  - Removed ~6,700 lines of unbacked code.
  - Scope is strictly grounded in verifiable empirical data.
  - Operational limitations (e.g. absent lead times) are explicitly documented rather than masked.

---

## ADR-002: In-Process Analytical Storage (DuckDB & Parquet vs. Relational Servers)

- **Status**: Accepted
- **Context**: The initial architecture required a running PostgreSQL container with eight distinct schemas. However, every workload in retail demand research consists of analytical OLAP scans: melting 30,490 series $\times$ 1,913 days to long format, computing rolling aggregations, and scoring models across origins. There are no concurrent online transaction processing (OLTP) workloads or row-level key-value lookups.
- **Decision**: Use Parquet files on disk for persistent storage and DuckDB/Polars for analytical in-process transformations. Eliminate PostgreSQL and Docker container prerequisites.
- **Alternatives Considered**:
  - *PostgreSQL / TimescaleDB*: Heavy infrastructure requirement; slow bulk scans over 58 million rows.
  - *Raw CSV files only*: Excessive disk consumption and slow I/O for large tabular scans.
- **Consequences**:
  - Running experiments requires no external database service (`docker compose up` eliminated).
  - High compression: raw 229 MB tabular files compress efficiently into Parquet.
  - Analytical queries execute at in-memory C++ speed without network serialization.

---

## ADR-003: Nested Cross-Validation for Tuned Models

- **Status**: Accepted
- **Context**: Standard machine learning workflows often tune hyperparameters on the same validation fold used to report final model performance. When tuning high-capacity models with large Optuna search spaces, this induces severe optimistic selection bias (overfitting the test fold).
- **Decision**: Implement strict nested cross-validation. An outer `TimeSeriesSplit` reserves unseen temporal test windows for performance reporting; an inner `TimeSeriesSplit` over outer-training observations exclusively drives Optuna hyperparameter selection.
- **Alternatives Considered**:
  - *Standard K-Fold / TimeSeriesSplit*: Violates temporal isolation or leaks hyperparameter selection into reported metrics.
- **Consequences**:
  - Training compute increases proportionally ($k_{\text{outer}} \times n_{\text{trials}} \times k_{\text{inner}}$).
  - Reported metrics reflect true generalization performance on completely unseen temporal horizons (Cawley & Talbot, 2010).

---

## ADR-004: Horizon-Safe ("Direct") Feature Engineering

- **Status**: Accepted
- **Context**: Demand forecasting models frequently build lag features using `shift(1)`. While valid for 1-step-ahead daily re-forecasting, evaluating a multi-step horizon ($h=28$ days) with `shift(1)` features means that on test day 14, the model accesses the actual sales of test day 13—information impossible to know at forecast origin. Global ML models were effectively scored on 28 one-step forecasts, while baselines projected from origin.
- **Decision**: Enforce horizon-safe direct features: all backward-looking features (lags, rolling means, exponential moving averages) are shifted by the full forecast horizon $h=28$, dropping any lag shorter than $h$.
- **Alternatives Considered**:
  - *Recursive multi-step forecasting*: Feeding predictions back into lags iteratively accumulates autoregressive errors, resulting in severe degradation on sparse series.
- **Consequences**:
  - Completely eliminated future temporal leakage.
  - Corrected historical metrics: LightGBM WAPE adjusted from 86.76% to 88.79%, Tweedie from 81.95% to 85.72%, exposing the ~3.8 WAPE points that were illusory (historical figures from an earlier run; no committed artifact reproduces them).
  - Evaluation is 100% like-for-like against statistical baselines.

---

## ADR-005: Hypothesis Testing Battery Over Standalone Leaderboards

- **Status**: Accepted
- **Context**: Presenting a sorting leaderboard of point metrics (e.g. Model A at 75.16% WAPE vs Model B at 75.47% WAPE) invites readers to claim that Model A is superior. Across thousands of fat-tailed, intermittent series, small metric gaps are frequently statistical noise.
- **Decision**: Gate all model ranking claims through an econometric hypothesis testing battery:
  1. Omnibus Friedman test over per-series ranks.
  2. Post-hoc Nemenyi critical difference ranking if Friedman rejects equal performance.
  3. Paired series bootstrap confidence intervals for aggregate metric gaps.
  4. Per-series Diebold-Mariano tests with Holm-Bonferroni correction.
- **Alternatives Considered**:
  - *Uncorrected pairwise t-tests*: Invalid due to heavy right-skewed non-normality and uncontrolled family-wise error inflation.
- **Consequences**:
  - Revealed crucial negative results: LightGBM Tweedie vs standard LightGBM shows no statistically significant rank difference ($p=1.000$).
  - Clarified that ADIDA leads in volume-weighted aggregate error, while Croston SBA leads in average rank across individual series.

---

## ADR-006: Single-Process (`n_jobs=1`) Concurrency for StatsForecast on Windows

- **Status**: Accepted
- **Context**: `statsforecast` provides process-level parallelism via `n_jobs=-1`. On Windows architectures, Python relies on the `spawn` multiprocessing start method rather than Unix `fork`. In the absence of top-level `__main__` guards in all imported modules, child processes recursively re-imported the module tree, triggering immediate `BrokenProcessPool` crashes. Furthermore, spin-up overhead dominated runtimes for small series batches.
- **Decision**: Enforce `n_jobs=1` across all `statsforecast` invocations on Windows.
- **Alternatives Considered**:
  - *Enforce multiprocessing with complex spawn guards*: Fragile across different Windows PowerShell execution environments.
- **Consequences**:
  - Guarantees 100% crash-free execution on Windows.
  - Estimators are already Numba-compiled and vectorized across series in single-threaded C-speed, running 3,000 series in seconds.

---

## ADR-007: DVC as the Sole Pipeline Orchestrator

- **Status**: Accepted
- **Context**: The codebase previously carried an Apache Airflow DAG requiring a PostgreSQL metadata database, webserver, and scheduler. Multiple Airflow tasks invoked deleted modules or returned mocked dictionaries, while `dvc.yaml` sat abandoned with broken script paths.
- **Decision**: Delete Airflow completely. Repair `dvc.yaml` to serve as the single, lightweight pipeline orchestrator.
- **Alternatives Considered**:
  - *Maintain Airflow / Prefect*: Massive dependency footprint with no production scheduling requirement for a research benchmark.
- **Consequences**:
  - `dvc repro` reproduces data extraction, feature generation, backtesting, and reporting in a single command.
  - Pipeline stages track exact input and output file hashes, preventing stale artifacts.

---

## ADR-008: Deep Learning as an Empirical Hypothesis

- **Status**: Accepted
- **Context**: Early documentation prohibited neural networks, citing the 2020 M5 competition finding that top-50 solutions were tree-based. However, modern linear and patch-based time-series architectures (DLinear, PatchTST, N-HiTS) were published after 2020.
- **Decision**: Lift the dogmatic prohibition. Permit deep learning architectures as Tier 4 benchmarks evaluated under identical rolling-origin protocols.
- **Alternatives Considered**:
  - *Permanent exclusion of neural architectures*: Converts an empirical 2020 finding into an unchallengeable assumption.
- **Consequences**:
  - Formulated the exact empirical hypothesis: On zero-inflated (57.5% zeros) retail counts at Level 12, simple statistical estimators outperform or match complex neural models. DLinear serves as the direct test of whether linear mapping matches transformer complexity.

---

## ADR-009: Canonical Per-Series RMSSE Scaling Anchored to First Sale

- **Status**: Accepted
- **Context**: An earlier script aggregated RMSSE by pooling squared errors and naive scales across all series observations before taking the square root ($\sqrt{\text{mean}(e^2 / s)}$). This penalized sparse series with volatile spikes and deviated from the official M5 competition formula:
  $$\text{RMSSE}_i = \sqrt{\frac{\frac{1}{h}\sum (y_{it} - \hat{y}_{it})^2}{\frac{1}{T - d_{\text{first}}} \sum (y_{it} - y_{i,t-1})^2}}$$
  Additionally, calculating naive scale from Day 1 rather than the product launch date ($d_{\text{first}}$) padded the denominator with $0 - 0 = 0$ differences, artificially inflating RMSSE.
- **Decision**: Enforce canonical M5 scaling: calculate scale strictly from the first non-zero sale ($d_{\text{first}}$), compute RMSSE per (series, origin), and average across series.
- **Alternatives Considered**:
  - *Observation-pooled RMSSE*: Biased by high-volume series; breaks comparability with M5 benchmarks.
- **Consequences**:
  - Full metric parity restored across all evaluation tiers (AutoETS Level-12 RMSSE harmonized at $0.7675$).
  - Verified by unit tests `test_m5_01_scale_semantics.py` and `test_m5_03_tier1_parity.py`.

---

## ADR-010: Paired Series Bootstrap Over Pooled Diebold-Mariano

- **Status**: Accepted
- **Context**: In multi-series forecasting, standard Diebold-Mariano tests assume a single time series. Concatenating 3,000 series across 3 origins into a 252,000-row observation vector artificially inflates sample size $N$, rendering trivial metric differences statistically "significant" ($p < 10^{-10}$). Furthermore, pooling ignores cross-series variance and violates serial autocorrelation.
- **Decision**: Implement a paired series bootstrap (2,000 resamples). Resample whole time-series with replacement, evaluating aggregate metrics on each pseudo-panel to construct empirical 95% confidence intervals on metric deltas ($\Delta\text{WAPE}$, $\Delta\text{RMSSE}$).
- **Alternatives Considered**:
  - *Observation-level bootstrap*: Destroys temporal autocorrelation structure within series.
- **Consequences**:
  - Rigorous statistical defense: proved that Routed Ensemble is statistically indistinguishable from ADIDA on WAPE ($\Delta\text{WAPE} \in [-0.01\%, +1.04\%]$) and Moving Average is indistinguishable on RMSSE ($\Delta\text{RMSSE} \in [-0.0001, +0.0023]$).

---

## ADR-011: Canonical CSV as Source of Truth with Companion Parquet Sync

- **Status**: Accepted
- **Context**: The repository generated both `.csv` and `.parquet` companions for report outputs. In some cases, scripts updated the CSV while leaving Parquet companions stale, leading to conflicting numbers in dashboard views.
- **Decision**: Declare CSV the canonical source of truth for all human-readable report tables in `reports/`. Dashboard loaders read CSV first. A synchronization utility updates companion Parquets losslessly.
- **Alternatives Considered**:
  - *Parquet-only storage*: Eliminates easy human diffing in git pull requests.
- **Consequences**:
  - Git diffs clearly display metric changes in pull requests.
  - Zero drift between CSV and Parquet companions across all 38 report tables, verified by `test_m5_04_source_of_truth.py`.
