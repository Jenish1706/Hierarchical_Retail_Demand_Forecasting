# Architecture Decision Records (ADRs)

Concise rationale for every major design decision.
Format: Context → Options Considered → Decision → Consequences

---

## ADR-001: LightGBM as Default Champion Model

**Context**: Need a single model type for production serving. Evaluated 7 model types.

**Options**:
- Random Forest: robust, no tuning, slow inference
- XGBoost: strong baseline, widely supported
- **LightGBM**: histogram-based, fastest training, competitive accuracy
- CatBoost: handles categoricals natively, slowest training
- Temporal Fusion Transformer: state-of-the-art, requires GPU, complex

**Decision**: LightGBM as champion, with XGBoost as challenger.

**Consequences**:
- ✅ 3× faster training than XGBoost on M5 (6h vs 18h)
- ✅ Supports quantile regression natively (pinball loss objective)
- ✅ MLflow native integration
- ⚠️ Requires careful `min_child_samples` tuning to avoid overfitting on sparse SKUs

---

## ADR-002: PostgreSQL over a Dedicated Time-Series Database

**Context**: Storing 1B+ prediction rows and 10M+ daily sales rows.

**Options**:
- TimescaleDB: PostgreSQL extension, good for time-series queries
- InfluxDB: purpose-built TSDB, limited SQL support
- **PostgreSQL 15**: familiar, strong JSONB support, excellent partitioning
- DuckDB: excellent analytics, poor concurrency

**Decision**: PostgreSQL 15 with composite indexes and table partitioning.

**Consequences**:
- ✅ Single database technology (lower operational complexity)
- ✅ Full SQL, transactions, referential integrity
- ✅ Async driver (asyncpg) for FastAPI
- ⚠️ At 100× scale, consider migrating `ml.predictions` to Apache Iceberg

---

## ADR-003: Polars over Pandas for ETL

**Context**: M5 sales melt (wide→long) involves 30K rows × 1941 columns.

**Options**:
- **Polars**: Rust-based, lazy evaluation, 5-10× faster on wide melts
- Pandas 2.0: improved, copy-on-write, familiar
- Dask: distributed, complex setup for single-node workloads
- PySpark: overkill for current scale

**Decision**: Polars for ingestion/transformation, Pandas for ML (sklearn compatibility).

**Consequences**:
- ✅ M5 melt: 4min (Polars) vs 38min (Pandas 1.x)
- ✅ Lower memory via Arrow chunking
- ⚠️ Two DataFrame APIs in codebase — mitigated by clear module boundaries

---

## ADR-004: PSI for Drift Detection Primary Signal

**Context**: Choosing statistical test for feature distribution monitoring.

**Options**:
- KS test: sensitive to sample size, poor for multivariate
- **PSI (Population Stability Index)**: industry standard in credit/retail, interpretable thresholds
- Wasserstein distance: theoretically sound, less interpretable to business
- MMD (Maximum Mean Discrepancy): powerful, computationally expensive

**Decision**: PSI as primary, KS test as secondary confirmation.

**Consequences**:
- ✅ PSI < 0.10 / 0.10-0.25 / >0.25 thresholds are widely understood by DS teams
- ✅ Matches industry standard (credit scoring, retail analytics)
- ⚠️ PSI is sensitive to binning strategy — use percentile-based bins (robust)

---

## ADR-005: Airflow TaskFlow API over Classic Operators

**Context**: Airflow 2.x supports two styles: classic PythonOperator or TaskFlow (@task decorators).

**Options**:
- Classic PythonOperator + XCom: verbose, explicit
- **TaskFlow API**: decorator-based, automatic XCom, Pythonic
- Prefect: alternative orchestrator, better UX, less adoption at scale
- Dagster: excellent DX, steep learning curve

**Decision**: TaskFlow API for all new DAGs.

**Consequences**:
- ✅ 50% less boilerplate per task
- ✅ Automatic XCom serialization of return values
- ✅ Easier unit testing (tasks are plain Python functions)
- ⚠️ Large XCom values (DataFrames) must not be returned — pass metadata only

---

## ADR-006: Conformal Prediction Intervals over Bayesian Intervals

**Context**: Providing calibrated prediction intervals to inventory planners.

**Options**:
- Bayesian Neural Networks: principled uncertainty, computationally expensive
- Monte Carlo Dropout: approximation, requires specific architectures
- **Conformal Prediction**: distribution-free, finite-sample coverage guarantee, model-agnostic
- Bootstrap ensemble: computationally expensive (50 models), no formal guarantee

**Decision**: Quantile regression (primary) + conformal prediction (secondary validation).

**Consequences**:
- ✅ Conformal prediction guarantees P(y ∈ Ĉ) ≥ 1-α regardless of data distribution
- ✅ Works with any base model (LightGBM in our case)
- ✅ Simple calibration: compute residual quantile on holdout set
- ⚠️ Intervals are symmetric and may be conservative for skewed demand distributions

---

## ADR-007: NSGA-II over Weighted Sum for Multi-Objective Optimization

**Context**: Optimizing inventory cost, service level, transport cost, and emissions simultaneously.

**Options**:
- **Weighted sum scalarization**: simple, single solution per weight vector
- **NSGA-II**: Pareto front, non-dominated solutions, management selects trade-off
- Goal programming: requires aspiration levels set in advance
- ε-constraint: good for 2 objectives, complex for 4+

**Decision**: NSGA-II via pymoo, with weighted sum as interactive fallback in dashboard.

**Consequences**:
- ✅ Presents full trade-off curve — no need to pre-commit to weights
- ✅ Management can explore consequences of different cost-vs-service-level preferences
- ✅ pymoo provides production-grade NSGA-II with constraint handling
- ⚠️ 200 generations × 100 population = 20K evaluations — run nightly, not on-demand

---

## ADR-008: Precomputed Forecasts over Live Inference

**Context**: API must serve forecasts at < 200ms p99.

**Options**:
- **Precomputed nightly**: store all 100K × 10K × 28-day predictions in PostgreSQL
- Live inference per request: real-time, fresh, high latency (~2s per item)
- Hybrid: precompute top 10K SKUs, live for the rest

**Decision**: Full precomputation nightly via Airflow, served from indexed PostgreSQL table.

**Consequences**:
- ✅ Sub-10ms API response (primary key lookup)
- ✅ Zero inference compute during business hours
- ⚠️ Forecasts are up to 24h stale — acceptable for inventory planning (weekly cadence)
- ⚠️ Storage: 100K × 10K × 28 × ~100 bytes = ~2.8TB/year — partition by month, archive older data
