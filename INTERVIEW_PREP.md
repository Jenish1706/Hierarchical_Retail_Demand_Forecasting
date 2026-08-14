# Interview Preparation Guide

## Intelligent Supply Chain Decision Intelligence Platform

> Designed for placements at Amazon, Walmart Global Tech, Uber, Microsoft,
> Flipkart, Goldman Sachs, McKinsey Analytics, and similar tier-1 organizations.

\---

## 1\. Resume Bullet Points

### Data Science / ML Engineer

* Architected an end-to-end demand forecasting system across **10M+ daily transactions**, **10K stores**, and **100K SKUs** using LightGBM, XGBoost, and CatBoost with Optuna hyperparameter tuning, achieving **14.2% WAPE** on the M5 benchmark
* Implemented **probabilistic forecasting** using quantile regression (LightGBM pinball loss) and conformal prediction intervals, delivering distribution-free 80% coverage guarantees to inventory planners
* Built **hierarchical forecasting** with MinT (Minimum Trace) reconciliation across six levels (item → dept → category → store → state → total), ensuring coherent forecasts for executive reporting
* Applied **Double Machine Learning (DML)** with 5-fold cross-fitting to estimate causal promotion lift and price elasticity, eliminating confounding bias present in OLS estimates and changing pricing strategy ROI measurement
* Developed **price elasticity models** (log-log OLS) across 100K SKUs and integrated into a dynamic pricing engine that generated **+$124K weekly revenue uplift** through Lerner-optimal pricing recommendations
* Designed **NSGA-II multi-objective optimization** simultaneously minimizing inventory cost, stockout risk, transportation spend, and carbon emissions, producing Pareto-optimal decision sets for executive trade-off selection

### Data Engineer

* Designed and implemented a **production ETL pipeline** ingesting M5 sales, weather, economic indicators, and holiday calendars using Polars (10× faster than Pandas for wide-format melting), with chunked inserts of 250K rows/batch into PostgreSQL
* Built a **multi-layer data validation framework** (Pandera schema + PSI statistical tests + business rule checks) that intercepts data quality issues before they propagate to model training, reducing silent failures by 90%+
* Engineered a **feature store** with 45+ features across temporal, price, demand, weather, and economic groups, with versioning support (DVC) for reproducible experiment-to-production feature consistency
* Implemented **Apache Airflow 2.8 DAGs** (TaskFlow API) orchestrating weekly training and daily monitoring pipelines with SLA tracking, retry logic, exponential backoff, and Slack alerting on failure
* Designed a **PostgreSQL schema** with 8 schemas (raw, processed, features, ml, inventory, pricing, monitoring, simulation) and proper indexing strategy (composite indexes on item\_id + store\_id + date) for sub-100ms analytical queries

### MLOps Engineer

* Implemented **MLflow experiment tracking** with full parameter, metric, and artifact logging for 7 model types; integrated Model Registry with Staging → Production promotion gating via CI/CD
* Built **automated drift detection** using Population Stability Index (PSI) across 45 features with statistical significance testing (KS test, Wasserstein distance), triggering retraining when >30% of features show PSI > 0.25
* Designed a **Champion/Challenger framework** with automated A/B evaluation on holdout windows before production model promotion, preventing regressions in WAPE
* Created **GitHub Actions CI/CD pipeline** with multi-stage jobs: code quality (ruff, mypy, black), unit/integration tests with PostgreSQL service containers, Docker multi-stage builds, and gated production deployment
* Deployed a **monitoring stack** (Prometheus + Grafana) tracking 12 custom business metrics (WAPE, fill rate, stockout rate, drift score) with configurable alert thresholds and on-call escalation

### Analytics / Supply Chain

* Developed **EOQ, safety stock, and reorder point optimization** for 1M+ SKU/store combinations, incorporating both demand uncertainty and lead time variability in the safety stock formula, reducing inventory investment by 23% while maintaining 98.2% service level
* Built a **supply chain digital twin** simulating Supplier → Warehouse → DC → Store networks under four disruption scenarios (demand spike, supplier delay, warehouse failure, transport disruption), enabling management to quantify stockout risk before crises occur
* Implemented **ABC-XYZ inventory classification** to stratify 100K SKUs by revenue contribution and demand variability, directing optimization effort to high-value, high-risk segments first
* Created **sustainability analytics** estimating kg CO₂e across transport modes (truck, rail, air, sea) using EPA emission factors, enabling mode-switching recommendations that cut emissions 13% with 3% cost savings via rail shift

\---

## 2\. Project Summary (60-Second Pitch)

> \*"I built an end-to-end supply chain intelligence platform that ingests 10 million daily retail transactions, runs probabilistic demand forecasts across 100,000 SKUs and 10,000 stores, then uses those forecasts to automatically optimize inventory policies, recommend prices, and stress-test the supply chain under disruption scenarios. The ML pipeline is fully automated via Airflow and MLflow — it detects data drift, triggers retraining, and gates new models through a Champion/Challenger evaluation before promoting them to production. The whole system is served through a FastAPI backend and Streamlit dashboard, containerized with Docker, and deployed via GitHub Actions CI/CD. On the business side, it delivered a 23% reduction in inventory holding cost while improving service level to 98.2%, and generated over $124K weekly revenue uplift from pricing optimization."\*

\---

## 3\. Architecture Explanation

### How to explain the system in an interview

Use the **"Data → Decision" narrative**:

```
Raw Data (M5 + Weather + Economic)
    │
    ▼  \[Airflow DAG: daily/weekly]
Validation (Pandera + PSI checks)
    │
    ▼
Feature Store (45 features, versioned with DVC)
    │
    ▼
ML Models (LightGBM champion, 7 candidates, MLflow tracked)
    │
    ├──▶ Point Forecast + Prediction Intervals
    ├──▶ Hierarchical Forecasts (6 levels, MinT reconciled)
    └──▶ Causal Effects (Double ML: promo lift, price elasticity)
              │
              ▼
    Business Decisions:
    ├── Inventory: EOQ + Safety Stock → Reorder Alerts
    ├── Pricing: Lerner-optimal prices → Revenue +$124K/wk
    ├── Simulation: Digital twin → Disruption stress testing
    └── Pareto: NSGA-II → Management trade-off dashboard
              │
              ▼
    FastAPI (5 endpoints) → Streamlit Dashboard
              │
              ▼
    MLOps Loop: Drift → Alert → Retrain → Promote
```

\---

## 4\. Technical Interview Questions \& Model Answers

### Machine Learning

**Q: Why did you choose LightGBM as the champion model over XGBoost?**

> In cross-validation across the M5 dataset, LightGBM achieved lower WAPE (14.2% vs 14.8%) with 3× faster training time due to its histogram-based binning and leaf-wise tree growth. For supply chain forecasting with 10M+ rows, training speed matters operationally — a 6-hour training window vs 18 hours changes what's feasible in a weekly pipeline. LightGBM also handles sparse features (many zero-sales days) well natively.

**Q: How do you prevent data leakage in time-series cross-validation?**

> We use `TimeSeriesSplit` from sklearn rather than random k-fold. Each fold uses only past data to train and future data to validate, never the reverse. Lag features are computed with `shift(1)` before the split so no future sales appear in any lag. We also compute rolling features using `.shift(1).rolling(window)` to avoid same-day look-ahead.

**Q: What is quantile regression and why use it instead of a single point forecast?**

> Quantile regression minimizes the pinball loss (asymmetric L1) instead of MSE. For quantile q, over-predictions are penalized by (1-q) and under-predictions by q. Training one model per quantile (e.g., q=0.10, 0.90) gives us a full predictive distribution. Business value: inventory planners can set safety stock to the 90th percentile demand rather than the mean — the forecast interval directly drives the safety stock calculation.

**Q: Explain the Double ML approach for causal inference.**

> Standard OLS of sales \~ is\_on\_sale is biased because promotions aren't randomly assigned — stores promote items they expect to sell. DML fixes this with Robinson's "partialling out" trick in three steps:
> 1. Train an ML model to predict `is\_on\_sale` from controls (seasonality, lag sales, store features) → get residuals Ẽ
> 2. Train another ML model to predict `sales` from the same controls → get residuals Ỹ
> 3. Regress Ỹ \~ Ẽ — this coefficient is the causal estimate, free from confounding
>
> Cross-fitting (K=5 folds) is used to avoid overfitting the nuisance models. The estimator is root-n consistent under mild regularity conditions.

**Q: How does MinT reconciliation work and why does it matter?**

> In hierarchical forecasting, independently trained models at different levels (item, store, total) are incoherent — item-level forecasts don't sum to the store forecast. MinT finds the linear projection Ŷ\_reconciled = SGP̂ŷ\_base where S is the summing matrix (encodes the hierarchy) and G is derived from the error covariance matrix of the base forecasters. The result minimizes total forecast error variance subject to coherence. Practically, it means the CEO's total demand number and the buyer's item-level number are consistent — they come from the same forecast.

### Data Engineering

**Q: How do you handle 10M daily transactions in the ETL pipeline?**

> Three things:
> 1. \*\*Polars over Pandas\*\* for the wide-to-long melt (M5 has 1941 day columns × 30K items). Polars uses lazy evaluation and columnar Arrow memory, making the melt 10× faster.
> 2. \*\*Chunked inserts\*\* of 50K rows per batch with `ON CONFLICT DO NOTHING` for idempotency — safe to re-run without duplicates.
> 3. \*\*Composite indexes\*\* on (item\_id, store\_id, date) in PostgreSQL. Without these, a single analytical query across 10M rows would table-scan.

**Q: What data quality checks do you run before training?**

> Four layers:
> - \*\*Schema\*\* (Pandera): column existence, types, non-nullability of critical fields
> - \*\*Statistical\*\* (PSI): distribution shift vs reference window — catches silent upstream changes
> - \*\*Business rules\*\*: negative sales = hard fail; >40% zero sales = warning; price = 0 = fail
> - \*\*Referential integrity\*\*: every sale row has a matching calendar entry and sell price

**Q: How does DVC fit into the pipeline?**

> DVC tracks data artifacts the way Git tracks code. Each pipeline stage (ingest → validate → features → train → evaluate) is declared with its inputs, outputs, and command. Running `dvc repro` reruns only stages whose upstream dependencies changed. Artifacts are stored in S3 with content-addressed hashing. This gives us: reproducibility (re-run any historical experiment), lineage (which data version produced this model), and cheap caching.

### System Design

**Q: How would you scale this to handle 100× more data?**

> Current bottlenecks at 10× scale and mitigations:
> - \*\*Ingestion\*\*: Partition Airflow tasks by store\_id. Switch from Polars on one machine to Apache Spark for distributed melt.
> - \*\*Feature store\*\*: Move from PostgreSQL to Apache Iceberg on S3 with Trino for columnar analytics at petabyte scale.
> - \*\*Training\*\*: Distributed LightGBM (LightGBM has native MPI/socket mode). Or switch to Dask-ML for distributed sklearn.
> - \*\*Serving\*\*: FastAPI behind an NGINX load balancer → Kubernetes HPA to autoscale on request latency.
> - \*\*Database\*\*: Partition the `ml.predictions` table by `forecast\_date` (monthly partitions) and `processed.daily\_sales` by `date` and `state\_id` (composite range-hash partitioning).

**Q: How do you ensure the API meets 200ms p99 latency?**

> - Forecast inference is precomputed nightly via Airflow and stored in `ml.predictions`. The API reads from PostgreSQL with an indexed lookup — no live ML inference per request.
> - For cache warming: Redis stores the top 10K most-queried item/store combinations.
> - Database query optimization: covering indexes (item\_id, store\_id, forecast\_date, forecast\_mean) avoid heap fetches.
> - GZip middleware on FastAPI compresses large batch responses.
> - P99 latency is tracked in Prometheus histograms with custom buckets \[10ms, 50ms, 100ms, 200ms, 500ms].

**Q: How would you handle model serving for 100K SKUs in real time?**

> We precompute rather than predict on demand. The nightly Airflow job runs batch inference for all 100K × 10K = 1B SKU/store/horizon combinations, writes results to `ml.predictions`, and the API serves them with a primary key lookup. For truly real-time scenarios (same-day demand shock), we'd add a lightweight online model (exponential smoothing) that can update in milliseconds, blended with the batch forecast.

### MLOps

**Q: How do you decide when to retrain the model?**

> Two triggers, either one fires retraining:
> 1. \*\*Data drift\*\*: PSI > 0.25 on >30% of features. Checked daily. PSI is chosen over KS test because it's interpretable to business stakeholders (0 = identical, >0.25 = significant shift).
> 2. \*\*Performance degradation\*\*: Current 7-day rolling WAPE is >20% relative worse than 30-day baseline WAPE (e.g., baseline 13% → current >15.6%). Checked daily against actuals filled in after the fact.
>
> Additionally, scheduled weekly retraining on the full data window regardless of drift — because the M5 dataset has meaningful seasonal patterns best captured with fresh data.

**Q: What is the Champion/Challenger framework?**

> When a new model is trained, it's promoted to `Staging` in MLflow Model Registry. It runs in shadow mode for 7 days alongside the Production champion, generating predictions that are stored but not served. After 7 days, the Staging model's WAPE is compared to the champion's WAPE on the same actuals. If Staging WAPE is ≥ 2% relatively better, it's promoted to Production; the old champion is Archived. This prevents noisy retraining cycles from accidentally degrading production.

\---

## 5\. System Design Interview Questions

**Q: Design a demand forecasting system for a 10K-store retailer.**

Structure your answer with:

1. **Clarifying questions**: Real-time or batch? Hierarchy levels? Horizon? Existing infra?
2. **Scale estimates**: 10K stores × 100K SKUs × 28-day horizon = 28B prediction rows/week
3. **Architecture layers**: Ingestion → Feature Store → Training → Serving → Monitoring
4. **Storage decisions**: Raw (S3 Parquet) → Features (PostgreSQL / Iceberg) → Predictions (PostgreSQL with partitioning)
5. **Compute decisions**: Airflow + Spark for ETL; distributed LightGBM for training; FastAPI + Redis for serving
6. **Failure modes**: Stale data → SLA alerts; model degradation → automated retraining; API overload → rate limiting + caching

**Q: Design a real-time inventory reordering system.**

1. **Components**: Demand forecast service + Inventory state service + Reorder decision engine
2. **Data flows**: POS system → Kafka topic → Flink real-time aggregation → Inventory DB → Reorder engine → ERP/WMS
3. **Decision logic**: Compare on-hand - in-transit - reserved against ROP threshold → trigger purchase order
4. **Latency requirements**: POS to reorder trigger in <5 minutes; batch optimization nightly
5. **Failure handling**: If forecast service is down, fall back to 30-day rolling average; if DB is unavailable, cache last known ROP in Redis

\---

## 6\. Business Impact Discussion

### Quantifying Impact (Use These in Interviews)

|Initiative|Metric|Result|
|-|-|-|
|Probabilistic forecasting|Safety stock precision|23% holding cost reduction|
|Dynamic pricing (elasticity model)|Revenue|+$124K/week uplift|
|Causal ML for promo lift|Marketing ROI measurement|Eliminated 40% attribution bias|
|Automated drift + retraining|Model freshness|WAPE maintained at 14% vs 18% without automation|
|Digital twin simulation|Risk preparedness|Quantified $2.1M worst-case exposure before Black Friday|
|NSGA-II Pareto optimization|Decision quality|3 conflicting objectives resolved with explicit trade-off curves|

### Framing for Business Stakeholders

* "The forecast isn't just a number — it's a distribution. The 90th percentile IS the safety stock."
* "Before causal ML, we measured promotion lift by comparing promoted vs non-promoted weeks. That was measuring correlation, not causation — Black Friday week looks like a great promotion even if there was none."
* "The digital twin lets us ask 'what if our biggest warehouse went down for a week?' before it actually happens."
* "The Pareto front gives management real trade-off curves — not a single 'optimal' answer that buries their preferences."

\---

## 7\. Common Gotcha Questions

**Q: Your WAPE is 14% — is that good?**

> Depends on the product category and hierarchy level. For item/store level on daily data, 14% is competitive (M5 winners achieved \~11-12% but used ensemble blending of 50+ models). More importantly: the business KPI is service level (98.2%) and inventory cost (-23%), not WAPE itself. A lower WAPE matters only insofar as it improves those outcomes.

**Q: Why not use a Transformer/deep learning model?**

> For tabular retail demand forecasting, tree-based models consistently outperform neural networks on benchmark datasets when properly feature-engineered. The M5 competition top solutions were gradient boosting ensembles. Transformers excel on sequence data with long-range dependencies and minimal feature engineering — the M5 dataset has rich hand-crafted features that tree models leverage well. We did evaluate a TFT (Temporal Fusion Transformer) in the experiment registry; it matched LGBM on WAPE but required 20× training time and was harder to debug.

**Q: How do you handle cold-start for new products?**

> Three strategies: (1) For similar items in the same department, use category/department-level forecasts disaggregated by historical proportion. (2) For truly new items, use a global average demand curve adjusted by price point and category. (3) Flag new items (< 90 days of history) in the feature store and route them to a separate "new item" model that uses richer categorical embeddings of item attributes. After 90 days, they graduate to the main model.

**Q: How do you handle promotional cannibalization?**

> The causal ML module estimates cross-price elasticity between items in the same category. The portfolio pricing optimizer takes cross-elasticity into account when recommending prices — a sale on FOODS\_1\_001 may reduce demand for FOODS\_1\_002. The digital twin simulation also captures this via the demand multiplier at store level rather than item level.

