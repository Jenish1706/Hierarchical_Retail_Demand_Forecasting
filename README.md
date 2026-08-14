# M5 Retail Demand Intelligence Platform

> **Hierarchical Forecasting · Probabilistic Intervals · Price Sensitivity · Causal SNAP Analysis · Automated MLOps**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange.svg)](https://mlflow.org)
[![Airflow](https://img.shields.io/badge/Airflow-2.8-red.svg)](https://airflow.apache.org)

---

## Honest Project Scope

This platform is built on the **M5 Forecasting Competition dataset** — exactly what the data supports, nothing more.

### What the M5 dataset actually contains
| Table | Contents |
|-------|----------|
| `Sales_train_validation` | 30,490 series × 1,913 days of **integer unit counts** |
| `Sales_prices` | Weekly sell price per (item, store) — no cost, no margin |
| `calendar` | 1,969 dates with named events, SNAP flags, weekday info |
| `Sales_submission` | 28-day forecast submission template |

**Scale:** 10 stores · 3,049 unique items · 3 states (CA, TX, WI) · 5.25 years of history

### What this platform does

| Module | Data Supported | Description |
|--------|---------------|-------------|
| Point forecasting | ✅ | LightGBM champion + XGBoost/CatBoost challengers |
| Probabilistic forecasting | ✅ | Quantile regression (q10/q50/q90) with calibration checks |
| Hierarchical forecasting | ✅ | MinT reconciliation: item → dept → cat → store → state → total |
| Intermittent demand | ✅ | Croston/ADIDA for zero-heavy series |
| Price sensitivity analysis | ✅ (with stated limits) | Log-log OLS elasticity, not profit optimization |
| SNAP causal analysis | ✅ | Double ML estimating causal demand lift on benefit days |
| Holiday lift estimation | ✅ | Event study across 161 named events |
| Demand scenario analysis | ✅ | What-if demand shock simulation |
| Inventory policy simulation | ⚠️ Assumed params | EOQ/SS/ROP with documented cost assumptions |
| MLOps (MLflow/Airflow) | ✅ | Full lifecycle automation |
| Drift detection | ✅ | PSI + KS on features and predictions |
| FastAPI serving | ✅ | Pre-computed forecasts, sub-10ms p99 |
| Streamlit dashboard | ✅ | 4-module business dashboard |

### What this platform does NOT claim
- ❌ Supply chain network simulation — M5 has no network data
- ❌ Carbon emissions analytics — M5 has no transport data
- ❌ Warehouse utilization — M5 has no warehouse data
- ❌ Profit optimization — M5 has no unit cost / COGS data
- ❌ Real inventory optimization — M5 has no inventory level data

---

## Quick Start

```bash
git clone https://github.com/your-org/m5-demand-intelligence.git
cd m5-demand-intelligence
cp .env.example .env

# Download M5 data (Kaggle CLI)
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/

# Launch infrastructure
docker-compose up -d postgres redis mlflow airflow-webserver airflow-scheduler

# Initialize database
python scripts/init_db.py

# Trigger full pipeline
python scripts/run_pipeline.py
```

## Services

| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |
| Airflow UI | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 |

---

## Dataset Assumptions (Documented)

When modules require data not in M5, assumptions are documented here:

| Assumption | Value | Source | Module |
|------------|-------|--------|--------|
| Store locations | CA→LA, TX→Dallas, WI→Milwaukee | Geographic inference | Weather joining |
| Unit cost | 60% of sell price | Retail industry benchmark | Inventory simulation |
| Ordering cost | $50/order | APICS benchmark | Inventory simulation |
| Holding cost rate | 25%/year | APICS benchmark | Inventory simulation |
| Lead time mean | 7 days | Typical DC-to-store | Inventory simulation |
| Lead time std | 1.5 days | Assumed | Inventory simulation |

---

## Architecture

```
M5 Sales + Weather (Open-Meteo) + Economics (FRED)
              │
              ▼ [Airflow: weekly]
    Validation (Pandera + PSI checks)
              │
              ▼
    Feature Store (45 features, DVC versioned)
              │
              ▼
    ML Models (MLflow tracked)
    ├── Baselines: Naive, Seasonal Naive, SES
    ├── Point: LightGBM (champion), XGBoost, CatBoost
    ├── Probabilistic: LightGBM quantile (q10/q50/q90)
    └── Hierarchical: MinT-reconciled across 6 levels
              │
    ┌─────────┴──────────┐
    ▼                    ▼
Analytics Layer     Inventory Simulation
├── Price elast.    (stated assumptions)
├── SNAP causal
└── Holiday lift
              │
              ▼
    FastAPI + Streamlit + Prometheus/Grafana
              │
              ▼
    MLOps Loop: Drift → Alert → Retrain → Promote
```
