# Architecture Decision Records

Rewritten for v3. The previous file recorded decisions for modules that no
longer exist (NSGA-II via pymoo, PostgreSQL for a FastAPI service, a Streamlit
dashboard). Those ADRs are not amended here — they are gone, because the
decisions they justified were reversed wholesale when the project was rescoped
from a "supply chain platform" to a forecasting laboratory.

---

## ADR-001 — Remove the serving and non-ML layers

**Status:** Accepted

**Context.** The project began as a "supply chain intelligence platform" with a
FastAPI service, a Streamlit dashboard, a digital-twin simulation, supply-chain
graph analytics, carbon-emissions accounting, and NSGA-II multi-objective
inventory optimisation.

M5 contains four tables: wide-format unit sales, weekly prices, a calendar, and
a submission template. It has no supplier network, no transport legs, no
distances or weights, no warehouse metadata, no inventory levels, no lead times
and no unit costs. Every one of those modules was therefore producing outputs
derived entirely from invented constants.

**Decision.** Delete them. Delete the API and dashboard with them: this is a
research project whose deliverables are notebooks, figures and written analysis,
so a serving layer adds surface area without adding evidence.

**Consequences.**
- The scope is now exactly what the data supports.
- ~6,700 lines removed.
- Anything requiring absent data is stated as absent rather than simulated.
- Trade-off accepted: the project no longer demonstrates API or frontend skills.
  That is the right trade for a study whose subject is forecasting.

---

## ADR-002 — DuckDB and Parquet instead of PostgreSQL

**Status:** Accepted

**Context.** The original design ran a PostgreSQL server with eight schemas.
Every workload in this project is an analytical scan over a columnar panel:
melt 30,490 × 1,913 to long form, filter, aggregate, score. There is no
concurrent writer, no transactional requirement, and no service reading rows by
key.

**Decision.** Parquet on disk as the storage format, DuckDB for analytical
queries over it. No database server.

**Consequences.**
- `docker compose up` is no longer a prerequisite for running an experiment.
- The 229 MB workbook extracts once to 42 MB of Parquet; a filtered panel load
  takes under a second.
- DuckDB is an in-process library, so it is a dependency rather than a service.
- Trade-off: no concurrent multi-user access. Nothing here needs it.

---

## ADR-003 — Nested cross-validation for every tuned model

**Status:** Accepted

**Context.** The original trainer selected hyperparameters on the last CV fold
and then reported cross-validated performance using those same hyperparameters.
The fold that chose the parameters was also scored with them, so the reported
metric was optimistically biased — and the bias grows with the size of the
search space, which is exactly the regime Optuna operates in.

**Decision.** Nested CV. The outer `TimeSeriesSplit` reports performance; an
inner `TimeSeriesSplit` over the outer-training rows only selects
hyperparameters. The outer test fold is not merely excluded from fitting, it is
absent from the frame the sampler can see.

**Consequences.**
- Cost rises from `n_trials` fits to `k_outer × n_trials × k_inner`.
- The reported number is the number you get on unseen data.
- Reference: Cawley & Talbot (2010), *JMLR* 11:2079–2107.

---

## ADR-004 — Horizon-safe ("direct") features

**Status:** Accepted

**Context.** Features were built with `shift(1)`, which is correct for a model
re-run nightly with yesterday's actuals in hand. Evaluated over a 28-day
window, however, `lag_1` on test day 14 is the *actual* sales of test day 13 —
unknowable at the origin. The global ML models were effectively being scored on
28 one-step forecasts while the baselines projected the whole horizon from the
origin.

**Decision.** Shift every backward-looking feature by the full horizon, and drop
lags shorter than it. `FeaturePipeline(horizon=h)`.

**Consequences.**
- Measured cost of the fix: LightGBM WAPE moved 86.76 → 88.79, Tweedie
  81.95 → 85.72. Every per-series baseline was unchanged to the decimal, which
  is what proved the leak was real and confined to the global models.
- ~3.8 WAPE points of the previous headline were illusory.
- The comparison against from-origin baselines is now like-for-like.

---

## ADR-005 — Significance testing gates every ranking claim

**Status:** Accepted

**Context.** A leaderboard spanning 75.17 to 77.46 WAPE invites the reading that
the top model is better. Across thousands of heterogeneous, fat-tailed series a
gap that size is frequently noise.

**Decision.** No ranking is reported as a finding without a test behind it.
Friedman gates the post-hoc tests; Nemenyi and MCB run only if it rejects;
Diebold-Mariano covers pairs, always Holm-corrected.

**Consequences.**
- Produced a genuine negative result: `lightgbm` vs `lightgbm_tweedie` at
  adjusted p = 1.0000, despite Tweedie being the textbook choice for
  zero-inflated counts.
- Surfaced that pooled and rank-based tests disagree on the leader — which
  turned out to match a documented finding in the M5 paper itself.
- Reported as "no evidence of a difference", never "evidence of no difference".

---

## ADR-006 — `n_jobs=1` for statsforecast

**Status:** Accepted

**Context.** statsforecast exposes process-level parallelism via `n_jobs`, and
`-1` is the obvious default for 30,490 series.

Measured behaviour: at 40 series, every model took ~12 seconds with `n_jobs=-1`,
essentially all of it process-pool spin-up. ADIDA and Theta additionally raised
`BrokenProcessPool` on Windows, where the `spawn` start method re-imports the
module in each worker and requires a `__main__` guard.

With `n_jobs=1`, the same 15 models at 200 series ran in 0.01–14.8 seconds.

**Decision.** Default to 1. The estimators are already Numba-compiled and
vectorised across series inside one process, so the pool only helps at very
large series counts.

**Consequences.**
- Reproducible on Windows without a `__main__` guard.
- Anyone reproducing this work will hit the same failure; documented here and in
  the model card.
- Raise `n_jobs` deliberately for full-panel runs from a guarded script — not by
  default.

---

## ADR-007 — DVC as the only orchestrator

**Status:** Accepted

**Context.** The project carried an Airflow DAG with two schedules. Several of
its tasks (`run_inventory_optimization`, `ingest_weather`,
`generate_reorder_alerts`) invoked deleted modules; others returned hardcoded
dicts. Meanwhile `dvc.yaml` referenced scripts that were never written, so
neither orchestrator actually ran.

**Decision.** Delete Airflow. Repair `dvc.yaml` so every stage maps to a script
that exists and has been executed.

**Consequences.**
- One orchestrator, and it works: `dvc repro` reproduces extract → EDA → tier1 →
  backtest → significance.
- No scheduler, metadata database or webserver to run for a project with no
  production schedule.
- Trade-off: no time-based triggering. Nothing here needs it.
- The `significance` stage depends on the backtest predictions, so a stale
  verdict cannot be quoted against fresh numbers.

---

## ADR-008 — Deep learning is permitted, and is a hypothesis to test

**Status:** Accepted

**Context.** Earlier versions prohibited neural models, citing the M5 finding
that all top-50 entries were gradient-boosting based.

That finding is from a competition that closed in 2020. N-HiTS, PatchTST,
DLinear and TiDE were all published afterwards. Treating a 2020 result as a
standing prohibition converts an empirical finding into an assumption.

**Decision.** The prohibition is void. Deep learning enters as Tier 4 and is
scored through the same harness as everything else.

**Consequences.**
- `torch` and `neuralforecast` added.
- Given 95.2% sparsity and RMSSE converging near 1.0 at item-store level, the
  realistic expectation is that DL does not win at level 12. The question worth
  answering is *where* it is competitive — high-volume series and higher
  aggregation levels — rather than what the overall average says.
- DLinear specifically tests the Zeng et al. (2023) claim that one linear layer
  matches transformers on forecasting benchmarks.
