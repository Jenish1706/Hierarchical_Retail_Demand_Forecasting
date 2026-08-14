-- =============================================================================
-- INTELLIGENT SUPPLY CHAIN DECISION INTELLIGENCE PLATFORM
-- Database Schema v1.0.0
-- PostgreSQL 15+
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "timescaledb"; -- Optional: for time-series performance

-- =============================================================================
-- SCHEMA DEFINITIONS
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS raw;         -- Raw ingested data
CREATE SCHEMA IF NOT EXISTS processed;   -- Cleaned & transformed data
CREATE SCHEMA IF NOT EXISTS features;    -- Feature store
CREATE SCHEMA IF NOT EXISTS ml;          -- ML models & predictions
CREATE SCHEMA IF NOT EXISTS inventory;   -- Inventory optimization
CREATE SCHEMA IF NOT EXISTS pricing;     -- Pricing engine
CREATE SCHEMA IF NOT EXISTS monitoring;  -- MLOps & monitoring
CREATE SCHEMA IF NOT EXISTS simulation;  -- Digital twin simulation
CREATE SCHEMA IF NOT EXISTS audit;       -- Audit trail

-- =============================================================================
-- RAW SCHEMA: Original ingested data
-- =============================================================================

CREATE TABLE raw.m5_sales (
    id              BIGSERIAL PRIMARY KEY,
    item_id         VARCHAR(50) NOT NULL,
    dept_id         VARCHAR(50) NOT NULL,
    cat_id          VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    state_id        VARCHAR(10) NOT NULL,
    date            DATE NOT NULL,
    sales           DECIMAL(12, 4) NOT NULL,
    source_file     VARCHAR(255),
    ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_valid        BOOLEAN DEFAULT TRUE,
    validation_errors JSONB
);

CREATE INDEX idx_m5_sales_item_store_date ON raw.m5_sales (item_id, store_id, date);
CREATE INDEX idx_m5_sales_date ON raw.m5_sales (date);
CREATE INDEX idx_m5_sales_store ON raw.m5_sales (store_id);
CREATE INDEX idx_m5_sales_cat ON raw.m5_sales (cat_id, dept_id);

CREATE TABLE raw.m5_calendar (
    date            DATE PRIMARY KEY,
    wm_yr_wk        INTEGER NOT NULL,
    weekday         VARCHAR(20) NOT NULL,
    wday            SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    year            SMALLINT NOT NULL,
    event_name_1    VARCHAR(100),
    event_type_1    VARCHAR(50),
    event_name_2    VARCHAR(100),
    event_type_2    VARCHAR(50),
    snap_ca         SMALLINT DEFAULT 0,
    snap_tx         SMALLINT DEFAULT 0,
    snap_wi         SMALLINT DEFAULT 0,
    ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE raw.m5_sell_prices (
    id              BIGSERIAL PRIMARY KEY,
    store_id        VARCHAR(50) NOT NULL,
    item_id         VARCHAR(50) NOT NULL,
    wm_yr_wk        INTEGER NOT NULL,
    sell_price      DECIMAL(10, 4) NOT NULL,
    ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sell_prices_item_store ON raw.m5_sell_prices (item_id, store_id, wm_yr_wk);

CREATE TABLE raw.weather_data (
    id              BIGSERIAL PRIMARY KEY,
    store_id        VARCHAR(50) NOT NULL,
    date            DATE NOT NULL,
    temp_max        DECIMAL(5, 2),
    temp_min        DECIMAL(5, 2),
    precipitation   DECIMAL(7, 2),
    snow_depth      DECIMAL(7, 2),
    wind_speed      DECIMAL(5, 2),
    humidity        DECIMAL(5, 2),
    source          VARCHAR(100),
    ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_weather_store_date ON raw.weather_data (store_id, date);

CREATE TABLE raw.economic_indicators (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    cpi             DECIMAL(10, 4),
    unemployment    DECIMAL(5, 2),
    consumer_confidence DECIMAL(8, 2),
    gdp_growth      DECIMAL(6, 3),
    region          VARCHAR(50),
    source          VARCHAR(100),
    ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- PROCESSED SCHEMA: Cleaned and normalized data
-- =============================================================================

CREATE TABLE processed.stores (
    store_id        VARCHAR(50) PRIMARY KEY,
    state_id        VARCHAR(10) NOT NULL,
    store_name      VARCHAR(200),
    store_type      VARCHAR(50),
    size_sqft       INTEGER,
    latitude        DECIMAL(10, 7),
    longitude       DECIMAL(10, 7),
    region          VARCHAR(100),
    open_date       DATE,
    is_active       BOOLEAN DEFAULT TRUE,
    metadata        JSONB,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE processed.products (
    item_id         VARCHAR(50) PRIMARY KEY,
    dept_id         VARCHAR(50) NOT NULL,
    cat_id          VARCHAR(50) NOT NULL,
    item_name       VARCHAR(500),
    brand           VARCHAR(200),
    unit_weight_kg  DECIMAL(8, 3),
    unit_volume_m3  DECIMAL(10, 6),
    shelf_life_days INTEGER,
    is_perishable   BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    metadata        JSONB,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE processed.daily_sales (
    id              BIGSERIAL PRIMARY KEY,
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    date            DATE NOT NULL,
    sales           DECIMAL(12, 4) NOT NULL,
    sell_price      DECIMAL(10, 4),
    revenue         DECIMAL(14, 4),
    is_zero_sale    BOOLEAN,
    is_outlier      BOOLEAN DEFAULT FALSE,
    outlier_method  VARCHAR(50),
    original_sales  DECIMAL(12, 4),
    dept_id         VARCHAR(50),
    cat_id          VARCHAR(50),
    state_id        VARCHAR(10),
    CONSTRAINT uq_sales_item_store_date UNIQUE (item_id, store_id, date)
);

CREATE INDEX idx_daily_sales_item_store_date ON processed.daily_sales (item_id, store_id, date);
CREATE INDEX idx_daily_sales_date ON processed.daily_sales (date DESC);
CREATE INDEX idx_daily_sales_cat ON processed.daily_sales (cat_id, dept_id, date);

-- =============================================================================
-- FEATURES SCHEMA: Feature store
-- =============================================================================

CREATE TABLE features.feature_definitions (
    feature_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_name    VARCHAR(200) NOT NULL UNIQUE,
    feature_group   VARCHAR(100) NOT NULL,
    description     TEXT,
    dtype           VARCHAR(50) NOT NULL,
    source_table    VARCHAR(200),
    computation_sql TEXT,
    version         INTEGER DEFAULT 1,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE features.feature_store (
    id              BIGSERIAL PRIMARY KEY,
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    date            DATE NOT NULL,
    feature_version INTEGER NOT NULL DEFAULT 1,
    -- Time features
    day_of_week     SMALLINT,
    day_of_month    SMALLINT,
    week_of_year    SMALLINT,
    month           SMALLINT,
    quarter         SMALLINT,
    year            SMALLINT,
    is_weekend      BOOLEAN,
    is_holiday      BOOLEAN,
    holiday_name    VARCHAR(100),
    days_to_holiday SMALLINT,
    days_after_holiday SMALLINT,
    -- Lag features
    lag_1           DECIMAL(12, 4),
    lag_7           DECIMAL(12, 4),
    lag_14          DECIMAL(12, 4),
    lag_28          DECIMAL(12, 4),
    lag_90          DECIMAL(12, 4),
    -- Rolling features
    rolling_mean_7  DECIMAL(12, 4),
    rolling_mean_28 DECIMAL(12, 4),
    rolling_mean_90 DECIMAL(12, 4),
    rolling_std_7   DECIMAL(12, 4),
    rolling_std_28  DECIMAL(12, 4),
    rolling_max_7   DECIMAL(12, 4),
    rolling_min_7   DECIMAL(12, 4),
    -- Price features
    sell_price      DECIMAL(10, 4),
    price_change    DECIMAL(10, 4),
    price_change_pct DECIMAL(8, 4),
    relative_price  DECIMAL(8, 4),
    discount_pct    DECIMAL(8, 4),
    is_on_sale      BOOLEAN,
    -- Demand features
    demand_velocity_7  DECIMAL(12, 4),
    demand_volatility_7 DECIMAL(12, 4),
    -- Event features
    snap_flag       BOOLEAN,
    event_type_1    VARCHAR(50),
    event_type_2    VARCHAR(50),
    -- Weather features
    temp_max        DECIMAL(5, 2),
    temp_min        DECIMAL(5, 2),
    precipitation   DECIMAL(7, 2),
    -- Economic features
    cpi             DECIMAL(10, 4),
    unemployment    DECIMAL(5, 2),
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_features_item_store_date UNIQUE (item_id, store_id, date, feature_version)
);

CREATE INDEX idx_features_item_store_date ON features.feature_store (item_id, store_id, date);
CREATE INDEX idx_features_date ON features.feature_store (date DESC);

-- =============================================================================
-- ML SCHEMA: Models, experiments, and predictions
-- =============================================================================

CREATE TABLE ml.experiments (
    experiment_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlflow_run_id   VARCHAR(100) UNIQUE,
    experiment_name VARCHAR(200) NOT NULL,
    model_type      VARCHAR(100) NOT NULL,
    description     TEXT,
    status          VARCHAR(50) DEFAULT 'running',  -- running, completed, failed
    started_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE,
    parameters      JSONB,
    metrics         JSONB,
    tags            JSONB,
    artifact_uri    VARCHAR(500),
    created_by      VARCHAR(100)
);

CREATE TABLE ml.model_registry (
    model_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name      VARCHAR(200) NOT NULL,
    model_version   INTEGER NOT NULL,
    model_type      VARCHAR(100) NOT NULL,
    mlflow_model_uri VARCHAR(500),
    experiment_id   UUID REFERENCES ml.experiments(experiment_id),
    stage           VARCHAR(50) DEFAULT 'Staging',  -- Staging, Production, Archived
    metrics         JSONB,
    parameters      JSONB,
    feature_names   JSONB,
    training_cutoff DATE,
    validation_start DATE,
    validation_end  DATE,
    is_champion     BOOLEAN DEFAULT FALSE,
    deployed_at     TIMESTAMP WITH TIME ZONE,
    retired_at      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_model_version UNIQUE (model_name, model_version)
);

CREATE TABLE ml.predictions (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   UUID DEFAULT uuid_generate_v4(),
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    forecast_date   DATE NOT NULL,
    prediction_date DATE NOT NULL,  -- when prediction was made
    horizon_days    SMALLINT NOT NULL,
    model_id        UUID REFERENCES ml.model_registry(model_id),
    -- Point forecasts
    forecast_mean   DECIMAL(12, 4) NOT NULL,
    -- Probabilistic
    forecast_p10    DECIMAL(12, 4),
    forecast_p25    DECIMAL(12, 4),
    forecast_p50    DECIMAL(12, 4),
    forecast_p75    DECIMAL(12, 4),
    forecast_p90    DECIMAL(12, 4),
    -- Actuals (filled in post-hoc)
    actual_sales    DECIMAL(12, 4),
    -- Errors
    mae             DECIMAL(12, 6),
    rmse            DECIMAL(12, 6),
    mape            DECIMAL(12, 6),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_predictions_item_store ON ml.predictions (item_id, store_id, forecast_date);
CREATE INDEX idx_predictions_forecast_date ON ml.predictions (forecast_date DESC);

CREATE TABLE ml.hierarchical_forecasts (
    id              BIGSERIAL PRIMARY KEY,
    level           VARCHAR(50) NOT NULL,  -- item, dept, cat, store, state, total
    entity_id       VARCHAR(100) NOT NULL,
    forecast_date   DATE NOT NULL,
    prediction_date DATE NOT NULL,
    horizon_days    SMALLINT NOT NULL,
    forecast_mean   DECIMAL(14, 4) NOT NULL,
    forecast_p10    DECIMAL(14, 4),
    forecast_p90    DECIMAL(14, 4),
    reconciled      BOOLEAN DEFAULT FALSE,
    reconciliation_method VARCHAR(50),
    model_id        UUID REFERENCES ml.model_registry(model_id),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- INVENTORY SCHEMA: Optimization results
-- =============================================================================

CREATE TABLE inventory.policies (
    id              BIGSERIAL PRIMARY KEY,
    policy_id       UUID DEFAULT uuid_generate_v4(),
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    effective_date  DATE NOT NULL,
    -- EOQ calculations
    annual_demand   DECIMAL(12, 4),
    ordering_cost   DECIMAL(10, 4),
    holding_cost_rate DECIMAL(6, 4),
    unit_cost       DECIMAL(10, 4),
    eoq             DECIMAL(12, 4),
    -- Safety stock
    demand_std_dev  DECIMAL(12, 4),
    lead_time_days  SMALLINT,
    lead_time_std   DECIMAL(6, 4),
    service_level_target DECIMAL(5, 4),
    z_score         DECIMAL(6, 4),
    safety_stock    DECIMAL(12, 4),
    -- Reorder point
    reorder_point   DECIMAL(12, 4),
    -- Max stock level
    max_stock       DECIMAL(12, 4),
    -- Metrics
    expected_holding_cost DECIMAL(12, 4),
    expected_stockout_cost DECIMAL(12, 4),
    expected_total_cost DECIMAL(12, 4),
    -- Status
    is_active       BOOLEAN DEFAULT TRUE,
    review_frequency VARCHAR(50) DEFAULT 'weekly',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_inventory_item_store_date UNIQUE (item_id, store_id, effective_date)
);

CREATE TABLE inventory.current_levels (
    id              BIGSERIAL PRIMARY KEY,
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    snapshot_date   DATE NOT NULL,
    on_hand_qty     DECIMAL(12, 4) NOT NULL,
    on_order_qty    DECIMAL(12, 4) DEFAULT 0,
    in_transit_qty  DECIMAL(12, 4) DEFAULT 0,
    reserved_qty    DECIMAL(12, 4) DEFAULT 0,
    available_qty   DECIMAL(12, 4),
    days_of_supply  DECIMAL(8, 2),
    reorder_flag    BOOLEAN DEFAULT FALSE,
    stockout_flag   BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_inventory_level UNIQUE (item_id, store_id, snapshot_date)
);

CREATE TABLE inventory.reorder_recommendations (
    id              BIGSERIAL PRIMARY KEY,
    recommendation_id UUID DEFAULT uuid_generate_v4(),
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    recommendation_date DATE NOT NULL,
    reorder_quantity DECIMAL(12, 4) NOT NULL,
    expected_lead_time_days SMALLINT,
    expected_delivery_date DATE,
    urgency_level   VARCHAR(20) DEFAULT 'normal',  -- critical, high, normal, low
    reason          TEXT,
    estimated_cost  DECIMAL(12, 4),
    status          VARCHAR(50) DEFAULT 'pending',  -- pending, approved, ordered, received
    approved_by     VARCHAR(100),
    approved_at     TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- PRICING SCHEMA
-- =============================================================================

CREATE TABLE pricing.elasticity_estimates (
    id              BIGSERIAL PRIMARY KEY,
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50),  -- NULL means all stores
    cat_id          VARCHAR(50),
    estimation_date DATE NOT NULL,
    method          VARCHAR(100) NOT NULL,  -- OLS, IV, DoubleML
    elasticity      DECIMAL(8, 4) NOT NULL,
    elasticity_se   DECIMAL(8, 4),
    elasticity_ci_lower DECIMAL(8, 4),
    elasticity_ci_upper DECIMAL(8, 4),
    p_value         DECIMAL(8, 6),
    r_squared       DECIMAL(6, 4),
    n_observations  INTEGER,
    model_id        UUID,
    is_significant  BOOLEAN,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE pricing.recommendations (
    id              BIGSERIAL PRIMARY KEY,
    rec_id          UUID DEFAULT uuid_generate_v4(),
    item_id         VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    recommendation_date DATE NOT NULL,
    current_price   DECIMAL(10, 4) NOT NULL,
    recommended_price DECIMAL(10, 4) NOT NULL,
    price_change_pct DECIMAL(8, 4),
    -- Predicted impact
    predicted_demand_change DECIMAL(8, 4),
    predicted_revenue_change DECIMAL(12, 4),
    predicted_profit_change DECIMAL(12, 4),
    -- Objective
    objective       VARCHAR(50),  -- maximize_revenue, maximize_profit, clear_inventory
    -- Constraints applied
    min_margin_pct  DECIMAL(6, 4),
    competitor_price DECIMAL(10, 4),
    -- Status
    status          VARCHAR(50) DEFAULT 'pending',
    applied_at      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- MONITORING SCHEMA: MLOps & drift detection
-- =============================================================================

CREATE TABLE monitoring.data_quality_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID DEFAULT uuid_generate_v4(),
    pipeline_name   VARCHAR(200) NOT NULL,
    run_date        DATE NOT NULL,
    dataset         VARCHAR(200) NOT NULL,
    total_rows      INTEGER,
    passed_checks   INTEGER,
    failed_checks   INTEGER,
    warning_checks  INTEGER,
    overall_status  VARCHAR(50),  -- passed, failed, warning
    results         JSONB,
    run_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE monitoring.model_performance (
    id              BIGSERIAL PRIMARY KEY,
    model_id        UUID REFERENCES ml.model_registry(model_id),
    evaluation_date DATE NOT NULL,
    evaluation_period VARCHAR(50),  -- daily, weekly, monthly
    n_predictions   INTEGER,
    mae             DECIMAL(12, 6),
    rmse            DECIMAL(12, 6),
    mape            DECIMAL(12, 6),
    wape            DECIMAL(12, 6),
    bias            DECIMAL(12, 6),
    coverage_80     DECIMAL(6, 4),
    coverage_95     DECIMAL(6, 4),
    is_degraded     BOOLEAN DEFAULT FALSE,
    degradation_pct DECIMAL(8, 4),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE monitoring.drift_detections (
    id              BIGSERIAL PRIMARY KEY,
    detection_id    UUID DEFAULT uuid_generate_v4(),
    model_id        UUID REFERENCES ml.model_registry(model_id),
    detection_date  DATE NOT NULL,
    drift_type      VARCHAR(50) NOT NULL,  -- data_drift, concept_drift, prediction_drift
    feature_name    VARCHAR(200),
    drift_score     DECIMAL(8, 6),
    drift_detected  BOOLEAN NOT NULL,
    threshold       DECIMAL(8, 6),
    statistical_test VARCHAR(100),
    p_value         DECIMAL(10, 8),
    action_taken    VARCHAR(200),
    details         JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE monitoring.retraining_events (
    id              BIGSERIAL PRIMARY KEY,
    event_id        UUID DEFAULT uuid_generate_v4(),
    trigger_type    VARCHAR(100) NOT NULL,  -- scheduled, drift_detected, performance_degraded
    trigger_details JSONB,
    old_model_id    UUID REFERENCES ml.model_registry(model_id),
    new_model_id    UUID REFERENCES ml.model_registry(model_id),
    status          VARCHAR(50) DEFAULT 'initiated',
    started_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE,
    improvement_pct DECIMAL(8, 4),
    notes           TEXT
);

CREATE TABLE monitoring.alerts (
    id              BIGSERIAL PRIMARY KEY,
    alert_id        UUID DEFAULT uuid_generate_v4(),
    alert_type      VARCHAR(100) NOT NULL,
    severity        VARCHAR(20) NOT NULL,  -- critical, warning, info
    title           VARCHAR(500) NOT NULL,
    message         TEXT,
    entity_type     VARCHAR(50),
    entity_id       VARCHAR(100),
    metric_name     VARCHAR(200),
    metric_value    DECIMAL(15, 6),
    threshold       DECIMAL(15, 6),
    status          VARCHAR(50) DEFAULT 'open',  -- open, acknowledged, resolved
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at     TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- SIMULATION SCHEMA: Digital twin
-- =============================================================================

CREATE TABLE simulation.scenarios (
    id              BIGSERIAL PRIMARY KEY,
    scenario_id     UUID DEFAULT uuid_generate_v4(),
    scenario_name   VARCHAR(200) NOT NULL,
    scenario_type   VARCHAR(100) NOT NULL,  -- demand_spike, supply_disruption, etc.
    description     TEXT,
    parameters      JSONB NOT NULL,
    start_date      DATE,
    end_date        DATE,
    status          VARCHAR(50) DEFAULT 'pending',
    created_by      VARCHAR(100),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE simulation.runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID DEFAULT uuid_generate_v4(),
    scenario_id     UUID REFERENCES simulation.scenarios(scenario_id),
    run_number      INTEGER NOT NULL,
    status          VARCHAR(50) DEFAULT 'running',
    started_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE,
    results         JSONB,
    kpis            JSONB,
    seed            INTEGER
);

CREATE TABLE simulation.results (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID REFERENCES simulation.runs(run_id),
    simulation_date DATE NOT NULL,
    entity_type     VARCHAR(50) NOT NULL,  -- store, warehouse, distribution_center
    entity_id       VARCHAR(100) NOT NULL,
    item_id         VARCHAR(50),
    -- KPIs at this timestep
    demand          DECIMAL(12, 4),
    supply          DECIMAL(12, 4),
    inventory_level DECIMAL(12, 4),
    fill_rate       DECIMAL(6, 4),
    stockout        BOOLEAN,
    backorder_qty   DECIMAL(12, 4),
    cost            DECIMAL(12, 4),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- AUDIT SCHEMA
-- =============================================================================

CREATE TABLE audit.pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID DEFAULT uuid_generate_v4(),
    pipeline_name   VARCHAR(200) NOT NULL,
    dag_run_id      VARCHAR(500),
    status          VARCHAR(50) NOT NULL,
    started_at      TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at    TIMESTAMP WITH TIME ZONE,
    records_processed BIGINT,
    error_message   TEXT,
    metadata        JSONB
);

CREATE TABLE audit.api_requests (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID DEFAULT uuid_generate_v4(),
    endpoint        VARCHAR(200) NOT NULL,
    method          VARCHAR(10) NOT NULL,
    status_code     SMALLINT,
    response_time_ms INTEGER,
    user_id         VARCHAR(100),
    ip_address      INET,
    request_body    JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- VIEWS: Business-facing views
-- =============================================================================

CREATE OR REPLACE VIEW processed.v_forecast_accuracy AS
SELECT
    p.item_id,
    p.store_id,
    p.forecast_date,
    p.forecast_mean,
    p.actual_sales,
    ABS(p.actual_sales - p.forecast_mean) AS abs_error,
    CASE WHEN p.actual_sales > 0
         THEN ABS(p.actual_sales - p.forecast_mean) / p.actual_sales * 100
         ELSE NULL END AS mape,
    mr.model_name,
    mr.model_version,
    mr.stage
FROM ml.predictions p
JOIN ml.model_registry mr ON p.model_id = mr.model_id
WHERE p.actual_sales IS NOT NULL;

CREATE OR REPLACE VIEW inventory.v_reorder_alerts AS
SELECT
    cl.item_id,
    cl.store_id,
    cl.snapshot_date,
    cl.on_hand_qty,
    cl.available_qty,
    ip.reorder_point,
    ip.safety_stock,
    ip.eoq,
    CASE
        WHEN cl.on_hand_qty <= ip.safety_stock THEN 'CRITICAL'
        WHEN cl.on_hand_qty <= ip.reorder_point THEN 'REORDER_NOW'
        WHEN cl.on_hand_qty <= ip.reorder_point * 1.2 THEN 'REORDER_SOON'
        ELSE 'ADEQUATE'
    END AS inventory_status,
    cl.days_of_supply
FROM inventory.current_levels cl
JOIN inventory.policies ip ON cl.item_id = ip.item_id
    AND cl.store_id = ip.store_id
    AND ip.is_active = TRUE
WHERE cl.snapshot_date = CURRENT_DATE;

-- =============================================================================
-- FUNCTIONS: Utility functions
-- =============================================================================

CREATE OR REPLACE FUNCTION processed.calculate_mape(
    p_actual DECIMAL,
    p_forecast DECIMAL
) RETURNS DECIMAL AS $$
BEGIN
    IF p_actual IS NULL OR p_actual = 0 THEN RETURN NULL; END IF;
    RETURN ABS(p_actual - p_forecast) / p_actual * 100;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION inventory.calculate_eoq(
    p_annual_demand DECIMAL,
    p_ordering_cost DECIMAL,
    p_holding_cost_per_unit DECIMAL
) RETURNS DECIMAL AS $$
BEGIN
    IF p_holding_cost_per_unit = 0 THEN RETURN NULL; END IF;
    RETURN SQRT(2 * p_annual_demand * p_ordering_cost / p_holding_cost_per_unit);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- TRIGGERS: Automatic timestamping
-- =============================================================================

CREATE OR REPLACE FUNCTION audit.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_stores_updated
    BEFORE UPDATE ON processed.stores
    FOR EACH ROW EXECUTE FUNCTION audit.update_timestamp();

CREATE TRIGGER trg_products_updated
    BEFORE UPDATE ON processed.products
    FOR EACH ROW EXECUTE FUNCTION audit.update_timestamp();
