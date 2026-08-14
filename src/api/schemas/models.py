"""
Pydantic v2 schemas for all API request/response models.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Any = None
    correlation_id: str | None = None


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=1000)


# ---------------------------------------------------------------------------
# Forecast Schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    item_id: str = Field(..., description="Product/SKU identifier", example="HOBBIES_1_001")
    store_id: str = Field(..., description="Store identifier", example="CA_1")
    horizon_days: int = Field(default=28, ge=1, le=90, description="Forecast horizon in days")
    include_intervals: bool = Field(default=True, description="Return prediction intervals")
    quantiles: list[float] = Field(
        default=[0.10, 0.50, 0.90],
        description="Quantile levels for probabilistic forecast",
    )
    as_of_date: date | None = Field(default=None, description="Forecast as of this date (default: today)")

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: list[float]) -> list[float]:
        for q in v:
            if not 0.0 < q < 1.0:
                raise ValueError(f"Quantile {q} must be between 0 and 1")
        return sorted(v)


class ForecastPoint(BaseModel):
    date: date
    forecast_mean: float = Field(..., ge=0)
    forecast_p10: float | None = None
    forecast_p25: float | None = None
    forecast_p50: float | None = None
    forecast_p75: float | None = None
    forecast_p90: float | None = None


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    model_name: str
    model_version: int
    forecast_generated_at: date
    horizon_days: int
    forecasts: list[ForecastPoint]
    calibration_note: str | None = None


class BatchForecastRequest(BaseModel):
    items: list[ForecastRequest] = Field(..., max_length=500)
    model_version: int | None = None


# ---------------------------------------------------------------------------
# Inventory Schemas
# ---------------------------------------------------------------------------

class InventoryPolicyRequest(BaseModel):
    item_id: str
    store_id: str
    mean_daily_demand: float = Field(..., gt=0)
    std_daily_demand: float = Field(..., ge=0)
    mean_lead_time_days: float = Field(default=7.0, gt=0)
    std_lead_time_days: float = Field(default=1.5, ge=0)
    unit_cost: float = Field(default=10.0, gt=0)
    service_level: float = Field(default=0.95, gt=0, le=0.999)
    ordering_cost: float = Field(default=50.0, ge=0)
    holding_cost_rate: float = Field(default=0.25, gt=0, le=1.0)


class InventoryPolicyResponse(BaseModel):
    item_id: str
    store_id: str
    eoq: float
    safety_stock: float
    reorder_point: float
    order_up_to_level: float
    expected_annual_orders: float
    expected_holding_cost: float
    expected_ordering_cost: float
    expected_total_cost: float
    inventory_turnover: float
    service_level_target: float


class ReorderAlertResponse(BaseModel):
    item_id: str
    store_id: str
    on_hand_qty: float
    reorder_point: float
    safety_stock: float
    reorder_quantity: float
    urgency: str   # critical, high, normal
    reason: str
    expected_lead_time_days: int


class CurrentInventoryRequest(BaseModel):
    store_id: str | None = None
    category: str | None = None
    urgency_filter: str | None = Field(
        default=None,
        description="Filter by urgency: critical, high, normal",
    )
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=500)


# ---------------------------------------------------------------------------
# Pricing Schemas
# ---------------------------------------------------------------------------

class PricingRecommendationRequest(BaseModel):
    item_id: str
    store_id: str
    current_price: float = Field(..., gt=0)
    unit_cost: float = Field(..., gt=0)
    current_demand: float = Field(..., ge=0)
    strategy: str = Field(
        default="maximize_profit",
        pattern="^(maximize_revenue|maximize_profit|clear_inventory|penetration)$",
    )
    min_margin_pct: float = Field(default=0.10, ge=0, le=1.0)
    competitor_price: float | None = None
    max_price_increase_pct: float = Field(default=0.20, ge=0)
    max_price_decrease_pct: float = Field(default=0.30, ge=0)


class PricingRecommendationResponse(BaseModel):
    item_id: str
    store_id: str
    current_price: float
    recommended_price: float
    price_change_pct: float
    predicted_demand: float
    predicted_demand_change_pct: float
    predicted_revenue: float
    revenue_change: float
    predicted_profit: float
    profit_change: float
    elasticity_used: float
    strategy: str


class ElasticityResponse(BaseModel):
    item_id: str
    store_id: str | None
    elasticity: float
    ci_lower: float
    ci_upper: float
    p_value: float
    r_squared: float
    n_obs: int
    is_significant: bool
    is_elastic: bool
    method: str


# ---------------------------------------------------------------------------
# Simulation Schemas
# ---------------------------------------------------------------------------

class SimulationScenarioRequest(BaseModel):
    scenario_name: str = Field(..., description="Descriptive name for this scenario")
    scenario_type: str = Field(
        ...,
        pattern="^(demand_spike|supplier_delay|warehouse_failure|baseline)$",
    )
    n_days: int = Field(default=90, ge=7, le=365)
    n_monte_carlo: int = Field(default=50, ge=10, le=500)
    disruption_start_day: int = Field(default=30, ge=0)
    disruption_duration_days: int = Field(default=7, ge=1)
    disruption_severity: float = Field(default=0.8, ge=0, le=1.0)
    affected_node_id: str | None = None


class SimulationKPISummary(BaseModel):
    scenario_name: str
    service_level_mean: float
    fill_rate_p10: float
    fill_rate_p90: float
    total_cost_mean: float
    total_cost_std: float
    stockout_risk_pct: float


class SimulationResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    status: str
    n_runs: int
    summary: SimulationKPISummary
    baseline_comparison: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Optimization Schemas
# ---------------------------------------------------------------------------

class MultiObjectiveRequest(BaseModel):
    n_items: int = Field(..., ge=1, le=10000)
    weight_holding_cost: float = Field(default=0.20, ge=0, le=1)
    weight_stockout_cost: float = Field(default=0.35, ge=0, le=1)
    weight_service_level: float = Field(default=0.30, ge=0, le=1)
    weight_transportation: float = Field(default=0.10, ge=0, le=1)
    weight_emissions: float = Field(default=0.05, ge=0, le=1)
    n_pareto_solutions: int = Field(default=50, ge=5, le=200)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "MultiObjectiveRequest":
        total = (self.weight_holding_cost + self.weight_stockout_cost
                 + self.weight_service_level + self.weight_transportation
                 + self.weight_emissions)
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return self


class ParetoSolution(BaseModel):
    solution_id: int
    holding_cost: float
    stockout_cost: float
    service_level: float
    transportation_cost: float
    carbon_emissions: float
    total_cost: float


class ParetoFrontResponse(BaseModel):
    n_solutions: int
    solutions: list[ParetoSolution]
    recommended_solution_id: int
    optimization_method: str


# ---------------------------------------------------------------------------
# Monitoring Schemas
# ---------------------------------------------------------------------------

class ModelPerformanceResponse(BaseModel):
    model_id: str
    model_name: str
    evaluation_date: date
    mae: float
    rmse: float
    mape: float
    wape: float
    bias: float
    n_predictions: int
    is_degraded: bool
    degradation_pct: float | None


class DriftStatusResponse(BaseModel):
    model_id: str
    detection_date: date
    data_drift_features: int
    total_features_checked: int
    drift_pct: float
    action_recommended: str
    retraining_triggered: bool
