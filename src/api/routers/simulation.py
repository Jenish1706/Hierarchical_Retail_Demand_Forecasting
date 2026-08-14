"""
Supply Chain Digital Twin Simulation API Router.
Provides discrete-event scenario simulation and stress testing.
"""
from __future__ import annotations

import uuid
from typing import Any
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.models import (
    APIResponse,
    SimulationKPISummary,
    SimulationResponse,
    SimulationScenarioRequest,
)
from src.simulation.digital_twin import SupplyChainDigitalTwin, SimulationConfig, DisruptionScenario
from src.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/run",
    response_model=APIResponse,
    summary="Run discrete-time Monte Carlo supply chain simulation",
)
async def run_simulation(req: SimulationScenarioRequest) -> APIResponse:
    """Run Monte Carlo digital twin simulation under specified disruption scenario."""
    config = SimulationConfig(
        n_days=req.n_days,
        n_monte_carlo=req.n_monte_carlo,
        random_seed=42,
    )
    disruptions = []
    if req.scenario_type != "baseline":
        disruptions.append(
            DisruptionScenario(
                scenario_type=req.scenario_type,
                start_day=req.disruption_start_day,
                duration_days=req.disruption_duration_days,
                severity=req.disruption_severity,
                target_node_id=req.affected_node_id or "STORE_CA_1",
            )
        )

    twin = SupplyChainDigitalTwin(config)
    results = twin.run_experiment(disruptions=disruptions)

    summary = SimulationKPISummary(
        scenario_name=req.scenario_name,
        service_level_mean=round(results["metrics"]["service_level_mean"], 4),
        fill_rate_p10=round(results["metrics"]["fill_rate_p10"], 4),
        fill_rate_p90=round(results["metrics"]["fill_rate_p90"], 4),
        total_cost_mean=round(results["metrics"]["total_cost_mean"], 2),
        total_cost_std=round(results["metrics"]["total_cost_std"], 2),
        stockout_risk_pct=round(results["metrics"]["stockout_risk_pct"], 2),
    )

    scenario_id = str(uuid.uuid4())[:8]
    res = SimulationResponse(
        scenario_id=scenario_id,
        scenario_name=req.scenario_name,
        status="completed",
        n_runs=req.n_monte_carlo,
        summary=summary,
        baseline_comparison={
            "service_level_drop_pct": round(0.985 - summary.service_level_mean, 4),
            "cost_increase_pct": round((summary.total_cost_mean - 10000.0) / 10000.0, 4),
        },
    )
    return APIResponse(success=True, data=res.model_dump())


@router.get(
    "/scenarios",
    response_model=APIResponse,
    summary="Get predefined stress testing scenario templates",
)
async def list_scenarios() -> APIResponse:
    """Get list of supported scenario templates."""
    scenarios = [
        {"name": "Baseline Normal Operations", "type": "baseline", "description": "Standard demand & lead times"},
        {"name": "Demand Spike (2x Holiday)", "type": "demand_spike", "description": "100% surge in daily demand"},
        {"name": "Supplier Lead Time Delay (+7d)", "type": "supplier_delay", "description": "Supplier lead time doubled"},
        {"name": "Warehouse Capacity Disruption", "type": "warehouse_failure", "description": "DC throughput reduced by 80%"},
    ]
    return APIResponse(success=True, data=scenarios)
