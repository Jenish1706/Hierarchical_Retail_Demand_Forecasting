"""
Multi-Objective Optimization API Router.
Provides NSGA-II Pareto optimization endpoints across cost, service level, and emissions.
"""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.models import (
    APIResponse,
    MultiObjectiveRequest,
    ParetoFrontResponse,
    ParetoSolution,
)
from src.optimization.multi_objective.nsga2 import MultiObjectiveInventoryOptimizer
from src.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/pareto-front",
    response_model=APIResponse,
    summary="Compute NSGA-II Pareto optimal front across cost, service, and emissions",
)
async def compute_pareto_front(req: MultiObjectiveRequest) -> APIResponse:
    """Run NSGA-II multi-objective optimization to generate non-dominated Pareto front."""
    weights = [
        req.weight_holding_cost,
        req.weight_stockout_cost,
        req.weight_service_level,
        req.weight_transportation,
        req.weight_emissions,
    ]
    optimizer = MultiObjectiveInventoryOptimizer(n_items=req.n_items)
    res_pareto = optimizer.optimize(
        n_solutions=req.n_pareto_solutions,
        weights=weights,
    )

    solutions = [
        ParetoSolution(
            solution_id=i,
            holding_cost=round(s["holding_cost"], 2),
            stockout_cost=round(s["stockout_cost"], 2),
            service_level=round(s["service_level"], 4),
            transportation_cost=round(s["transportation_cost"], 2),
            carbon_emissions=round(s["carbon_emissions"], 2),
            total_cost=round(s["total_cost"], 2),
        )
        for i, s in enumerate(res_pareto["solutions"])
    ]

    res = ParetoFrontResponse(
        n_solutions=len(solutions),
        solutions=solutions,
        recommended_solution_id=res_pareto.get("recommended_solution_id", 0),
        optimization_method="NSGA-II (via pymoo / fallback)",
    )
    return APIResponse(success=True, data=res.model_dump())
