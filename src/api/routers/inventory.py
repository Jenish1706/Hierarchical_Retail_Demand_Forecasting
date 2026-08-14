"""
Inventory Optimization API Router.
Provides EOQ, Safety Stock, Reorder Point, and Alerting endpoints.
"""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.models import (
    CurrentInventoryRequest,
    InventoryPolicyRequest,
    InventoryPolicyResponse,
    ReorderAlertResponse,
)
from src.optimization.inventory.optimizer import InventoryOptimizer, InventoryPolicy
from src.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/policy",
    response_model=InventoryPolicyResponse,
    summary="Compute EOQ, Safety Stock, and ROP",
)
async def compute_inventory_policy(req: InventoryPolicyRequest) -> InventoryPolicyResponse:
    """Compute optimal inventory policy parameters based on demand & lead time distribution."""
    policy = InventoryPolicy(
        item_id=req.item_id,
        store_id=req.store_id,
        mean_daily_demand=req.mean_daily_demand,
        std_daily_demand=req.std_daily_demand,
        annual_demand=req.mean_daily_demand * 365.0,
        mean_lead_time_days=req.mean_lead_time_days,
        std_lead_time_days=req.std_lead_time_days,
        unit_cost=req.unit_cost,
        holding_cost_rate=req.holding_cost_rate,
        ordering_cost=req.ordering_cost,
        service_level=req.service_level,
    )
    optimizer = InventoryOptimizer()
    policy = optimizer.optimize(policy)

    return InventoryPolicyResponse(
        item_id=req.item_id,
        store_id=req.store_id,
        eoq=round(policy.eoq, 2),
        safety_stock=round(policy.safety_stock, 2),
        reorder_point=round(policy.reorder_point, 2),
        order_up_to_level=round(policy.order_up_to_level, 2),
        expected_annual_orders=round(policy.expected_annual_orders, 2),
        expected_holding_cost=round(policy.expected_holding_cost, 2),
        expected_ordering_cost=round(policy.expected_ordering_cost, 2),
        expected_total_cost=round(policy.expected_total_cost, 2),
        inventory_turnover=round(policy.inventory_turnover, 2),
        service_level_target=req.service_level,
    )


@router.get(
    "/alerts",
    summary="Get reorder alerts across store inventory",
)
@router.get(
    "/reorder-alerts",
    summary="Get reorder alerts across store inventory",
)
async def get_reorder_alerts(
    store_id: str = Query(default="CA_1"),
    urgency: str | None = Query(default=None),
) -> dict[str, Any]:
    """List inventory items requiring reordering."""
    alerts = [
        ReorderAlertResponse(
            item_id="FOODS_1_001",
            store_id=store_id,
            on_hand_qty=12.0,
            reorder_point=45.0,
            safety_stock=18.0,
            reorder_quantity=150.0,
            urgency="critical",
            reason="On-hand stock (12) is below safety stock (18)",
            expected_lead_time_days=7,
        ),
        ReorderAlertResponse(
            item_id="HOBBIES_1_015",
            store_id=store_id,
            on_hand_qty=38.0,
            reorder_point=40.0,
            safety_stock=15.0,
            reorder_quantity=80.0,
            urgency="high",
            reason="On-hand stock (38) reached reorder point (40)",
            expected_lead_time_days=5,
        ),
    ]
    if urgency:
        alerts = [a for a in alerts if a.urgency == urgency]

    return {"alerts": [a.model_dump() for a in alerts]}


@router.post(
    "/current",
    summary="Get current inventory levels with pagination",
)
async def get_current_inventory(req: CurrentInventoryRequest) -> dict[str, Any]:
    """Fetch current inventory levels."""
    sample_inventory = [
        {
            "item_id": "FOODS_1_001",
            "store_id": req.store_id or "CA_1",
            "on_hand": 12,
            "on_order": 150,
            "safety_stock": 18,
            "reorder_point": 45,
            "urgency": "critical",
        },
        {
            "item_id": "HOBBIES_1_015",
            "store_id": req.store_id or "CA_1",
            "on_hand": 38,
            "on_order": 0,
            "safety_stock": 15,
            "reorder_point": 40,
            "urgency": "high",
        },
    ]
    return {
        "page": req.page,
        "page_size": req.page_size,
        "total_records": len(sample_inventory),
        "items": sample_inventory,
    }
