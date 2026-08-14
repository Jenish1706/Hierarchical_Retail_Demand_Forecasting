"""
Supply Chain Digital Twin Simulation.

Simulates the full supply chain network as an event-driven discrete system:
  Supplier → Warehouse → Distribution Center → Retail Store

Supports disruption scenarios:
  - Demand spike (Black Friday, viral product)
  - Supplier delay (port congestion, factory shutdown)
  - Warehouse failure (fire, flood, strike)
  - Transportation disruption (truck shortage, fuel crisis)

Design:
  - Entity-based simulation (each node is an agent)
  - Discrete time steps (daily)
  - Monte Carlo: run N simulations → distribution of outcomes
  - KPI tracking at each timestep
"""
from __future__ import annotations

import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Entity Definitions
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    SUPPLIER = "supplier"
    WAREHOUSE = "warehouse"
    DISTRIBUTION_CENTER = "distribution_center"
    STORE = "store"


@dataclass
class SupplyChainNode:
    """A node in the supply chain network."""
    node_id: str
    node_type: NodeType
    name: str
    capacity: float = float("inf")   # Max inventory units
    initial_inventory: float = 0.0
    lead_time_days: int = 3          # Days to receive from upstream
    lead_time_std: float = 0.5       # Variability in lead time
    is_active: bool = True
    location: tuple[float, float] = (0.0, 0.0)  # lat, lon

    # Runtime state
    current_inventory: float = field(init=False)
    orders_in_transit: list[dict] = field(default_factory=list)
    total_demand_served: float = field(default=0.0)
    total_stockouts: int = field(default=0)
    total_holding_cost: float = field(default=0.0)
    total_shortage_cost: float = field(default=0.0)

    def __post_init__(self):
        self.current_inventory = self.initial_inventory

    @property
    def fill_rate(self) -> float:
        total = self.total_demand_served + self.total_stockouts
        return self.total_demand_served / total if total > 0 else 1.0


@dataclass
class DisruptionEvent:
    """A disruption that affects supply chain performance."""
    start_day: int
    duration_days: int
    affected_node_id: str
    disruption_type: str            # "delay", "capacity_reduction", "shutdown"
    severity: float = 1.0           # 0=no effect, 1=total disruption
    demand_multiplier: float = 1.0  # For demand spikes
    supply_multiplier: float = 1.0  # For supply restrictions


@dataclass
class SimulationConfig:
    """Configuration for one simulation run."""
    n_days: int = 90
    n_monte_carlo: int = 100
    holding_cost_per_unit_per_day: float = 0.05
    shortage_cost_per_unit: float = 5.0
    ordering_cost: float = 50.0
    random_seed: int = 42
    demand_noise_std: float = 0.15   # 15% daily demand noise


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class SupplyChainSimulator:
    """
    Discrete-time supply chain simulator.

    Usage:
        sim = SupplyChainSimulator(nodes, edges, config)
        results = sim.run(demand_forecast, disruptions=[...])
        summary = sim.summarize(results)
    """

    def __init__(
        self,
        nodes: list[SupplyChainNode],
        edges: list[tuple[str, str]],   # (upstream_id, downstream_id)
        config: SimulationConfig,
    ):
        self.nodes = {n.node_id: n for n in nodes}
        self.edges = edges
        self.config = config
        self._upstream: dict[str, str] = {d: u for u, d in edges}
        self._downstream: dict[str, list[str]] = {}
        for u, d in edges:
            self._downstream.setdefault(u, []).append(d)
        self._log = get_logger("SupplyChainSimulator")

    def run(
        self,
        demand_by_store: dict[str, np.ndarray],  # {store_id: [daily_demand]}
        disruptions: list[DisruptionEvent] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Run a single simulation pass. Returns daily KPI DataFrame.
        """
        rng = np.random.default_rng(seed or self.config.random_seed)
        disruptions = disruptions or []
        nodes = deepcopy(self.nodes)
        records = []

        # Map disruptions by day
        disruption_map: dict[int, list[DisruptionEvent]] = {}
        for d in disruptions:
            for day in range(d.start_day, d.start_day + d.duration_days):
                disruption_map.setdefault(day, []).append(d)

        for day in range(self.config.n_days):
            active_disruptions = disruption_map.get(day, [])
            disrupted_nodes = {d.affected_node_id: d for d in active_disruptions}

            # Process each node
            for node_id, node in nodes.items():
                if not node.is_active:
                    continue

                disruption = disrupted_nodes.get(node_id)

                # ---- Receive pending shipments ----
                received = 0.0
                still_in_transit = []
                for order in node.orders_in_transit:
                    if day >= order["arrive_day"]:
                        # Apply supply disruption if active
                        supply_mult = disruption.supply_multiplier if disruption else 1.0
                        received += order["quantity"] * supply_mult
                    else:
                        still_in_transit.append(order)
                node.orders_in_transit = still_in_transit
                node.current_inventory = min(
                    node.current_inventory + received, node.capacity
                )

                # ---- Satisfy demand ----
                if node.node_type == NodeType.STORE:
                    base_demand = float(
                        demand_by_store.get(node_id, np.zeros(self.config.n_days))[day]
                        if node_id in demand_by_store
                        else rng.poisson(10)
                    )
                    # Add noise
                    noise = rng.normal(1.0, self.config.demand_noise_std)
                    demand_mult = disruption.demand_multiplier if disruption else 1.0
                    actual_demand = max(0.0, base_demand * noise * demand_mult)

                    served = min(actual_demand, node.current_inventory)
                    stockout = actual_demand - served
                    node.current_inventory -= served
                    node.total_demand_served += served
                    node.total_holding_cost += node.current_inventory * self.config.holding_cost_per_unit_per_day
                    node.total_shortage_cost += stockout * self.config.shortage_cost_per_unit
                    if stockout > 0:
                        node.total_stockouts += 1

                    records.append({
                        "day": day,
                        "node_id": node_id,
                        "node_type": node.node_type.value,
                        "demand": actual_demand,
                        "demand_served": served,
                        "stockout_qty": stockout,
                        "inventory_level": node.current_inventory,
                        "fill_rate": served / actual_demand if actual_demand > 0 else 1.0,
                        "holding_cost": node.current_inventory * self.config.holding_cost_per_unit_per_day,
                        "shortage_cost": stockout * self.config.shortage_cost_per_unit,
                        "is_disrupted": disruption is not None,
                        "disruption_type": disruption.disruption_type if disruption else None,
                    })

                # ---- Reorder logic (simple ROP-based) ----
                if node.node_type in (NodeType.STORE, NodeType.DISTRIBUTION_CENTER):
                    upstream_id = self._upstream.get(node_id)
                    if upstream_id and upstream_id in nodes:
                        upstream = nodes[upstream_id]
                        # Simple: reorder if below 7-day supply
                        daily_avg = float(
                            demand_by_store.get(node_id, np.array([10]))[max(0, day-7):day].mean()
                            if node_id in demand_by_store else 10
                        )
                        reorder_point = daily_avg * node.lead_time_days * 1.5
                        if node.current_inventory < reorder_point:
                            order_qty = daily_avg * 14  # 2-week order
                            actual_lt = max(1, int(rng.normal(node.lead_time_days, node.lead_time_std)))
                            if disruption and disruption.disruption_type == "delay":
                                actual_lt = int(actual_lt * (1 + disruption.severity * 2))
                            upstream.orders_in_transit.append({
                                "quantity": order_qty,
                                "arrive_day": day + actual_lt,
                            })

        return pd.DataFrame(records)

    def run_monte_carlo(
        self,
        demand_by_store: dict[str, np.ndarray],
        disruptions: list[DisruptionEvent] | None = None,
        n_runs: int | None = None,
    ) -> dict[str, Any]:
        """
        Run N Monte Carlo simulations and aggregate KPI distributions.
        """
        n_runs = n_runs or self.config.n_monte_carlo
        all_results = []
        kpi_runs = []

        for i in range(n_runs):
            df_run = self.run(demand_by_store, disruptions, seed=i)
            all_results.append(df_run)

            # Summary KPIs per run
            stores = df_run[df_run["node_type"] == "store"]
            kpi_runs.append({
                "run": i,
                "mean_fill_rate": stores["fill_rate"].mean(),
                "total_stockout_days": (stores["stockout_qty"] > 0).sum(),
                "total_holding_cost": stores["holding_cost"].sum(),
                "total_shortage_cost": stores["shortage_cost"].sum(),
                "total_demand": stores["demand"].sum(),
                "total_demand_served": stores["demand_served"].sum(),
            })

        df_kpis = pd.DataFrame(kpi_runs)

        return {
            "kpi_runs": df_kpis,
            "summary": {
                "fill_rate_p50": float(df_kpis["mean_fill_rate"].quantile(0.5)),
                "fill_rate_p10": float(df_kpis["mean_fill_rate"].quantile(0.1)),
                "fill_rate_p90": float(df_kpis["mean_fill_rate"].quantile(0.9)),
                "total_cost_mean": float(
                    (df_kpis["total_holding_cost"] + df_kpis["total_shortage_cost"]).mean()
                ),
                "total_cost_std": float(
                    (df_kpis["total_holding_cost"] + df_kpis["total_shortage_cost"]).std()
                ),
                "service_level_mean": float(df_kpis["mean_fill_rate"].mean()),
                "stockout_risk": float((df_kpis["total_stockout_days"] > 0).mean()),
            },
            "all_runs": all_results,
        }

    def compare_scenarios(
        self,
        demand_by_store: dict[str, np.ndarray],
        scenarios: dict[str, list[DisruptionEvent]],
    ) -> pd.DataFrame:
        """
        Compare KPIs across multiple disruption scenarios.

        Args:
            scenarios: {"scenario_name": [DisruptionEvent, ...]}

        Returns:
            DataFrame comparing key KPIs across scenarios
        """
        rows = []
        for scenario_name, disruptions in scenarios.items():
            self._log.info("running_scenario", name=scenario_name)
            mc_result = self.run_monte_carlo(
                demand_by_store, disruptions, n_runs=50
            )
            summary = mc_result["summary"]
            rows.append({
                "scenario": scenario_name,
                "service_level_mean": round(summary["service_level_mean"] * 100, 2),
                "fill_rate_p10": round(summary["fill_rate_p10"] * 100, 2),
                "fill_rate_p90": round(summary["fill_rate_p90"] * 100, 2),
                "total_cost_mean": round(summary["total_cost_mean"], 2),
                "total_cost_std": round(summary["total_cost_std"], 2),
                "stockout_risk_pct": round(summary["stockout_risk"] * 100, 2),
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-built Scenarios
# ---------------------------------------------------------------------------

def demand_spike_scenario(
    store_id: str, day: int, multiplier: float = 3.0, duration: int = 3
) -> list[DisruptionEvent]:
    """Sudden demand spike (Black Friday, viral product)."""
    return [DisruptionEvent(
        start_day=day, duration_days=duration,
        affected_node_id=store_id,
        disruption_type="demand_spike",
        demand_multiplier=multiplier,
    )]


def supplier_delay_scenario(
    warehouse_id: str, day: int, delay_severity: float = 0.8, duration: int = 14
) -> list[DisruptionEvent]:
    """Supplier shipment delay (port congestion, strikes)."""
    return [DisruptionEvent(
        start_day=day, duration_days=duration,
        affected_node_id=warehouse_id,
        disruption_type="delay",
        severity=delay_severity,
        supply_multiplier=1 - delay_severity,
    )]


def warehouse_failure_scenario(
    warehouse_id: str, day: int, capacity_loss: float = 1.0, duration: int = 7
) -> list[DisruptionEvent]:
    """Warehouse capacity loss (fire, flood, system failure)."""
    return [DisruptionEvent(
        start_day=day, duration_days=duration,
        affected_node_id=warehouse_id,
        disruption_type="shutdown",
        severity=capacity_loss,
        supply_multiplier=1 - capacity_loss,
    )]


# Aliases & High-level Wrapper for API compatibility
DisruptionScenario = DisruptionEvent


class SupplyChainDigitalTwin:
    """High-level Digital Twin interface wrapping SupplyChainSimulator."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.default_nodes = [
            SupplyChainNode("SUPPLIER_1", NodeType.SUPPLIER, "Main Supplier", capacity=10000.0, initial_inventory=5000.0),
            SupplyChainNode("WH_1", NodeType.WAREHOUSE, "Regional DC", capacity=5000.0, initial_inventory=2500.0),
            SupplyChainNode("STORE_CA_1", NodeType.STORE, "LA Store", capacity=1000.0, initial_inventory=500.0),
        ]
        self.default_edges = [("SUPPLIER_1", "WH_1"), ("WH_1", "STORE_CA_1")]
        self.simulator = SupplyChainSimulator(self.default_nodes, self.default_edges, self.config)

    def run_experiment(self, disruptions: list[DisruptionEvent] | None = None) -> dict[str, Any]:
        demand = {"STORE_CA_1": np.full(self.config.n_days, 20.0)}
        mc = self.simulator.run_monte_carlo(demand, disruptions=disruptions, n_runs=self.config.n_monte_carlo)
        return {
            "metrics": {
                "service_level_mean": mc["summary"]["service_level_mean"],
                "fill_rate_p10": mc["summary"]["fill_rate_p10"],
                "fill_rate_p90": mc["summary"]["fill_rate_p90"],
                "total_cost_mean": mc["summary"]["total_cost_mean"],
                "total_cost_std": mc["summary"]["total_cost_std"],
                "stockout_risk_pct": mc["summary"]["stockout_risk"] * 100.0,
            }
        }

