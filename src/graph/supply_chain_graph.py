"""
Supply Chain Graph Analytics.

Models the supply chain as a directed weighted graph:
  Nodes: Warehouses, Distribution Centers, Stores
  Edges: Transportation routes with capacity/cost/time attributes

Analyses:
  - Bottleneck detection (minimum cut / max flow)
  - Critical node identification (betweenness centrality)
  - Failure propagation simulation
  - Shortest path optimization
  - Network resilience scoring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class NetworkEdge:
    source: str
    target: str
    capacity: float          # Max units per day
    transit_time_days: float
    cost_per_unit: float
    distance_km: float
    transport_mode: str      # "truck", "rail", "air", "sea"
    co2_per_unit_km: float = 0.0  # kg CO₂ per unit per km


class SupplyChainGraph:
    """
    Directed graph representation of the supply chain network.

    Supports both nx.DiGraph (analysis) and nx.MultiDiGraph (multi-route).
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._log = get_logger("SupplyChainGraph")

    def add_node(
        self,
        node_id: str,
        node_type: str,
        **attrs: Any,
    ) -> None:
        self.G.add_node(node_id, node_type=node_type, **attrs)

    def add_edge(self, edge: NetworkEdge) -> None:
        self.G.add_edge(
            edge.source, edge.target,
            capacity=edge.capacity,
            transit_time=edge.transit_time_days,
            cost=edge.cost_per_unit,
            distance=edge.distance_km,
            mode=edge.transport_mode,
            co2=edge.co2_per_unit_km * edge.distance_km,
            weight=edge.cost_per_unit,  # Default weight for shortest path
        )

    def build_from_dataframes(
        self,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame,
    ) -> "SupplyChainGraph":
        """
        Build graph from DataFrames.

        df_nodes: node_id, node_type, [lat, lon, capacity, ...]
        df_edges: source, target, capacity, transit_time_days, cost_per_unit, distance_km
        """
        for _, row in df_nodes.iterrows():
            self.add_node(
                row["node_id"],
                row["node_type"],
                **{k: v for k, v in row.items() if k not in ("node_id", "node_type")},
            )

        for _, row in df_edges.iterrows():
            self.add_edge(NetworkEdge(
                source=row["source"],
                target=row["target"],
                capacity=float(row.get("capacity", 10000)),
                transit_time_days=float(row.get("transit_time_days", 1)),
                cost_per_unit=float(row.get("cost_per_unit", 1)),
                distance_km=float(row.get("distance_km", 100)),
                transport_mode=row.get("transport_mode", "truck"),
                co2_per_unit_km=float(row.get("co2_per_unit_km", 0.1)),
            ))

        self._log.info(
            "graph_built",
            nodes=self.G.number_of_nodes(),
            edges=self.G.number_of_edges(),
        )
        return self

    # -----------------------------------------------------------------------
    # Bottleneck Analysis
    # -----------------------------------------------------------------------

    def find_bottlenecks(
        self,
        source_nodes: list[str] | None = None,
        sink_nodes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Identify bottleneck edges using max-flow / min-cut analysis.

        A bottleneck is an edge whose removal most reduces max flow
        from suppliers to stores.
        """
        results = []

        # Get source and sink nodes by type
        if source_nodes is None:
            source_nodes = [n for n, d in self.G.nodes(data=True)
                           if d.get("node_type") in ("supplier", "warehouse")]
        if sink_nodes is None:
            sink_nodes = [n for n, d in self.G.nodes(data=True)
                         if d.get("node_type") == "store"]

        # Add virtual super-source and super-sink
        G_aug = self.G.copy()
        G_aug.add_node("__source__")
        G_aug.add_node("__sink__")
        for s in source_nodes:
            G_aug.add_edge("__source__", s, capacity=float("inf"))
        for t in sink_nodes:
            G_aug.add_edge(t, "__sink__", capacity=float("inf"))

        # Max flow
        try:
            flow_value, flow_dict = nx.maximum_flow(
                G_aug, "__source__", "__sink__", capacity="capacity"
            )

            # Find minimum cut edges
            _, partition = nx.minimum_cut(
                G_aug, "__source__", "__sink__", capacity="capacity"
            )
            reachable, non_reachable = partition
            cut_edges = [
                (u, v) for u in reachable for v in G_aug.successors(u)
                if v in non_reachable and u != "__source__"
            ]

            for u, v in cut_edges:
                edge_data = self.G.get_edge_data(u, v, default={})
                results.append({
                    "source": u,
                    "target": v,
                    "capacity": edge_data.get("capacity", 0),
                    "is_bottleneck": True,
                    "transport_mode": edge_data.get("mode", "unknown"),
                    "bottleneck_type": "min_cut",
                })

        except Exception as e:
            self._log.warning("max_flow_failed", error=str(e))

        return pd.DataFrame(results) if results else pd.DataFrame(
            columns=["source", "target", "capacity", "is_bottleneck"]
        )

    # -----------------------------------------------------------------------
    # Centrality Analysis
    # -----------------------------------------------------------------------

    def compute_centrality(self) -> pd.DataFrame:
        """
        Compute multiple centrality metrics for all nodes.
        High betweenness = critical relay node (single point of failure).
        """
        betweenness = nx.betweenness_centrality(self.G, weight="cost", normalized=True)
        in_degree = dict(self.G.in_degree())
        out_degree = dict(self.G.out_degree())

        # Closeness centrality (how quickly a node can reach all others)
        try:
            closeness = nx.closeness_centrality(self.G)
        except Exception:
            closeness = {n: 0.0 for n in self.G.nodes()}

        rows = []
        for node in self.G.nodes():
            node_data = self.G.nodes[node]
            rows.append({
                "node_id": node,
                "node_type": node_data.get("node_type", "unknown"),
                "betweenness_centrality": round(betweenness.get(node, 0), 6),
                "closeness_centrality": round(closeness.get(node, 0), 6),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "is_critical": betweenness.get(node, 0) > 0.3,
            })

        df = pd.DataFrame(rows).sort_values("betweenness_centrality", ascending=False)
        return df

    # -----------------------------------------------------------------------
    # Failure Propagation
    # -----------------------------------------------------------------------

    def simulate_node_failure(
        self, failed_node: str
    ) -> dict[str, Any]:
        """
        Simulate removing a node and measure network impact.
        Returns metrics on how many stores lose connectivity.
        """
        G_failed = self.G.copy()
        G_failed.remove_node(failed_node)

        all_stores = [n for n, d in self.G.nodes(data=True)
                     if d.get("node_type") == "store"]
        all_suppliers = [n for n, d in self.G.nodes(data=True)
                        if d.get("node_type") in ("supplier", "warehouse")]

        affected_stores = []
        for store in all_stores:
            if store not in G_failed:
                affected_stores.append(store)
                continue
            # Check if any supplier can still reach this store
            can_reach = any(
                nx.has_path(G_failed, supplier, store)
                for supplier in all_suppliers
                if supplier in G_failed
            )
            if not can_reach:
                affected_stores.append(store)

        return {
            "failed_node": failed_node,
            "total_stores": len(all_stores),
            "affected_stores": len(affected_stores),
            "pct_stores_affected": round(len(affected_stores) / max(len(all_stores), 1) * 100, 2),
            "affected_store_ids": affected_stores,
            "network_still_connected": nx.is_weakly_connected(G_failed) if G_failed.number_of_nodes() > 0 else False,
        }

    def rank_nodes_by_failure_impact(self) -> pd.DataFrame:
        """Rank all nodes by their failure impact score."""
        rows = []
        for node in self.G.nodes():
            impact = self.simulate_node_failure(node)
            rows.append({
                "node_id": node,
                "node_type": self.G.nodes[node].get("node_type", "unknown"),
                "pct_stores_affected": impact["pct_stores_affected"],
                "n_stores_affected": impact["affected_stores"],
            })
        return pd.DataFrame(rows).sort_values("pct_stores_affected", ascending=False)

    # -----------------------------------------------------------------------
    # Path Optimization
    # -----------------------------------------------------------------------

    def find_optimal_routes(
        self,
        source: str,
        targets: list[str],
        optimize_for: str = "cost",  # "cost", "time", "co2"
    ) -> pd.DataFrame:
        """Find lowest-cost/time/emission paths from source to all targets."""
        weight_map = {"cost": "cost", "time": "transit_time", "co2": "co2"}
        weight = weight_map.get(optimize_for, "cost")

        rows = []
        for target in targets:
            try:
                path = nx.shortest_path(self.G, source, target, weight=weight)
                path_weight = nx.shortest_path_length(self.G, source, target, weight=weight)
                rows.append({
                    "source": source,
                    "target": target,
                    "optimal_path": " → ".join(path),
                    "path_length": len(path) - 1,
                    "total_" + optimize_for: round(path_weight, 4),
                })
            except nx.NetworkXNoPath:
                rows.append({
                    "source": source,
                    "target": target,
                    "optimal_path": "NO_PATH",
                    "path_length": -1,
                    "total_" + optimize_for: float("inf"),
                })

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Network Summary
    # -----------------------------------------------------------------------

    def network_summary(self) -> dict[str, Any]:
        """Summary statistics for the supply chain network."""
        node_types = pd.Series(
            [d.get("node_type", "unknown") for _, d in self.G.nodes(data=True)]
        ).value_counts().to_dict()

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_types": node_types,
            "is_connected": nx.is_weakly_connected(self.G),
            "average_degree": round(
                sum(d for _, d in self.G.degree()) / max(self.G.number_of_nodes(), 1), 2
            ),
            "network_density": round(nx.density(self.G), 4),
            "diameter": nx.diameter(self.G.to_undirected()) if nx.is_connected(self.G.to_undirected()) else -1,
        }
